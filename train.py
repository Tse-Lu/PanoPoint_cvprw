# -*- coding: utf-8 -*-
'''
Stage 2: PanoMotion Adaptation + Joint Training (PanoPoint paper Section 3).

Pipeline:
  1. PanoMotion Adaptation (Section 3.3):
     N=20 random PanoMotion warps, Stage-1 BaseDetector detects keypoints,
     inverse-warp to source, NMS 4px → pseudo-ground-truth keypoints.

  2. Joint Training (Section 3.4):
     Source + warped panoramas → PanoPoint → detection loss (CE) + descriptor loss (hinge).
     L_total = L_loc + 5 * L_desc  (Section 4.3, omega_1=1, omega_2=5)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import glob
import numpy as np
from net import BaseDetector
from pano_motion import (
    random_Rt, warp_panorama, warp_points_forward, batch_warp_points_forward,
    inverse_warp_points,
    batch_euler_to_R, equi_to_sphere, sphere_to_equi,
    sphere_ray_to_cube, ray_cube_intersection,
    CUBE_DIST, T_RANGE, R_RANGE,
)


# ══════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════

class Stage2Dataset(Dataset):
    def __init__(self, img_dir, pseudo_gt_dict=None):
        self.files = sorted(
            glob.glob(os.path.join(img_dir, '**', '*.png'), recursive=True) +
            glob.glob(os.path.join(img_dir, '**', '*.jpg'), recursive=True)
        )
        self.pseudo_gt = pseudo_gt_dict if pseudo_gt_dict else {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (960, 480))
        img_t = torch.from_numpy(img).float().unsqueeze(0) / 255.0

        key = self.files[idx]
        if key in self.pseudo_gt:
            kpts = torch.from_numpy(self.pseudo_gt[key]).float()
        else:
            kpts = torch.zeros((0, 3))
        return img_t, kpts


def collate_fn(batch):
    images, kpts_list = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(kpts_list)


# ══════════════════════════════════════════════════════════════════════
# Detection helpers (for Stage-1 model, no calibration hack)
# ══════════════════════════════════════════════════════════════════════

def detect_keypoints_stage1(model, image_tensor, conf_thresh=0.015, nms_dist=4):
    '''
    Detect keypoints using Stage-1 model (no calibration hack).
    Returns: (N, 3) tensor of [x, y, score]
    '''
    with torch.no_grad():
        logits, _ = model(image_tensor)

    prob = F.softmax(logits, dim=1)
    nodust = prob[:, :-1, :, :]
    B, C, Hf, Wf = nodust.shape

    heatmap = nodust.permute(0, 2, 3, 1).reshape(B, Hf, Wf, 8, 8)
    heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, Hf * 8, Wf * 8)

    kernel = 2 * nms_dist + 1
    pad = nms_dist
    max_pooled = F.max_pool2d(heatmap.unsqueeze(1), kernel_size=kernel,
                               stride=1, padding=pad)
    keep = (heatmap.unsqueeze(1) == max_pooled).float()
    heatmap_nms = heatmap * keep.squeeze(1)

    mask = heatmap_nms > conf_thresh
    indices = torch.where(mask[0])
    if indices[0].shape[0] == 0:
        return torch.zeros((0, 3), device=image_tensor.device)

    scores = heatmap_nms[0][indices[0], indices[1]]
    pts = torch.stack([indices[1].float(), indices[0].float(), scores], dim=-1)
    return pts


def nms_keypoints(kpts, dist=4.0):
    '''NMS on keypoints with Euclidean distance threshold.'''
    if kpts.shape[0] <= 1:
        return kpts

    scores = kpts[:, 2]
    order = torch.argsort(scores, descending=True)
    kpts = kpts[order]

    keep = []
    while kpts.shape[0] > 0:
        keep.append(kpts[0:1])
        if kpts.shape[0] == 1:
            break
        d = torch.norm(kpts[1:, :2] - kpts[0:1, :2], dim=1)
        kpts = kpts[1:][d > dist]

    if keep:
        return torch.cat(keep, dim=0)
    return torch.zeros((0, 3), device=kpts.device)


# ══════════════════════════════════════════════════════════════════════
# PanoMotion Adaptation (Section 3.3)
# ══════════════════════════════════════════════════════════════════════

def generate_pseudo_gt(image_tensor, stage1_model, N=20, conf_thresh=0.015,
                       nms_dist=4, H=480, W=960, device='cuda'):
    '''
    Generate pseudo-ground-truth keypoints for one source panorama.

    Paper Section 3.3:
      N warps via PanoMotion → Stage-1 model detects keypoints on each warp →
      inverse-warp to source → collect all → NMS 4px.

    Args:
        image_tensor: (1, H, W) grayscale in [0, 1]
        stage1_model: BaseDetector with Stage-1 weights, in eval mode
    Returns:
        (M, 3) tensor of [x, y, score] on source image
    '''
    all_kpts = []

    for _ in range(N):
        R, t = random_Rt(1, device=device)
        R_s = R.squeeze(0)
        t_s = t.squeeze(0)

        img_4d = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            warped = warp_panorama(img_4d, R_s.unsqueeze(0), t_s.unsqueeze(0))

        kpts = detect_keypoints_stage1(stage1_model, warped, conf_thresh, nms_dist)
        if kpts.shape[0] == 0:
            continue

        kpts_src_xy = inverse_warp_points(kpts[:, :2], R_s, t_s, H, W)
        valid_x = (kpts_src_xy[:, 0] > 0) & (kpts_src_xy[:, 0] < W - 1)
        valid_y = (kpts_src_xy[:, 1] > 0) & (kpts_src_xy[:, 1] < H - 1)
        valid = valid_x & valid_y

        if valid.any():
            all_kpts.append(torch.cat([
                kpts_src_xy[valid], kpts[valid, 2:3]
            ], dim=-1))

    if not all_kpts:
        return torch.zeros((0, 3), device=device)

    merged = torch.cat(all_kpts, dim=0)
    return nms_keypoints(merged, dist=nms_dist)


def compute_pseudo_gt_all(img_dir, stage1_weights_path, N=20, device='cuda',
                          cache_dir=None):
    '''
    Pre-compute pseudo-GT for all images. Returns dict: path → np.array (M, 3).
    If cache_dir is provided, saves/loads .npy files.
    '''
    from net import BaseDetector

    model = BaseDetector().to(device)
    ckpt = torch.load(stage1_weights_path, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    files = sorted(
        glob.glob(os.path.join(img_dir, '**', '*.png'), recursive=True) +
        glob.glob(os.path.join(img_dir, '**', '*.jpg'), recursive=True)
    )

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    pseudo_gt = {}
    for i, fpath in enumerate(files):
        if cache_dir:
            cache_path = os.path.join(cache_dir, f'{i:08d}.npy')
            if os.path.exists(cache_path):
                pseudo_gt[fpath] = np.load(cache_path)
                continue

        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (960, 480))
        img_t = torch.from_numpy(img).float().unsqueeze(0) / 255.0

        kpts = generate_pseudo_gt(img_t, model, N=N, device=device)
        kpts_np = kpts.cpu().numpy()

        if cache_dir:
            np.save(os.path.join(cache_dir, f'{i:08d}.npy'), kpts_np)
        pseudo_gt[fpath] = kpts_np

        if (i + 1) % 100 == 0:
            print(f'  Pseudo-GT: {i+1}/{len(files)} ({kpts_np.shape[0]} kpts)')

    print(f'Pseudo-GT computed for {len(pseudo_gt)} images')
    return pseudo_gt


# ══════════════════════════════════════════════════════════════════════
# SuperPoint label encoding
# ══════════════════════════════════════════════════════════════════════

def pts_to_sp_labels(kpts_list, H=480, W=960, device='cuda'):
    '''
    Convert sparse keypoints to SuperPoint 65-class label grid.

    kpts_list: list of (N_k, 3) tensors [x, y, score]
    Returns: (B, H/8, W/8) long tensor, values 0..63 for sub-pixel position, 64 for dustbin.
    '''
    B = len(kpts_list)
    Hc, Wc = H // 8, W // 8
    labels = torch.full((B, Hc, Wc), 64, dtype=torch.long, device=device)

    for b, kpts in enumerate(kpts_list):
        if kpts.shape[0] == 0:
            continue
        # Sort ascending: last occurrence (highest score) wins in scatter
        order = torch.argsort(kpts[:, 2])
        kpts = kpts[order]
        x, y = kpts[:, 0], kpts[:, 1]
        cx = (x / 8).long().clamp(0, Wc - 1)
        cy = (y / 8).long().clamp(0, Hc - 1)
        sx = (x % 8).long()
        sy = (y % 8).long()

        indices = cy * Wc + cx
        labels[b].view(-1)[indices] = sy * 8 + sx

    return labels


# ══════════════════════════════════════════════════════════════════════
# Descriptor sampling
# ══════════════════════════════════════════════════════════════════════

def sample_descriptors(desc, points, H=480, W=960, cell_level=False):
    '''
    Sample descriptors at specified pixel coordinates.

    desc: (256, H_desc, W_desc) descriptor map (L2-normalized) [single image]
    points: (N, 2) tensor of (x, y) pixel coords
    cell_level: if True, desc is at H/8×W/8 and points are in pixel coords (auto-scaled)
    Returns: (N, 256) tensor of descriptors
    '''
    if points.shape[0] == 0:
        return torch.zeros((0, 256), device=desc.device)

    H_desc, W_desc = desc.shape[1], desc.shape[2]

    if cell_level:
        grid_x = (points[:, 0] / 8) / (W_desc - 1) * 2 - 1
        grid_y = (points[:, 1] / 8) / (H_desc - 1) * 2 - 1
    else:
        grid_x = points[:, 0] / (W - 1) * 2 - 1
        grid_y = points[:, 1] / (H - 1) * 2 - 1

    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(2)

    sampled = F.grid_sample(desc.unsqueeze(0), grid, mode='bilinear',
                            align_corners=True)
    return sampled.squeeze(0).squeeze(-1).T


# ══════════════════════════════════════════════════════════════════════
# Descriptor loss (Section 4.3, Eq. 11)
# ══════════════════════════════════════════════════════════════════════

def descriptor_loss(desc_s, desc_d, kpts_list, R, t,
                    H=480, W=960, mp=1.0, mn=0.2,
                    max_neg_pairs=250, cell_level=True):
    '''
    Hinge loss for descriptors.

    Paper Eq. 11:
      L_d = sum(c_k * max(0, 1 - f_s^T * f_d) + (1-c_k) * max(0, f_s^T * f_d - 0.2))

    Args:
        desc_s, desc_d: (B, 256, Hd, Wd) descriptor maps
        kpts_list: list of (N_k, 3) tensors — pseudo-GT keypoints on source
        R: (B, 3, 3) training rotation matrices
        t: (B, 3) training translation vectors
        mp: positive margin (default 1.0)
        mn: negative margin (default 0.2)
        cell_level: if True, desc is at H/8×W/8 (cell resolution)
    Returns:
        scalar loss
    '''
    B = desc_s.shape[0]
    device = desc_s.device
    pos_loss = torch.tensor(0.0, device=device)
    neg_loss = torch.tensor(0.0, device=device)

    pts_src_list = [k[:, :2] for k in kpts_list]
    pts_dst_list = batch_warp_points_forward(pts_src_list, R, t, H, W)

    for b in range(B):
        kpts = kpts_list[b]
        if kpts.shape[0] == 0:
            continue

        pts_src = pts_src_list[b]
        pts_dst = pts_dst_list[b]

        # Filter points with valid correspondences
        valid_x = (pts_dst[:, 0] > 0) & (pts_dst[:, 0] < W - 1)
        valid_y = (pts_dst[:, 1] > 0) & (pts_dst[:, 1] < H - 1)
        valid = valid_x & valid_y

        if not valid.any():
            continue

        pts_src_v = pts_src[valid]
        pts_dst_v = pts_dst[valid]
        K = pts_src_v.shape[0]

        # Sample descriptors
        d_s = sample_descriptors(desc_s[b], pts_src_v, H, W, cell_level=cell_level)
        d_d = sample_descriptors(desc_d[b], pts_dst_v, H, W, cell_level=cell_level)

        # Positive loss: push dot product above mp
        sim = (d_s * d_d).sum(dim=-1)  # (K,)
        pos_loss = pos_loss + F.relu(mp - sim).sum()

        # Negative loss: randomly pair non-corresponding points
        if K > 1:
            n_neg = min(K, max_neg_pairs)
            idx_a = torch.randperm(K, device=device)[:n_neg]
            idx_b = torch.randperm(K, device=device)[:n_neg]
            # Re-sample if any indices match
            same = (idx_a == idx_b)
            while same.any():
                idx_b[same] = torch.randint(0, K, (same.sum().item(),), device=device)
                same = (idx_a == idx_b)

            neg_sim = (d_s[idx_a] * d_d[idx_b]).sum(dim=-1)
            neg_loss = neg_loss + F.relu(neg_sim - mn).sum()

    Hc, Wc = desc_s.shape[2], desc_s.shape[3]
    pos = pos_loss / (Hc * Wc + 1e-10)
    neg = neg_loss / (Hc * Wc + 1e-10)
    lambda_d = 250.0
    return lambda_d * (pos + neg)


# ══════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════

def train_panopoint_stage2(
    img_dir,
    stage1_weights,
    pseudo_gt_dir=None,
    num_epochs=100,
    batch_size=4,
    lr=1e-3,
    omega_1=1.0,
    omega_2=5.0,
    mp=1.0,
    mn=0.2,
    device='cuda',
    resume_from=None,
):
    '''
    Stage 2 joint training of PanoPoint following the paper.
    If resume_from is provided, load full model + optimizer state to continue.
    '''
    # ── 1. Load or compute pseudo-GT ──
    print('=' * 60)
    print('Stage 2: PanoPoint Joint Training')
    print('=' * 60)

    if pseudo_gt_dir and os.path.isdir(pseudo_gt_dir) and os.listdir(pseudo_gt_dir):
        print(f'Loading cached pseudo-GT from {pseudo_gt_dir} ...')
        files = sorted(
            glob.glob(os.path.join(img_dir, '**', '*.png'), recursive=True) +
            glob.glob(os.path.join(img_dir, '**', '*.jpg'), recursive=True)
        )
        pseudo_gt = {}
        for i, fpath in enumerate(files):
            cache_path = os.path.join(pseudo_gt_dir, f'{i:08d}.npy')
            if os.path.exists(cache_path):
                pseudo_gt[fpath] = np.load(cache_path)
        print(f'  Loaded {len(pseudo_gt)}/{len(files)} images with pseudo-GT')
    else:
        print('Computing pseudo-GT via PanoMotion Adaptation (N=20) ...')
        pseudo_gt = compute_pseudo_gt_all(
            img_dir, stage1_weights, N=20, device=device,
            cache_dir=pseudo_gt_dir
        )

    # ── 2. Dataset & DataLoader ──
    dataset = Stage2Dataset(img_dir, pseudo_gt)
    if len(dataset) == 0:
        raise FileNotFoundError(f'No images found in {img_dir}')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, collate_fn=collate_fn)

    print(f'Dataset: {len(dataset)} images, batch_size={batch_size}')

    # ── 3. Model ──
    model = BaseDetector().to(device)
    start_epoch = 0
    optimizer = None

    if resume_from and os.path.exists(resume_from):
        print(f'Resuming from {resume_from} ...')
        ckpt = torch.load(resume_from, map_location=device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            start_epoch = ckpt.get('epoch', 0)
            if 'optimizer' in ckpt:
                optimizer = optim.Adam(model.parameters(), lr=lr)
                optimizer.load_state_dict(ckpt['optimizer'])
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                print(f'  Loaded model + optimizer, starting from epoch {start_epoch}')
            else:
                print(f'  Loaded model (no optimizer), starting from epoch {start_epoch}')
        else:
            # Old format: raw state_dict
            model.load_state_dict(ckpt, strict=False)
            print('  Loaded model (old format, no optimizer state)')
    elif stage1_weights and os.path.exists(stage1_weights):
        print(f'Loading Stage-1 weights from {stage1_weights}')
        ckpt = torch.load(stage1_weights, map_location=device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    else:
        print('No weights provided, training from scratch')

    # ── 4. Optimizer (if not loaded from resume) ──
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda')

    # ── 5. Training ──
    print(f'Training: epochs {start_epoch+1}-{start_epoch+num_epochs}, lr={lr}, mp={mp}, mn={mn}')
    print('-' * 60)

    for epoch in range(num_epochs):
        current_epoch = start_epoch + epoch
        torch.cuda.empty_cache()
        model.train()
        epoch_loss = 0.0
        epoch_loss_loc = 0.0
        epoch_loss_desc = 0.0

        for batch_idx, (images_src, kpts_list) in enumerate(loader):
            images_src = images_src.to(device, non_blocking=True)

            kpts_list = [k.to(device) if k.device != torch.device(device)
                         else k for k in kpts_list]

            B = images_src.shape[0]

            R_train, t_train = random_Rt(B, device=device)

            images_warped = warp_panorama(images_src, R_train, t_train)

            with torch.amp.autocast('cuda'):
                logits_s, desc_s = model(images_src)
                logits_d, desc_d = model(images_warped)

                labels = pts_to_sp_labels(kpts_list, device=device)
                loss_loc = F.cross_entropy(logits_s, labels)
                Hc, Wc = logits_s.shape[2], logits_s.shape[3]
                loss_loc = loss_loc * (64.0 / (Hc * Wc))

                loss_desc = descriptor_loss(
                    desc_s, desc_d, kpts_list, R_train, t_train,
                    H=480, W=960, mp=mp, mn=mn, cell_level=False
                )

                loss = omega_1 * loss_loc + omega_2 * loss_desc

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_loss_loc += loss_loc.item()
            epoch_loss_desc += loss_desc.item()

        avg = epoch_loss / len(loader)
        avg_loc = epoch_loss_loc / len(loader)
        avg_desc = epoch_loss_desc / len(loader)

        print(f'Epoch {current_epoch+1:3d}  '
              f'loss={avg:.4f}  loc={avg_loc:.4f}  desc={avg_desc:.4f}')

        if (current_epoch + 1) % 10 == 0:
            save_path = f'panopoint_final.pth'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': current_epoch + 1,
            }, save_path)
            print(f'  -> Saved {save_path} (epoch {current_epoch+1})')

    # Save final checkpoint
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': start_epoch + num_epochs,
    }, 'panopoint_final.pth')
    print('Training complete. Model saved to panopoint_final.pth')
    return model


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PanoPoint Stage 2 Training')
    parser.add_argument('--img_dir', type=str, default='train_CVPG',
                        help='Directory of training images')
    parser.add_argument('--stage1', type=str, default='panopoint_stage1_last.pth',
                        help='Path to Stage-1 weights (ignored if --resume is set)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from a checkpoint (panopoint_final.pth)')
    parser.add_argument('--pseudo_gt_dir', type=str, default='./pseudo_gt_cache',
                        help='Directory to cache pseudo-GT')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (4 for ~8GB GPU, increase if more VRAM)')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.device = 'cpu'
        print('CUDA not available, using CPU')

    train_panopoint_stage2(
        img_dir=args.img_dir,
        stage1_weights=args.stage1,
        pseudo_gt_dir=args.pseudo_gt_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        resume_from=args.resume,
    )
