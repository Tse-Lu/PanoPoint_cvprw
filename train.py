'''
train on real-world dataset
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
from torch.amp import autocast, GradScaler
from net import BaseDetector


class PanoMotionParallel:
    @staticmethod
    def get_batch_grid(B, N, H, W, device):
        yaws  = (torch.rand(B * N, 1, 1, 1, device=device) * 2 - 1)
        pitchs = (torch.rand(B * N, 1, 1, 1, device=device) * 0.4 - 0.2)
        rolls  = (torch.rand(B * N, 1, 1, 1, device=device) * 0.2 - 0.1)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
        grid = grid.repeat(B * N, 1, 1, 1)

        # yaw shifts x; pitch shifts y; roll (small) distorts
        grid[:, :, :, 0] = grid[:, :, :, 0] + yaws.squeeze(-1)
        grid[:, :, :, 0] = torch.remainder(grid[:, :, :, 0] + 1, 2) - 1
        grid[:, :, :, 1] = torch.clamp(grid[:, :, :, 1] + pitchs.squeeze(-1), -1, 1)

        return grid, (yaws, pitchs, rolls)

    @classmethod
    def generate_pseudo_gt(cls, model, images, N=20):
        B, C, H, W = images.shape
        device = images.device
        h_f, w_f = H // 8, W // 8

        images_ext = images.repeat_interleave(N, dim=0)

        grid_forward, (yaws, pitch, roll) = cls.get_batch_grid(B, N, H, W, device)
        warped_images = F.grid_sample(images_ext, grid_forward, mode='bilinear', align_corners=True)

        model.eval()
        with torch.no_grad():
            feat = model.backbone(warped_images)
            logits = model.detector(feat)
            probs = F.softmax(logits, dim=1)

        grid_inverse = cls.get_inverse_grid(B, N, h_f, w_f, device, yaws, pitch, roll)

        probs_restored = F.grid_sample(probs, grid_inverse, mode='bilinear', align_corners=True)

        avg_probs = probs_restored.view(B, N, 65, h_f, w_f).mean(dim=1)

        pseudo_labels = torch.argmax(avg_probs, dim=1)

        # Suppress edge artifacts: set dustbin for left/right 2 cells (16px)
        edge_cells = 2
        pseudo_labels[:, :, :edge_cells] = 64
        pseudo_labels[:, :, -edge_cells:] = 64

        return pseudo_labels

    @staticmethod
    def get_inverse_grid(B, N, H, W, device, yaws, pitch, roll):
        inv_yaws  = -yaws
        inv_pitch = -pitch
        inv_roll  = -roll
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B * N, 1, 1, 1)
        grid[:, :, :, 0] = torch.remainder(grid[:, :, :, 0] + inv_yaws.squeeze(-1) + 1, 2) - 1
        grid[:, :, :, 1] = torch.clamp(grid[:, :, :, 1] + inv_pitch.squeeze(-1), -1, 1)
        return grid

    @staticmethod
    def get_batch_grid_with_yaw(B, N, H, W, device, yaws):
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B * N, 1, 1, 1)
        grid[:, :, :, 0] = torch.remainder(grid[:, :, :, 0] + yaws.squeeze(-1) + 1, 2) - 1
        return grid, yaws


class SimplePanoDataset(Dataset):
    def __init__(self, img_dir):
        self.files = sorted(
            glob.glob(os.path.join(img_dir, "**", "*.png"), recursive=True) +
            glob.glob(os.path.join(img_dir, "**", "*.jpg"), recursive=True)
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (960, 480))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0)


def train_panopoint_full(img_dir, weights_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaseDetector().to(device)

    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        # Reset detector BN stats (stage1 BN running_var was broken)
        detector_bn = model.detector.head[1]
        detector_bn.reset_running_stats()
        model.train()
        dummy_feat = torch.randn(4, 128, 60, 120, device=device)
        for _ in range(50):
            _ = model.detector(dummy_feat)
        model.eval()

    loader = DataLoader(SimplePanoDataset(img_dir), batch_size=8, shuffle=True, num_workers=2)

    # Separate LR: detector head 0.1x to prevent logit drift
    backbone_params = []
    detector_params = []
    descriptor_params = []
    for name, param in model.named_parameters():
        if 'detector' in name:
            detector_params.append(param)
        elif 'descriptor' in name:
            descriptor_params.append(param)
        else:
            backbone_params.append(param)
    optimizer = optim.Adam([
        {'params': backbone_params},
        {'params': detector_params, 'lr': 1e-5},
        {'params': descriptor_params, 'lr': 3e-5},
    ], lr=3e-5)
    scaler = GradScaler('cuda')

    cls_weights = torch.ones(65, device=device)
    cls_weights[64] = 0.1

    for epoch in range(100):
        optimizer.zero_grad()
        epoch_loss = 0.0

        for batch_idx, images in enumerate(loader):
            images = images.to(device)
            B = images.shape[0]

            with torch.no_grad():
                v_gt_original = PanoMotionParallel.generate_pseudo_gt(model, images, N=10)

            grid_train, (yaws_train, _, _) = PanoMotionParallel.get_batch_grid(B, 1, 480, 960, device)
            images_warped = F.grid_sample(images, grid_train, mode='bilinear', align_corners=True)

            v_gt_float = v_gt_original.unsqueeze(1).float()
            grid_train_feat, _ = PanoMotionParallel.get_batch_grid_with_yaw(B, 1, 60, 120, device, yaws_train)
            v_gt_warped = F.grid_sample(v_gt_float, grid_train_feat, mode='nearest', align_corners=True)
            v_gt_warped = v_gt_warped.squeeze(1).long()

            model.train()
            with autocast('cuda'):
                logits, _ = model(images_warped)
                loss = F.cross_entropy(logits, v_gt_warped, weight=cls_weights)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        print(f"Epoch {epoch} with average Loss: {epoch_loss / len(loader):.8f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), "panopoint_final.pth")


train_panopoint = train_panopoint_full  # alias for train_first import

if __name__ == "__main__":
    train_panopoint_full(img_dir="train_CVPG", weights_path="panopoint_stage1_last.pth")
