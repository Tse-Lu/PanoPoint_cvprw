'''
Stage 1: Train BaseDetector on synthetic equirectangular dataset.
Uses CSG-based generator from gen_syn_pano.py for proper occlusion & intersection handling.
'''
import torch
import os
import numpy as np
import random

from torch.utils.data import Dataset, DataLoader
from net import BaseDetector
from gen_syn_pano import create_equirectangular_pano

class SyntheticPanoGenerator:
    def __init__(self, width=960, height=480, face_size=240, num_shapes_per_face=5):
        self.width = width
        self.height = height
        self.face_size = face_size
        self.num_shapes_per_face = num_shapes_per_face

    def generate_sample(self):
        img, kpts, sp_labels = create_equirectangular_pano(
            W=self.width, H=self.height,
            face_size=self.face_size,
            num_shapes_per_face=self.num_shapes_per_face
        )
        # BGR → grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_tensor = torch.from_numpy(gray).float().unsqueeze(0) / 255.0
        label_tensor = torch.from_numpy(sp_labels).long()
        return img_tensor, label_tensor


class SyntheticPanoDataset(Dataset):
    def __init__(self, num_samples=10000, width=960, height=480, cache_dir=None):
        self.num_samples = num_samples
        if cache_dir and os.path.isdir(cache_dir) and os.listdir(cache_dir):
            print(f"Loading pre-generated data from {cache_dir} ...")
            self.cache_dir = cache_dir
            self._use_cache = True
        else:
            self.generator = SyntheticPanoGenerator(width=width, height=height)
            self._use_cache = False

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self._use_cache:
            path = os.path.join(self.cache_dir, f"{idx:08d}.pt")
            return torch.load(path)
        return self.generator.generate_sample()


def get_pano_synthetic_dataloader(batch_size=8, num_workers=2, num_samples=10000):
    dataset = SyntheticPanoDataset(num_samples=num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True)


import cv2
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


def train_panopoint_synthetic(model, dataloader, num_epochs=200, device="cuda"):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    model.train()
    print("first stage (CSG generator):")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast():
                prob, _ = model(images)
                loss = F.cross_entropy(prob, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg = epoch_loss / len(dataloader)
        print(f"  Epoch {epoch+1}/{num_epochs}  |  loss = {avg:.4f}")
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"panopoint_stage1_epoch{epoch}.pth")


if __name__ == "__main__":
    import os, time
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    CACHE_DIR = "./syn_cache"
    NUM_SAMPLES = 2000

    # ── Pre-generate synthetic data to disk (CPU bottleneck solved) ──
    if not os.path.isdir(CACHE_DIR) or not os.listdir(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)
        gen = SyntheticPanoGenerator()
        print(f"Pre-generating {NUM_SAMPLES} samples to {CACHE_DIR} ...")
        t0 = time.time()
        for i in range(NUM_SAMPLES):
            img_t, label_t = gen.generate_sample()
            torch.save((img_t, label_t), os.path.join(CACHE_DIR, f"{i:08d}.pt"))
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{NUM_SAMPLES} ({NUM_SAMPLES/elapsed*(i+1)/60:.0f} min remaining)")
        print(f"Done in {time.time()-t0:.0f}s")
    else:
        print(f"Using cached data from {CACHE_DIR}/")

    # # ── Quick visual check (3 samples) ──
    # gen_viz = SyntheticPanoGenerator()
    # for i in range(3):
    #     img_t, label_map = gen_viz.generate_sample()
    #     img = img_t.squeeze().numpy()
    #     gt_pts = []
    #     Hf, Wf = label_map.shape
    #     for cy in range(Hf):
    #         for cx in range(Wf):
    #             v = int(label_map[cy, cx])
    #             if v < 64:
    #                 rx, ry = v % 8, v // 8
    #                 x, y = cx * 8 + rx, cy * 8 + ry
    #                 gt_pts.append([x, y])
    #     gt_pts = np.array(gt_pts)
    #     plt.figure(figsize=(12, 6))
    #     plt.imshow(img, cmap='gray')
    #     if len(gt_pts) > 0:
    #         plt.scatter(gt_pts[:, 0], gt_pts[:, 1], c='red', s=15, marker='x', linewidths=0.5)
    #     plt.title(f'Sample {i+1}: {len(gt_pts)} feature points')
    #     plt.show()


    # ── Training from cached data ──
    dataset = SyntheticPanoDataset(num_samples=NUM_SAMPLES, cache_dir=CACHE_DIR)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    base_detector = BaseDetector().to(device)
    train_panopoint_synthetic(base_detector, train_loader)
