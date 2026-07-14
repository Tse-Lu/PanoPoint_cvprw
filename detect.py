import torch
import torch.nn.functional as F
import cv2
import numpy as np
from .net import BaseDetector  

class PanoPointDetector:
    def __init__(self, weights_path, device='cuda', conf_thresh=0.01, nms_dist=8, max_keypoints=1500):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = BaseDetector().to(self.device)
        
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        self.conf_thresh = conf_thresh
        self.nms_dist = nms_dist
        self.max_keypoints = max_keypoints

    def preprocess(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        orig_shape = img.shape
        img_resized = cv2.resize(img, (960, 480))
        img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        return img_tensor.to(self.device), img_resized, orig_shape

    def nms(self, corners, dist_thresh):
        if len(corners) == 0: return []
        indices = np.argsort(corners[:, 2])[::-1]
        corners = corners[indices]
        
        keep = []
        while len(corners) > 0:
            curr = corners[0]
            keep.append(curr)
            if len(corners) == 1: break
           
            dist = np.linalg.norm(corners[1:, :2] - curr[:2], axis=1)
            corners = corners[1:][dist > dist_thresh]
        return np.array(keep)

    def detect(self, img_tensor):
        with torch.no_grad():
            logits, desc = self.model(img_tensor)

            # manual calibration: the detector head BN is poorly trained (running_var ~2000),
            # causing dustbin to dominate softmax. We calibrate by scaling the dustbin logit down.
            dustbin_logit = logits[:, 64:65, :, :]
            spatial_logits = logits[:, :64, :, :]
            # reduce dustbin logit so it doesn't suppress spatial bins
            calibrated = torch.cat([spatial_logits * 2.0, dustbin_logit * 0.5], dim=1)

            prob = F.softmax(calibrated, dim=1)
            nodust = prob[:, :-1, :, :]
            B, C, H, W = nodust.shape
            heatmap = nodust.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
            heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, H*8, W*8) # [1, 480, 960]

            def fast_nms(heatmap, dist):
                kernel = 2 * dist + 1
                pad = dist
                max_pooled = F.max_pool2d(heatmap.unsqueeze(1), kernel_size=kernel, stride=1, padding=pad)
                keep = (heatmap.unsqueeze(1) == max_pooled).float()
                return heatmap * keep.squeeze(1)

            heatmap_nms = fast_nms(heatmap, self.nms_dist)
            
            mask = heatmap_nms > self.conf_thresh
            pts_indices = torch.where(mask[0]) 
            
            if pts_indices[0].shape[0] == 0:
                return np.array([]), np.array([])
            
            scores = heatmap_nms[0][pts_indices[0], pts_indices[1]] # [N]
            pts = torch.stack([pts_indices[1], pts_indices[0], scores], dim=-1)

            # Global top-K selection to prevent clustering in small regions
            if pts.shape[0] > self.max_keypoints:
                _, keep_idx = torch.topk(pts[:, 2], self.max_keypoints)
                pts = pts[keep_idx]

            N = pts.shape[0]
            grid_x = (pts[:, 0:1] / (W*8 - 1)) * 2 - 1
            grid_y = (pts[:, 1:2] / (H*8 - 1)) * 2 - 1
            grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, 1, N, 2) # [1, 1, N, 2]
            
            descriptors = F.grid_sample(desc, grid, mode='bilinear', align_corners=True)
            descriptors = F.normalize(descriptors.reshape(256, N), p=2, dim=0).t() 

            return pts.cpu().numpy(), descriptors.cpu().numpy()

def visualize_matches(img1_path, img2_path, weights_path):
    detector = PanoPointDetector(weights_path)
    
    tensor1, show1, _ = detector.preprocess(img1_path)
    pts1, desc1 = detector.detect(tensor1)
    
    tensor2, show2, _ = detector.preprocess(img2_path)
    pts2, desc2 = detector.detect(tensor2)
    

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    kp1 = [cv2.KeyPoint(p[0], p[1], 1) for p in pts1]
    kp2 = [cv2.KeyPoint(p[0], p[1], 1) for p in pts2]

    res = cv2.drawMatches(show1, kp1, show2, kp2, good_matches, None,
                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv2.imshow("PanoPoint Matches", res)
    cv2.waitKey(0)
    cv2.imwrite("matches_result.jpg", res)

if __name__ == "__main__":
    visualize_matches("hongkong_wanchai_32.jpg", "hongkong_wanchai_33.jpg", "panopoint_final.pth")