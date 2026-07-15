# -*- coding: utf-8 -*-
'''
PanoMotion: 3D cube-geometry warping for 360 panoramas.
Section 3.2 of the PanoPoint paper (CVPR 2023 Workshop).

Geometry:
  S1 (source) at world origin, S2 (warped) at position t with orientation R.
  Cube Π centered at origin, faces at distance cube_dist=10.
  Equirectangular → sphere → cube intersection → R/t transform → sphere → equirectangular.
'''
import torch
import torch.nn.functional as F
import numpy as np


CUBE_DIST = 10
T_RANGE = 6.0
R_RANGE = np.pi

# Module-level equirectangular-to-sphere direction grid cache
_cached_d_cam = None
_cached_HW = None
_cached_device = None


def _get_cached_d_cam(H, W, device):
    global _cached_d_cam, _cached_HW, _cached_device
    if (_cached_d_cam is not None and _cached_HW == (H, W)
            and str(_cached_device) == str(device)):
        return _cached_d_cam

    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

    U = (grid_x + 1) / 2 * (W - 1)
    V = (grid_y + 1) / 2 * (H - 1)
    theta = U / (W - 1) * 2 * torch.pi - torch.pi
    phi = torch.pi / 2 - V / (H - 1) * torch.pi

    dx = torch.cos(phi) * torch.sin(theta)
    dy = torch.sin(phi)
    dz = torch.cos(phi) * torch.cos(theta)

    _cached_d_cam = torch.stack([dx, dy, dz], dim=-1)
    _cached_HW = (H, W)
    _cached_device = str(device)
    return _cached_d_cam


# ─── Rotation utilities ────────────────────────────────────────────────

def batch_euler_to_R(yaw, pitch, roll):
    '''
    yaw, pitch, roll: (B,) tensors, radians
    yaw   = rotation around Y (vertical)
    pitch = rotation around X (horizontal)
    roll  = rotation around Z (forward)
    R = Rz(roll) @ Rx(pitch) @ Ry(yaw)
    Returns: (B, 3, 3)
    '''
    B = yaw.shape[0]
    device = yaw.device
    dtype = yaw.dtype
    zeros = torch.zeros(B, device=device, dtype=dtype)
    ones = torch.ones(B, device=device, dtype=dtype)

    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)

    Ry00, Ry01, Ry02 = cy, zeros, sy
    Ry10, Ry11, Ry12 = zeros, ones, zeros
    Ry20, Ry21, Ry22 = -sy, zeros, cy

    Rx00, Rx01, Rx02 = ones, zeros, zeros
    Rx10, Rx11, Rx12 = zeros, cp, -sp
    Rx20, Rx21, Rx22 = zeros, sp, cp

    Rz00, Rz01, Rz02 = cr, -sr, zeros
    Rz10, Rz11, Rz12 = sr, cr, zeros
    Rz20, Rz21, Rz22 = zeros, zeros, ones

    Ry = torch.stack([
        torch.stack([Ry00, Ry01, Ry02], dim=-1),
        torch.stack([Ry10, Ry11, Ry12], dim=-1),
        torch.stack([Ry20, Ry21, Ry22], dim=-1),
    ], dim=1)
    Rx = torch.stack([
        torch.stack([Rx00, Rx01, Rx02], dim=-1),
        torch.stack([Rx10, Rx11, Rx12], dim=-1),
        torch.stack([Rx20, Rx21, Rx22], dim=-1),
    ], dim=1)
    Rz = torch.stack([
        torch.stack([Rz00, Rz01, Rz02], dim=-1),
        torch.stack([Rz10, Rz11, Rz12], dim=-1),
        torch.stack([Rz20, Rz21, Rz22], dim=-1),
    ], dim=1)

    return Rz @ Rx @ Ry


def rotation_matrix_to_euler(R):
    '''R: (3,3) or (B,3,3) -> (yaw, pitch, roll) each (B,) or scalar'''
    if R.dim() == 2:
        R = R.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    pitch = torch.asin(-R[:, 2, 0].clamp(-1, 1))
    yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    roll = torch.atan2(R[:, 2, 1], R[:, 2, 2])

    if squeeze:
        return yaw[0], pitch[0], roll[0]
    return yaw, pitch, roll


# ─── Equirectangular ↔ Sphere ──────────────────────────────────────────

def equi_to_sphere(pts, H=480, W=960):
    '''
    pts: (N, 2) tensor of (x, y) pixel coordinates in [0, W-1] x [0, H-1]
    Returns: (N, 3) unit direction vectors on the sphere
    '''
    theta = pts[..., 0] / (W - 1) * 2 * torch.pi - torch.pi
    phi = torch.pi / 2 - pts[..., 1] / (H - 1) * torch.pi

    dx = torch.cos(phi) * torch.sin(theta)
    dy = torch.sin(phi)
    dz = torch.cos(phi) * torch.cos(theta)
    return torch.stack([dx, dy, dz], dim=-1)


def sphere_to_equi(dirs, H=480, W=960):
    '''
    dirs: (N, 3) unit direction vectors
    Returns: (N, 2) pixel (x, y) coordinates
    '''
    dx, dy, dz = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    theta = torch.atan2(dx, dz)
    phi = torch.asin(dy.clamp(-1, 1))

    U = (theta + torch.pi) / (2 * torch.pi) * (W - 1)
    V = (torch.pi / 2 - phi) / torch.pi * (H - 1)
    return torch.stack([U, V], dim=-1)


# ─── Cube intersection ─────────────────────────────────────────────────

def sphere_ray_to_cube(directions, cube_dist=CUBE_DIST):
    '''
    Distance from origin along each direction to cube face.
    directions: (..., 3) unit vectors from origin
    Returns: (...) distance values μ
    '''
    abs_d = directions.abs()
    max_comp = abs_d.max(dim=-1)[0]
    return cube_dist / (max_comp + 1e-10)


def ray_cube_intersection(origins, directions, cube_dist=CUBE_DIST):
    '''
    Ray: origin + λ * direction intersects axis-aligned cube [-cube_dist, cube_dist]^3.

    origins: (..., 3) ray start positions
    directions: (..., 3) ray directions
    Returns: (...) λ values, inf for no intersection
    '''
    shape = origins.shape[:-1]
    device = directions.device
    lam_min = torch.full(shape, float('inf'), device=device)

    faces = [
        (0, 1, cube_dist),
        (0, -1, -cube_dist),
        (1, 1, cube_dist),
        (1, -1, -cube_dist),
        (2, 1, cube_dist),
        (2, -1, -cube_dist),
    ]

    for face_dim, sign_dir, target in faces:
        v = directions[..., face_dim]
        o_dim = origins[..., face_dim]
        lam = (target - o_dim) / (v + 1e-10)

        toward_face = v * sign_dir > 1e-6

        other_dims = [d for d in (0, 1, 2) if d != face_dim]
        od0, od1 = other_dims[0], other_dims[1]
        ip0 = origins[..., od0] + lam * directions[..., od0]
        ip1 = origins[..., od1] + lam * directions[..., od1]

        in_bounds = (ip0 >= -cube_dist - 1e-5) & (ip0 <= cube_dist + 1e-5) & \
                    (ip1 >= -cube_dist - 1e-5) & (ip1 <= cube_dist + 1e-5)

        mask = (lam > 1e-6) & toward_face & in_bounds
        lam_min = torch.where(mask & (lam < lam_min), lam, lam_min)

    return lam_min


# ─── Warp grid (inverse mapping: warped pixel → source pixel) ──────────

def build_warp_grid(R, t, H=480, W=960, cube_dist=CUBE_DIST):
    '''
    Build flow field for grid_sample: maps each warped pixel to source pixel.

    R: (B, 3, 3) rotation matrices
    t: (B, 3) translation vectors
    Returns: flow (B, H, W, 2) in normalized [-1, 1] coords for grid_sample
    '''
    B = R.shape[0]
    device = R.device

    d_cam = _get_cached_d_cam(H, W, device).unsqueeze(0).expand(B, H, W, 3)

    d_cam_flat = d_cam.reshape(B, H * W, 3)
    d_world_flat = torch.bmm(d_cam_flat, R.transpose(1, 2))
    d_world = d_world_flat.reshape(B, H, W, 3)

    t_exp = t.view(B, 1, 1, 3)
    lam = ray_cube_intersection(t_exp, d_world, cube_dist)

    lam_exp = lam.unsqueeze(-1)
    P_w = t_exp + lam_exp * d_world

    norm = torch.norm(P_w, dim=-1, keepdim=True)
    d_s1 = P_w / (norm + 1e-8)

    pts_src = sphere_to_equi(d_s1.reshape(-1, 3), H, W).reshape(B, H, W, 2)

    flow_x = pts_src[..., 0] / (W - 1) * 2 - 1
    flow_y = pts_src[..., 1] / (H - 1) * 2 - 1

    valid = lam < float('inf') - 1
    flow_x = torch.where(valid, flow_x, torch.full_like(flow_x, -2.0))
    flow_y = torch.where(valid, flow_y, torch.full_like(flow_y, -2.0))

    return torch.stack([flow_x, flow_y], dim=-1)


# ─── Warp images ────────────────────────────────────────────────────────

def warp_panorama(images, R, t, cube_dist=CUBE_DIST):
    '''
    Warp a batch of equirectangular panoramas using PanoMotion.

    images: (B, C, H, W) or (C, H, W)
    R: (B, 3, 3) or (3, 3)
    t: (B, 3) or (3,)
    Returns: warped images of same shape as input
    '''
    if images.dim() == 3:
        images = images.unsqueeze(0)
        squeeze_out = True
    else:
        squeeze_out = False
    if R.dim() == 2:
        R = R.unsqueeze(0)
    if t.dim() == 1:
        t = t.unsqueeze(0)

    B, C, H, W = images.shape
    flow = build_warp_grid(R, t, H, W, cube_dist)

    warped = F.grid_sample(images, flow, mode='bilinear',
                           padding_mode='zeros', align_corners=True)

    if squeeze_out:
        warped = warped.squeeze(0)
    return warped


# ─── Keypoint warping ───────────────────────────────────────────────────

def warp_points_forward(pts_src, R, t, H=480, W=960, cube_dist=CUBE_DIST):
    '''
    Forward warp: source keypoints → warped keypoints (Eq. 8 in paper).
    pts_src: (N, 2) tensor of (x, y) pixel coords on source
    R: (3, 3) rotation matrix
    t: (3,) translation vector
    Returns: (N, 2) tensor of (x, y) pixel coords on warped
    '''
    device = pts_src.device
    N = pts_src.shape[0]
    if N == 0:
        return pts_src

    d_src = equi_to_sphere(pts_src, H, W)

    mu = sphere_ray_to_cube(d_src, cube_dist)
    P_world = d_src * mu.unsqueeze(-1)

    R_inv = R.T
    P_s2 = (P_world - t.unsqueeze(0)) @ R_inv.T

    norm = torch.norm(P_s2, dim=-1, keepdim=True)
    d_s2 = P_s2 / (norm + 1e-8)

    pts_dst = sphere_to_equi(d_s2, H, W)
    return pts_dst


def batch_warp_points_forward(pts_src_list, R, t, H=480, W=960, cube_dist=CUBE_DIST):
    '''
    Batched forward warp: list of source keypoint sets → list of warped keypoint sets.

    pts_src_list: list of (N_b, 2) tensors of pixel coords on source, one per batch element
    R: (B, 3, 3) rotation matrices
    t: (B, 3) translation vectors
    Returns: list of (N_b, 2) tensors of pixel coords on warped
    '''
    device = R.device
    sizes = [p.shape[0] for p in pts_src_list]
    total_N = sum(sizes)
    if total_N == 0:
        return pts_src_list

    pts_all = torch.cat([p.to(device) for p in pts_src_list], dim=0)

    batch_indices = torch.cat([
        torch.full((sz,), i, dtype=torch.long, device=device)
        for i, sz in enumerate(sizes)
    ], dim=0)

    R_batch = R[batch_indices]
    t_batch = t[batch_indices]

    d_src = equi_to_sphere(pts_all, H, W)

    mu = sphere_ray_to_cube(d_src, cube_dist)
    P_world = d_src * mu.unsqueeze(-1)

    R_inv_batch = R_batch.transpose(1, 2)
    P_s2 = torch.bmm((P_world - t_batch).unsqueeze(1), R_inv_batch.transpose(1, 2)).squeeze(1)

    norm = torch.norm(P_s2, dim=-1, keepdim=True)
    d_s2 = P_s2 / (norm + 1e-8)

    pts_dst_all = sphere_to_equi(d_s2, H, W)

    pts_dst_list = list(torch.split(pts_dst_all, sizes, dim=0))
    return pts_dst_list


def inverse_warp_points(pts_dst, R, t, H=480, W=960, cube_dist=CUBE_DIST):
    '''
    Inverse warp: warped keypoints → source keypoints.
    pts_dst: (N, 2) tensor of (x, y) pixel coords on warped
    R: (3, 3) rotation matrix
    t: (3,) translation vector
    Returns: (N, 2) tensor of (x, y) pixel coords on source
    '''
    device = pts_dst.device
    N = pts_dst.shape[0]
    if N == 0:
        return pts_dst

    d_cam = equi_to_sphere(pts_dst, H, W)

    d_world = d_cam @ R.T

    lam = ray_cube_intersection(t.unsqueeze(0), d_world.unsqueeze(0), cube_dist).squeeze(0)

    P_world = t.unsqueeze(0) + lam.unsqueeze(-1) * d_world

    norm = torch.norm(P_world, dim=-1, keepdim=True)
    d_s1 = P_world / (norm + 1e-8)

    pts_src = sphere_to_equi(d_s1, H, W)
    return pts_src


# ─── Random (R, t) generation ───────────────────────────────────────────

def random_Rt(B, device='cuda', r_range=R_RANGE, t_range=T_RANGE):
    '''
    Generate B random (R, t) pairs for PanoMotion.
    R: yaw/pitch/roll ∈ [-r_range, r_range] → (B, 3, 3)
    t: each axis ∈ [-t_range, t_range] → (B, 3)
    '''
    yaw = (torch.rand(B, device=device) * 2 - 1) * r_range
    pitch = (torch.rand(B, device=device) * 2 - 1) * r_range
    roll = (torch.rand(B, device=device) * 2 - 1) * r_range
    R = batch_euler_to_R(yaw, pitch, roll)
    t = (torch.rand(B, 3, device=device) * 2 - 1) * t_range
    return R, t


# ─── Test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import cv2
    print('=== PanoMotion test ===')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    img = torch.ones(1, 1, 480, 960, device=device)

    R, t = random_Rt(1, device=str(device))
    yaw, pitch, roll = rotation_matrix_to_euler(R.squeeze(0))
    print(f'Yaw={yaw.item():.3f}  Pitch={pitch.item():.3f}  Roll={roll.item():.3f}')
    print(f't={t.squeeze(0).tolist()}')

    warped = warp_panorama(img, R, t)
    print(f'Input shape: {img.shape} → Warped shape: {warped.shape}')

    valid_pct = (warped > 0).float().mean().item() * 100
    print(f'Valid pixels in warped: {valid_pct:.1f}%')

    pts_src = torch.tensor([[480.0, 240.0]], device=device)
    pts_dst = warp_points_forward(pts_src, R.squeeze(0), t.squeeze(0))
    pts_back = inverse_warp_points(pts_dst, R.squeeze(0), t.squeeze(0))
    print(f'Source point: {pts_src.tolist()}')
    print(f'Warped point: {pts_dst.tolist()}')
    print(f'Round-trip:   {pts_back.tolist()}')

    print('\n=== Testing identity warp (R=I, t=0) ===')
    R_id = torch.eye(3, device=device).unsqueeze(0)
    t_zero = torch.zeros(1, 3, device=device)
    warped_id = warp_panorama(img, R_id, t_zero)
    diff = (warped_id - img).abs().max().item()
    print(f'Identity warp max diff: {diff:.6f}')
