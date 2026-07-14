import cv2
import numpy as np
import os
import random
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

def extract_vertices(geom):
    pts = []
    if geom.is_empty: return pts
    if geom.geom_type == 'Polygon':
        pts.extend(list(geom.exterior.coords)[:-1])
        for interior in geom.interiors:
            pts.extend(list(interior.coords)[:-1])
    elif geom.geom_type in ['MultiPolygon', 'GeometryCollection']:
        for g in geom.geoms:
            pts.extend(extract_vertices(g))
    return pts

def generate_single_face_csg(face_size=512, num_shapes=8, texture_dir=None):
    noise = (np.random.normal(0, 20, (face_size, face_size, 3))-20).astype(np.int16)
    if texture_dir and os.path.exists(texture_dir):
        texture_files = [f for f in os.listdir(texture_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if texture_files:
            tex_path = os.path.join(texture_dir, random.choice(texture_files))
            bg_img = cv2.imread(tex_path)
            if bg_img is not None:
                h, w = bg_img.shape[:2]
                if h > face_size and w > face_size:
                    top = random.randint(0, h - face_size)
                    left = random.randint(0, w - face_size)
                    img = bg_img[top:top+face_size, left:left+face_size].copy()
                else:
                    img = cv2.resize(bg_img, (face_size, face_size))
            else:
                img = np.clip(np.ones((face_size, face_size, 3), dtype=np.uint8) * 255 - noise, 0, 255).astype(np.uint8)
        else:
            img = np.clip(np.ones((face_size, face_size, 3), dtype=np.uint8) * 255 - noise, 0, 255).astype(np.uint8)
    else:
        img = np.clip(np.ones((face_size, face_size, 3), dtype=np.uint8) * 255 - noise, 0, 255).astype(np.uint8)

    polygons_list = []
    colors_list = []
    
    pad = 40 
    
    for _ in range(num_shapes):
        shape_type = random.choice(['rect', 'poly'])
        color = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
        
        if shape_type == 'rect':
            w, h = random.randint(50, 150), random.randint(50, 150)
            x, y = random.randint(pad, face_size-w-pad), random.randint(pad, face_size-h-pad)
            coords = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
        else:
            num_pts = random.randint(3, 5)
            cx, cy = random.randint(pad+60, face_size-pad-60), random.randint(pad+60, face_size-pad-60)
            coords = []
            for i in range(num_pts):
                angle = i * (2 * np.pi / num_pts) + random.uniform(0, 0.5)
                r = random.randint(40, 100)
                coords.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))
                
        poly = Polygon(coords)
        if poly.is_valid:
            polygons_list.append(poly)
            colors_list.append(color)

    for poly, color in zip(polygons_list, colors_list):
        cv2.fillPoly(img, [np.array(poly.exterior.coords, np.int32)], color)

    all_pts = []
    for i in range(len(polygons_list)):
        visible_geom = polygons_list[i]
        for j in range(i + 1, len(polygons_list)):
            visible_geom = visible_geom.difference(polygons_list[j])
        if not visible_geom.is_empty:
            clean_geom = visible_geom.simplify(0.01, preserve_topology=True)
            all_pts.extend(extract_vertices(clean_geom))

    valid_kpts = [(px, py) for px, py in all_pts if 0 <= px < face_size and 0 <= py < face_size]
    return img, list(set(valid_kpts))

def create_equirectangular_pano(W=1024, H=512, face_size=512, num_shapes_per_face=8):
    faces_imgs = []
    faces_kpts = []
    
    for i in range(6):
        img, kpts = generate_single_face_csg(face_size, num_shapes_per_face)
        faces_imgs.append(img)
        faces_kpts.append(kpts)
        
    U, V = np.meshgrid(np.arange(W), np.arange(H))
    theta = (U / (W - 1)) * 2 * np.pi - np.pi   
    phi = np.pi / 2 - (V / (H - 1)) * np.pi     

    X = np.cos(phi) * np.sin(theta)
    Y = np.sin(phi)
    Z = np.cos(phi) * np.cos(theta)

    xyz = np.stack([X, Y, Z], axis=-1)
    abs_xyz = np.abs(xyz)
    max_idx = np.argmax(abs_xyz, axis=-1)
    max_val = np.max(abs_xyz, axis=-1)
    max_val[max_val == 0] = 1
    
    x, y, z = X/max_val, Y/max_val, Z/max_val

    masks = [
        (max_idx == 2) & (Z > 0),  # Front
        (max_idx == 0) & (X > 0),  # Right
        (max_idx == 2) & (Z < 0),  # Back
        (max_idx == 0) & (X < 0),  # Left
        (max_idx == 1) & (Y > 0),  # Top
        (max_idx == 1) & (Y < 0)   # Bot
    ]

    u_face = np.zeros_like(x)
    v_face = np.zeros_like(y)

    u_face[masks[0]], v_face[masks[0]] = x[masks[0]], -y[masks[0]]
    u_face[masks[1]], v_face[masks[1]] = -z[masks[1]], -y[masks[1]]
    u_face[masks[2]], v_face[masks[2]] = -x[masks[2]], -y[masks[2]]
    u_face[masks[3]], v_face[masks[3]] = z[masks[3]], -y[masks[3]]
    u_face[masks[4]], v_face[masks[4]] = x[masks[4]], z[masks[4]]
    u_face[masks[5]], v_face[masks[5]] = x[masks[5]], -z[masks[5]]

    px_map = (u_face + 1) / 2 * (face_size - 1)
    py_map = (v_face + 1) / 2 * (face_size - 1)

    pano_img = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(6):
        remapped = cv2.remap(faces_imgs[i], px_map.astype(np.float32), py_map.astype(np.float32), 
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        pano_img[masks[i]] = remapped[masks[i]]

    pano_kpts = []
    for face_idx, kpts in enumerate(faces_kpts):
        for (px, py) in kpts:
            u = (px / (face_size - 1)) * 2 - 1
            v = (py / (face_size - 1)) * 2 - 1

            if face_idx == 0:   cX, cY, cZ = u, -v, 1
            elif face_idx == 1: cX, cY, cZ = 1, -v, -u
            elif face_idx == 2: cX, cY, cZ = -u, -v, -1
            elif face_idx == 3: cX, cY, cZ = -1, -v, u
            elif face_idx == 4: cX, cY, cZ = u, 1, v
            elif face_idx == 5: cX, cY, cZ = u, -1, -v

            R = np.sqrt(cX**2 + cY**2 + cZ**2)
            cX, cY, cZ = cX/R, cY/R, cZ/R

            lat_phi = np.arcsin(np.clip(cY, -1.0, 1.0))
            lon_theta = np.arctan2(cX, cZ)

            U_pano = (lon_theta + np.pi) / (2 * np.pi) * (W - 1)
            V_pano = (np.pi / 2 - lat_phi) / np.pi * (H - 1)

            pano_kpts.append((round(U_pano, 2), round(V_pano, 2)))

    gt_mask = np.zeros((H, W), dtype=np.float32)
    for p in pano_kpts:
        px, py = int(round(p[0])), int(round(p[1]))
        if 0 <= px < W and 0 <= py < H:
            gt_mask[py, px] = 1.0

    sp_labels = encode_superpoint_labels(gt_mask)

    return pano_img, np.array(pano_kpts), sp_labels

def encode_superpoint_labels(gt_mask):
    H, W = gt_mask.shape
    Hc, Wc = H // 8, W // 8
    
    out = gt_mask.reshape(Hc, 8, Wc, 8)
    out = out.transpose(0, 2, 1, 3).reshape(Hc, Wc, 64)
    
    max_idx = np.argmax(out, axis=2)
    
    prob_nodot = np.sum(out, axis=2) == 0
    
    labels = max_idx.copy()
    labels[prob_nodot] = 64
    
    return labels.astype(np.int64)

if __name__ == "__main__":
    img, kpts, sp_labels = create_equirectangular_pano(W=1024, H=512, face_size=512, num_shapes_per_face=10)
    
    vis = img.copy()
    for p in kpts:
        cv2.circle(vis, (int(p[0]), int(p[1])), 4, (0, 255, 0), 2)
        
    cv2.imwrite("pano_equi_vis.png", vis)