import os
import cv2
import torch
import time
import numpy as np
import open3d as o3d
import math
from depth_anything_v2.dpt import DepthAnythingV2
from typing import List, Union


DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# ------------------- Depth Estimation -------------------
def run_depth_anything(raw_img, encoder='vits', checkpoint_dir='checkpoints', save_npy=False, output_dir='set1', filename="frame"):
    start_time = time.time()

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'{checkpoint_dir}/depth_anything_v2_{encoder}.pth', map_location=DEVICE))
    model = model.to(DEVICE).eval()

    if raw_img is None or not isinstance(raw_img, np.ndarray):
        raise ValueError("Input must be a valid NumPy image array")
    
    if raw_img.dtype == np.float64:
        raw_img = (raw_img * 255).clip(0, 255).astype(np.uint8)

    depth = model.infer_image(raw_img)

    depth_path = None
    if save_npy:
        os.makedirs(output_dir, exist_ok=True)
        depth_filename = f"{filename}_depth.npy"
        depth_path = os.path.join(output_dir, depth_filename)
        np.save(depth_path, depth)

    # ------------------- Visualization -------------------
    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = depth_vis.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

    # Save as PNG
    os.makedirs(output_dir, exist_ok=True)
    depth_img_filename = f"{filename}_depth.png"
    cv2.imwrite(os.path.join(output_dir, depth_img_filename), depth_color)

    elapsed_time_ms = (time.time() - start_time) * 1000
    print(f"Depth estimation took {elapsed_time_ms:.2f} ms. Saved visualization to {depth_img_filename}")

    return depth, depth_path


# ------------------- Mask Resize Helper -------------------
def safe_mask_resize(mask, target_h, target_w):
    """
    Safely resize a boolean mask to match target resolution.
    Works whether mask is a torch.Tensor or numpy.ndarray.
    """
    import torch.nn.functional as F

    # Convert torch tensor to numpy if needed
    if hasattr(mask, "cpu"):  # it's a torch tensor
        mask = mask.cpu().numpy()

    if mask.ndim == 2:
        mask = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return mask.astype(bool)
    elif mask.ndim == 3:  # e.g. [N, H, W]
        resized = []
        for m in mask:
            m_resized = cv2.resize(m.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            resized.append(m_resized.astype(bool))
        return np.stack(resized, axis=0)
    else:
        raise ValueError(f"Unsupported mask dimensions: {mask.shape}")


# ------------------- BEV Projection -------------------
import torch.nn.functional as F

def get_bev(depth: np.ndarray,
            results_or_dets: Union[object, List[List[float]]],
            fx: float, fy: float, cx: float, cy: float,
            resolution: float = 0.02) -> np.ndarray:
    """
    Create a BEV map from a depth map and detections.

    Supports:
      - YOLO Results object with .masks and .boxes (preferred, pixel-wise)
      - Plain list-of-detections: [x1, y1, x2, y2, label, conf] (box-wise)

    Pixel coloring:
      - label == "path" → green, intensity by confidence
      - other labels     → red/blue mix, intensity by confidence (red by default here)
    """
    h, w = depth.shape
    bev_map = np.zeros((500,500, 3), dtype=np.uint8)
    bev_map[:] = (0, 0, 0)

    # Build camera grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy  # Y is currently unused, but kept for completeness

    def project_and_paint(mask_bool: np.ndarray, label: str, conf: float):
        """Project selected pixels to BEV and paint with color by label/conf."""
        if mask_bool.shape != depth.shape:
            mask_bool = safe_mask_resize(mask_bool, h, w)

        X_sel = X[mask_bool]
        Z_sel = Z[mask_bool]
        if X_sel.size == 0:
            return

        bx = (X_sel / resolution + bev_map.shape[1] // 2).astype(int)
        bz = (Z_sel / resolution).astype(int)

        valid = (bx >= 0) & (bx < bev_map.shape[1]) & (bz >= 0) & (bz < bev_map.shape[0])
        bx, bz = bx[valid], bz[valid]
        if bx.size == 0:
            return

        # Color: path → green; others → red (you can switch to blue easily)
        intensity = 255;#int(max(0, min(255, round(255.0 * float(conf)))))
        if str(label).lower() == "path":
            color = (0, intensity, 0)
        else:
            color = (0, 0, intensity)
        bev_map[bz, bx] = color

        # --- Fill inside holes as blue ---
        # Compute filled mask in image plane
        from scipy.ndimage import binary_fill_holes
        mask_filled = binary_fill_holes(mask_bool)

        # Hole region = filled - original
        hole_mask = mask_filled & (~mask_bool)

        X_hole = X[hole_mask]
        Z_hole = Z[hole_mask]
        bx_h = (X_hole / resolution + bev_map.shape[1] // 2).astype(int)
        bz_h = (Z_hole / resolution).astype(int)

        valid_h = (bx_h >= 0) & (bx_h < bev_map.shape[1]) & (bz_h >= 0) & (bz_h < bev_map.shape[0])
        bx_h, bz_h = bx_h[valid_h], bz_h[valid_h]

        #bev_map[bz_h, bx_h] = (255, 0, 0)  # blue holes only


    # Case A: YOLO Results object
    is_results_obj = hasattr(results_or_dets, "boxes")
    if is_results_obj:
        results = results_or_dets

        # Pull names dict if present
        names = getattr(results, "names", None)
        # Boxes/classes/conf
        cls_ids = results.boxes.cls.detach().cpu().numpy() if hasattr(results.boxes, "cls") else None
        confs   = results.boxes.conf.detach().cpu().numpy() if hasattr(results.boxes, "conf") else None

        # Preferred: pixel-wise masks
        from scipy.ndimage import binary_fill_holes
        
        if hasattr(results, "masks") and results.masks is not None and getattr(results.masks, "data", None) is not None:
            masks = results.masks.data  # shape [N, Hm, Wm] torch tensor
            masks_np = masks.detach().cpu().numpy()  # (N, Hm, Wm)
 
            for i in range(masks_np.shape[0]):
                mask_bool = masks_np[i] > 0.5
                mask_uint8 = (mask_bool.astype(np.uint8) * 255)
                kernel = np.ones((5,5), np.uint8)   # you can adjust size
                mask_filled = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
                mask_bool = mask_filled > 0
                mask_bool = binary_fill_holes(mask_bool).astype(bool)  # fill holes

                # Label & conf (fall back gracefully)
                label = str(names[int(cls_ids[i])]) if (names is not None and cls_ids is not None) else "object"
                conf  = float(confs[i]) if confs is not None else 0.6

                project_and_paint(mask_bool, label, conf)

        else:
            # Fallback to boxes → rasterize each box as a mask
            for i, box in enumerate(results.boxes):
                xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
                x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
                y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                mask_bool = np.zeros((h, w), dtype=bool)
                mask_bool[y1:y2, x1:x2] = True

                label = str(names[int(box.cls.item())]) if (names is not None and hasattr(box, "cls")) else "object"
                conf  = float(box.conf.item()) if hasattr(box, "conf") else 0.6
                project_and_paint(mask_bool, label, conf)

    else:
        # Case B: list-of-detections [[x1,y1,x2,y2,label,conf], ...]
        dets = results_or_dets
        if not isinstance(dets, list):
            raise TypeError(f"Unsupported detections type: {type(dets)}. Pass YOLO Results or list-of-dets.")

        for det in dets:
            if len(det) < 6:
                # tolerate shorter records but skip if malformed
                continue
            x1, y1, x2, y2, label, conf = det
            # clamp & int
            x1 = int(max(0, min(w - 1, round(x1))))
            x2 = int(max(0, min(w - 1, round(x2))))
            y1 = int(max(0, min(h - 1, round(y1))))
            y2 = int(max(0, min(h - 1, round(y2))))
            if x2 <= x1 or y2 <= y1:
                continue

            mask_bool = np.zeros((h, w), dtype=bool)
            mask_bool[y1:y2, x1:x2] = True
            project_and_paint(mask_bool, str(label), float(conf))



    # Conventional BEV orientation (forward ↑): flip vertical so near→bottom
    flipped = cv2.flip(bev_map, 0)
    return flipped



# ------------------- Unified Pipeline -------------------
def depth_yolo_to_bev(image: np.ndarray,
                      results_or_dets: Union[object, List[List[float]]],
                      encoder: str = 'vits',
                      checkpoint_dir: str = 'checkpoints',
                      out_name: str = "bev.jpg") -> np.ndarray:
    depth, _ = run_depth_anything(image, encoder, checkpoint_dir, save_npy=False)

    h, w = depth.shape
    fx = fy = 0.8 * w
    cx, cy = w / 2, h / 2

    bev_map = get_bev(depth, results_or_dets, fx, fy, cx, cy)
    cv2.imwrite(out_name, bev_map)
    return bev_map


# ------------------- Multi-BEV + 360 Merge -------------------
def generate_multiple_bevs(list_images, detections_list, encoder, checkpoint_dir, output_dir):
    """
    list_images: list of np.ndarray images
    detections_list: list of YOLO Results objects OR list-of-detections (one per image)
    """
    os.makedirs(output_dir, exist_ok=True)
    bev_maps = []

    for idx, (img, r) in enumerate(zip(list_images, detections_list)):
        try:
            print(f"🔎 Processing BEV {idx+1}")
            if hasattr(img, "shape"):
                print(f"   Image shape: {img.shape}")
            else:
                print(f"   Image type: {type(img)}")

            print(f"   Detection type: {type(r)}{' (YOLO Results)' if hasattr(r, 'boxes') else ' (list-of-dets)'}")

            out_path = os.path.join(output_dir, f"bev_{idx+1}.jpg")
            bev = depth_yolo_to_bev(img, r, encoder=encoder, checkpoint_dir=checkpoint_dir, out_name=out_path)

            if bev is None or getattr(bev, "size", 0) == 0:
                print(f"⚠️ Skipped Image {idx+1}: BEV was empty.")
            else:
                bev_maps.append(bev)
                print(f"✅ Saved BEV for Image {idx+1} at {out_path}")

        except Exception as e:
            print(f"❌ Error generating BEV for Image {idx+1}: {e}")

    if len(bev_maps) == 0:
        raise RuntimeError("❌ No BEV maps were generated. Something went wrong in integrated.generate_multiple_bevs.")

    return bev_maps



def stitch_bevs(bev_maps, output_path="circle_of_triangles.png"):
    if len(bev_maps) == 0:
        raise ValueError("bev_maps is empty")

    # === CONFIG ===
    canvas_size = 1500       # size of the square canvas
    radius = 300             # radius of the circle
    center = (canvas_size // 2, canvas_size // 2)

    # === Create base canvas ===
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    n = len(bev_maps)
    angle_step = 360 / n

    # === Helper: Rotate image around its center ===
    def rotate_image(img, angle_deg):
        h, w = img.shape[:2]
        c = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        return rotated

    # === Helper: Paste image with transparency handling ===
    def paste_img(base, overlay, x, y):
        h, w = overlay.shape[:2]
        H, W = base.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(x + w, W)
        y2 = min(y + h, H)

        x1_overlay = x1 - x
        y1_overlay = y1 - y
        x2_overlay = x1_overlay + (x2 - x1)
        y2_overlay = y1_overlay + (y2 - y1)

        for i in range(y1, y2):
            for j in range(x1, x2):
                oi = y1_overlay + (i - y1)
                oj = x1_overlay + (j - x1)
                val = overlay[oi, oj, :]
                if (base[i, j, 1] == 0) and (base[i, j, 2] == 0):
                    base[i, j] = overlay[oi, oj]
        return base

    # === Arrange BEVs around circle ===
    for i, bev in enumerate(bev_maps):
        h, w = bev.shape[:2]

        # Angle for this BEV's left edge
        angle_deg = i * angle_step

        # Rotate so left edge aligns radially
        rotated = rotate_image(bev, -angle_deg + 90)

        # Compute placement
        angle_rad = math.radians(angle_deg)
        x_offset = int(center[0] + radius * math.cos(angle_rad) - w // 2)
        y_offset = int(center[1] + radius * math.sin(angle_rad) - h // 2)

        # Paste on canvas
        canvas = paste_img(canvas, rotated, x_offset, y_offset)

    # Save final circular BEV arrangement
    cv2.imwrite(output_path, canvas)
    return canvas
# ------------------- Segmentation Overlay -------------------
def save_segmentation_overlay(image, results, save_path, alpha=0.5):
    """
    Overlay segmentation masks on the original image using YOLO Results.
    """
    if not hasattr(results, "plot"):
        raise TypeError(
            f"❌ Expected YOLO Results object (with .plot()), but got {type(results)}. "
            f"Pass results[0] from model(img)."
        )

    vis = results.plot()  # numpy array with masks/boxes drawn by Ultralytics
    vis_rgb = cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_BGR2RGB)
    original_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(original_rgb, 1 - alpha, vis_rgb, alpha, 0)

    cv2.imwrite(save_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    print(f"💾 Saved segmentation overlay at {save_path}")


def bev_to_occupancy(
    bev: np.ndarray,
    meters_per_cell: float = 0.02,      # must match get_bev() 'resolution'
    g_thresh: int = 40,                  # green threshold → free/path
    o_thresh: int = 40,                  # red/blue threshold → obstacle
    inflation_radius_m: float = 0.10,    # grow obstacles by robot radius
    smoothing_kernel: int = 3,           # small morphology to denoise
    save_png_path: str = None
) -> np.ndarray:
    """
    Convert a color BEV (BGR) into an occupancy grid:
      0   = free, 100 = occupied, 255 = unknown   (ROS-style)
    Path is green in your BEV, obstacles are red (or blue if you switch).
    """
    if bev is None or bev.ndim != 3 or bev.shape[2] != 3:
        raise ValueError("bev must be an HxWx3 BGR image")

    B, G, R = cv2.split(bev)

    # Masks (tune thresholds if needed)
    path_mask = (G > g_thresh)
    obst_mask = (R > o_thresh) | (B > o_thresh)  # handle red or blue obstacles

    occ = np.full(bev.shape[:2], 100, dtype=np.uint8)  # unknown by default
    occ[path_mask] = 255
    occ[obst_mask] = 0

    if save_png_path:
        cv2.imwrite(save_png_path, occ)
        print(f"💾 Saved occupancy PNG at {save_png_path}")

    return occ



def occ_to_grid(occ: np.ndarray) -> np.ndarray:
    """
    Convert occupancy map (0=free,100=occupied,255=unknown) into
    a simple grid for path planning:
      0 = occupied
      1 = free
     -1 = unknown
    """
    # Start with a copy
    grid = np.zeros_like(occ, dtype=np.int8)

    grid[occ == 0]   = 0    # occupied
    grid[occ == 255] = 1    # free
    grid[occ == 100] = -1   # unknown

    return grid