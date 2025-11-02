import cv2
import torch
import time
from depth_anything_v2.dpt import DepthAnythingV2
 
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

start_time = time.time()

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits'
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_vits.pth', map_location=DEVICE))
model = model.to(DEVICE).eval()

image_path = 'set1/001.jpg'
raw_img = cv2.imread(image_path)


depth = model.infer_image(raw_img)
end_time = time.time()

# Optional: Save the depth map
depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_colored = cv2.applyColorMap(depth_normalized.astype('uint8'), cv2.COLORMAP_INFERNO)

elapsed_time_ms = (end_time - start_time) * 1000
print(f"Depth estimation took {elapsed_time_ms:.2f} ms.")

depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_colored = cv2.applyColorMap(depth_normalized.astype('uint8'), cv2.COLORMAP_INFERNO)

# Resize original if needed
if raw_img.shape[:2] != depth_colored.shape[:2]:
    raw_img_resized = cv2.resize(raw_img, (depth_colored.shape[1], depth_colored.shape[0]))
else:
    raw_img_resized = raw_img

# Side-by-side comparison
side_by_side = cv2.hconcat([raw_img_resized, depth_colored])

cv2.imwrite('set1/depth_comparison(large).png', side_by_side)
print("Saved: depth_output.png (colored depth)")
print("Saved: depth_comparison.png (originallarge | depth)")