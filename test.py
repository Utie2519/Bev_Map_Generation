import cv2
import os
import glob
from ultralytics import YOLO
from PIL import Image
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import integrated


# ------------------ SAFE IMAGE LOADER ------------------
def load_images_safe(folder, limit=5):
    files = glob.glob(os.path.join(folder, "*"))
    files = sorted(files)[:limit]  # Only take up to 'limit' images

    images = []
    valid_files = []
    for f in files:
        img = cv2.imread(f)
        img = cv2.resize(img, (672, 864))
        if img is not None:
            images.append(img)
            valid_files.append(f)
        else:
            print(f"⚠️ Skipped (not an image or unreadable): {f}")

    return images, valid_files


# ------------------ MAIN PIPELINE ------------------
if __name__ == "__main__":
    image_folder = "lab_captured"        # Input images
    output_folder = "bev_outputs"
    overlay_folder = os.path.join(output_folder, "overlays")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(overlay_folder, exist_ok=True)

    # Load YOLO model
    model = YOLO("checkpoints/fine_tuned_model_best.pt")

    # Load up to 5 images safely
    list_images, image_paths = load_images_safe(image_folder, limit=5)

    print(f"📂 Looking for images in: {image_folder}")
    print(f"🔎 Found {len(list_images)} images: {image_paths}")

    if len(list_images) == 0:
        raise RuntimeError(
            f"❌ No images found in {image_folder}. "
            f"Make sure it contains valid image files (.jpg/.png)."
        )

    # Run YOLO detections + save overlays
    detections_list = []

for idx, (img, path) in enumerate(zip(list_images, image_paths)):
    results = model(img)   # Run YOLO
    r = results[0]         # Single Results object

    print(f"\n📸 Processing Image {idx+1}: {path}")
    print(f"✅ Found {len(r.boxes)} objects, {len(r.masks) if r.masks is not None else 0} masks")

    # Collect detections
    detections_list.append(r)

    # --- Save segmentation overlay ---
    overlay_path = os.path.join(overlay_folder, f"overlay_{idx+1}.jpg")
    integrated.save_segmentation_overlay(img, r, overlay_path)

    # --- Save Depth Map ---
    depth_out_dir = os.path.join(output_folder, "depth_maps")
    os.makedirs(depth_out_dir, exist_ok=True)

    depth, depth_path = integrated.run_depth_anything(
        img,
        encoder="vits",
        checkpoint_dir="checkpoints",
        save_npy=True,              # also save .npy file
        output_dir=depth_out_dir,
        filename=f"image_{idx+1}"
    )
    print(f"💾 Saved depth map for Image {idx+1} at {depth_out_dir}")

    # --- Save BEV ---
    bev_path = os.path.join(output_folder, f"bev_{idx+1}.jpg")
    integrated.depth_yolo_to_bev(
        img, r, encoder="vits",
        checkpoint_dir="checkpoints",
        out_name=bev_path
    )


    # ✅ Now generate BEVs for all images
    bev_maps = integrated.generate_multiple_bevs(
        list_images,
        detections_list,   # pass collected results
        encoder="vits",
        checkpoint_dir="checkpoints",
        output_dir=output_folder
    )

    print(f"\n🗺️ Generated {len(bev_maps)} BEV maps")

    if len(bev_maps) == 0:
        raise RuntimeError("❌ No BEV maps were generated. Something went wrong in integrated.generate_multiple_bevs.")

    # Create stitched 360° BEV
    bev360 = integrated.stitch_bevs(
        bev_maps,
        output_path=os.path.join(output_folder, "bev_360.jpg")
    )

    print(f"\n✅ Finished! Saved {len(bev_maps)} BEV maps, overlays, and a combined 360° BEV at {output_folder}")

# -------------------- NEW: BEV -> Occupancy & A* Pathfinding --------------------

input_folder = "bev_outputs"
output_folder = "occupancy_outputs"
path_folder = "path_outputs"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(path_folder, exist_ok=True)

meters_per_cell = 0.02  # Must match get_bev() resolution

# --- Process the stitched 360 BEV map ---
bev_file = "bev_360.jpg"
bev_path = os.path.join(input_folder, bev_file)
print(f"\n➡️ Processing the stitched 360 BEV map: {bev_path} ...")

# Load BEV image
bev_img = cv2.imread(bev_path)
if bev_img is None:
    print(f"⚠️ Skipping {bev_path}, could not read image.")
else:
    # Occupancy output (grayscale overlay)
    occ_png = os.path.join(output_folder, f"occupancy_360.png")

    occ = integrated.bev_to_occupancy(
        bev_img,
        meters_per_cell=meters_per_cell,
        g_thresh=50,
        o_thresh=40,
        inflation_radius_m=0.15,
        smoothing_kernel=3,
        save_png_path=occ_png
    )

    print("\n✅ Done! Grayscale occupancy map saved at:", occ_png)

    # --- Run A* on the 360 BEV occupancy map ---
    print("\n➡️ Running A* on the 360 BEV occupancy map ...")

    # Convert to planning grid
    occupancy_grid = integrated.occ_to_grid(occ)

    # Define start/end (these are pixel coords in the occupancy grid)
    # You may need to tune these depending on your stitched map's dimensions
    start = (1214, 696)
    end = (421, 688)

    # Build pathfinding grid
    grid = Grid(matrix=occupancy_grid)

    # Create nodes
    start_node = grid.node(start[0], start[1])
    end_node   = grid.node(end[0], end[1])

# Run A* search
finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
path, runs = finder.find_path(start_node, end_node, grid)
print(f"Path length: {len(path)} | Runs: {runs}")

# Convert to RGB for drawing path
arr = cv2.cvtColor(occ, cv2.COLOR_GRAY2BGR)

# Draw path as a thick blue line
for i in range(len(path) - 1):
    pt1 = (path[i].x, path[i].y)
    pt2 = (path[i + 1].x, path[i + 1].y)
    cv2.line(arr, pt1, pt2, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)

# Save visualization
out_path = os.path.join(path_folder, f"top_view_360.png")
Image.fromarray(arr).save(out_path)
print(f"✅ Saved path overlay at {out_path}")
