import os
from PIL import Image
import shutil
import re
from natsort import natsorted

def parse_filename(filename):
    match = re.match(r"0_\d+_\d+_(\d+)x(\d+)\.jpg", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def process_wsi_bag(bag_path, output_root):
    filenames = [f for f in os.listdir(bag_path) if f.endswith('.jpg')]
    if not filenames:
        return

    N, M = None, None
    for f in filenames:
        N, M = parse_filename(f)
        if N is not None:
            break

    if N is None or M is None:
        return

    bag_name = os.path.basename(bag_path)
    output_dir = os.path.join(output_root, bag_name)
    os.makedirs(output_dir, exist_ok=True)

    for n in range(1, N + 1):
        for m in range(1, M + 1):
            target_name = f"0_{n}_{m}_{N}x{M}.jpg"
            src_path = os.path.join(bag_path, target_name)
            dst_path = os.path.join(output_dir, target_name)

            if os.path.exists(src_path):
                shutil.copyfile(src_path, dst_path)
            else:

                white_img = Image.new("RGB", (224,224), color=(255, 255, 255))
                white_img.save(dst_path)

def process_all_bags(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    for bag_name in os.listdir(input_root):
        bag_path = os.path.join(input_root, bag_name)
        if os.path.isdir(bag_path):
            process_wsi_bag(bag_path, output_root)

def stitch_wsi(bag_folder, output_path):
    patches = [f for f in os.listdir((bag_folder)) if f.endswith(".jpg")]
    if not patches:
        return

    sample_patch = patches[0]
    first_img = Image.open(os.path.join(bag_folder, sample_patch))
    patch_width, patch_height = first_img.size

    _, _, N, M = parse_filename(sample_patch)

    stitched_img = Image.new("RGB", (M * patch_width, N * patch_height))

    for fname in patches:
        result = parse_filename(fname)
        if result is None:
            continue
        n, m, _, _ = result
        patch_path = os.path.join(bag_folder, fname)
        patch_img = Image.open(patch_path).convert("RGB")
        x = (m - 1) * patch_width
        y = (n - 1) * patch_height
        stitched_img.paste(patch_img, (x, y))

    bag_name = os.path.basename(bag_folder)
    stitched_img.save(os.path.join(output_path, f"{bag_name}.jpg"))

def stitch_all_bags(patch_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    for bag_name in os.listdir(patch_root):
        bag_path = os.path.join(patch_root, bag_name)
        if os.path.isdir(bag_path):
            stitch_wsi(bag_path, output_root)

input_root = "path_to_cropped_pathces"
complemented_root = "save_dir"

process_all_bags(input_root, complemented_root)

patch_root = "path_to_complemented_pathces"
output_root = "save_dir"
stitch_all_bags(patch_root, output_root)



