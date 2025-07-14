import os
from PIL import Image

import numpy as np

# 设置路径
base_dir = 'path_to_LSCC_or_CSCC_pathces'
save_dir = 'save_dir'
os.makedirs(save_dir, exist_ok=True)

# 图像参数
grid_size = 31
tile_size = 96
white_tile = Image.new('RGB', (tile_size, tile_size), (255, 255, 255))

for bag_name in os.listdir(base_dir):
    bag_path = os.path.join(base_dir, bag_name)
    if not os.path.isdir(bag_path):
        continue

    final_image = Image.new('RGB', (grid_size * tile_size, grid_size * tile_size))

    for idx in range(grid_size * grid_size):
        row = idx // grid_size
        col = idx % grid_size
        tile_path = os.path.join(bag_path, f'{idx}.jpg')

        if os.path.exists(tile_path):
            tile = Image.open(tile_path).convert('RGB')
            tile_np = np.array(tile)
            if np.all(tile_np == 0):
                tile = white_tile
        else:
            tile = white_tile

        final_image.paste(tile, (col * tile_size, row * tile_size))
    save_path = os.path.join(save_dir, f'{bag_name}.jpg')
    final_image.save(save_path)
    print(f'Saved: {save_path}')