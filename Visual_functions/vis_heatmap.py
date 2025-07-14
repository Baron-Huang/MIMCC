import torch
from natsort import natsorted

import cv2
import numpy as np
import re
import os
from PIL import Image


def interpret_bag(mil_feature=None, mil_head=None, test_loader=None):
    mil_feature.eval()
    mil_head.eval()
    bag_weight_list = []
    for img_list, label in test_loader:
        with torch.no_grad():
            pre_y = torch.zeros((1,768)).cuda()
            for img in img_list:
                pre_y = torch.cat((pre_y, mil_feature(img.cuda())))
            pre_y = pre_y[1:]
            bag_w = mil_head(pre_y)  # batch_size * 1042 * 768
            bag_weight_list.append(bag_w[:, :961, :] @ mil_head.head.weight.T )
    bag_weight_sum = torch.cat(bag_weight_list,dim=0).cpu().numpy() # 138 * 961 * 3

    count = 0
    for cate in ['I','II','III']:
        merged_img_dir = f'path_to_your_WSI/{cate}'
        save_dir = f'save_dir/{cate}'

        bag_names = sorted(os.listdir(merged_img_dir))
        grid_size = 31
        tile_size = 96
        alpha = 0.3  #


        for idx, bag_img_name in enumerate(bag_names):

            img_path = os.path.join(merged_img_dir, bag_img_name)
            weights = bag_weight_sum[count+idx][:,len(cate)-1].flatten()

            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)

            weights -= weights.min()
            weights /= weights.max() + 1e-8
            heatmap_small = weights.reshape(grid_size, grid_size)
            heatmap_large = cv2.resize(heatmap_small, (grid_size * tile_size, grid_size * tile_size),
                                       interpolation=cv2.INTER_CUBIC)
            heatmap_color = cv2.applyColorMap((heatmap_large * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_color, alpha, 0)

            save_name = os.path.splitext(bag_img_name)[0] + '_overlay.jpg'
            save_path = os.path.join(save_dir, save_name)
            Image.fromarray(overlay).save(save_path)
        count += len(bag_names)


def interpret_bag_for_lung(mil_feature=None, mil_head=None,test_loader=None):
    mil_feature.eval()
    mil_head.eval()
    bag_weight_list = []
    for img_list, label in test_loader:
        label = label.cuda()
        with torch.no_grad():
            pre_y = torch.zeros((1,768)).cuda()
            for img in img_list:
                pre_y = torch.cat((pre_y, mil_feature(img.cuda())))
            pre_y = pre_y[1:]
            bag_w = mil_head(pre_y)  # 1 * N * 768
            bag_weight_list.append(bag_w[:, :-5, :] @ mil_head.head.weight.T )

    count = 0
    tile_size = 224
    alpha = 0.3
    classes_name = ['acinar','solid']

    for k, cate in enumerate(classes_name):
        merged_img_dir = f'path_to_WSI/{cate}'
        save_dir = f'save_dir/{cate}'

        bag_names = natsorted(os.listdir(merged_img_dir))
        patch_img_dir = f'path_to_cropped_pathces/{cate}'

        for idx, bag_img_name in enumerate(bag_names):
            img_path = os.path.join(merged_img_dir, bag_img_name)
            patch_name_prefix = os.path.splitext(bag_img_name)[0]
            patch_dir = os.path.join(patch_img_dir, patch_name_prefix)
            patch_list = natsorted(os.listdir(patch_dir))[:-5]
            for fname in patch_list:
                match = re.match(r'0_\d+_\d+_(\d+)x(\d+)\.jpg', fname)
                if match:
                    grid_rows = int(match.group(1))
                    grid_cols = int(match.group(2))
                    break
            else:
                print(f"[Warning] No valid patch found for {bag_img_name}, skipping.")
                continue

            patch_scores = bag_weight_list[count + idx].squeeze(0).cpu().numpy()  # [N, C]
            heat_vector = patch_scores[:, k]
            heatmap_full = np.zeros((grid_rows, grid_cols), dtype=np.float32)
            mask_map = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

            for i, fname in enumerate(patch_list):
                match = re.match(r'0_(\d+)_(\d+)_\d+x\d+\.jpg', fname)
                if not match:
                    continue
                row = int(match.group(1)) - 1
                col = int(match.group(2)) - 1
                if 0 <= row < grid_rows and 0 <= col < grid_cols:
                    heatmap_full[row, col] = heat_vector[i]
                    mask_map[row, col] = 1

            if mask_map.sum() > 0:
                valid_vals = heatmap_full[mask_map == 1]
                heatmap_full[mask_map == 1] = (valid_vals - valid_vals.min()) / (
                            valid_vals.max() - valid_vals.min() + 1e-8)

            heatmap_large = cv2.resize(heatmap_full, (grid_cols * tile_size, grid_rows * tile_size),
                                       interpolation=cv2.INTER_CUBIC)
            heatmap_color = cv2.applyColorMap((heatmap_large * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_color, alpha, 0)

            save_name = os.path.splitext(bag_img_name)[0] + '_overlay.jpg'
            save_path = os.path.join(save_dir, save_name)
            Image.fromarray(overlay).save(save_path)
        count += len(bag_names)