from natsort import natsorted
import os
import re
from PIL import Image

def Cluster_vis_lung(mil_feature=None, mil_head=None, test_loader=None):
    # batch size ======1 !!!!!!!!
    mil_feature.eval()
    mil_head.eval()
    count = 0
    acinar_list = natsorted(os.listdir('path_to_Test_acinar'))
    n1 = len(acinar_list)
    solid_list = natsorted(os.listdir('path_to_Test_solid'))
    for img_list, label in test_loader:
        with torch.no_grad():
            pre_y = torch.zeros((1,768)).cuda()
            for img in img_list:
                pre_y = torch.cat((pre_y, mil_feature(img.cuda())))
            pre_y = pre_y[1:]
            each_list = mil_head(pre_y)

        patch_size = 224  # 每个 patch 的像素大小
        colors = [
            (121, 43, 166),  # color1
            (203, 104, 180),  # color2
            (166, 166, 166)  # color3
        ]


        if count < n1:
            whole_list = natsorted(os.listdir('path_to_complete_patches_of_test_acinar' + '/' + acinar_list[count]))
            cropped_list = natsorted(os.listdir('path_to_Test_acinar' + '/' + acinar_list[count]))
        else:
            whole_list = natsorted(os.listdir('path_to_complete_patches_of_test_solid' + '/' + solid_list[count-n1]))
            cropped_list = natsorted(os.listdir('path_to_Test_solid' + '/' + solid_list[count-n1]))

        for fname in whole_list:
            match = re.match(r'0_\d+_\d+_(\d+)x(\d+)\.jpg', fname)
            if match:
                grid_rows = int(match.group(1))
                grid_cols = int(match.group(2))
                break
        else:
            print(f"[Warning] No valid patch file found in: {whole_list}")
            continue
        wsi_image = Image.new("RGB", (grid_cols * patch_size, grid_rows * patch_size), color=(255, 255, 255))

        cropped_index_map = {name: idx for idx, name in enumerate(cropped_list[:-5])}

        for fname in whole_list:
            match = re.match(r'0_(\d+)_(\d+)_\d+x\d+\.jpg', fname)
            if not match:
                continue
            n, m = int(match.group(1)), int(match.group(2))

            row = n - 1
            col = m - 1
            x0 = col * patch_size
            y0 = row * patch_size

            if fname not in cropped_index_map:
                patch = Image.new("RGB", (patch_size, patch_size), color=(255, 255, 255))
            else:
                idx = cropped_index_map[fname]
                for class_id, cluster_indices in enumerate(each_list):
                    if idx in cluster_indices:
                        patch = Image.new("RGB", (patch_size, patch_size), color=colors[class_id])
                        break
                else:
                    patch = Image.new("RGB", (patch_size, patch_size), color=(255, 255, 255))

            wsi_image.paste(patch, (x0, y0))

        if count < n1:
            wsi_image.save(f"save_dir_to/acinar/{count}.jpg")
        else:
            wsi_image.save(f"save_dir_to/solid/{count}.jpg")
        count += 1

def Cluster_vis(mil_feature=None, mil_head=None, test_loader=None):
    # batch size =1
    mil_feature.eval()
    mil_head.eval()
    count = 0
    for img_list, label in test_loader:
        with torch.no_grad():
            pre_y = torch.zeros((1,768)).cuda()
            for img in img_list:
                pre_y = torch.cat((pre_y, mil_feature(img.cuda())))
            pre_y = pre_y[1:]
            each_list = mil_head(pre_y)  # batch_size * 1042 * 1
        grid_size = 31
        patch_size = 96
        colors = [
            (121, 43, 166),  # color1
            (203, 104, 180),  # color2
            (166, 166, 166)  # color3
        ]
        wsi_image = Image.new("RGB", (grid_size * patch_size, grid_size * patch_size), color=(255, 255, 255))

        # 依次绘制每一类
        for class_id, index_list in enumerate(each_list):
            if index_list != []:
                color = colors[class_id]
                for idx in index_list:
                    row = idx // grid_size
                    col = idx % grid_size

                # 计算左上角坐标
                    x0 = col * patch_size
                    y0 = row * patch_size

                # 创建纯色 patch
                    patch = Image.new("RGB", (patch_size, patch_size), color=color)
                    wsi_image.paste(patch, (x0, y0))
            wsi_image.save(f"your_path")
        count += 1