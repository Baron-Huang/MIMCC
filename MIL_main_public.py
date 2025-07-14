############################# TicMIL Demo main function ##############################
#### Author: Dr.Mingrui Ma
#### Email: XXx
#### Department: XJU
#### Attempt: Testing TicMIL Demo
#### TicMIL: A Tumor-guiding Instance Clustered Multi-instance Learning Network for Cervix Grading from Whole-slide Images

########################## API Section #########################
import warnings

warnings.filterwarnings("ignore")

import torch
from torch import nn
from pytorch_lightning import seed_everything
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
import random

from SwinT_models.flash_models.swin_transformer import SwinTransformer
from .SwinT_models.models.Simple_MIL_model_modules import MIL_Parallel_Feature,MIL_Parallel_Head_ub
from Utils.fit_functions import testing_for_parallel, training_for_parallel,extracting_feat_for_c16,interpret_bag_for_lung,Cluster_vis_lung
from Utils.Setup_Seed import setup_seed
from Utils.Read_MIL_Datasets import Read_MIL_Datasets
from Utils.ablation_experiments import save_model, acc_scores, to_np_category
import cv2
from skimage import io
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
import PIL
import seaborn as sns
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import to_pil_image
from torch.nn.parallel import DataParallel
import argparse


sns.set(font='Times New Roman', font_scale=0.6)


def create_swin_base():
    if args.img_size[0] == 224:
        swinT_base = SwinTransformer(img_size=args.img_size[0], patch_size=4, in_chans=3, num_classes=args.class_num,
                                     embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                                     window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                     norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                                     use_checkpoint=False, fused_window_process=False, is_flash=args.is_flash)
    else:
        swinT_base = SwinTransformer(img_size=args.img_size[0], patch_size=4, in_chans=3,
                                     num_classes=args.class_num,
                                     embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                                     window_size=3, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                     norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                                     use_checkpoint=False, fused_window_process=False, is_flash=args.is_flash)

    checkpoint = torch.load(args.pretrained_weights_path, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = swinT_base.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            # logger.warning(f"Error in loading {k}, passing......")
            pass
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = swinT_base.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            # logger.warning(f"Error in loading {k}, passing......")
            pass
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = swinT_base.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            # logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(swinT_base.head.bias, 0.)
            torch.nn.init.constant_(swinT_base.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']

    swinT_base.load_state_dict(state_dict, strict=False)
    nn.init.trunc_normal_(swinT_base.head.weight, std=.02)
    # print(swinT_base.layers[0].blocks[0].mlp.fc2.weight)

    return swinT_base

def worker_init_fn(worker_id):
    random.seed(7 + worker_id)
    np.random.seed(7 + worker_id)
    torch.manual_seed(7 + worker_id)
    torch.cuda.manual_seed(7 + worker_id)
    torch.cuda.manual_seed_all(7 + worker_id)

########################## main_function #########################
if __name__ == '__main__':
    # dataset_root = 'CAMELYON16_SefMIC'  # 2 classes
    dataset_root = 'DHMC_Lung_new'   # 2 classes
    ########################## Hyparameters #########################
    paras = argparse.ArgumentParser(description='MIL Hyparameters')
    paras.add_argument('--random_seed', type=int, default=0)
    paras.add_argument('--gpu_device', type=int, default=0)
    paras.add_argument('--class_num', type=int, default=2)
    paras.add_argument('--batch_size', type=int, default=1)
    paras.add_argument('--epochs', type=int, default=100)
    paras.add_argument('--img_size', type=list, default=[96,96])
    paras.add_argument('--bags_len', type=int, default=2000)
    paras.add_argument('--feat_extract', default=False, action='store_true')
    paras.add_argument('--cluster_vis', default=False, action='store_true')
    paras.add_argument('--bag_weight', default=False, action='store_true')
    paras.add_argument('--is_flash', default=False, action='store_true')
    paras.add_argument('--use_amp', default=False, action='store_true')
    paras.add_argument('--max_input_len', type=int, default=2000)
    paras.add_argument('--num_workers', type=int, default=16)
    paras.add_argument('--worker_time_out', type=int, default=0)
    paras.add_argument('--run_mode', type=str, default='train')
    paras.add_argument('--parallel_gpu_ids', type=list, default=[0,1,2,3])
    paras.add_argument('--train_read_path', type=str,
                default=f'/data/MIL/Datasets/{dataset_root}/Train')
    paras.add_argument('--test_read_path', type=str,
                default=f'/data/MIL/Datasets/{dataset_root}/Test')
    paras.add_argument('--val_read_path', type=str,
                default=f'/data/MIL/Datasets/{dataset_root}/Test')
    ### Parallel save
    paras.add_argument('--weights_save_path', type=str,
                        default=f'/data/MIL/MMvC/Weights_Result/{dataset_root}')
    ### Parallel test
    paras.add_argument('--test_weights_feature', type=str,
                        default=r'/data/MIL/MMvC/Weights_Result/DHMC_Lung_new/SwinT_sota_Feature_ValAcc_0.8444444444444444_Epoch76.pth')
    paras.add_argument('--test_weights_head', type=str,
                        default=r'/data/MIL/MMvC/Weights_Result/DHMC_Lung_new/SwinT_sota_Head_ValAcc_0.8444444444444444_Epoch76.pth')

    ### Pretrained
    paras.add_argument('--pretrained_weights_path', type=str,
                default=r'/data/MIL/MMvC/Weights/SwinT/swin_tiny_patch4_window7_224_22k.pth')

    args = paras.parse_args()
    seed_everything(args.random_seed)
    os.makedirs(args.weights_save_path,exist_ok=True)

    # print('########################## reading datas and processing datas #########################')
    train_data = Read_MIL_Datasets(read_path=args.train_read_path ,img_size=args.img_size, bags_len=args.bags_len)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              timeout=args.worker_time_out)

    test_data = Read_MIL_Datasets(read_path=args.test_read_path, img_size=args.img_size, bags_len=args.bags_len)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                              timeout=args.worker_time_out)

    val_data = Read_MIL_Datasets(read_path=args.val_read_path, img_size=args.img_size, bags_len=args.bags_len)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                              timeout=args.worker_time_out)

    swinT_base = create_swin_base()


    ### creating a SPE-MIL model
    mil_feature = MIL_Parallel_Feature(base_model=swinT_base, img_size=args.img_size)
    mil_feature = mil_feature.cuda()
    mil_feature = DataParallel(mil_feature, device_ids=args.parallel_gpu_ids)
    mil_head = MIL_Parallel_Head_ub(base_model=swinT_base, class_num=args.class_num,
                                    model_stats=args.run_mode, seed=args.random_seed,
                                    batch_size=args.batch_size, bags_len=args.bags_len,
                                    feat_extract=args.feat_extract, bag_weight=args.bag_weight,
                                    cluster_vis=args.cluster_vis)
    mil_head = mil_head.cuda()


    ########################## fitting models and testing models #########################
    if args.run_mode == 'train':
        print('########################## fitting models #########################')
        training_for_parallel(mil_feature=mil_feature, mil_head=mil_head, train_loader=train_loader,
                              val_loader=val_loader, test_loader=test_loader,
                              lr_fn='vit', epoch=args.epochs, gpu_device=args.gpu_device,
                              weight_path=args.weights_save_path, max_input_len=args.max_input_len,
                              bags_len=args.bags_len, batch_size=args.batch_size, use_amp=args.use_amp)

    if args.run_mode == 'test':
        # print('########################## testing function #########################')
        head_weight = torch.load(args.test_weights_head, map_location='cuda:0')
        feature_weight = torch.load(args.test_weights_feature, map_location='cuda:0')
        mil_feature.load_state_dict(feature_weight, strict=True)
        mil_head.load_state_dict(head_weight, strict=True)
        if args.feat_extract:
        # print('########################## testing function #########################')
            extracting_feat_for_c16(mil_feature=mil_feature, mil_head=mil_head, train_loader=train_loader,
                                    batch_size=args.batch_size, class_num=args.class_num,
                                    bags_len=args.bags_len, val_loader=val_loader, test_loader=test_loader)
        elif args.bag_weight:
            interpret_bag_for_lung(mil_feature=mil_feature, mil_head=mil_head, train_loader=train_loader,
                                   batch_size=args.batch_size, class_num=args.class_num,
                                   bags_len=args.bags_len, val_loader=val_loader, test_loader=test_loader)
        elif args.cluster_vis:
            Cluster_vis_lung(mil_feature=mil_feature, mil_head=mil_head, train_loader=train_loader,
                             batch_size=args.batch_size, class_num=args.class_num,
                             bags_len=args.bags_len, val_loader=val_loader, test_loader=test_loader)
        else:
            testing_for_parallel(mil_feature=mil_feature, mil_head=mil_head, train_loader=train_loader,
                                 batch_size=args.batch_size, class_num=args.class_num,
                                 bags_len=args.bags_len, val_loader=val_loader, test_loader=test_loader)





