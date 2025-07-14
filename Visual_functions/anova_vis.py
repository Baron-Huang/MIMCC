import torch
import umap
import matplotlib.pyplot as plt
import pandas as pd

def extracting_feat(mil_feature=None, mil_head=None, gpu_device=None,test_loader=None):


    mil_feature.eval()
    mil_head.eval()
    sum_label = torch.zeros(2).cuda(gpu_device)
    sum_feat = []
    for img_list, label in test_loader:
        label = label.cuda()
        with torch.no_grad():
            pre_y = torch.zeros((1, 768)).cuda()
            for img in img_list:
                pre_y = torch.cat((pre_y, mil_feature(img.cuda())))
            pre_y = pre_y[1:]
            pre_y = mil_head(pre_y)
            sum_feat.append(pre_y)
            sum_label = torch.cat((sum_label, label))
    sum_label = sum_label[2:]
    sum_feat = torch.cat(sum_feat,dim=0)


    plt.rcParams['font.family'] = 'DejaVu Sans'

    sum_label = sum_label.cpu().numpy()
    sum_feat = sum_feat.cpu().numpy()

    # for anova
    reducer = umap.UMAP(n_components=1)
    embedding = reducer.fit_transform(sum_feat)

    dfs = []

    for label, name in zip([0, 1, 2], ['Grade I', 'Grade II', 'Grade III']):
        idx = sum_label == label
        data = embedding[idx]

        df = pd.DataFrame({
            f'{name}_x': data[:, 0],
        })
        dfs.append(df)

    max_len = max(df.shape[0] for df in dfs)
    for i in range(len(dfs)):
        dfs[i] = dfs[i].reindex(range(max_len))

    final_df = pd.concat(dfs, axis=1)
    final_df.to_csv('save_dir/anova.csv', index=False, encoding='utf-8')


def extracting_feat_for_c16(mil_feature=None, mil_head=None, gpu_device=None,test_loader=None):


    mil_feature.eval()
    mil_head.eval()

    sum_label = torch.zeros(2).cuda(gpu_device)
    sum_feat = []
    for img_list, label in test_loader:
        label = label.cuda()
        with torch.no_grad():
            pre_y = torch.zeros((1, 768)).cuda()
            for img in img_list:
                pre_y = torch.cat((pre_y, mil_feature(img.cuda())))
            pre_y = pre_y[1:]
            pre_y = mil_head(pre_y)
            sum_feat.append(pre_y)
            sum_label = torch.cat((sum_label, label))
    sum_label = sum_label[2:]
    sum_feat = torch.cat(sum_feat,dim=0)

    plt.rcParams['font.family'] = 'DejaVu Sans'

    sum_label = sum_label.cpu().numpy()
    sum_feat = sum_feat.cpu().numpy()

    # for anova
    reducer = umap.UMAP(n_components=1)
    embedding = reducer.fit_transform(sum_feat)

    dfs = []

    for label, name in zip([0, 1], ['Normal', 'Tumor']):
        idx = sum_label == label
        data = embedding[idx]
        count = data.shape[0]

        df = pd.DataFrame({
            f'{name}_x': data[:, 0],
        })
        dfs.append(df)

    max_len = max(df.shape[0] for df in dfs)
    for i in range(len(dfs)):
        dfs[i] = dfs[i].reindex(range(max_len))

    # 拼接
    final_df = pd.concat(dfs, axis=1)
    final_df.to_csv('save/dir/anova.csv', index=False, encoding='utf-8')