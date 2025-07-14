############################# TicMIL_model_modules ##############################
#### Author: Dr. Mingrui Ma
#### Email: xxx
#### Department: Xinjiang University
#### Attempt: creating TicMIL model by loading pretrained weight for searching the best learning rate
import time

from MMvC.Loss_functions.mmd import MMDLoss
########################## API Section #########################
from MMvC.Models.SwinT_models.models.swin_transformer import SwinTransformer
from MMvC.matrix_transformation.GFK_torch import GFK
from MMvC.matrix_transformation.LLE import stable_lle
from torch_pca import PCA
from torch import nn
import torch
import random
from MMvC.Loss_functions.adv import AdversarialLoss
import torch.nn.functional as F
import math


class TgiClustering():
    def __init__(self, k_nums = 3, sel_dis = 'l1', train_iters = 50, p = 1):
        super(TgiClustering, self).__init__()
        self.k_nums = k_nums
        self.sel_dis = sel_dis
        self.train_iters = train_iters
        self.p = p

    def l2_distance(self, x, y):
        return torch.sqrt((x - y).permute(1, 0) @ (x - y))

    def l1_distance(self, x, y):
        return torch.sum(torch.abs(x - y))

    def lmax_distance(self, x, y):
        return torch.max(torch.abs(x - y))

    def lp_distance(self, x, y, p):
        lp_sum = 0
        for i in range(int(x.shape[0])):
            lp_sum += (x[i] - y[i]) ** p
        lp_sum = torch.abs(lp_sum) ** (1 / p)
        return lp_sum

    def init_cluster_centre(self, x, k_num, mode='random'):
        if mode == 'random':
            y_shape = x.shape[1]
            clus_center = torch.zeros((1, y_shape)).cuda()
            for i in range(k_num):
                clus_center_k = torch.zeros((1, 1)).cuda()
                for j in range(y_shape):
                    clus_center_inter = random.uniform(torch.max(x[:, j]), torch.min(x[:, j]))
                    clus_center_inter = torch.reshape(clus_center_inter, (1, 1)).cuda()
                    clus_center_k = torch.cat((clus_center_k, clus_center_inter), dim=1)
                clus_center_k = clus_center_k[:, 1:]
                clus_center = torch.cat((clus_center, clus_center_k))
            clus_center = clus_center[1:, :]
            return clus_center

        elif mode == 'kpp':
            dims = x.shape[1]
            init = torch.zeros((k_num, dims)).cuda()

            r = torch.distributions.uniform.Uniform(0, 1)
            for i in range(k_num):
                if i == 0:
                    init[i, :] = x[torch.randint(x.shape[0], [1])]

                else:
                    D2 = torch.cdist(init[:i, :][None, :], x[None, :], p=2)[0].amin(dim=0)
                    probs = D2 / torch.sum(D2)
                    cumprobs = torch.cumsum(probs, dim=0)
                    init[i, :] = x[torch.searchsorted(
                        cumprobs, r.sample([1]).cuda())]

            return init
        return None

    def init_cluster_centre_simple(self, x, k_num):
        y_shape = x.shape[1]
        clus_center = torch.randn((k_num, y_shape)).cuda()
        return clus_center

    def assign_data_point(self, x, init_cluster_cen):
        assigned_set = {}
        for i in range(init_cluster_cen.shape[0]):
            assigned_set[str(i)] = []

        for i in range(x.shape[0]):
            cont_dis = torch.zeros((1, 1)).cuda()
            for j in range(init_cluster_cen.shape[0]):
                if self.sel_dis == 'l2':
                    dis_value = \
                    self.l2_distance(x[i, :].reshape(x.shape[1], 1), init_cluster_cen[j, :].reshape(x.shape[1], 1))
                elif self.sel_dis == 'l1':
                    dis_value = \
                    self.l1_distance(x[i, :].reshape(x.shape[1], 1), init_cluster_cen[j, :].reshape(x.shape[1], 1))
                elif self.sel_dis == 'lp':
                    dis_value = \
                self.lp_distance(x[i, :].reshape(x.shape[1], 1), init_cluster_cen[j, :].reshape(x.shape[1], 1), p=self.p)
                elif self.sel_dis == 'lmax':
                    dis_value = \
                    self.lmax_distance(x[i, :].reshape(x.shape[1], 1), init_cluster_cen[j, :].reshape(x.shape[1], 1))
                else:
                    pass
                cont_dis = torch.cat((cont_dis, dis_value.reshape(1, 1)))
            cont_dis = cont_dis[1:, :]
            max_id = torch.argmin(cont_dis).cpu().numpy()
            assigned_set[str(max_id)].append(i)
        return assigned_set

    def assign_data_point_mat_ver(self, x, init_cluster_cen):
        assigned_set = {}
        init_cluster_order_matrix = torch.zeros((x.shape[0], 1)).cuda()
        for i in range(init_cluster_cen.shape[0]):
            assigned_set[str(i)] = []
            x_y = x - init_cluster_cen[i].expand(x.shape[0], -1)
            x_y_2 = x_y ** 2
            xxx = torch.sum(x_y_2, dim=1)
            xxx_sqrt = torch.sqrt(xxx)
            xxx_sqrt = xxx_sqrt.reshape(xxx.shape[0], 1)
            init_cluster_order_matrix = torch.cat((init_cluster_order_matrix, xxx_sqrt), dim=1)
        init_cluster_order_matrix = init_cluster_order_matrix[:, 1:]
        init_cluster_order = torch.argmin(init_cluster_order_matrix, dim=1)
        for i in range(init_cluster_cen.shape[0]):
            k = torch.nonzero(init_cluster_order == torch.tensor(i)).detach().cpu().numpy()
            k = list(k.reshape((k.shape[0])))
            assigned_set[str(i)] = k
        return assigned_set

    def upgrade_cluster_centre(self, x, assigned_set):
        new_centre = torch.zeros((1, x.shape[1])).cuda()
        for i in range(len(assigned_set)):
            new_inter = torch.mean(x[assigned_set[str(i)], :], dim=0)
            new_centre = torch.cat((new_centre, new_inter.reshape(1, x.shape[1])))
        new_centre = new_centre[1:, :]
        return new_centre

    def forward(self, x):
        # k = self.k_nums
        clus_center = self.init_cluster_centre(x, self.k_nums,mode='kpp')
        for train_i in range(self.train_iters):
            assiged_set = self.assign_data_point_mat_ver(x, clus_center)
            # assiged_set = self.assign_data_point(x, clus_center)
            new_centre = self.upgrade_cluster_centre(x, assiged_set)
            if torch.mean(new_centre) == torch.mean(clus_center):
                break
            else:
                clus_center = new_centre
        #print('train_i:', train_i)
        return assiged_set

class Sef_Cluster(nn.Module):
    def __init__(self, k_nums = 3, sel_dis = 'liad', train_iters = 50, p = 1, feature_lens = 768):
        super(Sef_Cluster, self).__init__()
        self.k_nums = k_nums
        self.sel_dis = sel_dis
        self.train_iters = train_iters
        self.p = p
        self.feature_lens = feature_lens
        self.inst_adapt_w = torch.nn.Parameter(self.create_adapt_w(),  requires_grad=True)
        self.inst_adapt_lamda = 0.1

    def create_adapt_w(self):
        rand_vect = torch.abs(torch.randn(self.feature_lens))
        rand_mat = torch.diag(rand_vect)
        adapt_w = rand_mat.cuda(0)
        return adapt_w

    def instance_adaptive_distance(self, x, y):
        dis_mat = torch.sqrt(torch.nn.ReLU()((x - y).permute(1, 0) @ self.inst_adapt_w @ (x - y)))
        dis_value = torch.squeeze(dis_mat)
        return dis_value

    def instance_adaptive_distance_mat_ver(self, x, y):
        x_y_org = x - y.expand(x.shape[0], -1)
        x_y = x_y_org @ torch.exp(self.inst_adapt_w)
        x_y_2 = x_y ** 2
        dis_sum = torch.sum(x_y_2, dim=1)
        dis_sqrt = torch.sqrt(torch.nn.ReLU()(dis_sum))
        dis_value = dis_sqrt.reshape(dis_sqrt.shape[0], 1)
        #print(dis_value.shape)
        return dis_value

    def l2_distance(self, x, y):
        return torch.squeeze(torch.sqrt((x - y).permute(1, 0) @ (x - y)))

    def l2_distance_mat_ver(self, x, y):
        x_y = x - y.expand(x.shape[0], -1)
        x_y_2 = x_y ** 2
        dis_sum = torch.sum(x_y_2, dim=1)
        dis_sqrt = torch.sqrt(dis_sum)
        dis_value = dis_sqrt.reshape(dis_sqrt.shape[0], 1)
        return dis_value


    def l1_distance(self, x, y):
        return torch.sum(torch.abs(x - y))

    def lmax_distance(self, x, y):
        return torch.max(torch.abs(x - y))

    def lp_distance(self, x, y, p):
        lp_sum = 0
        for i in range(int(x.shape[0])):
            lp_sum += (x[i] - y[i]) ** p
        lp_sum = torch.abs(lp_sum) ** (1 / p)
        return lp_sum

    def init_cluster_centre(self, x, k_num):
        y_shape = x.shape[1]
        clus_center = torch.zeros((1, y_shape)).cuda()
        for i in range(k_num):
            clus_center_k = torch.zeros((1, 1)).cuda()
            for j in range(y_shape):
                clus_center_inter = torch.tensor(random.uniform(torch.max(x[:, j]), torch.min(x[:, j])))
                clus_center_inter = torch.reshape(clus_center_inter, (1, 1)).cuda()
                clus_center_k = torch.cat((clus_center_k, clus_center_inter), dim=1)
            clus_center_k = clus_center_k[:, 1:]
            clus_center = torch.cat((clus_center, clus_center_k))
        clus_center = clus_center[1:, :]
        return clus_center

    def init_cluster_centre_simple(self, x, k_num):
        y_shape = x.shape[1]
        clus_center = torch.randn((k_num, y_shape)).cuda()
        return clus_center

    def assign_data_point(self, x, init_cluster_cen):
        assigned_set = {}
        for i in range(init_cluster_cen.shape[0]):
            assigned_set[str(i)] = []

        for i in range(x.shape[0]):
            cont_dis = torch.zeros((1, 1)).cuda()
            for j in range(init_cluster_cen.shape[0]):
                if self.sel_dis == 'l2':
                    dis_value = \
                    self.l2_distance(x[i, :].reshape(x.shape[1], 1), init_cluster_cen[j, :].reshape(x.shape[1], 1))
                elif self.sel_dis == 'l1':
                    dis_value = \
                    self.l1_distance(x[i, :].reshape(x.shape[1], 1), init_cluster_cen[j, :].reshape(x.shape[1], 1))
                elif self.sel_dis == 'lp':
                    dis_value = \
                self.lp_distance(x[i, :].reshape(x.shape[1], 1), init_cluster_cen[j, :].reshape(x.shape[1], 1), p=self.p)
                elif self.sel_dis == 'lmax':
                    dis_value = \
                    self.lmax_distance(x[i, :].reshape(x.shape[1], 1), init_cluster_cen[j, :].reshape(x.shape[1], 1))
                elif self.sel_dis == 'liad':
                    self.instance_adaptive_distance(x[i, :].reshape(x.shape[1], 1), init_cluster_cen[j, :].reshape(x.shape[1], 1))
                else:
                    print('assign_data_point error!!!')
                cont_dis = torch.cat((cont_dis, dis_value.reshape(1, 1)))
            cont_dis = cont_dis[1:, :]
            max_id = torch.argmin(cont_dis).cpu().numpy()
            assigned_set[str(max_id)].append(i)
        return assigned_set

    def assign_data_point_mat_ver(self, x, init_cluster_cen):
        assigned_set = {}
        init_cluster_order_matrix = torch.zeros((x.shape[0], 1)).cuda()
        for i in range(init_cluster_cen.shape[0]):
            assigned_set[str(i)] = []
            if self.sel_dis == 'l2':
                dis_value = self.l2_distance_mat_ver(x, init_cluster_cen[i])
            elif self.sel_dis == 'liad':
                dis_value = self.instance_adaptive_distance_mat_ver(x, init_cluster_cen[i])
            else:
                print('assign_data_point_mat_ver error!!!')
            init_cluster_order_matrix = torch.cat((init_cluster_order_matrix, dis_value), dim=1)
        init_cluster_order_matrix = init_cluster_order_matrix[:, 1:]
        init_cluster_order = torch.argmin(init_cluster_order_matrix, dim=1)
        for i in range(init_cluster_cen.shape[0]):
            k = torch.nonzero(init_cluster_order == torch.tensor(i)).detach().cpu().numpy()
            k = list(k.reshape((k.shape[0])))
            assigned_set[str(i)] = k
        return assigned_set

    def upgrade_cluster_centre(self, x, assigned_set):
        new_centre = torch.zeros((1, x.shape[1])).cuda()
        for i in range(len(assigned_set)):
            new_inter = torch.mean(x[assigned_set[str(i)], :], dim=0)
            new_centre = torch.cat((new_centre, new_inter.reshape(1, x.shape[1])))
        new_centre = new_centre[1:, :]
        return new_centre

    def forward(self, x):
        k = self.k_nums
        clus_center = self.init_cluster_centre(x, self.k_nums)
        for train_i in range(self.train_iters):
            assiged_set = self.assign_data_point_mat_ver(x, clus_center)
            new_centre = self.upgrade_cluster_centre(x, assiged_set)
            if torch.mean(new_centre) == torch.mean(clus_center):
                break
            else:
                clus_center = new_centre
        #print('train_i:', train_i)
        return assiged_set


class MIL_Parallel_Feature(nn.Module):
    def __init__(self, base_model=None, img_size=None):
        super(MIL_Parallel_Feature, self).__init__()
        self.layers_0 = base_model.layers[0]
        self.layers_1 = base_model.layers[1]
        self.layers_2 = base_model.layers[2]
        self.layers_3 = base_model.layers[3]
        self.patch_embed = base_model.patch_embed
        self.pos_drop = base_model.pos_drop
        self.norm = base_model.norm
        self.img_size = img_size
        if self.img_size[0] == 224:
            self.avgp = nn.AvgPool1d(kernel_size=49, stride=49)  # for 224* 224 input
        else:
            self.avgp = nn.AvgPool1d(kernel_size=9, stride=9)  # only for 96*96 input


    def forward(self, x):
        y = self.patch_embed(x)
        y = self.pos_drop(y)
        y = self.layers_0(y)
        y = self.layers_1(y)
        y = self.layers_2(y)
        y = self.layers_3(y)
        y = self.norm(y)
        y = self.avgp(y.permute(0, 2, 1))
        y = torch.reshape(y, (y.shape[0], y.shape[1]))
        return y


class MIL_Parallel_Head(nn.Module):
    def __init__(self, base_model = None, class_num = 3, seed=None, batch_size = 2, bags_len = 1042, model_stats = 'train',abla_type='tic',feat_extract=False, bag_weight=False, cluster_vis=False):
        super(MIL_Parallel_Head, self).__init__()
        self.head = base_model.head
        self.batch_size = batch_size
        self.bags_len = bags_len
        self.feat_extract = feat_extract
        self.bag_weight = bag_weight
        self.cluster_vis = cluster_vis
        self.seed = seed
        self.model_stats = model_stats
        self.abla_type = abla_type
        self.class_num = class_num
        self.tgi_clustering_block_Vanilla = TgiClustering(k_nums=3,sel_dis='l2')
        # self.tgi_clustering_block = Sef_Cluster(k_nums=3, sel_dis='liad',feature_lens=20)
        self.tgi_clustering_block = Sef_Cluster(k_nums=3, sel_dis='liad')



    def forward(self, x):
        # clustering guiding
        y = torch.reshape(x, (int(x.shape[0] / self.bags_len), self.bags_len, x.shape[1]))
        if self.bag_weight:
            bag_w = y[:]
            return bag_w
        min_dis = torch.zeros((1)).cuda()
        max_dis = torch.zeros((1)).cuda()
        for i in range(y.shape[0]):
            gfk = GFK(dim=20)
            y_s = y[i][0:961, :]
            y_t = y[i][961:, :]
            with torch.no_grad():
                _, unlabelled_y, target_guiding_y = gfk.fit(y_s, y_t, norm_inputs=True)

            assigned_sets = self.tgi_clustering_block.forward(unlabelled_y)
            assign_y_0 = unlabelled_y[assigned_sets['0'], :]
            assign_y_1 = unlabelled_y[assigned_sets['1'], :]
            assign_y_2 = unlabelled_y[assigned_sets['2'], :]

            if assign_y_0.shape == (0, 768) or assign_y_1.shape == (0, 768) or assign_y_2.shape == (0, 768):
                dis_0_tar = torch.tensor(0).cuda(0)
                dis_1_tar = torch.tensor(0).cuda(0)
                dis_2_tar = torch.tensor(0).cuda(0)
            else:
                dis_0_tar = self.tgi_clustering_block.instance_adaptive_distance(
                    torch.mean(target_guiding_y, dim=0, keepdim=True).permute(1, 0),
                    torch.mean(assign_y_0, dim=0, keepdim=True).permute(1, 0))
                dis_1_tar = self.tgi_clustering_block.instance_adaptive_distance(
                    torch.mean(target_guiding_y, dim=0, keepdim=True).permute(1, 0),
                    torch.mean(assign_y_1, dim=0, keepdim=True).permute(1, 0))
                dis_2_tar = self.tgi_clustering_block.instance_adaptive_distance(
                    torch.mean(target_guiding_y, dim=0, keepdim=True).permute(1, 0),
                    torch.mean(assign_y_2, dim=0, keepdim=True).permute(1, 0))

                ###adaptive dis
            if (dis_0_tar == torch.tensor(0).cuda(0)) or (dis_1_tar == torch.tensor(0).cuda(0)) or (
                    dis_2_tar == torch.tensor(0).cuda(0)):
                dis_0_rate = torch.tensor(0).cuda(0)
                dis_1_rate = torch.tensor(0).cuda(0)
                dis_2_rate = torch.tensor(0).cuda(0)
            else:
                dis_0_rate = (dis_0_tar / (dis_0_tar + dis_1_tar + dis_2_tar)) * 0.1
                dis_1_rate = (dis_1_tar / (dis_0_tar + dis_1_tar + dis_2_tar)) * 0.1
                dis_2_rate = (dis_2_tar / (dis_0_tar + dis_1_tar + dis_2_tar)) * 0.1

            #  Cluser idx
            dis_rates = {
                '0': dis_0_rate,
                '1': dis_1_rate,
                '2': dis_2_rate
            }

            sorted_assigned_sets = [assigned_sets[k] for k in sorted(dis_rates, key=dis_rates.get)]

            min_dis += min([dis_0_tar, dis_1_tar, dis_2_tar])
            max_dis += max([dis_0_tar, dis_1_tar, dis_2_tar])

            y[i][0:961, :][assigned_sets['0'], :] = (1 - dis_0_rate) * y[i][0:961, :][assigned_sets['0'], :]
            y[i][0:961, :][assigned_sets['1'], :] = (1 - dis_1_rate) * y[i][0:961, :][assigned_sets['1'], :]
            y[i][0:961, :][assigned_sets['2'], :] = (1 - dis_2_rate) * y[i][0:961, :][assigned_sets['2'], :]

            if self.cluster_vis:
                return sorted_assigned_sets

        min_dis = min_dis / y.shape[0]
        max_dis = max_dis / y.shape[0]
        final_y = torch.mean(y, dim=1, keepdim=True)
        final_y = torch.reshape(final_y, (final_y.shape[0], final_y.shape[2]))
        y_logits = self.head(final_y)
        if self.feat_extract:
            return final_y
        else:
            return y_logits, min_dis, max_dis

class MIL_Parallel_Head_ub(nn.Module):
    def __init__(self, base_model = None, class_num = 3, seed=None, batch_size = 2, bags_len = 1042, model_stats = 'train',abla_type='tic',feat_extract=False, cluster_vis=False, bag_weight=False):
        super(MIL_Parallel_Head_ub, self).__init__()
        self.head = base_model.head
        self.batch_size = batch_size
        self.bags_len = bags_len
        self.feat_extract = feat_extract
        self.bag_weight = bag_weight
        self.cluster_vis = cluster_vis
        self.seed = seed
        self.model_stats = model_stats
        self.abla_type = abla_type
        self.class_num = class_num
        self.tgi_clustering_block_Vanilla = TgiClustering(k_nums=3, sel_dis='l2')
        # self.tgi_clustering_block = Sef_Cluster(k_nums=3, sel_dis='liad',feature_lens=20)
        self.tgi_clustering_block = Sef_Cluster(k_nums=3, sel_dis='liad')



    def forward(self, x):

        if self.abla_type == 'sota':
            # clustering guiding
            y = x.unsqueeze(dim=0)
            if self.bag_weight:
                bag_w = y[:]
                return bag_w
            min_dis = torch.zeros((1)).cuda()
            max_dis = torch.zeros((1)).cuda()
            for i in range(y.shape[0]):
                gfk = GFK(dim=20)
                y_s = y[i][:-5, :]
                y_t = y[i][-5:, :]
                with torch.no_grad():
                    _, unlabelled_y, target_guiding_y = gfk.fit(y_s, y_t, norm_inputs=True)

                assigned_sets = self.tgi_clustering_block.forward(unlabelled_y)
                assign_y_0 = unlabelled_y[assigned_sets['0'], :]
                assign_y_1 = unlabelled_y[assigned_sets['1'], :]
                assign_y_2 = unlabelled_y[assigned_sets['2'], :]

                if assign_y_0.shape == (0, 768) or assign_y_1.shape == (0, 768) or assign_y_2.shape == (0, 768):
                    dis_0_tar = torch.tensor(0).cuda(0)
                    dis_1_tar = torch.tensor(0).cuda(0)
                    dis_2_tar = torch.tensor(0).cuda(0)
                else:
                    dis_0_tar = self.tgi_clustering_block.instance_adaptive_distance(
                        torch.mean(target_guiding_y, dim=0, keepdim=True).permute(1, 0),
                        torch.mean(assign_y_0, dim=0, keepdim=True).permute(1, 0))
                    dis_1_tar = self.tgi_clustering_block.instance_adaptive_distance(
                        torch.mean(target_guiding_y, dim=0, keepdim=True).permute(1, 0),
                        torch.mean(assign_y_1, dim=0, keepdim=True).permute(1, 0))
                    dis_2_tar = self.tgi_clustering_block.instance_adaptive_distance(
                        torch.mean(target_guiding_y, dim=0, keepdim=True).permute(1, 0),
                        torch.mean(assign_y_2, dim=0, keepdim=True).permute(1, 0))

                    ###adaptive dis
                if (dis_0_tar == torch.tensor(0).cuda(0)) or (dis_1_tar == torch.tensor(0).cuda(0)) or (
                        dis_2_tar == torch.tensor(0).cuda(0)):
                    dis_0_rate = torch.tensor(0).cuda(0)
                    dis_1_rate = torch.tensor(0).cuda(0)
                    dis_2_rate = torch.tensor(0).cuda(0)
                else:
                    dis_0_rate = (dis_0_tar / (dis_0_tar + dis_1_tar + dis_2_tar)) * 0.1
                    dis_1_rate = (dis_1_tar / (dis_0_tar + dis_1_tar + dis_2_tar)) * 0.1
                    dis_2_rate = (dis_2_tar / (dis_0_tar + dis_1_tar + dis_2_tar)) * 0.1

                #  Cluser idx
                dis_rates = {
                        '0': dis_0_rate,
                        '1': dis_1_rate,
                        '2': dis_2_rate
                    }

                sorted_assigned_sets = [assigned_sets[k] for k in sorted(dis_rates, key=dis_rates.get)]

                min_dis += min([dis_0_tar, dis_1_tar, dis_2_tar])
                max_dis += max([dis_0_tar, dis_1_tar, dis_2_tar])

                y[i][:-5, :][assigned_sets['0'], :] = (1-dis_0_rate) * y[i][:-5, :][assigned_sets['0'], :]
                y[i][:-5, :][assigned_sets['1'], :] = (1-dis_1_rate) * y[i][:-5, :][assigned_sets['1'], :]
                y[i][:-5, :][assigned_sets['2'], :] = (1-dis_2_rate) * y[i][:-5, :][assigned_sets['2'], :]

                if self.cluster_vis:
                    return sorted_assigned_sets

            min_dis = min_dis / y.shape[0]
            max_dis = max_dis / y.shape[0]
            # final_y = 0.1 * torch.mean(y, dim=1,keepdim=True) + torch.mean(original_y, dim=1,keepdim=True)  # residual
            final_y = torch.mean(y, dim=1, keepdim=True)
            final_y = torch.reshape(final_y, (final_y.shape[0], final_y.shape[2]))
            y_logits = self.head(final_y)
            if self.feat_extract:
                return final_y
            else:
                return y_logits, min_dis, max_dis # SOTA:tic+attn
            # return y1, min_dis, non_min_dis   # tic: del max and only mean agg
        elif self.abla_type == 'tic_mani':
            # clustering guiding
            y = x.unsqueeze(dim=0)
            if self.bag_weight:
                bag_w = torch.mean(y, dim=2, keepdim=True)
                return bag_w
            min_dis = torch.zeros((1)).cuda()
            max_dis = torch.zeros((1)).cuda()
            for i in range(y.shape[0]): 
                y_s = y[i][:-5, :]
                y_t = y[i][-5:, :]
                with torch.no_grad():
                    pca_model = PCA(n_components=20)
                    unlabelled_y = pca_model.fit_transform(y_s)
                    target_guiding_y = pca_model.transform(y_t)

                if unlabelled_y.shape[0] < 20:
                    final_y = torch.mean(y, dim=1, keepdim=True)
                    final_y = torch.reshape(final_y, (final_y.shape[0], final_y.shape[2]))
                    y_logits = self.head(final_y)
                    return y_logits, min_dis, max_dis
                
                assigned_sets = self.tgi_clustering_block.forward(unlabelled_y)
                assign_y_0 = unlabelled_y[assigned_sets['0'], :]
                assign_y_1 = unlabelled_y[assigned_sets['1'], :]
                assign_y_2 = unlabelled_y[assigned_sets['2'], :]

                if assign_y_0.shape == (0, 20) or assign_y_1.shape == (0, 20) or assign_y_2.shape == (0, 20):
                    dis_0_tar = torch.tensor(0).cuda(0)
                    dis_1_tar = torch.tensor(0).cuda(0)
                    dis_2_tar = torch.tensor(0).cuda(0)
                else:
                    dis_0_tar = self.tgi_clustering_block.instance_adaptive_distance(
                        torch.mean(target_guiding_y, dim=0, keepdim=True).permute(1, 0),
                        torch.mean(assign_y_0, dim=0, keepdim=True).permute(1, 0))
                    dis_1_tar = self.tgi_clustering_block.instance_adaptive_distance(
                        torch.mean(target_guiding_y, dim=0, keepdim=True).permute(1, 0),
                        torch.mean(assign_y_1, dim=0, keepdim=True).permute(1, 0))
                    dis_2_tar = self.tgi_clustering_block.instance_adaptive_distance(
                        torch.mean(target_guiding_y, dim=0, keepdim=True).permute(1, 0),
                        torch.mean(assign_y_2, dim=0, keepdim=True).permute(1, 0))

                if (dis_0_tar == torch.tensor(0).cuda(0)) or (dis_1_tar == torch.tensor(0).cuda(0)) or (
                        dis_2_tar == torch.tensor(0).cuda(0)):
                    dis_0_rate = torch.tensor(0).cuda(0)
                    dis_1_rate = torch.tensor(0).cuda(0)
                    dis_2_rate = torch.tensor(0).cuda(0)
                else:
                    dis_0_rate = (dis_0_tar / (dis_0_tar + dis_1_tar + dis_2_tar)) * 0.1
                    dis_1_rate = (dis_1_tar / (dis_0_tar + dis_1_tar + dis_2_tar)) * 0.1
                    dis_2_rate = (dis_2_tar / (dis_0_tar + dis_1_tar + dis_2_tar)) * 0.1

                y[i][:-5, :][assigned_sets['0'], :] = (1 - dis_0_rate) * y[i][:-5, :][assigned_sets['0'], :]
                y[i][:-5, :][assigned_sets['1'], :] = (1 - dis_1_rate) * y[i][:-5, :][assigned_sets['1'], :]
                y[i][:-5, :][assigned_sets['2'], :] = (1 - dis_2_rate) * y[i][:-5, :][assigned_sets['2'], :]
            final_y = torch.mean(y, dim=1, keepdim=True)
            final_y = torch.reshape(final_y, (final_y.shape[0], final_y.shape[2]))
            y_logits = self.head(final_y)
            if self.feat_extract:
                return final_y
            else:
                return y_logits, min_dis, max_dis  # SOTA:tic+attn
        elif self.abla_type == 'tic':
            # clustering guiding
            y = x.unsqueeze(dim=0)
            if self.bag_weight:
                bag_w = torch.mean(y, dim=2, keepdim=True)
                return bag_w
            min_dis = torch.zeros((1)).cuda()
            max_dis = torch.zeros((1)).cuda()
            for i in range(y.shape[0]):
                
                unlabelled_y = y[i][:-5, :]
                target_guiding_y = y[i][-5:, :]

                assigned_sets = self.tgi_clustering_block_Vanilla.forward(unlabelled_y)
                assign_y_0 = unlabelled_y[assigned_sets['0'], :]
                assign_y_1 = unlabelled_y[assigned_sets['1'], :]
                assign_y_2 = unlabelled_y[assigned_sets['2'], :]

                if assign_y_0.shape == (0, 768) or assign_y_1.shape == (0, 768) or assign_y_2.shape == (0, 768):
                    dis_0_tar = torch.tensor(0).cuda(0)
                    dis_1_tar = torch.tensor(0).cuda(0)
                    dis_2_tar = torch.tensor(0).cuda(0)
                else:
                    dis_0_tar = self.tgi_clustering_block_Vanilla.l2_distance(
                        torch.mean(target_guiding_y, dim=0, keepdim=True).permute(1, 0),
                        torch.mean(assign_y_0, dim=0, keepdim=True).permute(1, 0))
                    dis_1_tar = self.tgi_clustering_block_Vanilla.l2_distance(
                        torch.mean(target_guiding_y, dim=0, keepdim=True).permute(1, 0),
                        torch.mean(assign_y_1, dim=0, keepdim=True).permute(1, 0))
                    dis_2_tar = self.tgi_clustering_block_Vanilla.l2_distance(
                        torch.mean(target_guiding_y, dim=0, keepdim=True).permute(1, 0),
                        torch.mean(assign_y_2, dim=0, keepdim=True).permute(1, 0))

                if (dis_0_tar == torch.tensor(0).cuda(0)) or (dis_1_tar == torch.tensor(0).cuda(0)) or (
                        dis_2_tar == torch.tensor(0).cuda(0)):
                    dis_0_rate = torch.tensor(0).cuda(0)
                    dis_1_rate = torch.tensor(0).cuda(0)
                    dis_2_rate = torch.tensor(0).cuda(0)
                else:
                    dis_0_rate = (dis_0_tar / (dis_0_tar + dis_1_tar + dis_2_tar)) * 0.1
                    dis_1_rate = (dis_1_tar / (dis_0_tar + dis_1_tar + dis_2_tar)) * 0.1
                    dis_2_rate = (dis_2_tar / (dis_0_tar + dis_1_tar + dis_2_tar)) * 0.1

                y[i][:-5, :][assigned_sets['0'], :] = (1 - dis_0_rate) * y[i][:-5, :][assigned_sets['0'], :]
                y[i][:-5, :][assigned_sets['1'], :] = (1 - dis_1_rate) * y[i][:-5, :][assigned_sets['1'], :]
                y[i][:-5, :][assigned_sets['2'], :] = (1 - dis_2_rate) * y[i][:-5, :][assigned_sets['2'], :]
            final_y = torch.mean(y, dim=1, keepdim=True)
            final_y = torch.reshape(final_y, (final_y.shape[0], final_y.shape[2]))
            y_logits = self.head(final_y)
            if self.feat_extract:
                return final_y
            else:
                return y_logits, min_dis, max_dis  # SOTA:tic+attn
        else:
            y = x[:-5,:] # unbalanced batch
            y = torch.mean(y, dim=0, keepdim=True)  # unbalanced batch
            y = self.head(y)
            return y


