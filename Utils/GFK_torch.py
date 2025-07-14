# This is an implementation of GFK in Pytorch with reference to
# Gong B, Shi Y, Sha F, et al. Geodesic flow kernel for unsupervised domain adaptation. CVPR2012
# Note: The numpy version can be found in package bob.learn.linear.GFK
import torch


def degenerated_GSVD(A,B):
    """
    Implementation of the Generalized SVD algorithm where matrix A is invertible

    :param A:  matrix A with shape [n,n] 10 10
    :param B:  matrix B with shape [b,n] 758 10
    :return: [V1,V2,V,Gam,Sig] where A = V1*Gam*V^T and B = V2*Sig*V^T are both SVD.
    """
    B_InvA = B @ torch.linalg.inv(A)
    V2,S,V1_h = torch.linalg.svd(B_InvA)
    V1 = V1_h.H
    # assert torch.dist(B_InvA, V2[:, :A.shape[1]] @ torch.diag(S) @ V1_h) < 1e-3

    Gam_i = 1 / torch.sqrt(1+S**2)
    Sig_i = S / torch.sqrt(1+S**2)

    Sig = torch.zeros_like(B)# shape [b,n]
    for i in range(len(S)):
        Sig[i,i] = Sig_i[i]
    V = A.T @ V1 @ torch.diag(1 / Gam_i)
    Gam = torch.diag(Gam_i)

    # assert torch.dist(A, V1 @ Gam @ V.H) < 1e-3
    # assert torch.dist(B, V2 @ Sig @ V.H) < 1e-3
    return V1,V2,V,Gam,Sig

def null_space(A):
    # A is colum Orthogonal matrix
    U, _, _ = torch.linalg.svd(A, full_matrices=True)

    return U


def sqrt_SymTensor(t):
    eigvals, eigvecs = torch.linalg.eigh(t)
    eigvals = eigvals.to(torch.complex64)
    eigvecs = eigvecs.to(torch.complex64)
    t_half = eigvecs @ torch.diag(eigvals.pow(0.5)) @ eigvecs.T
    t_half = torch.real(t_half)
    return t_half

class GFK:
    def __init__(self, dim=20):
        """
        Init func
        :param dim: dimension after GFK
        """
        self.dim = dim
        self.eps = 1e-20

    def znorm(self, data):
        """
        Z-Normaliza
        """
        mu = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        data = (data - mu) / std
        return data

    def pca_proj(self,X, threshold=0.99):
        """
        使用 PyTorch 计算 PCA，并自动选择维度，使得累计贡献率超过 threshold（默认为 99%）。

        参数：
        X : torch.Tensor (n, d) - 输入数据，每行是一个样本
        threshold : float - 贡献率阈值（默认为 0.99）

        返回：
        P : torch.Tensor (d, k) - 投影矩阵（主成分）
        """
        # 1. 均值中心化
        X_mean = X.mean(dim=0, keepdim=True)  # 计算均值
        X_centered = X - X_mean  # 去中心化

        # 2. 计算协方差矩阵
        cov_matrix = torch.mm(X_centered.T, X_centered) / (X.shape[0] - 1)

        # 3. 进行 SVD 分解
        U, S, Vh = torch.linalg.svd(cov_matrix)

        # 4. 计算贡献率
        explained_variance_ratio = S / S.sum()  # 贡献率
        cumulative_variance = torch.cumsum(explained_variance_ratio, dim=0)  # 累积贡献率

        # 5. 选择满足贡献率超过 threshold 的最小维度 k
        k = torch.where(cumulative_variance >= threshold)[0][0].item() + 1  # 第一个超过 threshold 的索引 +1

        # 6. 选择前 k 个主成分
        P = U[:, :k]  # 投影矩阵（前 k 个特征向量）


        return P

    def fit(self, Xs, Xt, norm_inputs=None):
        """
        Obtain the kernel G
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :param norm_inputs: normalize the inputs or not
        :return: GFK kernel G
        """
        if norm_inputs:
            source = self.znorm(Xs)
            target = self.znorm(Xt)
        else:
            source = Xs
            target = Xt
        Ps = self.pca_proj(source,threshold=0.99)
        Pt = self.pca_proj(target, threshold=0.99)
        Ps_complement = null_space(Ps)[:, Ps.shape[1]:]
        Ps = torch.hstack((Ps, Ps_complement))

        # assert torch.dist(torch.eye(Ps.shape[0]).cuda(), Ps.T @ Ps) < 1e-3

        Pt = Pt[:, :self.dim]
        N = Ps.shape[1]
        dim = Pt.shape[1]

        # Principal angles between subspaces
        QPt = Ps.T @ Pt

        # [V1,V2,V,Gam,Sig] = gsvd(QPt(1:dim,:), QPt(dim+1:end,:));
        A = QPt[0:dim, :]
        # assert torch.dist(torch.linalg.inv(A) @ A, torch.eye(dim)) < 1e-3 # check the orthogonality of A
        B = QPt[dim:, :]
        V1, V2, V, Gam, Sig = degenerated_GSVD(A, B) # Since A is invertible, GSVD can be degenerated in a simpler way

        V2 = -V2
        # print(torch.diag(torch.diag(Gam)))
        theta = torch.arccos(torch.diag(Gam))

        # Equation (6)
        B1 = torch.diag(0.5 * (1 + (torch.sin(2 * theta) / (2. * torch.maximum(theta, torch.tensor(self.eps))))))
        B2 = torch.diag(0.5 * ((torch.cos(2 * theta) - 1) / (2 * torch.maximum(theta, torch.tensor(self.eps)))))
        B3 = B2.clone()  # B3 和 B2 相同
        B4 = torch.diag(0.5 * (1 - (torch.sin(2 * theta) / (2. * torch.maximum(theta, torch.tensor(self.eps))))))

        # Equation (9) of the suplementary matetial
        delta1_1 = torch.hstack((V1, torch.zeros((dim, N - dim)).cuda()))
        delta1_2 = torch.hstack((torch.zeros((N - dim, dim)).cuda(), V2))
        delta1 = torch.vstack((delta1_1, delta1_2))

        delta2_1 = torch.hstack((B1, B2, torch.zeros((dim, N - 2 * dim)).cuda()))
        delta2_2 = torch.hstack((B3, B4, torch.zeros((dim, N - 2 * dim)).cuda()))
        delta2_3 = torch.zeros((N - 2 * dim, N)).cuda()
        delta2 = torch.vstack((delta2_1, delta2_2, delta2_3))

        delta3_1 = torch.hstack((V1, torch.zeros((dim, N - dim)).cuda()))
        delta3_2 = torch.hstack((torch.zeros((N - dim, dim)).cuda(), V2))
        delta3 = torch.vstack((delta3_1, delta3_2)).T

        delta = delta1 @ delta2 @ delta3
        G = Ps @ delta @ Ps.T
        sqG = torch.real(sqrt_SymTensor(G))
        Xs_new = (sqG @ Xs.T).T
        Xt_new = (sqG @ Xt.T).T

        return G, Xs_new, Xt_new




# =======================
# 下面是一个简单测试示例
# =======================
if __name__ == "__main__":
    A = torch.randn(10, 768).cuda()
    B = torch.randn(10, 768).cuda()
    gfk = GFK(dim=20)
    G, A_new, B_new = gfk.fit(A,B,norm_inputs=True)
    # print(A_new)
    # print(B_new)
    # print(G)
    # print(G.shape, A_new.shape, B_new.shape)