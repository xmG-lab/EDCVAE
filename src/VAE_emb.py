import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import torch.distributions as D
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
import anndata as ad
import networkx as nx
from sklearn.preprocessing import StandardScaler
import scglue
from itertools import chain

torch.cuda.empty_cache()
Tensor = torch.cuda.FloatTensor

# ----------------------- GraphEncoder -----------------------
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.leaky_relu = nn.LeakyReLU(0.2)
    def forward(self, x, adj):
        support = self.linear(x)
        out = torch.matmul(adj, support)
        out = self.leaky_relu(out)
        return out

class GraphEncoder(nn.Module):
    def __init__(self, n_nodes, z_dim):
        """
        n_nodes: 图中节点数 = n_genes + n_peaks
        z_dim: 每个节点的隐变量维度
        """
        super(GraphEncoder, self).__init__()
        self.n_nodes = n_nodes
        self.z_dim = z_dim
        # 初始化每个节点的初始特征，维度设为 z_dim
        self.node_features = nn.Parameter(torch.randn(n_nodes, z_dim))
        # 两层GCN层，使用 LeakyReLU 激活
        self.gcn1 = GCNLayer(z_dim, z_dim)
        self.gcn2 = GCNLayer(z_dim, z_dim)
        # 生成均值和对数方差
        self.fc_mu = nn.Linear(z_dim, z_dim)
        self.fc_logvar = nn.Linear(z_dim, z_dim)

    def forward(self, adj):
        """
        adj: 图的邻接矩阵，形状 (n_nodes, n_nodes)
        返回一个 Normal 分布，其均值为所有节点隐变量扁平化后的向量
        """
        norm_adj = adj
        # 利用GCN更新节点特征
        x = self.gcn1(self.node_features, norm_adj)
        x = self.gcn2(x, norm_adj)
        # 计算均值和对数方差
        mu = self.fc_mu(x)       # 形状 (n_nodes, z_dim)
        logvar = self.fc_logvar(x)  # 形状 (n_nodes, z_dim)
        # 扁平化为一个向量
        mu_flat = mu.view(-1)
        logvar_flat = logvar.view(-1)
        return D.Normal(mu_flat, torch.exp(0.5 * logvar_flat))
    

# ----------------------- GraphDecoder (改进版) -----------------------
class GraphDecoder(nn.Module):
    def __init__(self, z_dim, n_nodes):
        super(GraphDecoder, self).__init__()
        self.n_nodes = n_nodes
        self.z_dim = z_dim
        self.fc1 = nn.Linear(n_nodes, n_nodes)
        self.fc2 = nn.Linear(n_nodes, n_nodes)
        self.fc3 = nn.Linear(n_nodes, n_nodes)
        self.fc_mu = nn.Linear(n_nodes, n_nodes)
        self.fc_logvar = nn.Linear(n_nodes, n_nodes)
        self.tanh = nn.Tanh()

    def forward(self, latent_flat):
        """
        latent_flat: 扁平化后的节点隐变量向量，形状 (n_nodes * z_dim,)
        返回一个 Normal 分布，其均值为重构的邻接矩阵（通过非线性层和内积计算）
        """
        # 将扁平向量恢复为 (n_nodes, z_dim)
        node_latent = latent_flat.view(self.n_nodes, self.z_dim)

        # 内积生成初步重构邻接矩阵
        recon = torch.matmul(node_latent, node_latent.T)  # (n_nodes, n_nodes)

        # 三层非线性变换
        h = self.tanh(self.fc1(recon))
        h = self.tanh(self.fc2(h))
        h = self.tanh(self.fc3(h))

        # 输出均值与方差
        mu = self.fc_mu(h)
        sigma = torch.exp(0.5 * self.fc_logvar(h))  # 使用标准差构造正态分布
        return D.Normal(mu, sigma)




# ----------------------- DataEncoder -----------------------
class DataEncoder(nn.Module):
    """数据编码器：从基因表达数据编码细胞的潜在分布"""
    def __init__(self, n_genes, z_dim, hidden_dim=128):
        super(DataEncoder, self).__init__()
        # 增加额外的全连接层形成深层网络，并使用残差连接
        self.fc1 = nn.Linear(n_genes, hidden_dim)
        self.fc_res = nn.Linear(hidden_dim, hidden_dim)  # 残差分支
        self.fc2 = nn.Linear(hidden_dim, z_dim)
        self.fc_mu = nn.Linear(z_dim, z_dim)
        self.fc_logvar = nn.Linear(z_dim, z_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)


    def encode_data(self, modality, data):
        if modality not in self.x2u:
            raise ValueError(f"Modality {modality} not found.")
        with torch.no_grad():
            data = torch.from_numpy(data).float().to(self.device())
            dist = self.x2u[modality](data)
            return dist.mean.cpu().numpy()

    def forward(self, x):
        # x: (batch, n_genes)
        h1 = self.leaky_relu(self.fc1(x))
        # 残差连接
        h_res = self.leaky_relu(self.fc_res(h1))
        h1 = h1 + h_res
        h1 = self.dropout(h1)
        h2 = self.leaky_relu(self.fc2(h1))
        h2 = self.dropout(h2)
        mu = self.fc_mu(h2)
        logvar = self.fc_logvar(h2)
        return D.Normal(mu, torch.exp(0.5 * logvar))
    

# ----------------------- DataDecoder Definitions -----------------------
    
class ExpressionDecoder(nn.Module):
    """专门给 RNA 用的解码器，结构可以与 ATAC 不同"""
    def __init__(self, z_dim, n_genes):
        super(ExpressionDecoder, self).__init__()
        # 例如：RNA 解码器多一层 FC
        self.fc1 = nn.Linear(n_genes, n_genes)
        self.fc2 = nn.Linear(n_genes, n_genes)
        self.fc3 = nn.Linear(n_genes, n_genes)
        self.fc_mu = nn.Linear(n_genes, n_genes)
        self.fc_logvar = nn.Linear(n_genes, n_genes)
        self.tanh = nn.Tanh()
    def forward(self, u, v):
        recon = u @ v.T                          # (B, n_genes)
        h = self.tanh(self.fc1(recon))
        h = self.tanh(self.fc2(h))
        h = self.tanh(self.fc3(h))
        mu = self.fc_mu(h)
        sigma = torch.exp(0.5 * self.fc_logvar(h))
        return D.Normal(mu, sigma)

class AtacDecoder(nn.Module):
    """专门给 ATAC 用的解码器，结构可以不同于 RNA"""
    def __init__(self, z_dim, n_peaks):
        super(AtacDecoder, self).__init__()
        # 例如：ATAC 解码器只有两层 FC
        self.fc1 = nn.Linear(n_peaks, n_peaks)
        self.fc2 = nn.Linear(n_peaks, n_peaks)
        self.fc_mu = nn.Linear(n_peaks, n_peaks)
        self.fc_logvar = nn.Linear(n_peaks, n_peaks)
        self.relu = nn.ReLU()
    def forward(self, u, v):
        recon = u @ v.T                          # (B, n_peaks)
        h = self.relu(self.fc1(recon))
        h = self.relu(self.fc2(h))
        mu = self.fc_mu(h)
        sigma = torch.exp(0.5 * self.fc_logvar(h))
        return D.Normal(mu, sigma)


class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        # 定义隐藏层尺寸，例如扩大2倍，再逐步降维
        hidden_dim1 = z_dim * 2
        hidden_dim2 = z_dim
        hidden_dim3 = z_dim // 2
        
        self.fc1 = nn.Linear(z_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, 1)  # 输出单一 logit
        
        # 使用 LeakyReLU 激活函数，使梯度在负区间也能流动
        self.leaky_relu = nn.LeakyReLU(0.2)
        # 添加 Dropout 层防止过拟合
        self.dropout = nn.Dropout(0.3)

    def forward(self, u):
        x = self.leaky_relu(self.fc1(u))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc3(x))
        x = self.dropout(x)
        logits = self.fc4(x).squeeze(1)
        return logits




class Prior(nn.Module):
    """先验分布：标准正态分布"""
    def __init__(self, z_dim):
        super(Prior, self).__init__()
        self.z_dim = z_dim

    def forward(self):
        return D.Normal(torch.zeros(self.z_dim).cuda(), torch.ones(self.z_dim).cuda())


import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.distributions as D
from torch.cuda.amp import autocast, GradScaler

torch.cuda.empty_cache()
Tensor = torch.cuda.FloatTensor

# ----------------------- SharedEncoder -----------------------
class SharedEncoder(nn.Module):
    def __init__(self, hidden_dim, z_dim):
        super(SharedEncoder, self).__init__()
        # 从隐藏层到潜在空间的共享网络
        self.fc_res = nn.Linear(hidden_dim, hidden_dim)  # 残差分支
        self.fc2 = nn.Linear(hidden_dim, z_dim)
        self.fc_mu = nn.Linear(z_dim, z_dim)
        self.fc_logvar = nn.Linear(z_dim, z_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, h):
        # 残差连接
        h_res = self.leaky_relu(self.fc_res(h))
        h = h + h_res
        h = self.dropout(h)
        h2 = self.leaky_relu(self.fc2(h))
        h2 = self.dropout(h2)
        mu = self.fc_mu(h2)
        logvar = self.fc_logvar(h2)
        return D.Normal(mu, torch.exp(0.5 * logvar))

# ----------------------- DataEncoder -----------------------
class DataEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, shared_encoder):
        super(DataEncoder, self).__init__()
        # 模态特异的输入层
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        # 共享的编码器
        self.shared_encoder = shared_encoder
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 输入层映射到共享隐藏维度
        h = self.leaky_relu(self.input_layer(x))
        h = self.dropout(h)
        # 通过共享编码器生成潜在分布
        return self.shared_encoder(h)

# ------------------------ GLUE Network Definition -------------------------
class GLUE(nn.Module):
    def __init__(self, adj_A_init, n_genes, n_peaks, z_dim, hidden_dim=128):
        super(GLUE, self).__init__()
        self.n_genes = n_genes
        self.n_peaks = n_peaks
        self.n_nodes = adj_A_init.shape[0]
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        # 共享的编码器模块
        self.shared_encoder = SharedEncoder(hidden_dim, z_dim)

        # 数据编码器：为 RNA 和 ATAC 定义专属输入层，共享后续编码器
        self.x2u = nn.ModuleDict({
            "expression": DataEncoder(n_genes, hidden_dim, self.shared_encoder),
            "atac": DataEncoder(n_peaks, hidden_dim, self.shared_encoder)
        })

        # 将初始邻接矩阵转为可训练参数
        self.adj_A = nn.Parameter(torch.from_numpy(adj_A_init).float().cuda(), requires_grad=True)
        # 图编码器和解码器
        self.g2v = GraphEncoder(self.n_nodes, z_dim)
        self.v2g = GraphDecoder(z_dim, self.n_nodes)
        
        # 数据解码器
        self.u2x = nn.ModuleDict({
            "expression": ExpressionDecoder(z_dim, n_genes),
            "atac": AtacDecoder(z_dim, n_peaks)
        })

        self.idx = {
            "expression": torch.arange(n_genes).cuda(),
            "atac": torch.arange(n_genes, n_genes + n_peaks).cuda()
        }
        self.du = Discriminator(z_dim)
        self.prior_data = Prior(z_dim)
        self.prior_graph = Prior(self.n_nodes * z_dim)
        self.keys = list(self.x2u.keys())

    def encode_data(self, modality, data):
        if modality not in self.x2u:
            raise ValueError(f"Modality {modality} not found.")
        with torch.no_grad():
            data = torch.from_numpy(data).float().to(self.device())
            dist = self.x2u[modality](data)
            return dist.mean.cpu().numpy()

    def encode_graph(self):
        with torch.no_grad():
            v_dist = self.g2v(self.adj_A)
            v_mean = v_dist.mean
            v_mean = v_mean.view(self.n_nodes, self.z_dim)
            return v_mean.cpu().numpy()
        

    def get_shared_embeddings(self, modality, data):
        """
        获取共享权重编码器后的嵌入表示
        modality: "expression" 或 "atac"
        data: 输入数据 (numpy array)
        """
        if modality not in self.x2u:
            raise ValueError(f"Modality {modality} not found.")
        with torch.no_grad():
            # 将数据转换为 Tensor 并移到 GPU
            data = torch.from_numpy(data).float().to(self.device())
            # 通过模态特异输入层
            h = self.x2u[modality].input_layer(data)
            h = self.x2u[modality].leaky_relu(h)
            h = self.x2u[modality].dropout(h)
            # 通过共享编码器获取潜在分布均值
            dist = self.shared_encoder(h)
            return dist.mean.cpu().numpy()

    def device(self):
        return self.adj_A.device

# 以下部分（如 GLUETrainer、其他类）保持不变



class GLUETrainer:
    """GLUE训练器"""
    def __init__(self, net, lam_data=1.0, lam_kl=1.0, lam_graph=1.0, lam_align=1.0,
                 modality_weight={"expression": 1.0, "atac": 1.0}, lr=0.0001):
        self.net = net.cuda()
        self.lam_data = lam_data
        self.lam_kl = lam_kl
        self.lam_graph = lam_graph
        self.lam_align = lam_align
        self.modality_weight = modality_weight
        
        # 优化器分别更新图编码器、数据编码器、解码器
        self.vae_optim = optim.Adam(
            itertools.chain(net.g2v.parameters(), net.v2g.parameters(),
                            net.x2u.parameters(), net.u2x.parameters()), lr=lr)
        self.disc_optim = optim.Adam(net.du.parameters(), lr=lr)
        self.adj_optim = optim.Adam([net.adj_A], lr=lr * 0.02)
        self.scheduler = optim.lr_scheduler.StepLR(self.vae_optim, step_size=20, gamma=0.5)

    def compute_losses(self, data_expression, data_atac, epoch, dsc_only=False):
        net = self.net
        # 构造数据字典
        x = {"expression": data_expression, "atac": data_atac}
        # 构造模态标记：expression 为 0，atac 为 1
        modality_flags = {
            "expression": torch.zeros(data_expression.size(0), dtype=torch.int64).cuda(),
            "atac": torch.ones(data_atac.size(0), dtype=torch.int64).cuda()
        }
        
        # 数据编码：先通过模态特异输入层，再通过共享编码器
        h = {mod: net.x2u[mod].input_layer(x[mod]) for mod in net.keys}
        h = {mod: net.x2u[mod].leaky_relu(h[mod]) for mod in net.keys}
        h = {mod: net.x2u[mod].dropout(h[mod]) for mod in net.keys}
        q_z_x = {mod: net.shared_encoder(h[mod]) for mod in net.keys}
        
        # 重参数采样
        z_sample = {mod: q_z_x[mod].rsample() for mod in net.keys}
        prior_data = net.prior_data()   # p(z)
        prior_graph = net.prior_graph() # p(v)
        
        # 判别器损失（对齐不同模态的潜变量）
        concat_means = torch.cat([q_z_x[mod].mean for mod in net.keys])
        concat_flags = torch.cat([modality_flags[mod] for mod in net.keys])
        dsc_loss = F.binary_cross_entropy_with_logits(net.du(concat_means), concat_flags.float())
        if dsc_only:
            return {"dsc_loss": self.lam_align * dsc_loss}
        
        # ---- 图编码及图解码部分 ----
        q_v = net.g2v(net.adj_A)
        v_sample = q_v.rsample()
        p_G_v = net.v2g(v_sample)
        G_recon_mean = p_G_v.mean
        
        nonzero_idx = torch.nonzero(net.adj_A, as_tuple=False)
        G_target = net.adj_A[nonzero_idx[:, 0], nonzero_idx[:, 1]]
        G_pred = G_recon_mean[nonzero_idx[:, 0], nonzero_idx[:, 1]]
        g_nll = F.mse_loss(G_pred, G_target)
        g_kl = D.kl_divergence(q_v, prior_graph).mean()
        g_elbo = g_nll + self.lam_kl * g_kl

        # ---- 拆分图潜变量为 RNA 和 ATAC 部分 ----
        num_rna_nodes = net.n_genes
        num_atac_nodes = net.n_peaks
        v_rna = v_sample[:num_rna_nodes * net.z_dim].view(num_rna_nodes, net.z_dim)
        v_atac = v_sample[num_rna_nodes * net.z_dim:].view(num_atac_nodes, net.z_dim)
        
        # ---- 数据重构损失 ----
        data_recon_loss = {}
        for mod in net.keys:
            if mod == "expression":
                v_mod = v_rna
            elif mod == "atac":
                v_mod = v_atac
            else:
                raise ValueError(f"Unsupported modality: {mod}")
            p_x_z = net.u2x[mod](z_sample[mod], v_mod)
            data_recon_loss[mod] = F.mse_loss(p_x_z.mean, x[mod])
        
        # ---- KL 损失（数据部分） ----
        data_kl_loss = {}
        data_total_loss = {}
        for mod in net.keys:
            kl_val = D.kl_divergence(q_z_x[mod], prior_data).mean()
            data_kl_loss[mod] = kl_val
            data_total_loss[mod] = data_recon_loss[mod] + self.lam_kl * kl_val
        
        # 总数据部分的 ELBO 损失
        total_data_loss = sum(self.modality_weight[mod] * data_total_loss[mod] for mod in net.keys)
        
        # 总 VAE 损失：数据部分加上图部分，再加上邻接矩阵稀疏正则
        vae_loss = self.lam_data * total_data_loss + self.lam_graph * g_elbo + \
                self.lam_data * torch.mean(torch.abs(net.adj_A))
        # 生成器损失：包含对抗损失
        generator_loss = vae_loss - self.lam_align * dsc_loss
        
        return {
            "dsc_loss": dsc_loss,
            "vae_loss": vae_loss,
            "gen_loss": generator_loss,
            "g_nll": g_nll,
            "g_kl": g_kl,
            "g_elbo": g_elbo,
            "x_rna_nll": data_recon_loss["expression"],
            "x_rna_kl": data_kl_loss["expression"],
            "x_rna_elbo": data_total_loss["expression"],
            "x_atac_nll": data_recon_loss["atac"],
            "x_atac_kl": data_kl_loss["atac"],
            "x_atac_elbo": data_total_loss["atac"],
            "sparsity_loss": self.lam_data * torch.mean(torch.abs(net.adj_A))
        }

    def train_step(self, batch, epoch):
        data_expression, data_atac = batch
        data_expression = data_expression.cuda()
        data_atac = data_atac.cuda()
        # 判别器优化
        disc_losses = self.compute_losses(data_expression, data_atac, epoch, dsc_only=True)
        self.net.zero_grad()
        disc_losses["dsc_loss"].backward()
        self.disc_optim.step()
        # 生成器 (VAE) 优化
        gen_losses = self.compute_losses(data_expression, data_atac, epoch)
        self.net.zero_grad()
        gen_losses["gen_loss"].backward()
        self.vae_optim.step()
        if epoch % 10 >= 5:
            self.adj_optim.step()
        return gen_losses

    def fit(self, train_loader, val_loader, n_epochs=10):
        self.net.train()
        # 打开文件（以追加模式）
        with open("loss_log.txt", "a") as log_file:
            for epoch in range(n_epochs):
                # 训练阶段
                train_losses = []
                for batch in train_loader:
                    data_expression, data_atac, _, _ = batch
                    data_expression = Variable(data_expression.type(Tensor))
                    data_atac = Variable(data_atac.type(Tensor))
                    losses = self.train_step((data_expression, data_atac), epoch)
                    train_losses.append(losses)
                # 计算训练损失均值
                avg_train = {key: np.mean([loss[key].item() if isinstance(loss[key], torch.Tensor) else loss[key]
                                            for loss in train_losses])
                            for key in train_losses[0]}
                
                self.scheduler.step()
                # 验证阶段
                self.net.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        data_expression, data_atac, _, _ = batch
                        data_expression = data_expression.cuda()
                        data_atac = data_atac.cuda()
                        loss_val = self.compute_losses(data_expression, data_atac, epoch)
                        val_losses.append(loss_val)
                avg_val = {key: np.mean([loss[key].item() if isinstance(loss[key], torch.Tensor) else loss[key]
                                        for loss in val_losses])
                        for key in val_losses[0]}
                self.net.train()
                
                # 拼接输出字符串（变量名称更贴实际）
                log_str = (
                    f"Epoch {epoch}: Train -> g_mse: {avg_train['g_nll']:.3f}, g_kl: {avg_train['g_kl']:.3f}, g_loss: {avg_train['g_elbo']:.3f}, "
                    f"x_rna_mse: {avg_train['x_rna_nll']:.3f}, x_rna_kl: {avg_train['x_rna_kl']:.3f}, x_rna_loss: {avg_train['x_rna_elbo']:.3f}, "
                    f"x_atac_mse: {avg_train['x_atac_nll']:.3f}, x_atac_kl: {avg_train['x_atac_kl']:.3f}, x_atac_loss: {avg_train['x_atac_elbo']:.3f}, "
                    f"dsc_loss: {avg_train['dsc_loss']:.3f}, vae_loss: {avg_train['vae_loss']:.3f}, gen_loss: {avg_train['gen_loss']:.3f}, "
                    f"sparsity: {avg_train['sparsity_loss']:.3f} || "
                    f"Val -> g_mse: {avg_val['g_nll']:.3f}, g_kl: {avg_val['g_kl']:.3f}, g_loss: {avg_val['g_elbo']:.3f}, "
                    f"x_rna_mse: {avg_val['x_rna_nll']:.3f}, x_rna_kl: {avg_val['x_rna_kl']:.3f}, x_rna_loss: {avg_val['x_rna_elbo']:.3f}, "
                    f"x_atac_mse: {avg_val['x_atac_nll']:.3f}, x_atac_kl: {avg_val['x_atac_kl']:.3f}, x_atac_loss: {avg_val['x_atac_elbo']:.3f}, "
                    f"dsc_loss: {avg_val['dsc_loss']:.3f}, vae_loss: {avg_val['vae_loss']:.3f}, gen_loss: {avg_val['gen_loss']:.3f}, "
                    f"sparsity: {avg_val['sparsity_loss']:.3f}\n"
                )

                # 打印到控制台
                print(log_str.strip())
                # 写入文件
                log_file.write(log_str)
            # 保存最后的邻接矩阵
            final_adj = (self.net.adj_A.cpu().detach().numpy() != 0).astype(int)
            np.save("GRN_regulatory_matrix.npy", final_adj)


# ----------------------- 读取数据和构建邻接矩阵 -----------------------

rna = ad.read_h5ad("rna-pp.h5ad")
atac = ad.read_h5ad("atac-pp.h5ad")
guidance = nx.read_graphml("guidance.graphml.gz")
gene_names = list(rna.var[rna.var["highly_variable"]].index)
peak_names = list(atac.var[atac.var["highly_variable"]].index)
guidance_hvf = nx.read_graphml("guidance-hvf.graphml.gz")
ordered_nodes = gene_names + peak_names

adj_matrix_sparse = nx.to_scipy_sparse_array(guidance_hvf, nodelist=ordered_nodes)
adj_matrix = adj_matrix_sparse.toarray()
def normalize_adj_gcn(adj: np.ndarray) -> np.ndarray:
    A = adj + np.eye(adj.shape[0], dtype=adj.dtype)
    deg = np.sum(A, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat

A_norm = normalize_adj_gcn(adj_matrix)
adj_A_init = A_norm
print(adj_A_init)

# ----------------------- 筛选高变异基因和峰值 -----------------------
rna_hv = rna[:, rna.var["highly_variable"]]
atac_hv = atac[:, atac.var["highly_variable"]]

from scipy.sparse import issparse

if issparse(rna_hv.X):
    rna_data = rna_hv.X.toarray()
else:
    rna_data = rna_hv.X

if issparse(atac_hv.X):
    atac_data = atac_hv.X.toarray()
else:
    atac_data = atac_hv.X

# ----------------------- 转换为 PyTorch 张量 -----------------------
rna_tensor = torch.tensor(rna_data, dtype=torch.float32)
atac_tensor = torch.tensor(atac_data, dtype=torch.float32)

# 定义基因和峰数
n_genes = rna_tensor.shape[1]
n_peaks = atac_tensor.shape[1]

# 获取细胞数
num_rna = rna_tensor.shape[0]
num_atac = atac_tensor.shape[0]

# ----------------------- 下采样（保持两者细胞数一致） -----------------------
if num_atac < num_rna:
    indices = torch.randperm(num_rna)[:num_atac]
    rna_tensor_sub = rna_tensor[indices]
else:
    rna_tensor_sub = rna_tensor
print(rna_tensor_sub.shape, atac_tensor.shape)

# ----------------------- 数据标准化 -----------------------
rna_numpy = rna_tensor_sub.cpu().numpy()
rna_scaler = StandardScaler()
rna_data_std = rna_scaler.fit_transform(rna_numpy)

atac_numpy = atac_tensor.cpu().numpy()
atac_scaler = StandardScaler()
atac_data_std = atac_scaler.fit_transform(atac_numpy)


# 将标准化后的数据转为 GPU 张量
rna_tensor_std = torch.tensor(rna_data_std, dtype=torch.float32).cuda()
atac_tensor_std = torch.tensor(atac_data_std, dtype=torch.float32).cuda()


# ----------------------- 构建数据集和划分训练/验证集 -----------------------
dataset = TensorDataset(rna_tensor_std, atac_tensor_std,
                        torch.ones_like(rna_tensor_std), torch.ones_like(atac_tensor_std))

# 这里划分 80% 训练集，20% 验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 检查第一个 batch
for batch in train_loader:
    data_expression, data_atac, _, _ = batch
    print("Expression data contains NaN:", torch.isnan(data_expression).any())
    print("ATAC data contains NaN:", torch.isnan(data_atac).any())
    print("Expression data contains Inf:", torch.isinf(data_expression).any())
    print("ATAC data contains Inf:", torch.isinf(data_atac).any())
    break

# ----------------------- 初始化 GLUE 模型和 Trainer -----------------------
net = GLUE(adj_A_init, n_genes, n_peaks, z_dim=250)
trainer = GLUETrainer(net)

# 训练，并同时打印训练和验证集的损失
trainer.fit(train_loader, val_loader)

# 下面继续获取数据嵌入和保存（此处只保存训练集对应的细胞）
rna_embs = net.get_shared_embeddings("expression", rna_data)  # 注意：如果你希望的是扩展后的数据，可以使用扩展后的数据
atac_embs = net.get_shared_embeddings("atac", atac_data)
rna.obsm["X_glue"] = rna_embs
atac.obsm["X_glue"] = atac_embs

feature_embs = net.encode_graph()
v_rna = feature_embs[:net.n_genes, :]
v_atac = feature_embs[net.n_genes:, :]

v_rna_full = np.zeros((rna.n_vars, net.z_dim))
hv_gene_indices = [rna.var_names.get_loc(gene) for gene in gene_names]
for i, idx in enumerate(hv_gene_indices):
    v_rna_full[idx, :] = v_rna[i, :]
rna.varm["X_glue"] = v_rna_full
print(rna.varm["X_glue"].shape)

v_atac_full = np.zeros((atac.n_vars, net.z_dim))
hv_peak_indices = [atac.var_names.get_loc(peak) for peak in peak_names]
for i, idx in enumerate(hv_peak_indices):
    v_atac_full[idx, :] = v_atac[i, :]
atac.varm["X_glue"] = v_atac_full
print(atac.varm["X_glue"].shape)

# 如果做了下采样，则同步筛选 rna 的 AnnData 对象
if num_atac < num_rna:
    rna = rna[rna.obs_names[indices.cpu().numpy()]]

rna.write("rna-emb.h5ad", compression="gzip")
atac.write("atac-emb.h5ad", compression="gzip")


