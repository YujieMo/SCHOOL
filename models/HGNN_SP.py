from utils import process, data_load
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from embedder import embedder
from layers import FullyConnect, Discriminator, Linear_layer, SemanticAttention
from evaluation import evaluation_metrics

INF = 1e-8

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class HGNN_SP(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def training(self):
        self.features_orth = self.features.to(self.args.device)
        self.features = self.features.to(self.args.device)
        self.graph = {t: [m.to(self.args.device) for m in ms] for t, ms in self.graph.items()}
        # self.mlp = MLP([self.args.ft_size, self.args.g_dim])
        model = modeler(self.args).to(self.args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr)
        cnt_wait = 0;
        best = 1e9;
        self.args.batch_size = 1
        self.criterion = SpectralNetLoss()
        self.g = nn.Sequential(nn.Linear(self.args.out_ft, self.args.g_dim, bias=False),
                               nn.ReLU(inplace=True)).to(self.args.device)

        X = self.features[0:self.args.node_num].cpu()
        distX = process.pairwise_distance(X)
        # Sort the distances and get the sorted indices
        distX_sorted, idx = torch.sort(distX, dim=1)
        num = self.features[0:self.args.node_num].shape[0]
        A = torch.zeros(num, num)
        rr = torch.zeros(num)
        for i in range(num):
            di = distX_sorted[i, 1:self.args.k + 1]
            rr[i] = 0.5 * (self.args.k * di[self.args.k - 1] - torch.sum(di[:self.args.k]))
            id = idx[i, 1:self.args.k + 1]
            A[i, id] = (di[self.args.k - 1] - di) / (
                        self.args.k * di[self.args.k - 1] - torch.sum(di[:self.args.k]) + torch.finfo(torch.float).eps)
        alpha = rr.mean()
        # r = 0

        beta = rr.mean()

        for epoch in tqdm(range(self.args.nb_epochs)):

            model.train()
            optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(True)


            emb_het, emb_hom, A, Y = model(self.graph, self.features, self.features_orth,
                                           idx, beta, alpha)

            # The first term in Eq. (13): spectral loss
            sploss = self.criterion(A, Y)
            p_i = Y.sum(0).view(-1)
            p_i = (p_i + INF) / (p_i.sum() + INF)
            p_i = torch.abs(p_i)
            # The second term in Eq. (13): entropy loss
            entrpoy_loss = math.log(p_i.size(0) + INF) + ((p_i + INF) * torch.log(p_i + INF)).sum()
            sploss = sploss + self.args.gamma * entrpoy_loss

            ###########################################
            embs_P1 = self.g(emb_het)
            embs_P2 = self.g(emb_hom)
            #######################################################################
            # The first term in Eq. (15): invariance loss
            inter_c = embs_P1.T @ embs_P2.detach()
            inter_c = F.normalize(inter_c, p=2, dim=1)
            loss_inv = -torch.diagonal(inter_c).sum()

            # The second term in Eq. (15): uniformity loss
            intra_c = (embs_P1).T @ (embs_P1).contiguous()
            intra_c = torch.exp(F.normalize(intra_c, p=2, dim=1)).sum()
            loss_uni = torch.log(intra_c).mean()

            intra_c_2 = (embs_P2).T @ (embs_P2).contiguous()
            intra_c_2 = torch.exp(F.normalize(intra_c_2, p=2, dim=1)).sum()
            loss_uni += torch.log(intra_c_2).mean()
            loss_consistency = loss_inv + self.args.eta * loss_uni

            # The second term in Eq. (13): cluster-level loss
            Y_hat = torch.argmax(Y, dim=1)
            cluster_center = torch.stack([torch.mean(embs_P2[Y_hat == i], dim=0) for i in
                                          range(self.args.cluster)])  # Shape: (num_clusters, embedding_dim)
            # Gather positive cluster centers
            positive = cluster_center[Y_hat]
            # The first term in Eq. (11)
            inter_c = positive.T @ embs_P1
            inter_c = F.normalize(inter_c, p=2, dim=1)
            loss_spe_inv = -torch.diagonal(inter_c).sum()



            # loss =  (10* sploss  +1 * loss_consistency + 0.5 * (loss_spe_inv))
            loss = (sploss + self.args.mu * loss_consistency + self.args.delta * (loss_spe_inv))
            print(loss)
            # with torch.autograd.detect_anomaly(True):
            loss.backward()
            optimizer.step()

            train_loss = loss.item()

            if (train_loss < best):
                best = train_loss
                cnt_wait = 0
            else:
                cnt_wait += 1

            if cnt_wait == self.args.patience:
                break

        model.load_state_dict(
            torch.load('saved_model/best_{}_{}.pkl'.format(self.args.dataset, self.args.custom_key)))

        embs_het, emb_hom = model.embed(self.graph, self.features,  self.features_orth,
                                        idx, alpha, beta)

        h_concat = []
        h_concat.append(embs_het)
        h_concat.append(emb_hom)
        h_concat = torch.cat(h_concat, 1)
        test_out = h_concat.detach().cpu().numpy()

        if self.args.dataset in ['freebase', 'Aminer', 'imdb', 'Freebase']:  # , 'ogbn-mag'
            ev = evaluation_metrics(test_out, self.labels, self.args, self.train_idx, self.val_idx, self.test_idx)
        else:
            ev = evaluation_metrics(test_out, self.labels, self.args)
        fis, fas, k1, st = ev.evalutation(self.args)
        return fis, fas, k1, st





class modeler(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bnn = nn.ModuleDict()
        self.disc2 = nn.ModuleDict()
        self.fc = nn.ModuleDict()
        self.mlp = MLP([self.args.ft_size, self.args.out_ft])
        self.semanticatt = nn.ModuleDict()
        self.spectral_net = SpectralNetModel(
            self.args.sp_arch, input_dim=self.args.ft_size
        ).to(self.args.device)

        for t, rels in self.args.nt_rel.items():  # {note_type: [rel1, rel2]}
            self.fc[t] = FullyConnect(args.hid_units2 + args.ft_size, args.out_ft)
            self.disc2[t] = Discriminator(args.ft_size, args.out_ft)
            for rel in rels:
                self.bnn['0' + rel] = Linear_layer(args.ft_size, args.hid_units, act=nn.ReLU(), isBias=False)
                self.bnn['1' + rel] = Linear_layer(args.hid_units, args.hid_units2, act=nn.ReLU(), isBias=False)

            self.semanticatt['0' + t] = SemanticAttention(args.hid_units, args.hid_units // 4)
            self.semanticatt['1' + t] = SemanticAttention(args.hid_units2, args.hid_units2 // 4)

    def forward(self, graph, features, features_orth, idx, beta, alpha):
        # lambda_ = 10
        embs1 = torch.zeros((self.args.node_size, self.args.hid_units)).to(self.args.device)
        embs2 = torch.zeros((self.args.node_size, self.args.out_ft)).to(self.args.device)

        for n, rels in self.args.nt_rel.items():  # p: [[Nei(p-a), Nei(p-c)]]  (Np, Nprel)
            vec = []
            for j, rel in enumerate(rels):
                t = rel.split('-')[1]
                mean_neighbor = torch.spmm(graph[n][j], features[self.args.node_cnt[t]])
                v = self.bnn['0' + rel](mean_neighbor)
                vec.append(v)  # (1, Nt, ft)

            vec = torch.stack(vec, 0)  # (rel_size, Nt, ft_size)
            v_summary = torch.mean(vec, 0)  # (Nt, hd)

            embs1[self.args.node_cnt[n]] = v_summary

        for n, rels in self.args.nt_rel.items():  # p: [[Nei(p-a), Nei(p-c)]]  (Np, Nprel)
            vec = []
            for j, rel in enumerate(rels):
                t = rel.split('-')[-1]

                mean_neighbor = torch.spmm(graph[n][j], embs1[self.args.node_cnt[t]])
                v = self.bnn['1' + rel](mean_neighbor)
                vec.append(v)  # (1, Nt, ft)

            vec = torch.stack(vec, 0)  # (rel_size, Nt, ft_size)
            v_summary = torch.mean(vec, 0)  # (Nt, hd)
            v_cat = torch.hstack((v_summary, features[self.args.node_cnt[n]]))
            v_summary = self.fc[n](v_cat)

            embs2[self.args.node_cnt[n]] = v_summary

        if self.args.dataset in ['ACM']:
            embs_het = embs1
        else:
            embs_het = embs2

        embs_het = embs_het[0:self.args.node_num]

        # self.spectral_net.eval()
        self.spectral_net(self.args, features_orth[0:self.args.node_num], should_update_orth_weights=True)
        # Gradient step
        num = embs_het.shape[0]
        Y, Y_2, Y_2_orth = self.spectral_net(self.args, features[0:self.args.node_num], should_update_orth_weights=False)
        A = torch.zeros((num, num)).to(self.args.device)
        idxa0 = idx[:, 1:self.args.k + 1]
        dfi = torch.sqrt(torch.sum((Y.unsqueeze(1) - Y[idxa0]) ** 2, dim=2) + 1e-8).to(self.args.device)
        dxi = torch.sqrt(torch.sum((Y_2_orth.unsqueeze(1) - Y_2_orth[idxa0]) ** 2, dim=2) + 1e-8).to(self.args.device)
        ad = -(dxi + beta * dfi) / (2 * alpha)
        A.scatter_(1, idxa0.to(self.args.device), process.EProjSimplex_new_matrix(ad))
        embs_hom = torch.mm(A, Y_2)


        return embs_het, embs_hom, A, Y

    def embed(self, graph, features, features_orth, idx, beta, alpha):
        # lambda_ = 10
        embs1 = torch.zeros((self.args.node_size, self.args.hid_units)).to(self.args.device)
        embs2 = torch.zeros((self.args.node_size, self.args.out_ft)).to(self.args.device)
        for n, rels in self.args.nt_rel.items():  # p: [[Nei(p-a), Nei(p-c)]]  (Np, Nprel)
            vec = []
            for j, rel in enumerate(rels):
                t = rel.split('-')[1]

                mean_neighbor = torch.spmm(graph[n][j], features[self.args.node_cnt[t]])
                v = self.bnn['0' + rel](mean_neighbor)
                vec.append(v)  # (1, Nt, ft)
            vec = torch.stack(vec, 0)  # (rel_size, Nt, ft_size)
            v_summary = torch.mean(vec, 0)  # (Nt, hd)

            embs1[self.args.node_cnt[n]] = v_summary

        for n, rels in self.args.nt_rel.items():  # p: [[Nei(p-a), Nei(p-c)]]  (Np, Nprel)
            vec = []
            for j, rel in enumerate(rels):
                t = rel.split('-')[-1]

                mean_neighbor = torch.spmm(graph[n][j], embs1[self.args.node_cnt[t]])
                v = self.bnn['1' + rel](mean_neighbor)
                vec.append(v)  # (1, Nt, ft)

            vec = torch.stack(vec, 0)  # (rel_size, Nt, ft_size)
            v_summary = torch.mean(vec, 0)  # (Nt, hd)
            v_cat = torch.hstack((v_summary, features[self.args.node_cnt[n]]))
            v_summary = self.fc[n](v_cat)

            embs2[self.args.node_cnt[n]] = v_summary
        if self.args.dataset in ['ACM']:# , 'Freebase'
            embs_het = embs1
        else:
            embs_het = embs2

        embs_het = embs_het[0:self.args.node_num]

        # self.spectral_net.eval()
        self.spectral_net(self.args, features_orth[0:self.args.node_num], should_update_orth_weights=True)
        # Gradient step
        num = embs_het.shape[0]
        Y, Y_2, Y_2_orth = self.spectral_net(self.args, features[0:self.args.node_num], should_update_orth_weights=False)
        A = torch.zeros((num, num)).to(self.args.device)
        idxa0 = idx[:, 1:self.args.k + 1]
        dfi = torch.sqrt(torch.sum((Y.unsqueeze(1) - Y[idxa0]) ** 2, dim=2)).to(self.args.device)
        dxi = torch.sqrt(torch.sum((Y_2_orth.unsqueeze(1) - Y_2_orth[idxa0]) ** 2, dim=2) + 1e-8).to(self.args.device)
        ad = -(dxi + beta * dfi) / (2 * alpha)
        A.scatter_(1, idxa0.to(self.args.device), process.EProjSimplex_new_matrix(ad))
        embs_hom = torch.mm(A, Y_2)

        return embs_het.detach(), embs_hom.detach()


class SpectralNetModel(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(SpectralNetModel, self).__init__()
        self.architecture = architecture
        self.layers = nn.ModuleList()
        self.input_dim = input_dim

        current_dim = self.input_dim
        for i, layer in enumerate(self.architecture):
            next_dim = layer
            if i == len(self.architecture) - 1:
                self.layers.append(
                    nn.Sequential(nn.Linear(current_dim, next_dim), nn.Tanh())
                )
            else:
                self.layers.append(
                    nn.Sequential(nn.Linear(current_dim, next_dim), nn.LeakyReLU())
                )
                current_dim = next_dim

    def _make_orthonorm_weights(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Orthonormalize the output of the network using the Cholesky decomposition.

        Parameters
        ----------
        Y : torch.Tensor
            The output of the network.

        Returns
        -------
        torch.Tensor
            The orthonormalized output.

        Notes
        -----
        This function applies QR decomposition to orthonormalize the output (`Y`) of the network.
        The inverse of the R matrix is returned as the orthonormalization weights.
        """

        m = Y.shape[0]
        _, R = torch.linalg.qr(Y)
        orthonorm_weights = np.sqrt(m + 1e-8) * torch.inverse(R)
        return orthonorm_weights

    def forward(
            self,args, x: torch.Tensor, should_update_orth_weights: bool = True
    ) -> torch.Tensor:
        """
        Perform the forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        should_update_orth_weights : bool, optional
            Whether to update the orthonormalization weights using the Cholesky decomposition or not.

        Returns
        -------
        torch.Tensor
            The output tensor.

        Notes
        -----
        This function takes an input tensor `x` and computes the forward pass of the model.
        If `should_update_orth_weights` is set to True, the orthonormalization weights are updated
        using the QR decomposition. The output tensor is returned.
        """
        i = 0
        for layer in self.layers:
            x = layer(x)
            if self.architecture[i] == args.out_ft:
                x_1 = x
            i += 1

        Y_tilde = x
        Y2_tilde = x_1
        if should_update_orth_weights:
            self.orthonorm_weights = self._make_orthonorm_weights(Y_tilde)
            self.orthonorm_weights2 = self._make_orthonorm_weights(Y2_tilde)
        Y = Y_tilde @ self.orthonorm_weights
        Y_2 = Y2_tilde @ self.orthonorm_weights2

        return Y, x_1, Y_2



class SpectralNetLoss(nn.Module):
    def __init__(self):
        super(SpectralNetLoss, self).__init__()

    def forward(self, W: torch.Tensor, Y: torch.Tensor, is_normalized: bool = False
                ) -> torch.Tensor:
        """
        This function computes the loss of the SpectralNet model.
        The loss is the rayleigh quotient of the Laplacian matrix obtained from W,
        and the orthonormalized output of the network.

        Args:
            W (torch.Tensor):               Affinity matrix
            Y (torch.Tensor):               Output of the network
            is_normalized (bool, optional): Whether to use the normalized Laplacian matrix or not.

        Returns:
            torch.Tensor: The loss
        """
        m = Y.size(0)
        if is_normalized:
            D = torch.sum(W, dim=1)
            Y = Y / torch.sqrt(D + 1e-8)[:, None]

        Dy = torch.cdist(Y, Y)
        loss = torch.sum(W * Dy.pow(2)) / (2 * m)

        return loss


class MLP(nn.Module):
    def __init__(self, dim, dropprob=0.0):
        super(MLP, self).__init__()
        self.net = nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropprob)
        struc = []
        for i in range(len(dim)):
            struc.append(dim[i])
        for i in range(len(struc) - 1):
            self.net.append(nn.Linear(struc[i], struc[i + 1]))

    def forward(self, x):
        for i in range(len(self.net) - 1):
            x = F.relu(self.net[i](x))
            x = self.dropout(x)

        # For the last layer
        y = self.net[-1](x)

        return y
