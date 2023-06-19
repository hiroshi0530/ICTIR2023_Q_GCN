import torch
import itertools
import numpy as np
import scipy.sparse as sp

from torch import nn
from torch import optim

import networkx as nx

from util.loss import BPRLoss
from util.loss import EmbLoss
from util.util import xavier_uniform_initialization

from tqdm import tqdm
from loguru import logger

from trainer.Trainer import Trainer


class QGCN(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()

        self.config = config

        self.user_num = dataset.user_num
        self.item_num = dataset.item_num

        self.user_id_name = config["user_id_name"]
        self.item_id_name = config["item_id_name"]

        self.user_idx_name = config["user_idx_name"]
        self.item_idx_name = config["item_idx_name"]
        self.neg_item_idx_name = config["neg_item_idx_name"]

        self.early_stop_num = config["early_stop_num"]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dataset = dataset
        self.epochs = config["epochs"]
        self.topk = config["topk"]

        self.alpha_list = config["alpha_list"]
        self.alpha_list = list(map(lambda x: x / sum(self.alpha_list), self.alpha_list))  # 正規化

        # load dataset info
        self.interaction_matrix = self.dataset.get_train_coo_matrix()

        self.embedding_size = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.learning_rate = config["learning_rate"]
        self.reg_weight = config["reg_weight"]

        assert len(self.alpha_list) == self.n_layers + 1

        # define layers and loss
        self.user_embedding_01 = torch.nn.Embedding(num_embeddings=self.user_num, embedding_dim=self.embedding_size).to(
            self.device
        )
        self.item_embedding_01 = torch.nn.Embedding(num_embeddings=self.item_num, embedding_dim=self.embedding_size).to(
            self.device
        )

        self.user_embedding_02 = torch.nn.Embedding(num_embeddings=self.user_num, embedding_dim=self.embedding_size).to(
            self.device
        )
        self.item_embedding_02 = torch.nn.Embedding(num_embeddings=self.item_num, embedding_dim=self.embedding_size).to(
            self.device
        )

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.norm_adj_matrix = self.get_normalized_adjacency_matrix().to(self.device)

        # eigenvalue decomposition
        l, p = torch.linalg.eigh(self.norm_adj_matrix.to_dense().to(self.device))

        pp = torch.pow(p, 2)

        l = l.to(self.device)
        p = p.to(self.device)
        pp = pp.to(self.device)

        self.Q0 = self.get_Q0(l, p, pp)

        self.apply(xavier_uniform_initialization)

        # optimizer
        self.set_optimizer()

    def get_normalized_adjacency_matrix(self):
        B = nx.Graph()

        user_nodes = [i for i in range(self.dataset.user_num)]
        item_nodes = [(i + self.dataset.user_num) for i in range(self.dataset.item_num)]

        B.add_nodes_from(user_nodes, bipartite=0)
        B.add_nodes_from(item_nodes, bipartite=1)
        B.add_edges_from(
            [
                (row, self.dataset.user_num + col)
                for row, col in zip(self.interaction_matrix.row, self.interaction_matrix.col)
            ]
        )

        A = nx.adjacency_matrix(B).todense()
        A = torch.tensor(A, dtype=float)

        D = torch.diag(torch.sum(A, dim=1)).type(torch.LongTensor)
        D_0_5_pow = torch.pow(torch.where(D == 0, 1, D), -1 / 2) - torch.where(D == 0, 1, 0)

        A = A.to(self.device)
        D = D.to(self.device)
        D_0_5_pow = D_0_5_pow.to(self.device)
        A_hat = D_0_5_pow.to(float) @ A.to(float) @ D_0_5_pow.to(float)
        A_hat = A_hat.float()

        return A_hat.to_sparse()

    def get_Q0(self, l, p, pp):
        logger.info("start get Q0")
        Q0 = torch.zeros(len(p), len(p)).to(self.device)
        for i in tqdm(range(len(p))):
            Q0 += torch.multiply(pp[:, i : i + 1], pp[:, i : i + 1].T)

        logger.info("end get Q0")
        return Q0

    def get_embeddings_01(self):
        user_embeddings = self.user_embedding_01.weight
        item_embeddings = self.item_embedding_01.weight
        embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return embeddings

    def get_embeddings_02(self):
        user_embeddings = self.user_embedding_02.weight
        item_embeddings = self.item_embedding_02.weight
        embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return embeddings

    def forward(self):
        all_embeddings_01 = self.get_embeddings_01()
        embeddings_list_01 = [self.alpha_list[0] * all_embeddings_01]

        for _, alpha in zip(range(self.n_layers), self.alpha_list[1:]):
            all_embeddings_01 = torch.sparse.mm(self.norm_adj_matrix, all_embeddings_01)
            embeddings_list_01.append(alpha * all_embeddings_01)

        all_embeddings_02 = self.get_embeddings_02()
        embeddings_list_02 = [self.alpha_list[0] * all_embeddings_02]

        for _, alpha in zip(range(self.n_layers), self.alpha_list[1:]):
            all_embeddings_02 = torch.sparse.mm(self.Q0, all_embeddings_02)
            embeddings_list_02.append(alpha * all_embeddings_02)

        embeddings_list = []
        for e1, e2 in zip(embeddings_list_01, embeddings_list_02):
            embeddings_list.append((e1 + e2) / 2)

        temp = torch.stack(embeddings_list, dim=0)
        lightgcn_all_embeddings = torch.sum(temp, dim=0)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.user_num, self.item_num])
        return user_all_embeddings, item_all_embeddings

    def get_loss(self, interaction):
        user = interaction[self.user_idx_name]
        pos_item = interaction[self.item_idx_name]
        neg_item = interaction[self.neg_item_idx_name]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_embeddings_01 = self.user_embedding_01(user)
        pos_embeddings_01 = self.item_embedding_01(pos_item)
        neg_embeddings_01 = self.item_embedding_01(neg_item)

        u_embeddings_02 = self.user_embedding_02(user)
        pos_embeddings_02 = self.item_embedding_02(pos_item)
        neg_embeddings_02 = self.item_embedding_02(neg_item)

        reg_loss_01 = self.reg_loss(u_embeddings_01, pos_embeddings_01, neg_embeddings_01)
        reg_loss_02 = self.reg_loss(u_embeddings_02, pos_embeddings_02, neg_embeddings_02)

        loss = mf_loss + self.reg_weight * (reg_loss_01 + reg_loss_02)

        return loss

    def predict(self, interaction):
        user = interaction[self.user_idx_name]
        item = interaction[self.item_idx_name]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def all_user_predict(self, interaction):
        user = interaction[self.user_idx_name]

        self.restore_user_e, self.restore_item_e = self.forward()
        u_embeddings = self.restore_user_e[user]

        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.0)

    def fit(self):
        trainer = Trainer(self.config, self, self.dataset)
        self.best_valid_result, self.test_result = trainer.train()

    def calculate_gcn_diversity(self):
        user_embedding = (self.user_embedding_01.weight + self.user_embedding_02.weight) / 2
        item_embedding = (self.item_embedding_01.weight + self.item_embedding_02.weight) / 2

        # 解析
        score = torch.matmul(user_embedding, item_embedding.T)
        sorted_list, idx = torch.sort(score, dim=1)

        idx = torch.fliplr(idx)
        sorted_list = torch.fliplr(sorted_list)

        cdist_top_5_list = []
        cdist_middle_5_list = []
        cdist_middle_1_5_list = []
        cdist_middle_2_5_list = []
        cdist_middle_3_5_list = []
        cdist_middle_4_5_list = []
        cdist_middle_5_5_list = []

        combinations_top_5_list = []
        combinations_middle_5_list = []
        combinations_middle_1_5_list = []
        combinations_middle_2_5_list = []
        combinations_middle_3_5_list = []
        combinations_middle_4_5_list = []
        combinations_middle_5_5_list = []

        for i, e in enumerate(user_embedding):
            top_5 = item_embedding[idx[i, 0:5]]
            middle_5 = item_embedding[idx[i, 25:30]]
            middle_1_5 = item_embedding[idx[i, 50:55]]
            middle_2_5 = item_embedding[idx[i, 75:80]]
            middle_3_5 = item_embedding[idx[i, 100:105]]
            middle_4_5 = item_embedding[idx[i, 150:155]]
            middle_5_5 = item_embedding[idx[i, 250:255]]

            top_5_mean = torch.mean(top_5, dim=0).reshape(1, -1).to(torch.float)
            middle_5_mean = torch.mean(middle_5, dim=0).reshape(1, -1).to(torch.float)
            middle_1_5_mean = torch.mean(middle_1_5, dim=0).reshape(1, -1).to(torch.float)
            middle_2_5_mean = torch.mean(middle_2_5, dim=0).reshape(1, -1).to(torch.float)
            middle_3_5_mean = torch.mean(middle_3_5, dim=0).reshape(1, -1).to(torch.float)
            middle_4_5_mean = torch.mean(middle_4_5, dim=0).reshape(1, -1).to(torch.float)
            middle_5_5_mean = torch.mean(middle_5_5, dim=0).reshape(1, -1).to(torch.float)

            _cdist_top_5 = torch.cdist(top_5, top_5_mean).mean()
            _cdist_middle_5 = torch.cdist(middle_5, middle_5_mean).mean()
            _cdist_middle_1_5 = torch.cdist(middle_1_5, middle_1_5_mean).mean()
            _cdist_middle_2_5 = torch.cdist(middle_2_5, middle_2_5_mean).mean()
            _cdist_middle_3_5 = torch.cdist(middle_3_5, middle_3_5_mean).mean()
            _cdist_middle_4_5 = torch.cdist(middle_4_5, middle_4_5_mean).mean()
            _cdist_middle_5_5 = torch.cdist(middle_5_5, middle_5_5_mean).mean()

            cdist_top_5_list.append(_cdist_top_5.item())
            cdist_middle_5_list.append(_cdist_middle_5.item())
            cdist_middle_1_5_list.append(_cdist_middle_1_5.item())
            cdist_middle_2_5_list.append(_cdist_middle_2_5.item())
            cdist_middle_3_5_list.append(_cdist_middle_3_5.item())
            cdist_middle_4_5_list.append(_cdist_middle_4_5.item())
            cdist_middle_5_5_list.append(_cdist_middle_5_5.item())

            ## combination
            def get_combinations_norm(mat):
                temp = list(itertools.combinations(mat, 2))
                temp_list = []
                for i in temp:
                    temp_list.append(torch.dist(i[0], i[1]).item())
                return np.mean(temp_list)

            combinations_top_5_list.append(get_combinations_norm(top_5))
            combinations_middle_5_list.append(get_combinations_norm(middle_5))
            combinations_middle_1_5_list.append(get_combinations_norm(middle_1_5))
            combinations_middle_2_5_list.append(get_combinations_norm(middle_2_5))
            combinations_middle_3_5_list.append(get_combinations_norm(middle_3_5))
            combinations_middle_4_5_list.append(get_combinations_norm(middle_4_5))
            combinations_middle_5_5_list.append(get_combinations_norm(middle_5_5))

        self.cdist_top_5_list = float(np.mean(cdist_top_5_list))
        self.cdist_middle_5_list = float(np.mean(cdist_middle_5_list))
        self.cdist_middle_1_5_list = float(np.mean(cdist_middle_1_5_list))
        self.cdist_middle_2_5_list = float(np.mean(cdist_middle_2_5_list))
        self.cdist_middle_3_5_list = float(np.mean(cdist_middle_3_5_list))
        self.cdist_middle_4_5_list = float(np.mean(cdist_middle_4_5_list))
        self.cdist_middle_5_5_list = float(np.mean(cdist_middle_5_5_list))
        self.combinations_top_5_list = float(np.mean(combinations_top_5_list))
        self.combinations_middle_5_list = float(np.mean(combinations_middle_5_list))
        self.combinations_middle_1_5_list = float(np.mean(combinations_middle_1_5_list))
        self.combinations_middle_2_5_list = float(np.mean(combinations_middle_2_5_list))
        self.combinations_middle_3_5_list = float(np.mean(combinations_middle_3_5_list))
        self.combinations_middle_4_5_list = float(np.mean(combinations_middle_4_5_list))
        self.combinations_middle_5_5_list = float(np.mean(combinations_middle_5_5_list))
