import os
import torch
import numpy as np

from loguru import logger

from util.util import is_train_stop
from util.util import get_yyyymmddhhmmss
from util.util import create_dir

from tqdm import tqdm


class Evaluator:
    def __init__(self, val_user_idx, score_matrix, user_idx2viewed_idx_dict, val_user_idx2item_idx_dict, topk) -> None:
        self.val_user_idx = val_user_idx
        self.score_matrix = score_matrix
        self.user_idx2viewed_idx_dict = user_idx2viewed_idx_dict
        self.val_user_idx2item_idx_dict = val_user_idx2item_idx_dict
        self.topk = topk

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        _viewed_row_idx_list = []
        _viewed_col_idx_list = []

        _val_viewed_item_idx_list = []

        for i, _idx in enumerate(self.val_user_idx):
            _viewed_item_idx = self.user_idx2viewed_idx_dict[_idx.item()]
            _viewed_row_idx_list.extend([i] * _viewed_item_idx.shape[0])
            _viewed_col_idx_list.extend(_viewed_item_idx)

            _val_viewed_item_idx_list.append(self.val_user_idx2item_idx_dict[_idx.item()])

        _viewed_idx_tuple = (_viewed_row_idx_list, _viewed_col_idx_list)

        self.score_matrix[_viewed_idx_tuple] = -np.inf

        _, self.topk_idx_matrix = torch.topk(score_matrix, self.topk, dim=1)

        _val_matrix_row_idx_list = []
        _val_matrix_col_idx_list = []

        self.topk_ndcg_score_list = []

        for i, (_arg_idx, _val_idx) in enumerate(zip(self.topk_idx_matrix, _val_viewed_item_idx_list)):
            _val_matrix_row_idx_list.extend([i] * len(_val_idx))
            _val_matrix_col_idx_list.extend(_val_idx)

            self.topk_ndcg_score_list.append(
                [1 / np.log2(2 + k) if k < min(self.topk, len(_val_idx.tolist())) else 0 for k in range(self.topk)]
            )

        # recall, precision計算用の 0-1 matrix
        _val_matrix_idx_tuple = (_val_matrix_row_idx_list, _val_matrix_col_idx_list)

        # val binary matrix
        self.val_binary_matrix = torch.zeros(score_matrix.shape).to(self.device)
        self.val_binary_matrix[_val_matrix_idx_tuple] = 1

        self.topk_binary_matrix = torch.gather(self.val_binary_matrix, dim=1, index=self.topk_idx_matrix)

        self.topk_ndcg_score_matrix = torch.FloatTensor(self.topk_ndcg_score_list).to(self.device)

    def get_topk_binary_matrix_cat_valid_num(self):
        return torch.cat([self.topk_binary_matrix, self.val_binary_matrix.sum(dim=1, keepdim=True)], dim=1)

    def get_recall(self):
        return self.topk_binary_matrix.sum(dim=1) / self.val_binary_matrix.sum(dim=1)

    def get_precision(self):
        return self.topk_binary_matrix.sum(dim=1) / self.topk

    def get_ndcg(self):
        temp = torch.FloatTensor(
            [[1 / np.log2(2 + i) for i in range(self.topk)] for _ in range(self.topk_binary_matrix.shape[0])]
        ).to(self.device)
        return torch.mul(self.topk_binary_matrix, temp).sum(dim=1) / self.topk_ndcg_score_matrix.sum(dim=1)

    def get_mrr(self):
        temp = torch.LongTensor(
            [
                [i for i in range(self.topk_binary_matrix.shape[1], 0, -1)]
                for j in range(self.topk_binary_matrix.shape[0])
            ]
        ).to(self.device)
        temp = torch.mul(temp, self.topk_binary_matrix)

        temp = torch.cat([torch.zeros(self.topk_binary_matrix.shape[0], 1).to(self.device), temp], dim=1)
        _, temp_indices = torch.topk(temp, k=1, dim=1)
        temp_indices = temp_indices.type(torch.double)

        return torch.reciprocal(torch.where(temp_indices == 0, float("inf"), temp_indices))

    def get_map(self):
        """
        topkとvalのminで平均を取ることに注意
        """
        temp1 = torch.mul(self.topk_binary_matrix, torch.cumsum(self.topk_binary_matrix, dim=1))
        temp2 = torch.reciprocal(
            torch.tensor(
                [
                    [i for i in range(1, self.topk_binary_matrix.shape[1] + 1)]
                    for i in range(self.topk_binary_matrix.shape[0])
                ]
            )
        ).to(self.device)

        temp3 = self.val_binary_matrix.sum(dim=1).reshape(-1, 1).type(torch.long)
        temp4 = torch.LongTensor([self.topk] * self.val_binary_matrix.shape[0]).reshape(-1, 1).to(self.device)
        temp5, _ = torch.min(torch.cat([temp3, temp4], dim=1), dim=1)

        return torch.mul(temp1, temp2).sum(dim=1) / torch.where(temp5 == 0, 1, temp5)


class Trainer:
    def __init__(self, config, model, dataset) -> None:
        self.model = model
        self.dataset = dataset
        self.config = config
        self.epochs = self.config["epochs"]
        self.topk = self.config["topk"]

        self.checkpoint_dir = self.config["checkpoint_dir_name"]
        create_dir(self.config["checkpoint_dir_name"])
        self.checkpoint_model_file = os.path.join(
            self.checkpoint_dir, f"{get_yyyymmddhhmmss()}_{self.config['model']}_{self.dataset.name}.pt"
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self):
        train_dl, val_dl, test_dl = self.dataset.get_dataloader()

        self.best_valid_metric = 0
        self.best_valid_recall = 0
        self.best_valid_precision = 0
        self.best_valid_ndcg = 0
        current_epoch = 0

        for epoch in range(self.epochs):
            self.model.train()

            train_dl = tqdm(
                train_dl,
                total=len(train_dl),
                ncols=100,
                desc=f"\033[31mTrain {epoch:>4}\033[0m",
            )

            for _user_idx, _item_idx in train_dl:
                # negative sampling

                ret_user_idx_list = []
                ret_pos_item_idx_list = []
                ret_neg_item_idx_list = []

                user_idx_array = np.array(_user_idx)
                pos_item_idx_array = np.array(_item_idx)

                user_idx_array_bool_list = [i for i in range(len(user_idx_array))]
                temp_user_idx_array_bool_list = user_idx_array_bool_list.copy()

                while len(temp_user_idx_array_bool_list) > 0:
                    temp_negative_item_idx = np.random.choice(
                        self.dataset.item_num, size=len(temp_user_idx_array_bool_list)
                    )

                    for u, i, p, n in zip(
                        user_idx_array[temp_user_idx_array_bool_list],
                        temp_user_idx_array_bool_list,
                        pos_item_idx_array[temp_user_idx_array_bool_list],
                        temp_negative_item_idx,
                    ):
                        if n in self.dataset.user_idx2neg_item_idx_set_dict[u]:
                            try:
                                user_idx_array_bool_list[i] = None
                                ret_user_idx_list.append(u)
                                ret_pos_item_idx_list.append(p)
                                ret_neg_item_idx_list.append(n)
                            except:
                                raise Exception

                    temp_user_idx_array_bool_list = list(filter(lambda x: x is not None, user_idx_array_bool_list))

                interaction = {
                    "user_idx": torch.LongTensor(ret_user_idx_list).to(self.device),
                    "item_idx": torch.LongTensor(ret_pos_item_idx_list).to(self.device),
                    "neg_item_idx": torch.LongTensor(ret_neg_item_idx_list).to(self.device),
                }

                self.model.optimizer.zero_grad()
                loss = self.model.get_loss(interaction)
                loss.backward()
                self.model.optimizer.step()

            # eval start
            self.model.eval()

            recall = 0
            precision = 0
            ndcg = 0
            mrr = 0
            map_ = 0

            with torch.no_grad():
                val_dl = tqdm(
                    val_dl,
                    total=len(val_dl),
                    ncols=100,
                    desc=f"\033[31mValid {epoch:>4}\033[0m",
                )

                for i, _val_user_idx in enumerate(val_dl):
                    interaction = {
                        "user_idx": _val_user_idx[0].to(self.device),
                    }

                    score_matrix = self.model.all_user_predict(interaction)
                    score_matrix = score_matrix.view(-1, self.dataset.item_num)

                    evaluator = Evaluator(
                        _val_user_idx[0],
                        score_matrix,
                        self.dataset.train_user_idx2viewed_idx_dict,
                        self.dataset.val_user_idx2item_idx_dict,
                        self.topk,
                    )

                    recall += evaluator.get_recall().sum().item() / self.dataset.val_user_num
                    precision += evaluator.get_precision().sum().item() / self.dataset.val_user_num
                    ndcg += evaluator.get_ndcg().sum().item() / self.dataset.val_user_num
                    mrr += evaluator.get_mrr().sum().item() / self.dataset.val_user_num
                    map_ += evaluator.get_map().sum().item() / self.dataset.val_user_num

                logger.info(
                    f"epoch : {epoch}, recall@{self.topk} : {recall:.4f}, precision@{self.topk} : {precision:.4f}, ndcg@{self.topk} : {ndcg:.4f}, mrr@{self.topk} : {mrr:.4f}, map@{self.topk} : {map_:.4f}"
                )

                if recall > self.best_valid_metric:
                    self.best_valid_epoch = epoch
                    self.best_valid_recall = recall
                    self.best_valid_precision = precision
                    self.best_valid_ndcg = ndcg
                    self.best_valid_mrr = mrr
                    self.best_valid_map = map_

                    logger.info(f"best valid score improved : {recall:.4f}")
                    logger.info(f"save checkpoint file : {self.checkpoint_model_file}")
                    torch.save(self.model.state_dict(), self.checkpoint_model_file)

                is_stop, current_epoch, self.best_valid_metric = is_train_stop(
                    recall, self.best_valid_metric, current_epoch, self.model.early_stop_num
                )

                if is_stop:
                    logger.info("=== early stop break ===")
                    logger.info(f"best valid score : {self.best_valid_metric:.4f}")
                    break

        self.model.eval()

        recall = 0
        precision = 0
        ndcg = 0
        mrr = 0
        map_ = 0

        with torch.no_grad():
            logger.info("load best valid model")
            self.model.load_state_dict(torch.load(self.checkpoint_model_file))

            for _test_user_idx in test_dl:
                interaction = {
                    "user_idx": _test_user_idx[0].to(self.device),
                }

                score_matrix = self.model.all_user_predict(interaction)
                score_matrix = score_matrix.view(-1, self.dataset.item_num)

                evaluator = Evaluator(
                    _test_user_idx[0],
                    score_matrix,
                    self.dataset.train_val_user_idx2viewed_idx_dict,
                    self.dataset.test_user_idx2item_idx_dict,
                    self.topk,
                )

                recall += evaluator.get_recall().sum().item() / self.dataset.test_user_num
                precision += evaluator.get_precision().sum().item() / self.dataset.test_user_num
                ndcg += evaluator.get_ndcg().sum().item() / self.dataset.test_user_num
                mrr += evaluator.get_mrr().sum().item() / self.dataset.test_user_num
                map_ += evaluator.get_map().sum().item() / self.dataset.test_user_num

            self.test_recall = recall
            self.test_precision = precision
            self.test_ndcg = ndcg
            self.test_mrr = mrr
            self.test_map = map_

            logger.info(
                f"BEST VALID RESULT : recall@{self.topk} : {self.best_valid_recall:.4f}, precision@{self.topk} : {self.best_valid_precision:.4f}, ndcg@{self.topk} : {self.best_valid_ndcg:.4f}, mrr@{self.topk} : {self.best_valid_mrr:.4f}, map@{self.topk} : {self.best_valid_map:.4f}"
            )

            logger.info(
                f"TEST RESULT       : recall@{self.topk} : {self.test_recall:.4f}, precision@{self.topk} : {self.test_precision:.4f}, ndcg@{self.topk} : {self.test_ndcg:.4f}, mrr@{self.topk} : {self.test_mrr:.4f}, map@{self.topk} : {self.test_map:.4f}"
            )

            best_valid_result = {
                f"recall@{self.topk}": self.best_valid_recall,
                f"precision@{self.topk}": self.best_valid_precision,
                f"ndcg@{self.topk}": self.best_valid_ndcg,
                f"mrr@{self.topk}": self.best_valid_mrr,
                f"map@{self.topk}": self.best_valid_map,
            }

            test_result = {
                f"recall@{self.topk}": self.test_recall,
                f"precision@{self.topk}": self.test_precision,
                f"ndcg@{self.topk}": self.test_ndcg,
                f"mrr@{self.topk}": self.test_mrr,
                f"map@{self.topk}": self.test_map,
            }

            return best_valid_result, test_result
