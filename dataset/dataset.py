# TODO: 時系列にそって分割
# TODO: groupbyをitemでもするべき？

import torch
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp

from util.logger import logger

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, config, dataset_info) -> None:
        self.file_name = dataset_info["input_file_path"]
        self.config = config
        self.usecols = config["usecols"]
        self.name = self.config["dataset"]
        self.lower = float(self.config["dataset_lower"])
        self.upper = float(self.config["dataset_upper"])
        self.train_batch_size = self.config["train_batch_size"]
        self.eval_batch_size = self.config["eval_batch_size"]

        self.random_state = self.config["random_state"]

        self.user_id_name = self.config["user_id_name"]
        self.item_id_name = self.config["item_id_name"]

        self.user_idx_name = self.config["user_idx_name"]
        self.item_idx_name = self.config["item_idx_name"]
        self.neg_item_idx_name = self.config["neg_item_idx_name"]

        self.random_state = self.config["random_state"]

        self.train_size = self.config["train_val_test"][0]
        self.val_size = self.config["train_val_test"][1]
        self.test_size = self.config["train_val_test"][2]

        self.orig_df = pd.read_csv(self.file_name, usecols=self.usecols).reset_index(drop=True)
        self.orig_df_before_filtering_shape = self.orig_df.shape
        logger.debug(f"orig_df shape : {self.orig_df.shape}")

        # 重複削除
        logger.debug("drop duplicate")
        self.orig_df = self.orig_df.drop_duplicates(subset=self.usecols).reset_index(drop=True)
        self.orig_df_after_filtering_shape = self.orig_df.shape
        logger.debug(f"orig_df shape : {self.orig_df.shape}")

        self.data_split = self.config["data_split"]

        # interactionに応じてfilter
        self.df = self.get_filtered_df()

        # TODO: debug 後に削除
        logger.debug(f"user_num   : {self.df.user_id.unique().size:,}")
        logger.debug(f"item_num   : {self.df.item_id.unique().size:,}")
        logger.debug(f"df.head(5) : \n{self.df.head(5)}")
        logger.debug(f"df.tail(5) : \n{self.df.tail(5)}")

        self.user_id2user_idx_dict = {}
        self.item_id2item_idx_dict = {}

        self.user_idx, _ = pd.factorize(self.df.user_id, sort=True)
        self.item_idx, _ = pd.factorize(self.df.item_id, sort=True)

        # id => arrayのidxへ変換するdict
        self.user_id2user_idx_dict = dict(zip(self.df.user_id, self.user_idx))
        self.item_id2item_idx_dict = dict(zip(self.df.item_id, self.item_idx))

        self.user_idx2user_id_dict = dict(zip(self.user_idx, self.df.user_id))
        self.item_idx2item_id_dict = dict(zip(self.item_idx, self.df.item_id))

        self.df[self.user_idx_name] = self.df.user_id.map(lambda x: self.user_id2user_idx_dict[x])
        self.df[self.item_idx_name] = self.df.item_id.map(lambda x: self.item_id2item_idx_dict[x])

        # train, val, test split
        logger.debug("data split")
        if self.data_split == "sort":
            self.split_train_val_test_by_sort()
        elif self.data_split == "no_sort":
            self.split_train_val_test()
        else:
            raise Exception

        # self.df = pd.concat([self.train_df, self.val_df, self.test_df]).reset_index(drop=True)
        self.all_item_idx_set = set(self.train_df.item_idx.values)

        # negative samplingの準備
        logger.debug("set_user_idx2neg_item_idx_set_dict")
        self.set_user_idx2neg_item_idx_set_dict()

        # 評価用のtrainのidx
        logger.debug("set_train_viewed_item_idx")
        self.set_train_viewed_item_idx()

        # 評価用のtrainとvalのidx
        self.set_train_val_viewed_item_idx()

        # 基本情報をset
        self.set_basic_info()

        # TODO:後に削除
        # データを保存
        self.save_train_valid_test_data()

    def set_basic_info(self):
        # basic info
        self.user_num = self.df.user_idx.unique().size
        self.item_num = self.df.item_idx.unique().size
        self.nnz = self.df.shape[0]

        self.density = self.nnz / self.user_num / self.item_num
        self.sparsity = 1 - self.density

        self.user_interaction_mean = self.df.groupby(self.user_idx_name).count().item_idx.mean()
        self.item_interaction_mean = self.df.groupby(self.item_idx_name).count().user_idx.mean()

        self.train_user_num = self.train_df.user_idx.unique().size
        self.train_item_num = self.train_df.item_idx.unique().size
        self.train_nnz = self.train_df.shape[0]

        self.val_user_num = self.val_df.user_idx.unique().size
        self.val_item_num = self.val_df.item_idx.unique().size
        self.val_nnz = self.val_df.shape[0]

        self.test_user_num = self.test_df.user_idx.unique().size
        self.test_item_num = self.test_df.item_idx.unique().size
        self.test_nnz = self.test_df.shape[0]

    def get_filtered_df(self) -> pd:
        _df = self.orig_df.copy(deep=True)

        # filterの条件を満たすまでloop
        logger.info("dataset filter loop start")
        while True:
            # TODO: 後に削除
            # 時間がかかるのでdebug用
            logger.info("inside filtering loop")

            start = time.time()

            logger.debug("  user_id groupby")
            _df = (
                _df.groupby(self.user_id_name)
                .filter(
                    lambda x: self.lower <= x[self.item_id_name].count() & x[self.item_id_name].count() <= self.upper
                )
                .reset_index(drop=True)
            )

            logger.debug("  item_id groupby")
            _df = (
                _df.groupby(self.item_id_name)
                .filter(
                    lambda x: self.lower <= x[self.user_id_name].count() & x[self.user_id_name].count() <= self.upper
                )
                .reset_index(drop=True)
            )

            if _df.shape[0] == 0:
                raise Exception("df data length is 0.")

            _item_id_groupby = _df.groupby(self.item_id_name).count()
            _user_id_groupby = _df.groupby(self.user_id_name).count()

            logger.debug(f"  groupby_item_id max : {_item_id_groupby.max()[0]}")
            logger.debug(f"  groupby_item_id min : {_item_id_groupby.min()[0]}")
            logger.debug(f"  groupby_user_id max : {_user_id_groupby.max()[0]}")
            logger.debug(f"  groupby_user_id min : {_user_id_groupby.min()[0]}")

            if (
                _item_id_groupby.max()[0] <= self.upper
                and _item_id_groupby.min()[0] >= self.lower
                and _user_id_groupby.max()[0] <= self.upper
                and _user_id_groupby.min()[0] >= self.lower
            ):
                break

            # TODO: 後に削除
            elapsed_time = time.time() - start
            logger.info(f"elapsed_time : {elapsed_time:.3f}")

        logger.info("dataset filter loop end")

        return _df

    def split_train_val_test(self):
        user_idx = self.df.user_idx
        item_idx = self.df.item_idx

        user_train, user_test, item_train, item_test = train_test_split(
            user_idx, item_idx, test_size=self.test_size, stratify=user_idx, random_state=self.random_state
        )

        user_train, user_val, item_train, item_val = train_test_split(
            user_train,
            item_train,
            test_size=self.val_size / (1 - self.test_size),
            stratify=user_train,
            random_state=self.random_state,
        )

        self.train_df = pd.DataFrame(
            {
                self.user_idx_name: user_train.values,
                self.item_idx_name: item_train.values,
            }
        )

        self.val_df = pd.DataFrame(
            {
                self.user_idx_name: user_val.values,
                self.item_idx_name: item_val.values,
            }
        )

        self.test_df = pd.DataFrame(
            {
                self.user_idx_name: user_test.values,
                self.item_idx_name: item_test.values,
            }
        )

    # TODO: 時系列データかつ層化抽出 train_test_splitでstratify=True & shuffle=False ができない
    def split_train_val_test_by_sort(self):
        # trainはvalとtestより前，valとtestは順番をつけない

        train_df = (
            self.df.sort_values(by="timestamp")
            .groupby("user_idx")
            .apply(lambda x: x.head(int(x.shape[0] * self.train_size)))
        )

        # multi-index解除
        train_df.index = train_df.index.droplevel(0)

        val_test_df = self.df[~self.df.index.isin(train_df.index)]

        val_df = (
            val_test_df.sample(frac=1)
            .groupby("user_idx")
            .apply(lambda x: x.head(int(x.shape[0] * self.val_size / (self.val_size + self.test_size))))
        )

        # multi-index解除
        val_df.index = val_df.index.droplevel(0)

        test_df = val_test_df[~val_test_df.index.isin(val_df.index)]

        # 時系列に分解できているか確認
        assert sum(train_df.groupby("user_idx").max()["timestamp"] > val_df.groupby("user_idx").min()["timestamp"]) == 0

        assert (
            sum(train_df.groupby("user_idx").max()["timestamp"] > test_df.groupby("user_idx").min()["timestamp"]) == 0
        )

        # 指定した割合で分割できているか確認
        assert (
            0.95 * self.train_size
            >= (train_df.groupby("user_idx").count()["item_idx"] / self.df.groupby("user_idx").count()["item_idx"])
        ).sum() == 0

        assert (
            1.05 * self.train_size
            <= (train_df.groupby("user_idx").count()["item_idx"] / self.df.groupby("user_idx").count()["item_idx"])
        ).sum() == 0

        self.train_df = train_df[["user_idx", "item_idx"]].reset_index(drop=True)
        self.val_df = val_df[["user_idx", "item_idx"]].reset_index(drop=True)
        self.test_df = test_df[["user_idx", "item_idx"]].reset_index(drop=True)

    def set_user_idx2neg_item_idx_set_dict(self):
        self.user_idx2neg_item_idx_set_dict = {}

        for _idx, _df in self.train_df.groupby(self.user_idx_name):
            _positive_item_set = set(_df.item_idx.values)
            self.user_idx2neg_item_idx_set_dict[_idx] = self.all_item_idx_set - _positive_item_set

    def get_dataloader(self):
        train_ds = TensorDataset(
            torch.tensor(self.train_df.user_idx.values, dtype=torch.long),
            torch.tensor(self.train_df.item_idx.values, dtype=torch.long),
        )

        # valのdatasetはuser_idxに対してitem_idxのlist
        _val_user_idx_list = []
        self.val_user_idx2item_idx_dict = {}

        for _idx, _df in self.val_df.groupby(self.user_idx_name):
            _val_user_idx_list.append(_idx)
            self.val_user_idx2item_idx_dict[_idx] = _df.item_idx.values
        val_ds = TensorDataset(torch.tensor(_val_user_idx_list))

        # testのdatasetはuser_idxに対してitem_idxのlist
        _test_user_idx_list = []
        self.test_user_idx2item_idx_dict = {}

        for _idx, _df in self.test_df.groupby(self.user_idx_name):
            _test_user_idx_list.append(_idx)
            self.test_user_idx2item_idx_dict[_idx] = _df.item_idx.values
        test_ds = TensorDataset(torch.tensor(_test_user_idx_list))

        # set dataloader
        train_dl = DataLoader(train_ds, batch_size=self.train_batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=self.eval_batch_size, shuffle=False)
        test_dl = DataLoader(test_ds, batch_size=self.eval_batch_size, shuffle=False)

        return train_dl, val_dl, test_dl

    def get_train_coo_matrix(self):
        return sp.coo_matrix(
            ([1] * self.train_nnz, (self.train_df.user_idx.values, self.train_df.item_idx.values)),
            shape=(self.train_user_num, self.train_item_num),
        ).astype(np.float32)

    def get_train_csr_matrix(self):
        return sp.csr_matrix(
            ([1] * self.train_nnz, (self.train_df.user_idx.values, self.train_df.item_idx.values)),
            shape=(self.train_user_num, self.train_item_num),
        ).astype(np.float32)

    # 評価のためにtrainで利用されたデータを除外するためのidx
    def set_train_viewed_item_idx(self):
        self.train_user_idx2viewed_idx_dict = {}

        for _idx, _df in self.train_df.groupby(self.user_idx_name):
            self.train_user_idx2viewed_idx_dict[_idx] = _df.item_idx.values

    # 評価のためにtrainとvalで利用されたデータを除外するためのidx
    def set_train_val_viewed_item_idx(self):
        self.train_val_user_idx2viewed_idx_dict = {}

        _train_val_df = pd.concat([self.train_df, self.val_df]).reset_index(drop=True)

        for _idx, _df in _train_val_df.groupby(self.user_idx_name):
            self.train_val_user_idx2viewed_idx_dict[_idx] = _df.item_idx.values

    def save_train_valid_test_data(self):
        # vscodeの確認用データの保存
        import os
        import datetime

        yyyymmdd_dir = datetime.datetime.now().strftime("%Y%m%d")
        os.makedirs("./result/" + yyyymmdd_dir + "_data/", exist_ok=True)

        self.train_df.to_csv(f'./result/{yyyymmdd_dir}_data/train_{self.name}_{self.config["seed"]}.csv')
        self.val_df.to_csv(f'./result/{yyyymmdd_dir}_data/valid_{self.name}_{self.config["seed"]}.csv')
        self.test_df.to_csv(f'./result/{yyyymmdd_dir}_data/test_{self.name}_{self.config["seed"]}.csv')
