# qgcnのgithub公開用
# コード修正デバッグ用
# ml-100k
# recall@10 : 0.2056, precision@10 : 0.2035, ndcg@10 : 0.2563, mrr@10 : 0.4578, map@10 : 0.1332
# recall@10 : 0.2402, precision@10 : 0.2462, ndcg@10 : 0.3237, mrr@10 : 0.5692, map@10 : 0.1917


import yaml
import json
import argparse
import dataset

from loguru import logger
from enum import Enum
from util.seed import set_seed
from util.logger import init_logger
from QGCN import QGCN


class DatasetInfo(Enum):
    ML_100k = {
        "name": "ml-100k",
        "input_file_path": "./data/csv/ml-100k.csv",
        "config_file_path": "./config/dataset/ml-100k.yaml",
        "default_columns": ["user_id", "item_id", "rating", "timestamp"],
    }

    ML_1M = {
        "name": "ml-1m",
        "input_file_path": "./data/csv/ml-1m.csv",
        "config_file_path": "./config/dataset/ml-1m.yaml",
        "default_columns": ["user_id", "item_id", "rating", "timestamp"],
    }

    Gowalla = {
        "name": "gowalla",
        "input_file_path": "./data/csv/gowalla.csv",
        "config_file_path": "./config/dataset/gowalla.yaml",
        "default_columns": ["user_id", "item_id", "timestamp"],
    }

    Yelp = {
        "name": "yelp",
        "input_file_path": "./data/csv/yelp.csv",
        "config_file_path": "./config/dataset/yelp.yaml",
        "default_columns": ["user_id", "item_id", "timestamp"],
    }

    Amazon_Books = {
        "name": "Amazon_Books",
        "input_file_path": "./data/csv/Amazon_Books.csv",
        "config_file_path": "./config/dataset/Amazon_Books.yaml",
        "default_columns": ["user_id", "item_id", "timestamp"],
    }

    @staticmethod
    def get_dataset_info(dataset_name):
        for i in DatasetInfo:
            if i.value["name"] == dataset_name:
                return i.value
        raise Exception("No dataset info.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", required=False, type=int, default=123)
    parser.add_argument("-d", "--dataset", required=False, type=str, default="ml-100k")
    # parser.add_argument("-d", "--dataset", required=False, type=str, default="ml-1m")
    # parser.add_argument("-d", "--dataset", required=False, type=str, default="yelp")
    # parser.add_argument("-d", "--dataset", required=False, type=str, default="gowalla")
    # parser.add_argument("-d", "--dataset", required=False, type=str, default="Amazon_Books")

    args, _ = parser.parse_known_args()

    return args


def main(args: argparse) -> None:
    print(args.dataset)
    dataset_info = DatasetInfo.get_dataset_info(args.dataset)

    with open("./config/common.yaml") as f:
        config = yaml.safe_load(f.read())

    init_logger(config)

    logger.info("1. read config file")

    with open(dataset_info["config_file_path"]) as f:
        model_config = yaml.safe_load(f.read())

    logger.info("2. update config")
    config.update(model_config)
    config.update(args.__dict__)
    logger.info(f"config : {json.dumps(config, indent=2)}")

    logger.info("3. set seed")
    set_seed(seed=config["seed"])

    logger.info("4. set dataset")
    ds = dataset.Dataset(config, dataset_info)

    logger.info(f"user_num  : {ds.user_num:,}")
    logger.info(f"item_num  : {ds.item_num:,}")
    logger.info(f"inter_num : {ds.nnz:,}")

    logger.info("5. set model")
    model = QGCN(config, ds)
    logger.info(model)

    logger.info("6. training and evaluate")
    model.fit()

    # embeddingのベクトル多様性の計算
    model.calculate_gcn_diversity()


if __name__ == "__main__":
    args = parse_args()
    main(args)
