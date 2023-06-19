import os
from datetime import datetime

from torch import nn
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_


def xavier_normal_initialization(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


def xavier_uniform_initialization(module):
    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


def is_train_stop(val, best_val, current_epoch, early_stop_num):
    """
    stop epoch by early stop

    """
    if val > best_val:
        best_val = val
        current_epoch = 0
        is_stop = False
    else:
        current_epoch = current_epoch + 1
        if current_epoch > early_stop_num:
            is_stop = True
        else:
            is_stop = False

    return is_stop, current_epoch, best_val


def convert_to_num(arg):
    try:
        if arg == "inf":
            return float(arg)
        return eval(arg)
    except:
        return arg


def get_yyyymmddhhmmss():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
