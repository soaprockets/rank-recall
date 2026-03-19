# coding: utf-8

"""
Utility functions
##########################
"""

import numpy as np
import torch
import importlib
import datetime, random
import pandas as pd 
from utils.enum_type import ModelType


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y-%H-%M-%S')

    return cur


def get_model_type(model_name):
    model_file_name = model_name.lower()
    module_paths = ['.'.join(['models', model_file_name]), '.'.join(['models', 'sequential', model_file_name])]
    for i, module_path in enumerate(module_paths):
        if importlib.util.find_spec(module_path, __name__):
            break
    return ModelType(i)


def get_model(model_name):
    r"""Automatically select model class based on model name
    Args:
        model_name (str): model name
    Returns:
        Recommender: model class
    """
    model_file_name = model_name.lower()
    module_path = '.'.join(['models', model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)

    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer():
    return getattr(importlib.import_module('models.common.trainer'), 'Trainer')


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r""" validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ': ' + '%.04f' % value + '    '
    return result_str


def load_csv_to_dict(file_path):
    """
    Read a CSV file using Pandas with tab as the separator and convert to a dictionary.
    Use the first column as keys and the second column as values.

    Args:
    file_path (str): The path to the CSV file.

    Returns:
    dict: A dictionary where the first column is the key and the second column is the value.
    """
    # 读取文件
    df = pd.read_csv(file_path, sep='\t',skiprows=1,header=None,index_col=False)
    
    # 将前两列转换为字典
    result_dict = pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0]).to_dict()

    return result_dict
