import importlib
import time

import torch
from pesq import pesq
import numpy as np
from pystoi.stoi import stoi
import os
import multiprocessing

def remove_extra_tail(m, size=256):
    assert m.shape[1] >= size, "len(y) should be large than size."
    return m[:, : - (m.shape[1] % size)]

def prepare_empty_dir(dirs, resume=False):
    """
    if resume experiment, assert the dirs exist,
    if not resume experiment, make dirs.

    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert dir_path.exists()
        else:
            dir_path.mkdir(parents=True, exist_ok=True)

class ExecutionTime:
    """
    Usage:
        timer = ExecutionTime()
        <Something...>
        print(f'Finished in {timer.duration()} seconds.')
    """


    def __init__(self):
        self.start_time = time.time()


    def duration(self):
        return int(time.time() - self.start_time)


def initialize_config(module_cfg):
    """
    According to config items, load specific module dynamically with params.

    eg，config items as follow：
        module_cfg = {
            "module": "models.model",
            "main": "Model",
            "args": {...}
        }

    1. Load the module corresponding to the "module" param.
    2. Call function (or instantiate class) corresponding to the "main" param.
    3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"])
    return getattr(module, module_cfg["main"])(**module_cfg["args"])

def compute_PESQ(clean_signal, noisy_signal, sr=16000):

    # pool = multiprocessing.Pool(1)`
    # rslt = pool.apply_async(_compute_PESQ_sub_task, args = (clean_signal, noisy_signal, sr))
    # pool.close() # 关闭进程池，不运行再向内添加进程
    # rslt.wait(timeout=1) # 子进程的处理时间上限为 1 秒钟，到时未返回结果，则终止子进程
    #
    # if rslt.ready(): # 在 1 秒钟内子进程处理完毕
    #     return rslt.get()
    # else: # 过了 1 秒了，但仍未处理完，返回 -1
    #     return -1
    return pesq(sr, clean_signal, noisy_signal, "wb")

def z_score(m):
    mean = torch.mean(m, [1,2])
    std_var = torch.std(m, [1,2])

    # size: [batch] => pad => [batch, T, F]
    mean = mean.expand(m.size()[::-1]).permute(2, 1, 0)
    std_var = std_var.expand(m.size()[::-1]).permute(2, 1, 0)

    return (m - mean) / std_var, mean, std_var

def reverse_z_score(m, mean, std_var):
    return m * std_var + mean

def min_max(m):
    m_max = np.max(m)
    m_min = np.min(m)

    return (m - m_min) / (m_max - m_min), m_max, m_min

def reverse_min_max(m, m_max, m_min):
    return m * (m_max - m_min) + m_min

def compute_STOI(clean_signal, noisy_signal, sr=16000):
    return stoi(clean_signal, noisy_signal, sr, extended=False)
