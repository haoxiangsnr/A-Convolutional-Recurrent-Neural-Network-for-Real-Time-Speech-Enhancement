import importlib
import os
import time

import numpy as np
import torch
from pesq import pesq
from pystoi.stoi import stoi


def load_checkpoint(checkpoint_path, device):
    _, ext = os.path.splitext(os.path.basename(checkpoint_path))
    assert ext in (".pth", ".tar"), "Only support ext and tar extensions of model checkpoint."
    model_checkpoint = torch.load(os.path.abspath(os.path.expanduser(checkpoint_path)), map_location=device)

    if ext == ".pth":
        print(f"Loading {checkpoint_path}.")
        return model_checkpoint
    else:  # tar
        print(f"Loading {checkpoint_path}, epoch = {model_checkpoint['epoch']}.")
        return model_checkpoint["model"]


def get_sub_band_bound(idx, n_bins, n_neighbor):
    """
    根据索引来获取上下界限

    Args:
        idx: 当前索引
        n_bins: 总共的频带数量
        n_neighbor: 每侧拓展的频率带数量

    Returns:
        (子带上界的索引，子带下界的索引)
    """
    # 随机取子带区间
    n_bins_bottom = np.min([(n_bins - 1) - idx, n_neighbor])
    n_bins_top = np.min([idx, n_neighbor])

    # 补齐上边或者下边的长度
    if n_bins_bottom < n_neighbor:
        n_bins_top += n_neighbor - n_bins_bottom
    elif n_bins_top < n_neighbor:
        n_bins_bottom += n_neighbor - n_bins_top
    else:
        pass

    idx_bottom_bound = idx + n_bins_bottom
    idx_top_bound = idx - n_bins_top
    return idx_top_bound, idx_bottom_bound


def overlap_cat(chunk_list, dim=-1):
    """
    按照 50% 的 overlap 沿着最后一个维度对 chunk_list 进行拼接

    Args:
        dim: 需要拼接的维度
        chunk_list(list): [[B, T], [B, T], ...]

    Returns:
        overlap 拼接后
    """
    overlap_output = []
    for i, chunk in enumerate(chunk_list):
        first_half, last_half = torch.split(chunk, chunk.size(-1) // 2, dim=dim)
        if i == 0:
            overlap_output += [first_half, last_half]
        else:
            overlap_output[-1] = (overlap_output[-1] + first_half) / 2
            overlap_output.append(last_half)

    overlap_output = torch.cat(overlap_output, dim=dim)
    return overlap_output


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


def initialize_config(module_cfg, pass_args=True):
    """
    According to config items, load specific module dynamically with params.
    eg，config items as follow：
        module_cfg = {
            "module": "model.model",
            "main": "Model",
            "args": {...}
        }
    1. Load the module corresponding to the "module" param.
    2. Call function (or instantiate class) corresponding to the "main" param.
    3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"])

    if pass_args:
        return getattr(module, module_cfg["main"])(**module_cfg["args"])
    else:
        return getattr(module, module_cfg["main"])


def compute_PESQ(clean_signal, noisy_signal, sr=16000):
    return pesq(sr, clean_signal, noisy_signal, "wb")


def z_score(m):
    mean = np.mean(m)
    std_var = np.std(m)
    return (m - mean) / std_var, mean, std_var


def reverse_z_score(m, mean, std_var):
    return m * std_var + mean


def min_max(m):
    m_max = np.max(m)
    m_min = np.min(m)

    return (m - m_min) / (m_max - m_min), m_max, m_min


def reverse_min_max(m, m_max, m_min):
    return m * (m_max - m_min) + m_min


def sample_fixed_length_data_aligned(data_a, data_b, sample_length):
    """
    从某个随机位置开始，从两个样本中取出固定长度的片段
    """
    assert data_a.shape == data_b.shape, "Inconsistent dataset size."
    dim = np.ndim(data_a)
    assert dim == 1 or dim == 2, "Only support 1D or 2D."

    if data_a.shape[-1] > sample_length:
        frames_total = data_a.shape[-1]
        start = np.random.randint(frames_total - sample_length + 1)
        end = start + sample_length
        if dim == 1:
            return data_a[start:end], data_b[start:end]
        else:
            return data_a[:, start:end], data_b[:, start:end]
    elif data_a.shape[-1] == sample_length:
        return data_a, data_b
    else:
        frames_total = data_a.shape[-1]
        if dim == 1:
            return np.append(
                data_a,
                np.zeros(sample_length - frames_total, dtype=np.float32)
            ), np.append(
                data_b,
                np.zeros(sample_length - frames_total, dtype=np.float32)
            )
        else:
            return np.append(
                data_a,
                np.zeros(shape=(data_a.shape[0], sample_length - frames_total), dtype=np.float32),
                axis=-1
            ), np.append(
                data_b,
                np.zeros(shape=(data_a.shape[0], sample_length - frames_total), dtype=np.float32),
                axis=-1
            )


def compute_STOI(clean_signal, noisy_signal, sr=16000):
    return stoi(clean_signal, noisy_signal, sr, extended=False)


def print_tensor_info(tensor, flag="Tensor"):
    floor_tensor = lambda float_tensor: int(float(float_tensor) * 1000) / 1000
    print(flag)
    print(
        f"\tmax: {floor_tensor(torch.max(tensor))}, min: {float(torch.min(tensor))}, mean: {floor_tensor(torch.mean(tensor))}, std: {floor_tensor(torch.std(tensor))}")


def set_requires_grad(nets, requires_grad=False):
    """
    Args:
        nets: list of networks
        requires_grad
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def prepare_device(n_gpu: int, cudnn_deterministic=False):
    """Choose to use CPU or GPU depend on "n_gpu".
    Args:
        n_gpu(int): the number of GPUs used in the experiment.
            if n_gpu is 0, use CPU;
            if n_gpu > 1, use GPU.
        cudnn_deterministic (bool): repeatability
            cudnn.benchmark will find algorithms to optimize training. if we need to consider the repeatability of experiment, set use_cudnn_deterministic to True
    """
    if n_gpu == 0:
        print("Using CPU in the experiment.")
        device = torch.device("cpu")
    else:
        if cudnn_deterministic:
            print("Using CuDNN deterministic mode in the experiment.")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        device = torch.device("cuda:0")

    return device
