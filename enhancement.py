import argparse
import json
import os
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import soundfile as sf

from utils.stft import STFT
from utils.utils import initialize_config, compute_PESQ


def main(config, epoch):
    root_dir = Path(config["experiments_dir"]) / config["name"]
    enhancement_dir = root_dir / "enhancements"
    checkpoints_dir = root_dir / "checkpoints"

    """============== 加载数据集 =============="""
    dataset = initialize_config(config["dataset"])
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
    )

    """============== 加载模型断点（"best"，"latest"，通过数字指定） =============="""
    model = initialize_config(config["model"])
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    stft = STFT(
        filter_length=320,
        hop_length=160
    ).to("cpu")

    if epoch == "best":
        model_path = checkpoints_dir / "best_model.tar"
        model_checkpoint = torch.load(model_path.as_posix(), map_location=device)
        model_static_dict = model_checkpoint["model"]
        checkpoint_epoch = model_checkpoint['epoch']
    elif epoch == "latest":
        model_path = checkpoints_dir / "latest_model.tar"
        model_checkpoint = torch.load(model_path.as_posix(), map_location=device)
        model_static_dict = model_checkpoint["model"]
        checkpoint_epoch = model_checkpoint['epoch']
    else:
        model_path = checkpoints_dir / f"model_{str(epoch).zfill(4)}.pth"
        model_checkpoint = torch.load(model_path.as_posix(), map_location=device)
        model_static_dict = model_checkpoint
        checkpoint_epoch = epoch

    print(f"Loading model checkpoint, epoch = {checkpoint_epoch}")
    model.load_state_dict(model_static_dict)
    model.to(device)
    model.eval()

    """============== 增强语音 =============="""
    if epoch == "best" or epoch == "latest":
        results_dir = enhancement_dir / f"{epoch}_checkpoint_{checkpoint_epoch}_epoch"
    else:
        results_dir = enhancement_dir / f"checkpoint_{epoch}_epoch"

    results_dir.mkdir(parents=True, exist_ok=True)

    for i, (mixture, clean, _, names) in enumerate(dataloader):
        print(f"Enhance {i + 1}th speech")
        name = names[0]

        # Mixture mag and Clean mag
        print("\tSTFT...")
        mixture_D = stft.transform(mixture)
        mixture_real = mixture_D[:, :, :, 0]
        mixture_imag = mixture_D[:, :, :, 1]
        mixture_mag = torch.sqrt(mixture_real ** 2 + mixture_imag ** 2) # [1, T, F]

        print("\tEnhancement...")
        mixture_mag_chunks = torch.split(mixture_mag, mixture_mag.size()[1] // 5, dim=1)
        mixture_mag_chunks = mixture_mag_chunks[:-1]
        enhanced_mag_chunks = []
        for mixture_mag_chunk in tqdm(mixture_mag_chunks):
            mixture_mag_chunk = mixture_mag_chunk.to(device)
            enhanced_mag_chunks.append(model(mixture_mag_chunk).detach().cpu()) # [T, F]

        enhanced_mag = torch.cat(enhanced_mag_chunks, dim=0).unsqueeze(0) # [1, T, F]

        # enhanced_mag = enhanced_mag.detach().cpu().data.numpy()
        # mixture_mag = mixture_mag.cpu()

        enhanced_real = enhanced_mag * mixture_real[:, :enhanced_mag.size(1), :] / mixture_mag[:, :enhanced_mag.size(1), :]
        enhanced_imag = enhanced_mag * mixture_imag[:, :enhanced_mag.size(1), :] / mixture_mag[:, :enhanced_mag.size(1), :]

        enhanced_D = torch.stack([enhanced_real, enhanced_imag], 3)
        enhanced = stft.inverse(enhanced_D)

        enhanced = enhanced.detach().cpu().squeeze().numpy()

        sf.write(f"{results_dir}/{name}.wav", enhanced, 16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Spectrogram mapping: Speech Enhancement")
    parser.add_argument("-C", "--config", type=str, required=True,
                        help="Specify the configuration file for enhancement (*.json).")
    parser.add_argument('-D', '--device', default=None, type=str, help="GPU for speech enhancement.")
    parser.add_argument("-E", "--epoch", default="best",
                        help="Model checkpoint for speech enhancement, can be set to 'best', 'latest' and specific epoch. (default: 'best')")
    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    config = json.load(open(args.config))
    config["name"] = os.path.splitext(os.path.basename(args.config))[0]
    main(config, args.epoch)
