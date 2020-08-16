import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
import torchaudio
from tqdm import tqdm

from trainer.base_trainer import BaseTrainer
from util.utils import overlap_cat

plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(self, config, resume: bool, model, loss_function, optimizer, train_dataloader, validation_dataloader):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0

        for noisy, clean, _ in self.train_dataloader:
            self.optimizer.zero_grad()

            noisy = noisy.to(self.device)  # [B, T]
            clean = clean.to(self.device)  # [B, T]

            noisy_d = torch.stft(
                noisy,
                n_fft=320,
                hop_length=160,
                win_length=320,
                window=torch.hann_window(320).to(self.device))  # [B, F, T, 2]

            noisy_mag, noisy_phase = torchaudio.functional.magphase(noisy_d)  # [B, F, T], [B, F, T]

            enhanced_mag = self.model(noisy_mag)

            enhanced_d = torch.cat([
                (enhanced_mag * torch.cos(noisy_phase)).unsqueeze(-1),
                (enhanced_mag * torch.sin(noisy_phase)).unsqueeze(-1)
            ], dim=-1)  # [B, F, T, 2]

            enhanced = torchaudio.functional.istft(
                enhanced_d,
                n_fft=320,
                hop_length=160,
                win_length=320,
                window=torch.hann_window(320).to(self.device),
                length=noisy.shape[1])

            loss = self.loss_function(clean, enhanced)
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()

        self.writer.add_scalar(f"Loss/Train", loss_total / len(self.train_dataloader), epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        noisy_list = []
        clean_list = []
        enhanced_list = []

        loss_total = 0.0

        visualization_limit = self.validation_custom_config["visualization_limit"]
        n_fft = self.validation_custom_config["n_fft"]
        hop_length = self.validation_custom_config["hop_length"]
        win_length = self.validation_custom_config["win_length"]
        batch_size = self.validation_custom_config["batch_size"]
        unfold_size = self.validation_custom_config["unfold_size"]

        for i, (noisy, clean, name) in tqdm(enumerate(self.validation_dataloader), desc="Inference"):
            assert len(name) == 1, "The batch size of inference stage must 1."
            name = name[0]
            padded_length = 0  # 用于后续的 pad

            noisy = noisy.to(self.device)  # [1, T]
            clean = clean.to(self.device)  # [1, T]

            noisy_d = torch.stft(
                noisy,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=torch.hann_window(win_length).to(self.device))  # [B, F, T, 2]
            noisy_mag, noisy_phase = torchaudio.functional.magphase(noisy_d)  # [1, F, T]

            """=== === === start overlap enhancement === === ==="""
            noisy_mag = noisy_mag[None, :, :, :]  # [1, F, T] => [1, 1, F, T]，多一个维度是为了 unfold

            if noisy_mag.size(-1) % unfold_size != 0:
                padded_length = unfold_size - (noisy_mag.size(-1) % unfold_size)
                noisy_mag = torch.cat([noisy_mag, torch.zeros(1, 1, noisy_mag.size(2), padded_length, device=self.device)], dim=-1)  # [1, 1, F, T]

            noisy_mag = functional.unfold(noisy_mag, kernel_size=(n_fft // 2 + 1, unfold_size), stride=unfold_size // 2)
            # [1, F, T, N] => [N, 1, F, T] => [N, F, T], where is #chunks.
            noisy_mag = noisy_mag.reshape(1, n_fft // 2 + 1, unfold_size, -1).permute(3, 0, 1, 2).squeeze(1)
            noisy_mag_chunks = torch.split(noisy_mag, batch_size, dim=0)  # [N, F, T] => ([B, F, T], ...), where the number is N // batch_size + 1

            enhanced_mag_chunks = []
            for noisy_mag_chunk in noisy_mag_chunks:
                enhanced_mag_chunk = self.model(noisy_mag_chunk)  # [1, F, T]
                enhanced_mag_chunks += torch.split(enhanced_mag_chunk, 1, dim=0)  # [B, F, T] => ([1, F, T], [1, F, T], ...)

            enhanced_mag = overlap_cat(enhanced_mag_chunks)  # ([1, F, T], [1, F, T], ...) => [1, F, T]
            enhanced_mag = enhanced_mag if padded_length == 0 else enhanced_mag[:, :, :-padded_length]  # [1, F, T]
            """=== === === end overlap enhancement === === ==="""

            enhanced_d = torch.cat([
                (enhanced_mag * torch.cos(noisy_phase)).unsqueeze(-1),
                (enhanced_mag * torch.sin(noisy_phase)).unsqueeze(-1)
            ], dim=-1)  # [B, F, T, 2]

            enhanced = torchaudio.functional.istft(
                enhanced_d,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=torch.hann_window(win_length).to(self.device),
                length=noisy.shape[1])  # [1, T]

            loss_total += self.loss_function(clean, enhanced).item()

            noisy = noisy.detach().squeeze(0).cpu().numpy()
            clean = clean.detach().squeeze(0).cpu().numpy()
            enhanced = enhanced.detach().squeeze(0).cpu().numpy()

            assert len(noisy) == len(clean) == len(enhanced)

            if i <= np.min([visualization_limit, len(self.validation_dataloader)]):
                """======= 可视化第 i 个结果 ======="""
                self.spec_audio_visualization(noisy, enhanced, clean, name, epoch)

            noisy_list.append(noisy)
            clean_list.append(clean)
            enhanced_list.append(enhanced)

        self.writer.add_scalar(f"Loss/Validation", loss_total / len(self.validation_dataloader), epoch)
        return self.metrics_visualization(noisy_list, clean_list, enhanced_list, epoch)
