import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from trainer.base_trainer import BaseTrainer

plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(self, config, resume: bool, model, loss_function, optimizer, train_dataloader, validation_dataloader):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0

        for noisy, clean, n_frames_list, _ in self.train_dataloader:
            self.optimizer.zero_grad()

            noisy = noisy.to(self.device).unsqueeze(1)  # [B, F, T] => [B, 1, F, T]
            clean = clean.to(self.device).unsqueeze(1)  # [B, F, T] => [B, 1, F, T]

            enhanced = self.model(noisy)  # [B, 1, F, T]

            loss = self.loss_function(enhanced, clean, n_frames_list, self.device)
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

        for i, (noisy, clean, name) in tqdm(enumerate(self.validation_dataloader), desc="Inference"):
            assert len(name) == 1, "The batch size of inference stage must 1."
            name = name[0]

            noisy = noisy.numpy().reshape(-1)
            clean = clean.numpy().reshape(-1)

            noisy_mag, noisy_phase = librosa.magphase(librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length))  # [T], [F, T]
            clean_mag, _ = librosa.magphase(librosa.stft(clean, n_fft=n_fft, hop_length=hop_length, win_length=win_length))  # [T] => [F, T]

            noisy_mag = torch.tensor(noisy_mag, device=self.device)[None, None, :, :]  # [F, T] => [1, 1, F, T]
            clean_mag = torch.tensor(clean_mag, device=self.device)[None, None, :, :]

            enhanced_mag = self.model(noisy_mag)

            loss_total += self.loss_function(clean_mag, enhanced_mag, [clean_mag.shape[-1], ], self.device).item()

            enhanced_mag = enhanced_mag.squeeze(0).squeeze(0).detach().cpu().numpy()  # [1, 1, F, T] => [F, T]
            enhanced = librosa.istft(enhanced_mag * noisy_phase, hop_length=hop_length, win_length=win_length, length=len(noisy))

            assert len(noisy) == len(clean) == len(enhanced)

            if i <= np.min([visualization_limit, len(self.validation_dataloader)]):
                """======= 可视化第 i 个结果 ======="""
                self.spec_audio_visualization(noisy, enhanced, clean, name, epoch)

            noisy_list.append(noisy)
            clean_list.append(clean)
            enhanced_list.append(enhanced)

        self.writer.add_scalar(f"Loss/Validation", loss_total / len(self.validation_dataloader), epoch)
        return self.metrics_visualization(noisy_list, clean_list, enhanced_list, epoch)
