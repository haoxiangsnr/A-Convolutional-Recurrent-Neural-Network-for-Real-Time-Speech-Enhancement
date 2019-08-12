import librosa
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from trainer.base_trainer import BaseTrainer
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import numpy as np
import librosa.display
from tqdm import tqdm

from utils.utils import compute_STOI, compute_PESQ, z_score, reverse_z_score


class Trainer(BaseTrainer):
    def __init__(self,
                 config,
                 resume,
                 model,
                 optimizer,
                 loss_function,
                 train_dataloader,
                 validation_dataloader):
        super(Trainer, self).__init__(config, resume, model, optimizer, loss_function)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0
        for mixture, clean, n_frames_list, _ in tqdm(self.train_dataloader, desc="Training"):
            self.optimizer.zero_grad()

            mixture = mixture.to(self.device)
            clean = clean.to(self.device)

            # Mixture mag and Clean mag
            mixture_D  = self.stft.transform(mixture)
            mixture_real = mixture_D[:, :, :, 0]
            mixture_imag = mixture_D[:, :, :, 1]
            mixture_mag = torch.sqrt(mixture_real ** 2 + mixture_imag ** 2) # [batch, T, F]

            clean_D  = self.stft.transform(clean)
            clean_real = clean_D[:, :, :, 0]
            clean_imag = clean_D[:, :, :, 1]
            clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)

            if self.z_score:
                mixture_mag, _, _ = z_score(mixture_mag)
                clean_mag, _, _ = z_score(clean_mag)

            enhanced_mag = self.model(mixture_mag)

            loss = self.loss_function(enhanced_mag, clean_mag, n_frames_list)
            loss.backward()

            self.optimizer.step()

            loss_total += float(loss)

        dataloader_len = len(self.train_dataloader)
        self.viz.writer.add_scalar("Loss/Train", loss_total / dataloader_len, epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        loss_total = 0.0
        mixture_mean = None
        mixture_std = None
        stoi_c_n = []
        stoi_c_d = []
        pesq_c_n = []
        pesq_c_d = []

        for mixture, clean, n_frames_list, names in tqdm(self.validation_dataloader):
            mixture = mixture.to(self.device)
            clean = clean.to(self.device)

            # Mixture mag and Clean mag
            mixture_D = self.stft.transform(mixture)
            mixture_real = mixture_D[:, :, :, 0]
            mixture_imag = mixture_D[:, :, :, 1]
            mixture_mag = torch.sqrt(mixture_real ** 2 + mixture_imag ** 2)

            clean_D = self.stft.transform(clean)
            clean_real = clean_D[:, :, :, 0]
            clean_imag = clean_D[:, :, :, 1]
            clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)

            if self.z_score:
                mixture_mag, mixture_mean, mixture_std = z_score(mixture_mag)
                clean_mag, _, _ = z_score(clean_mag)

            enhanced_mag = self.model(mixture_mag)

            loss = self.loss_function(enhanced_mag, clean_mag, n_frames_list)
            loss_total += loss

            if self.z_score:
                enhanced_mag = reverse_z_score(enhanced_mag, mixture_mean, mixture_std)

            masks = []
            len_list = []
            for n_frames in n_frames_list:
                masks.append(torch.ones(n_frames, 161, dtype=torch.float32))
                len_list.append((n_frames - 1) * 160 + 320)

            masks = pad_sequence(masks, batch_first=True).to(self.device) # [batch, longest n_frame, n_fft]
            enhanced_mag = enhanced_mag * masks

            enhanced_real = enhanced_mag * mixture_real / mixture_mag.squeeze(1)
            enhanced_imag = enhanced_mag * mixture_imag / mixture_mag.squeeze(1)

            enhanced_D = torch.stack([enhanced_real, enhanced_imag], 3)
            enhanced = self.stft.inverse(enhanced_D)

            enhanced_speeches = enhanced.detach().cpu().numpy()
            mixture_speeches = mixture.detach().cpu().numpy()
            clean_speeches = clean.detach().cpu().numpy()

            for i in range(len(n_frames_list)):
                enhanced = enhanced_speeches[i][:len_list[i]]
                mixture = mixture_speeches[i][:len_list[i]]
                clean = clean_speeches[i][:len_list[i]]

                self.viz.writer.add_audio(f"Audio/{names[i]}_Mixture", mixture, epoch, sample_rate=16000)
                self.viz.writer.add_audio(f"Audio/{names[i]}_Enhanced", enhanced, epoch, sample_rate=16000)
                self.viz.writer.add_audio(f"Audio/{names[i]}_Clean", clean, epoch, sample_rate=16000)

                fig, ax = plt.subplots(3, 1)
                for j, y in enumerate([mixture, enhanced, clean]):
                    ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                        np.mean(y),
                        np.std(y),
                        np.max(y),
                        np.min(y)
                    ))
                    librosa.display.waveplot(y, sr=16000, ax=ax[j])
                plt.tight_layout()
                self.viz.writer.add_figure(f"Waveform/{names[i]}", fig, epoch)

                stoi_c_n.append(compute_STOI(clean, mixture, sr=16000))
                stoi_c_d.append(compute_STOI(clean, enhanced, sr=16000))
                pesq_c_n.append(compute_PESQ(clean, mixture, sr=16000))
                pesq_c_d.append(compute_PESQ(clean, enhanced, sr=16000))


        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        self.viz.writer.add_scalars(f"Metrics/STOI", {
            "clean and noisy": get_metrics_ave(stoi_c_n),
            "clean and denoisy": get_metrics_ave(stoi_c_d)
        }, epoch)
        self.viz.writer.add_scalars(f"Metrics/PESQ", {
            "clean and noisy": get_metrics_ave(pesq_c_n),
            "clean anddenoisy": get_metrics_ave(pesq_c_d)
        }, epoch)

        dataloader_len = len(self.validation_dataloader)
        self.viz.writer.add_scalar("Loss/Validation", loss_total / dataloader_len, epoch)

        score = (get_metrics_ave(stoi_c_d) + self._transform_pesq_range(get_metrics_ave(pesq_c_d))) / 2
        return score

