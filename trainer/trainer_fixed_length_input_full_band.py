import matplotlib.pyplot as plt
import numpy as np
import torch

from inferencer.inferencer import inference_wrapper
from trainer.base_trainer import BaseTrainer

plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(self, config, resume: bool, model, loss_function, optimizer, train_dataloader, validation_dataloader):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0

        for noisy_mag, clean_mag, _ in self.train_dataloader:
            noisy_mag = noisy_mag.to(self.device)
            clean_mag = clean_mag.to(self.device)

            self.optimizer.zero_grad()

            enhanced_mag = self.model(noisy_mag)

            loss = self.loss_function(clean_mag, enhanced_mag)
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()

        self.writer.add_scalar(f"Train/Loss", loss_total / len(self.train_dataloader), epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        noisy_list, enhanced_list, clean_list, name_list, loss = inference_wrapper(
            dataloader=self.validation_dataloader,
            model=self.model,
            loss_function=self.loss_function,
            device=self.device,
            inference_args=self.validation_custom_config,
            enhanced_dir=None
        )

        self.writer.add_scalar(f"Validation/Loss", loss, epoch)

        for i in range(np.min([self.validation_custom_config["visualization_limit"], len(self.validation_dataloader)])):
            self.spec_audio_visualization(
                noisy_list[i],
                enhanced_list[i],
                clean_list[i],
                name_list[i],
                epoch
            )

        score = self.metrics_visualization(noisy_list, clean_list, enhanced_list, epoch)
        return score
