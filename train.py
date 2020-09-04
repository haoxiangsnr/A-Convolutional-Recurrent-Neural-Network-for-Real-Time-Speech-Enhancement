import argparse
import os

import json5
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from util.utils import initialize_config


def main(config, resume):
    # Random seed for both CPU and GPU.
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    def collate_fn_pad(batch):
        """
        Returns:
            [B, F, T (Longest)]
        """
        noisy_list = []
        clean_list = []
        n_frames_list = []
        names = []

        for noisy, clean, n_frames, name in batch:
            noisy_list.append(torch.tensor(noisy).permute(1, 0))  # [F, T] => [T, F]
            clean_list.append(torch.tensor(clean).permute(1, 0))  # [1, T] => [T, 1]
            n_frames_list.append(n_frames)
            names.append(name)

        # seq_list = [(T1, F), (T2, F), ...]
        #   item.size() must be (T, *)
        #   return (longest_T, len(seq_list), *)
        noisy_list = pad_sequence(noisy_list).permute(1, 2, 0)  # ([T1, F], [T2, F], ...) => [T, B, F] => [B, F, T]
        clean_list = pad_sequence(clean_list).permute(1, 2, 0)  # ([T1, 1], [T2, 1], ...) => [T, B, 1] => [B, 1, T]

        return noisy_list, clean_list, n_frames_list, names

    train_dataloader = DataLoader(
        dataset=initialize_config(config["train_dataset"]),
        batch_size=config["train_dataloader"]["batch_size"],
        num_workers=config["train_dataloader"]["num_workers"],
        shuffle=config["train_dataloader"]["shuffle"],
        pin_memory=config["train_dataloader"]["pin_memory"],
        collate_fn=collate_fn_pad
    )

    valid_dataloader = DataLoader(
        dataset=initialize_config(config["validation_dataset"]),
        num_workers=0,
        batch_size=1
    )

    model = initialize_config(config["model"])

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"])
    )

    loss_function = initialize_config(config["loss_function"])
    trainer_class = initialize_config(config["trainer"], pass_args=False)

    trainer = trainer_class(
        config=config,
        resume=resume,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=valid_dataloader
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CRN")
    parser.add_argument("-C", "--configuration", required=True, type=str, help="Configuration (*.json).")
    parser.add_argument("-P", "--preloaded_model_path", type=str, help="Path of the *.Pth file of the model.")
    parser.add_argument("-R", "--resume", action="store_true", help="Resume experiment from latest checkpoint.")
    args = parser.parse_args()

    if args.preloaded_model_path:
        assert not args.resume, "Resume conflict with preloaded model. Please use one of them."

    configuration = json5.load(open(args.configuration))
    configuration["experiment_name"], _ = os.path.splitext(os.path.basename(args.configuration))
    configuration["config_path"] = args.configuration
    configuration["preloaded_model_path"] = args.preloaded_model_path

    main(configuration, resume=args.resume)
