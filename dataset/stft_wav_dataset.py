import os

import librosa
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset


class WavDataset(Dataset):
    """
    Define train dataset
    """

    def __init__(self,
                 mixture_dataset,
                 clean_dataset,
                 limit=None,
                 offset=0,
                 ):
        """
        Construct train dataset
        Args:
            mixture_dataset (str): mixture dir (wav format files)
            clean_dataset (str): clean dir (wav format files)
            limit (int): the limit of the dataset
            offset (int): the offset of the dataset
        """
        assert os.path.exists(mixture_dataset) and os.path.exists(clean_dataset)

        print("Search datasets...")
        mixture_wav_files = librosa.util.find_files(mixture_dataset, ext="wav", limit=limit, offset=offset)
        clean_wav_files = librosa.util.find_files(clean_dataset, ext="wav", limit=limit, offset=offset)

        assert len(mixture_wav_files) == len(clean_wav_files)
        print(f"\t Original length: {len(mixture_wav_files)}")

        self.length = len(mixture_wav_files)
        self.mixture_wav_files = mixture_wav_files
        self.clean_wav_files = clean_wav_files

        print(f"\t Offset: {offset}")
        print(f"\t Limit: {limit}")
        print(f"\t Final length: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        mixture_path = self.mixture_wav_files[item]
        clean_path = self.clean_wav_files[item]
        name = os.path.splitext(os.path.basename(clean_path))[0]

        mixture, sr = sf.read(mixture_path, dtype="float32")
        clean, sr = sf.read(clean_path, dtype="float32")
        assert sr == 16000
        assert mixture.shape == clean.shape

        n_frames = (len(mixture) - 320) // 160 + 1

        return mixture, clean, n_frames, name
