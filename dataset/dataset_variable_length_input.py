import os

import librosa
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(
            self,
            dataset_list,
            limit,
            offset,
            sr,
            n_fft,
            hop_length,
            train
    ):
        """
        dataset_list(*.txt):
            <noisy_path> <clean_path>\n
        e.g:
            noisy_1.wav clean_1.wav
            noisy_2.wav clean_2.wav
            ...
            noisy_n.wav clean_n.wav
        """
        super(Dataset, self).__init__()
        self.sr = sr
        self.train = train

        dataset_list = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(dataset_list)), "r")]
        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]

        self.dataset_list = dataset_list
        self.length = len(self.dataset_list)
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        noisy_path, clean_path = self.dataset_list[item].split(" ")
        name = os.path.splitext(os.path.basename(noisy_path))[0]
        noisy, _ = librosa.load(os.path.abspath(os.path.expanduser(noisy_path)), sr=self.sr)
        clean, _ = librosa.load(os.path.abspath(os.path.expanduser(clean_path)), sr=self.sr)

        if self.train:
            noisy_mag, _ = librosa.magphase(librosa.stft(noisy, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft))
            clean_mag, _ = librosa.magphase(librosa.stft(clean, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft))
            return noisy_mag, clean_mag, noisy_mag.shape[-1], name
        else:
            return noisy, clean, name
