import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class STFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=512):
        super(STFT, self).__init__()

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.forward_transform = None
        # scale = self.filter_length / self.hop_length
        # scale = 1
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        window = np.sqrt(np.hamming(filter_length)).astype(np.float32)
        fourier_basis = fourier_basis * window
        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])

        inv_fourier_basis = np.fft.fft(np.eye(self.filter_length))
        inv_fourier_basis = inv_fourier_basis / window
        cutoff = int((self.filter_length / 2 + 1))
        inv_fourier_basis = np.vstack([np.real(inv_fourier_basis[:cutoff, :]),
                                       np.imag(inv_fourier_basis[:cutoff, :])])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(inv_fourier_basis).T[:, None, :])

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        ##
        #
        input_data = input_data.view(num_batches, 1, num_samples)
        forward_transform = F.conv1d(input_data,
                                     Variable(self.forward_basis, requires_grad=False),
                                     stride=self.hop_length,
                                     padding=0)
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        # magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        # phase = torch.autograd.Variable(torch.atan2(imag_part.load_data, real_part.load_data))
        res = torch.stack([real_part, imag_part], dim=-1)
        res = res.permute([0, 2, 1, 3])
        return res

    # (B,T,F,2)
    def inverse(self, stft_res):
        real_part = stft_res[:, :, :, 0].permute(0, 2, 1)
        imag_part = stft_res[:, :, :, 1].permute(0, 2, 1)
        recombine_magnitude_phase = torch.cat([real_part, imag_part], dim=1)
        # recombine_magnitude_phase = stft_res.permute([0, 2, 3, 1]).contiguous().view([1, -1, stft_res.size()[]])

        inverse_transform = F.conv_transpose1d(recombine_magnitude_phase,
                                               Variable(self.inverse_basis, requires_grad=False),
                                               stride=self.hop_length,
                                               padding=0)
        return inverse_transform[:, 0, :]

    def forward(self, input_data):
        stft_res = self.transform(input_data)
        reconstruction = self.inverse(stft_res)
        return reconstruction