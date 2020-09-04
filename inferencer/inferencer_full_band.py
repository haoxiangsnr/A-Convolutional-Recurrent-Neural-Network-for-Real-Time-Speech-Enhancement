import librosa
import torch


def full_band_no_truncation(model, device, inference_args, noisy):
    """
    extract full_band spectra for inference, without truncation.
    """
    n_fft = inference_args["n_fft"]
    hop_length = inference_args["hop_length"]
    win_length = inference_args["win_length"]

    noisy_mag, noisy_phase = librosa.magphase(librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    noisy_mag = torch.tensor(noisy_mag, device=device)[None, None, :, :]  # [F, T] => [1, 1, F, T]
    enhanced_mag = model(noisy_mag)  # [1, 1, F, T] => [1, 1, F, T]
    enhanced_mag = enhanced_mag.squeeze(0).squeeze(0).detach().cpu().numpy()  # [1, 1, F, T] => [F, T]

    enhanced = librosa.istft(enhanced_mag * noisy_phase, hop_length=hop_length, win_length=win_length, length=len(noisy))

    assert len(noisy) == len(enhanced)

    return noisy, enhanced
