import os

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np

import torch
import torchaudio


if __name__ == "__main__":
    root = "FOLDER_PATH_TO_DATA_GOES_HERE"

    stem_dir = "01 the 1"
    accom_samples, accom_sr = torchaudio.load(
        os.path.join(root, stem_dir, "accompaniment.wav"))

    # Trim samples to specified length
    sample_seconds = 10
    n_fft = 1024
    n_mels = 128
    accom_length = int(sample_seconds * accom_sr)
    accom_segment = accom_samples[:, :accom_length]
    
    spec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=44100,
                                                          n_fft=n_fft,
                                                          n_mels=n_mels,
                                                          normalized=True)
    # spec_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft,
    #                                                    normalized=False,
    #                                                    power=None,
    #                                                    return_complex=True)
    gen_spectrogram = spec_transform(accom_segment)

    print(gen_spectrogram.shape, accom_length, gen_spectrogram.dtype)
    print(gen_spectrogram.min(), gen_spectrogram.max())

    gen_spectrogram = gen_spectrogram.mean(dim=0)
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1)
    plt.imshow(gen_spectrogram.numpy())
    plt.savefig('output.png')
    plt.close()

    # imel_transform = torchaudio.transforms.InverseMelScale(sample_rate=44100,
    #                                                        n_stft=((n_fft // 2) + 1),
    #                                                        n_mels=n_mels,
    #                                                        max_iter=10000)
    # gen_spectrogram = imel_transform(gen_spectrogram)

    # spectrogram = torch.randn(gen_spectrogram.shape, dtype=torch.cfloat)
    # spectrogram.real = gen_spectrogram
    # spectrogram.imag = gen_spectrogram

    # ispec_transform = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft)
    # waveform = ispec_transform(spectrogram, accom_length)

    # torchaudio.save('test.wav', waveform, accom_sr)