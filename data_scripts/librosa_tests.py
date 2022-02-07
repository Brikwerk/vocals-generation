import os

import torch
import librosa
from librosa.feature import melspectrogram
import librosa.display
from matplotlib import pyplot as plt
import numpy as np
import soundfile as sf
from skimage import io


if __name__ == "__main__":
    root = "FOLDER_PATH_TO_DATA_GOES_HERE"
    stem_dir = "SONG_NAME"
    hop_length = 1024
    n_mels = 216

    y, sr = librosa.load(
        os.path.join(root, stem_dir, "accompaniment.wav"))
    y = y[:(sr * 10)]
    
    spec = melspectrogram(y, sr=sr, hop_length=hop_length, n_mels=n_mels)

    S_dB = librosa.power_to_db(spec, ref=np.max)
    # Normalize
    S_dB = (S_dB / 80) + 1

    print(S_dB.shape, S_dB.min(), S_dB.max())

    S_dB = torch.from_numpy(S_dB)
    torch.save(S_dB, 'spec.pt')

    sr = 22100
    S_dB = torch.load("spec.pt")
    S_dB = S_dB.numpy()
    io.imsave('output1.png', S_dB)

    S_dB = (S_dB - 1) * 80

    # Convert S_dB back to power
    S = librosa.db_to_power(S_dB)

    # Convert mel-spectrogram to audio
    y = librosa.feature.inverse.mel_to_audio(S, sr=sr,
        hop_length=hop_length)

    # Save audio
    sf.write('output.wav', y, sr)
