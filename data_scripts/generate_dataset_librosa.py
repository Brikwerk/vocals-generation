import os

import torch
import librosa
from librosa.feature import melspectrogram
import librosa.display
import numpy as np


if __name__ == "__main__":
    root = "FOLDER_PATH_TO_DATA_GOES_HERE"
    output = "FOLDER_PATH_TO_DATA_GOES_HERE"

    segment_length = 10 # Seconds
    hop_length = 1024
    n_mels = 216

    # Get all folders in the root diretory
    stem_dirs = os.listdir(root)

    # Iterate over all stem dirs, load the accompaniment/vocals,
    # split the samples into segments, generate spectrograms from
    # the segemtns, and concatenate accompaniment/vocal spectrograms
    # into a single tensor.
    for i, stem_dir in enumerate(stem_dirs):
        print(f"({i}/{len(stem_dirs)}) Working on {stem_dir}")

        # Load the vocals and accompaniment
        vocals_samples, vocals_sr = librosa.load(
            os.path.join(root, stem_dir, "vocals.wav"))
        accom_samples, accom_sr = librosa.load(
            os.path.join(root, stem_dir, "accompaniment.wav"))

        vocals_length = int(segment_length * vocals_sr)
        accom_length = int(segment_length * accom_sr)

        # Iterate over all segments of the vocals and accompaniment
        for i in range(0, vocals_samples.shape[0] - vocals_length, vocals_length):
            vocals_segment = vocals_samples[i:i+vocals_length]
            accom_segment = accom_samples[i:i+accom_length]

            # Generate spectrograms
            vocals_spec = melspectrogram(vocals_segment, vocals_sr,
                                         hop_length=hop_length, n_mels=n_mels)
            accom_spec = melspectrogram(accom_segment, accom_sr,
                                        hop_length=hop_length, n_mels=n_mels)

            # Power-to-dB conversion
            vocals_spec = librosa.power_to_db(vocals_spec)
            accom_spec = librosa.power_to_db(accom_spec)

            # Normalize spectrograms
            vocals_spec = (vocals_spec / 80) + 1
            accom_spec = (accom_spec / 80) + 1

            # Convert to tensors
            vocals_spec = torch.from_numpy(vocals_spec).unsqueeze(0)
            accom_spec = torch.from_numpy(accom_spec).unsqueeze(0)

            # Concatenate spectrograms
            spec = torch.cat((vocals_spec, accom_spec), dim=0)

            # Save spectrograms
            torch.save(spec, os.path.join(output, f"{stem_dir}_{i}.pt"))

    print("Done!")
