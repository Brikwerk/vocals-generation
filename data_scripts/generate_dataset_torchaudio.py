import os

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T


if __name__ == "__main__":
    root = "FOLDER_PATH_TO_DATA_GOES_HERE"
    output = "FOLDER_PATH_TO_DATA_GOES_HERE"

    segment_length = 10 # Seconds
    n_fft = 566 # How many fft bins
    spec_transform = T.Spectrogram(n_fft=n_fft,
                                   normalized=False,
                                   power=None,
                                   return_complex=True)

    # Get all folders in the root diretory
    stem_dirs = os.listdir(root)

    # Iterate over all stem dirs, load the accompaniment/vocals,
    # split the samples into segments, generate spectrograms from
    # the segemtns, and concatenate accompaniment/vocal spectrograms
    # into a single tensor.
    dataset = None
    for i, stem_dir in enumerate(stem_dirs):
        print(f"({i}/{len(stem_dirs)}) Working on {stem_dir}")

        # Load the vocals and accompaniment
        vocals_samples, vocals_sr = torchaudio.load(
            os.path.join(root, stem_dir, "vocals.wav"), normalize=True)
        accom_samples, accom_sr = torchaudio.load(
            os.path.join(root, stem_dir, "accompaniment.wav"), normalize=True)

        vocals_length = int(segment_length * vocals_sr)
        accom_length = int(segment_length * accom_sr)

        # Iterate over all segments of the vocals and accompaniment
        for i in range(0, vocals_samples.shape[1] - vocals_length, vocals_length):
            vocals_segment = vocals_samples[:, i:i+vocals_length]
            accom_segment = accom_samples[:, i:i+accom_length]

            # Generate spectrograms
            vocals_spec = spec_transform(vocals_segment).unsqueeze(0)
            accom_spec = spec_transform(accom_segment).unsqueeze(0)

            # Concatenate spectrograms
            spec = torch.cat((vocals_spec, accom_spec), dim=0)

            # Save spectrograms
            torch.save(spec, os.path.join(output, f"{stem_dir}_{i}.pt"))

    print("Done!")
