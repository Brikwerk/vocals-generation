import os

import librosa
import soundfile as sf
from torch.utils.data import Dataset
import torch
from matplotlib import pyplot as plt


class StemsDataset(Dataset):

    def __init__(self, data_root, transforms=None, data_ext='.pt') -> None:
        super().__init__()

        self.data_root = data_root

        # Get a list of all spectrograms in the data_root
        self.spectrograms = [
            os.path.join(data_root, f)
            for f in os.listdir(data_root)
            if os.path.isfile(os.path.join(data_root, f)) and f.endswith(data_ext)
        ]

        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.spectrograms)
    
    def __getitem__(self, idx) -> torch.Tensor:
        spectrograms = torch.load(self.spectrograms[idx])

        if self.transforms:
            spectrograms = self.transforms(spectrograms)
        
        # Concatenate real/imaginary numbers into channels
        vocals = spectrograms[0].unsqueeze(0)
        accompaniment = spectrograms[1].unsqueeze(0)
        
        spectrograms = {
            'vocals': vocals,
            'accompaniment': accompaniment
        }

        return spectrograms
    

if __name__ == "__main__":
    spec_dataset = StemsDataset(
        data_root=FOLDER_PATH_TO_DATA_GOES_HERE) #FOLDER_PATH_TO_DATA_GOES_HERE
    print(len(spec_dataset))

    specs = spec_dataset.__getitem__(10)
    v_shape = specs['vocals'].shape
    print(specs['vocals'].min(), specs['vocals'].max())
    print(specs['accompaniment'].min(), specs['accompaniment'].max(), specs['accompaniment'].shape)

    accom_spec = specs['vocals'][0].numpy()

    fig = plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    plt.imshow(accom_spec)
    plt.savefig('output_ds.png')
    plt.close()

    accom_spec = (accom_spec - 1) * 80
    sr = 22100
    hop_length = 1024
    n_mels = 216

    # Convert S_dB back to power
    S = librosa.db_to_power(accom_spec)

    # Convert mel-spectrogram to audio
    y = librosa.feature.inverse.mel_to_audio(S, sr=sr,
                                             hop_length=hop_length)

    # Save audio
    sf.write('output.wav', y, sr)