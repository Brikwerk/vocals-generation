import torch
import torchaudio


def spec_tensor_to_spec(tensor, sr=8000, n_fft=566, audio_length=10):
    """
    Convert a 4-channel tensor to a spectrogram
    
    Tensors should be of shape (4, n_fft, X)
    Where X is the number of spectrogram frames.

    Args:
        tensor: 4-channel tensor
        sr: sample rate
        n_fft: number of fft bins
        audio_length: length of audio in seconds
    """
    spec = torch.randn(2, tensor.shape[1], tensor.shape[2],
                       dtype=torch.cfloat)
    spec.real = tensor[:2, :, :]
    spec.imag = tensor[:2, :, :]

    ispec_transform = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft)
    waveform = ispec_transform(spec, sr * audio_length)

    return waveform


def save_spec_tensor(tensor, filename, 
                     sr=8000, n_fft=566, audio_length=10):
    """
    Save a 4-channel tensor as a spectrogram
    
    Tensors should be of shape (4, n_fft, X)
    Where X is the number of spectrogram frames.

    Args:
        tensor: 4-channel tensor
        filename: filename to save to
        sr: sample rate
        n_fft: number of fft bins
        audio_length: length of audio in seconds
    """
    waveform = spec_tensor_to_spec(tensor, sr, n_fft, audio_length)

    torchaudio.save(filename, waveform, sr)
