# vocals-generation

## COSC490 Group 2 Term Project

## Group Members

- Corey Bond
- Paula Wong-Chung
- Reece Walsh

## Topic

### Generation of vocals for accompanying music

## General Outline

We were interested in researching methods/accuracy of generating vocals based on audio clips pulled from popular music, and determining if the vocal clips could be interpreted as accurate by the human ear.

The paper [Generation of lyrics lines conditioned on music audio clips](https://aclanthology.org/2020.nlp4musa-1.7.pdf) provided the motivation to investigate applications to other musical artists, as well as extending the results to generate vocal tracks. Upon initial trials, it was immediately evident that this would not be accurate enough for our needs, due to the noise in the generated spectrogram. An [AutoVC](http://proceedings.mlr.press/v97/qian19c/qian19c.pdf) and a [VAE/GAN](https://arxiv.org/pdf/1512.09300.pdf) approach were investigated as possible solutions to this accuracy issue. The AutoVC approach was significantly more effective.  

We identified the musical catalogues of Elton John and Taylor Swift as study subjects for the project. They each have more than 150 songs available, which allowed us to create a reasonable private data set based off of 10 second audio clips from their music. By having two artists form the basis for the data, we hoped to have enough consistency in the data to achieve interpretable results that could be identified as the respective artist. Songs and clips that are predominantly instrumental were removed from the dataset, as the focus is on reproducing vocals.  

We used the publicly available [spleeter](https://github.com/deezer/spleeter) tool that allows for us to split vocals out from accompanying (instrumental) audio, resulting in instrumental input files, and associated vocal output files. We then transformed those audio files into MEL spectrograms. Using this method, we were able to build a training and testing set of data. This process can also be used to generate the required splits should we switch to a publicly available dataset.  

Once the data was isolated, we constructed training scenarios based on the three papers linked above. This will allow us to generate [MEL spectrograms](https://towardsdatascience.com/learning-from-audio-the-mel-scale-mel-spectrograms-and-mel-frequency-cepstral-coefficients-f5752b6324a8) which can be compared for accuracy, and converted to audio files for our listening pleasure.

Future work: This is just a starting point. Now that the autoencoder approach has been determined to be effective, further work will be completed to improve the model.  

## Model Locations

```
    .
    ├── autovc_mod                          # Modified AutoVC architecture
    │   ├── accom_synthesis.ipynb           # Accompaniment synthesis notebook
    │   ├── autovc.md                       # README for the original AutoVC code
    │   ├── compile_dataset.py              # Compile dataset into .npy files
    │   ├── conditioned_main.py             # Main script for conditional vocal generation
    │   ├── conditioned_main_v3.py          # Main script for conditional vocal generation, v3 of the network
    │   ├── conversion.ipynb                # Convert .npy to .wav with the model
    │   ├── data_loader.py                  # Loads the dataset into a data loader
    │   ├── hparams.py                      # Hyperparameters for the model
    │   ├── LICENSE                         # MIT License
    │   ├── losses.py                       # Loss calculations for the VAE component
    │   ├── main.py                         # Original AutoVC main script
    │   ├── make_metadata.py                # Generate metadata for the dataset
    │   ├── make_spect.py                   # Generate spectrograms for the dataset
    │   ├── model_bl.py                     # Baseline model
    │   ├── model_vc.py                     # Vocal conditioning model
    │   ├── solver_encoder.py               # Solver for the VAE encoder
    │   ├── synthesis.py                    # Synthesis script
    │   ├── vocals_synthesis.ipynb          # Vocal synthesis notebook
    │   ├── vocals_synthesis_v2.ipynb       # Vocal synthesis notebook, v2 of the network
    │   ├── vocals_synthesis_v3.ipynb       # Vocal synthesis notebook, v3 of the network
    │   ├── vocoder.ipynb                   # Original AutoVC vocoder synthesis
    ├── data_scripts                        # Test scripts for generating the dataset
    │   ├── example_load_store.py           # Testing generating/storing spectrograms 
    │   ├── generate_dataset_librosa.py     # Dataset generation using librosa
    │   ├── generate_dataset_torchaudio.py  # Dataset generation using torchaudio
    │   ├── librosa_tests.py                # Testing the librosa package
    ├── vae                                 # VAE files
    │   ├── outputs                         # Files created after running inference will be found here
    │   ├── src                             # VAE source files
    │   |    ├── dataset.py                 # Loads the dataset as spectrograms in tensor form
    │   |    ├── losses.py                  # Loss calculations for the VAE
    │   |    ├── utils.py                   # For tensor to spectrogram conversion
    │   |    ├── vae_models.py              # Contains two different VAE models: LitVAE and SpecVAE
    │   ├── vae_inference.py                # Code for running inference using the VAE
    │   ├── vae_train.py                    # Trainer for the VAE
    └── README.md                           # This file!
```

