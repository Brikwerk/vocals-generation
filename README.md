# COSC490_Group_2_Term_Project

## Group Members

- Corey Bond
- Paula Wong-Chung
- Reece Walsh

## Topic

### Investigate the possibility of using a Variational Autoencoder with a Convolutional Neural Network (VAE CNN) to generate vocals and lyrics based on music audio clips

## General Outline

We are interested in researching methods/accuracy of generating vocals based on audio clips pulled from popular music, and determining if the vocal clips can be interpreted as accurate by the human ear.

The paper [Generation of lyrics lines conditioned on music audio clips](https://aclanthology.org/2020.nlp4musa-1.7.pdf) provided the motivation to investigate applications to other musical arrtists, as well as extending the results to generate vocal tracks.

We identified the musical catalogues of Elton John and Taylor Swift as study subjects. They each have more than 150 songs available, which will allow for us to create a reasonable data set based off of audio clips from their music.

We will use the publicly available [spleeter](https://github.com/deezer/spleeter) tool that allows for us to split vocals out from accompanying (instrumental) audio, resulting in instrumental input files, and associated vocal output files. We can then transform those audio files into MEL spectrograms. Using this method, we are able to build a training and testing set of data.

Once the data is isolated, we will construct a dual VAE CNN training scenario based on the paper linked above. This will allow us to generate spectrograms which can be compared for accuracy, and converted to audio files for our listening pleasure.

## Expected Outcomes

- Generation of a private database composed of separated instrumental (input) and vocal (output) audio files.
- Generation of a database composed of separated instrumental (input) and vocal (output) spectrogram files.
- Trained and tested dual VAE RNN model.
- Generated spectrograms to compare to existing audio tracks.
- (*Optional - time dependent*) Vocals generated for a **new** song, perhaps overlaid over a similar music video, that can be enjoyed by the class.

## Tentative Schedule

| Milestone                | Date          |
| ------------------------ | ------------- |
| Outline Complete         | Feb 18, 2022  |
| Music Database Generated | Feb 18, 2022  |
| Create Train/Test Data   |  |
| Construct Model          |  |
| Report Draft #1          |  |
| Implementation Complete  |  |
| Report Draft # 2         |  |
| Presentation             | Apr 4-8, 2022 |
| Final Report Complete    | Apr 11, 2022? |