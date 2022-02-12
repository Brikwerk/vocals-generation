# COSC490_Group_2_Term_Project

## Group Members

- Corey Bond
- Paula Wong-Chung
- Reece Walsh

## Topic

### Investigating the effectiveness of different neural network architecture to generate vocals and lyrics based on music audio clips

## General Outline

We are interested in researching methods/accuracy of generating vocals based on audio clips pulled from popular music, and determining if the vocal clips can be interpreted as accurate by the human ear.

The paper [Generation of lyrics lines conditioned on music audio clips](https://aclanthology.org/2020.nlp4musa-1.7.pdf) provided the motivation to investigate applications to other musical arrtists, as well as extending the results to generate vocal tracks. Upon initial trials, it was immediately evident that this would not be accurate enough for our needs, so an [AutoVC](http://proceedings.mlr.press/v97/qian19c/qian19c.pdf) and a [VAE/GAN](https://arxiv.org/pdf/1512.09300.pdf) approach will also be investigated.  

We identified the musical catalogues of Elton John and Taylor Swift as study subjects. They each have more than 150 songs available, which will allow for us to create a reasonable data set based off of 10 second audio clips from their music. Songs and clips that are predominantly instrumental will be removed from the dataset.  

We will use the publicly available [spleeter](https://github.com/deezer/spleeter) tool that allows for us to split vocals out from accompanying (instrumental) audio, resulting in instrumental input files, and associated vocal output files. We can then transform those audio files into MEL spectrograms. Using this method, we are able to build a training and testing set of data.

Once the data is isolated, we will construct training scenarios based on the three papers linked above. This will allow us to generate [MEL spectrograms](https://towardsdatascience.com/learning-from-audio-the-mel-scale-mel-spectrograms-and-mel-frequency-cepstral-coefficients-f5752b6324a8) which can be compared for accuracy, and converted to audio files for our listening pleasure.

## Expected Outcomes

- Generation of a private database composed of separated instrumental (input) and vocal (output) 10 second audio files.
- Generation of a database composed of separated instrumental (input) and vocal (output) MEL spectrogram files.
- Trained and tested dual VAE RNN, AutoVC, and VAE/GAN model for comparison of effectiveness/accuracy.
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
