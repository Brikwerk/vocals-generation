# COSC490_Group_2_Term_Project

## Group Members

- Corey Bond
- Paula Wong-Chung
- Reece Walsh

## Topic

### Investigating the effectiveness of different neural network architecture to generate vocals based on music audio clips

## General Outline

We are interested in researching methods/accuracy of generating vocals based on audio clips pulled from popular music, and determining if the vocal clips can be interpreted as accurate by the human ear.

The paper [Generation of lyrics lines conditioned on music audio clips](https://aclanthology.org/2020.nlp4musa-1.7.pdf) provided the motivation to investigate applications to other musical artists, as well as extending the results to generate vocal tracks. Upon initial trials, it was immediately evident that this would not be accurate enough for our needs, due to the noise in the generated spectrogram. An [AutoVC](http://proceedings.mlr.press/v97/qian19c/qian19c.pdf) and a [VAE/GAN](https://arxiv.org/pdf/1512.09300.pdf) approach will be investigated as possible solutions to this accuracy issue.  

We identified the musical catalogues of Elton John and Taylor Swift as study subjects for the project. They each have more than 150 songs available, which will allow for us to create a reasonable private data set based off of 10 second audio clips from their music. By having two artists form the basis for the data, we hope to have enough consistency in the data to achieve interpretable results than can be identified as the respective artist. Songs and clips that are predominantly instrumental will be removed from the dataset, as the focus is on reproducing vocals.  

We will use the publicly available [spleeter](https://github.com/deezer/spleeter) tool that allows for us to split vocals out from accompanying (instrumental) audio, resulting in instrumental input files, and associated vocal output files. We can then transform those audio files into MEL spectrograms. Using this method, we are able to build a training and testing set of data. This process can also be used to generate the required splits should we switch to a publicly available dataset.  

Once the data is isolated, we will construct training scenarios based on the three papers linked above. This will allow us to generate [MEL spectrograms](https://towardsdatascience.com/learning-from-audio-the-mel-scale-mel-spectrograms-and-mel-frequency-cepstral-coefficients-f5752b6324a8) which can be compared for accuracy, and converted to audio files for our listening pleasure.

Future work considerations: This is computation heavy and requires retraining multiple models for each dataset and method. We will have to consider this as part of the comparison between the methods, and may lead to future recommended structural improvements.  

## Expected Outcomes

- Generation of a private database composed of separated instrumental (input) and vocal (output) 10 second audio files. (Placeholder as proof of concept)
- Generation of a database composed of separated instrumental (input) and vocal (output) MEL spectrogram files (**Note**: Check with library to determine if copyright would apply to generated spectrograms).
- Trained and tested dual VAE RNN, AutoVC, and VAE/GAN model for comparison of effectiveness/accuracy.
- Generated spectrograms to compare to existing audio tracks.
- (*Optional - time dependent*) Vocals generated for a **new** song, perhaps overlaid over a similar music video, that can be enjoyed by the class for the purpose of a more engaging demonstration.

## Tentative Schedule

| Milestone                | Date          |
| ------------------------ | ------------- |
| Outline Complete         | Feb 18, 2022  |
| Music Database Generated | Feb 18, 2022  |
| Create Train/Test Data   | TBD |
| Construct Model          | TBD |
| Report Draft #1          | TBD |
| Implementation Complete  | TBD |
| Report Draft # 2         | TBD |
| Presentation             | Apr 4-8, 2022 |
| Final Report Complete    | Apr 11, 2022? |
