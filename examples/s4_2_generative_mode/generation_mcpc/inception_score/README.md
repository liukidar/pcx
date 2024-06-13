# IS_MS_SS

In GANs, the objective function for the generator and the discriminator usually measures how well they are doing relative to the opponent. For example, we measure how well the generator is fooling the discriminator. It is not a good metric in measuring the image quality or its diversity. As part of the GAN series, we look into the Inception Score and Fréchet Inception Distance on how to compare results from different GAN models.

## Inception Score (IS)

IS uses two criteria in measuring the performance of GAN:
* The quality of the generated images, and
* their diversity.

## Incetipn Score

To calculate the inception score for cifar10 or the dataset in the imageNet dataset, the pre-trained model is Inception Netowrk.

## MNSIT Score

To calculate the inception score for MNIST data, the pre-trained model is Resnet18 that trained on MNSIT dataset.


## SVHN Score

To calculate the inception score for SVHN data, the pre-trained model is Resnet18 that trained on SVHN dataset.

## Usage
* Put the generated images in the data folder(the default grid size is 10x10)
* Execute the 'run.sh' script.


Any questions and suggestions are welcomed~~
