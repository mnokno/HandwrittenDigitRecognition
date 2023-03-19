# Handwritten Digit Recognition using Convolutional Neural Networks

This project is about recognizing handwritten digits using custom architecture of Convolutional Neural Networks (CNN). The CNNs have been trained on a dataset of 1.5 million images, resulting in an impressive accuracy of 99.625% on Kaggle.

## Dataset

The dataset used for training consists of images of handwritten digits, with each image being a 28x28 grayscale image. The dataset was sourced from Kaggle, and can be found [here](https://www.kaggle.com/c/digit-recognizer/data).

## Architecture

The custom architecture of CNNs used in this project has been designed specifically for recognizing handwritten digits. It consists of several layers, including convolutional layers, max pooling layers, and fully connected layers. The details of the architecture can be found in the code.

## Training

Training the CNNs requires a significant amount of resources, as the version trained on 1.5 million images required 32GB of RAM. However, a [lighter version](https://www.kaggle.com/code/mnokno/pytorch-cnn-data-argumentation-99-6) trained on 378K images can be trained on Kaggle, which reduces the resource requirements. The training process can take several hours to complete for the 1.5 million image verion, depending on the resources available.

## Results

The CNNs achieved a remarkable accuracy of 99.625% on the test set, demonstrating the effectiveness of the custom architecture for recognizing handwritten digits.

## Conclusion

The project demonstrates the effectiveness of custom architecture of CNNs for recognizing handwritten digits. The trained model can be used to recognize digits drawn by users, and can be further improved by training on additional data.
