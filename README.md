# waveletCodingCNN
master research archive
_______

Hello. Here is He.

The work is about predictive wavelet based image compression inspired by many of my Laboratory researches, like [Mr. takezawa](https://github.com/bamboosteam/n1cnn)


My workflow is

1. Use python to convert downloaded images to P2-PGM which is required format as input of our coding C program.
2. Run a python script to batchly run C program, decompose all images for train/validation
3. Extract training data from decomposed 5level DWT by python.
4. Train the CNN model.
5. Implement the model to C program.
- Noting that C file in this repository only shows implementation idea of my part



