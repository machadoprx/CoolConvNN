# CoolConvNN
Complete, simple and cool convolutional neural network, build from scratch, parallel(CPU only) and almost dependency free(OpenMP). Support custom layers, restricted to ReLU activation and [CONV-POOL-ACTIV]xM [FC]xN architecture.

## Introduction

This software was coded with only learning purposes, not for production uses. The aim of use is myself, optimized for run in a intel CPU.
Apart of that, it is fully functioning, the FC layers uses Batch normalization and ReLU, Conv Layers bias(BN todo) and ReLU and MaxPool layers. Learning is done using mini-batch sgd.
Accuracy and validation set, todo

It uses LodePNG (https://github.com/lvandeve/lodepng) for png to float arrays conversion.

External data is supported, following the format:

Number of labels(int), Samples per labels(int), Dimension of features(int)

Data mean(sizeof(float) * FeaturesDim)

Data deviation(sizeof(float) * FeaturesDim)

Actual data(sizeof(float) * labels * samplesPerLables * FeaturesDim)

The data samples must be ordered by labels and their actual names must be kept aside (labels number are generated automatically)

### Installing

Clone the repository and run make in the folder. (gcc)

#### How to use

Process data (save to binary and normalyze) with labels separated by folder (must be  .png and width == height):

```
./cnn_cpp getdata [samplesPerlabel] [samplesSize] [numberOfLabels] [sourceFolder] 
```

params.ini Define the cnn architecture:

```
2 // [conv+relu][pool] layers
1 32 1 3 1 28 28 // conv1: input depth, number of filters, stride, filterSize, padding, input width, input height
2 2 32 0 28 28 // pool1: stride, filterSize, depth, padding, input width, input height
32 64 1 3 1 14 14 // conv2
2 2 64 0 14 14 // pool2
0 // additional fully connected layers
3136 16 512 // features dimension, number of labels, size of fc
256 0.05 // batch size and learning rate
```

Start learning:

```
./cnn_cpp new [epochs]
```

Continue learning (you may set the learning rate in params.ini):

```
./cnn_cpp continue [epochs]
```

Test sample:
```
./cnn_cpp test [sample_path]
```



