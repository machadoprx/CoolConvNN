# CoolConvNN
Complete, simple and cool convolutional neural network, build from scratch, parallel(CPU only) and almost dependency free(OpenMP). Support custom layers, restricted to ReLU activation and [CONV-POOL-ACTIV]xM [FC]xN architecture.

## Introduction

This software was coded with only learning purposes, not for production uses. The aim of use is myself, optimized for run in a intel CPU.
Apart of that, it is fully functioning, the FC layers uses Batch normalization and ReLU, Conv Layers bias(BN todo) and ReLU and MaxPool layers. 

It uses the standard C++ library and LodePNG (https://github.com/lvandeve/lodepng) for png to float arrays conversion.

External data is supported, following the format:

Number of labels(int), Samples per labels(int), Dimension of features(int)

Data mean(sizeof(float) * FeaturesDim)

Data deviation(sizeof(float) * FeaturesDim)

Actual data(sizeof(float) * labels * samplesPerLables * FeaturesDim)

The data samples must be ordered by labels and their actual names must be kept aside (labels number are generated automatically)

### Installing

Clone the repository and run make in the folder. (gcc)


