# CoolConvNN
Complete, simple and cool convolutional neural network, build from scratch, parallel(CPU only) and almost dependency free(OpenMP). Support custom layers, restricted to ReLU activation.

## Introduction

This software was coded with only learning purposes, not for production uses. The aim of use is myself, optimized for run in a intel CPU.
Apart of that, it is fully functioning, the FC layers uses Batch normalization and ReLU, Conv Layers batch normalization and ReLU and MaxPool layers. Learning is done using mini-batch sgd.
Accuracy and validation set, todo

It uses LodePNG (https://github.com/lvandeve/lodepng) for png to float arrays conversion.

External data is supported, following the csv format:

835,4096,2,masked,unmasked
87.17098,84.20363,81.519516, ... ,79.42487,77.518654,76.04577,74.97219,74.087395,73.2
72.778564,71.41166,69.91917, ... ,68.847664,67.7942,67.16,66.50335,65.87927,65.37431
-0.33487654,-0.2857717,-0.38590893, ... ,-0.4879623,-0.56448114,-0.4101708,-0.3997688,0
-0.8728179,-0.9562261,-1.0026246, ... ,-1.0423806,-1.0312552,-1.0120616,-0.9987559,1
2.4470487,2.5051897,2.665211, ... ,2.7244022,2.8196306,2.8667905,2.9545586,2.404601,0
...

number_of_samples,features_size,number_of_labels,[labels separated by comma]
mean_of_features_separated_by_comma
std_of_features_separated_by_comma
sample_1_features_separated_by_comma,label
...
sample_n,label

### Installing

Clone the repository and run make in the folder. (gcc)

#### How to use

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
./cnn_c new [epochs]
```

Continue learning (you may set the learning rate in params.ini):

```
./cnn_c continue [epochs]
```

Test sample:
```
./cnn_c test [sample_path]
```



