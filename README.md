# CoolConvNN
Complete, simple and cool convolutional neural network framework, built from scratch, parallel able with OpenMP and almost dependency free. Supports custom architectures.

## Introduction

This project aims to be a dependency free implementation, in C, of a accessible deep learning framework.

Current features, activations, optimizer and layers:

* Convolutional layer
* MaxPool layer
* Fully Connected layer
* Batch normalization as default in Convolutional and fully-connected layers.
* ReLU activation
* Mini-batch stochastic gradient descent

To do:
* Deconvolutional layer
* Recurrent layer
* Dropout
* TanH activation
* Leaky ReLU activation
* Adam optimizer

It uses LodePNG (https://github.com/lvandeve/lodepng) for png to float arrays conversion.

External data is supported, following the csv format:

```
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

features_separated_by_comma,label

...
sample_n,label
```

Save the file as data.csv, in working directory

## Installing

You may need to specify the header of cblas_sgemm function in matrix.h

After that, just clone the repository and run make in the cloned folder

## How to use

params.ini Define the cnn architecture:

```
8
conv 1 32 1 5 2 64 64 relu
max_pool 2 2 32 0 64 64
conv 32 32 1 3 1 32 32 relu
max_pool 2 2 32 0 32 32
conv 32 32 1 3 1 16 16 relu
max_pool 2 2 32 0 16 16
fc 2048 512 relu
fc 512 2 none
32
```

```
[total number of layers]
[type(conv)] [input depth number_of_filters stride filter_size padding input_width input_height] [activation]
[type(max_pool)] [stride filter_size, depth, padding, input_width, input_height]
...
[type(fc)] [input_dim layer_size] [activation]
...
[batch size]
```

Start fitting the data:

```
./cnn_c new [validation_set_split] [learning_rate] [l2_reg_lambda] [epochs] 
```

Continue fitting :

```
./cnn_c continue [validation_set_split] [learning_rate] [l2_reg_lambda] [epochs]
```

Test sample:
```
./cnn_c test [sample_path]
```



