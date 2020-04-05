all:
	gcc -std=c17 -lopenblas -fopenmp -liomp5 -lpthread -Ofast main.c cnn.c matrix.c neural_net.c fc_layer.c conv_layer.c pool_layer.c image.c png2array/png2array.c png2array/lodepng.c utils.c -Wall -pedantic -lm -o cnn_c
