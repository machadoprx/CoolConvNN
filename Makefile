all:
	clang -std=c17 -lpthread -lm -ldl -fopenmp -O3 -flto main.c cool_nn.c matrix.c neural_net.c fc_layer.c conv_layer.c pool_layer.c image.c parse_data.c png2array/png2array.c png2array/lodepng.c utils.c activations_layer.c bn_layer.c -Wall -lm -o cnn_c
