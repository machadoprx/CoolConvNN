all:
	gcc --std=c11 -DMKL_ILP64 -m64 -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl -fopenmp -Ofast main.c cnn.c matrix.c neural_net.c fc_layer.c conv_layer.c pool_layer.c image.c png2array/png2array.c png2array/lodepng.c utils.c -Wall -pedantic -lm -o cnn_c
