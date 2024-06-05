TO compile this program:

nvcc -I/home/jovyan/cuda_sdk_samples/common/inc -o pa2 PA2.cu


To run:

Example ----- ./PA2 "lena/lena.pgm" "lenaout/lenaout.pgm" 16 4 4 "G"
                    
args order -- ./PA2, inputFilename, outputFilename,
                                   blockSize, edgeSize, strideSize, memType (G, S, T)

