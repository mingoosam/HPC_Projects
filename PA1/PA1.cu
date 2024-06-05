
/*
*  File : PA1.cu
*  Author: Andy Varner
*  Created: 02.06.2024
*  References: simpleTexture.cu (from SDK CUDA Toolkit samples)

*  To compile this program --- nvcc -I/home/jovyan/cuda_sdk_samples/common/inc -o pa1 PA1.cu
*  On marge --- nvcc -I/agkgd4/Documents/hpc_class/cuda-samples/Common -o pa1 PA1.cu
*  To run -------------------- ./pa1 -i "lena/lena.pgm" -o "lena/lenaout.pgm" 16
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

const char *projectName = "PA1.cu";
const char *imageFilename = NULL;
const char *outputFilename = NULL;

int blockSize = 0;



__global__ void transformKernel(float *inputData, float *outputData, int width, int height, int newWidth, int newHeight, int kernelSize){
    
 
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    int start_input_x = x * 4;
    int start_input_y = y * 4;

    if (x < newWidth && y < newHeight){
        
        float kernel[16];

        // fill up kernel array
        for (int xx = 0; xx < kernelSize; xx++){
            for (int yy = 0; yy < kernelSize; yy++){
                //int idx = (start_input_x + xx) * width + start_input_y + yy;
                int idx = (start_input_x + xx) + (start_input_y + yy)* height;
                kernel[xx * kernelSize + yy] = inputData[idx];
            }
        }
        // sort
        float temp;
        for (int i = 0; i < 16; i++){
            for (int j = 0; j < 16 - i - 1; j++){
                if (kernel[j] > kernel[j+1]){
                    temp = kernel[j];
                    kernel[j] = kernel[j+1];
                    kernel[j+1] = temp;
                }
            }
        }
        // then find the median and assign to output 
        float median = (kernel[7] + kernel[8]) / 2;
        outputData[y*newHeight + x] = median;
    }//end if
}


void runTest(int argc, char **argv);

/*
 * Program Main
 */
int main(int argc, char **argv){

    printf("\n%s starting...\n", projectName);    

    if (argc < 6){
        printf("Command line arg error: Please suppy thte input file name with -i flag followed by the output file name"
               "with -o flag and the block-edge size as an int.\n");
        exit(EXIT_FAILURE);
    }
  
    imageFilename = argv[2];
    outputFilename = argv[4];
    blockSize = atoi(argv[5]);

    runTest(argc, argv);

    printf("\nFile execution complete\n");
}


void runTest(int argc, char **argv) {

    printf("\nimageFilename = %s\n\n", imageFilename);
    int devID = findCudaDevice(argc, (const char **)argv);
    
    //load image from disk
    
    float *hData = NULL;
    unsigned int width, height;
    
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);
    if (imagePath == NULL) {
        printf("Can't source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }
    
    if (sdkLoadPGM(imagePath, &hData, &width, &height)){
        printf("\nLoaded '%s', %d x %d pixels\n\n", imageFilename, width, height);
    }
    
    unsigned int size = width * height * sizeof(float);

    // Get reduced size dimensions
    unsigned int newWidth = width / 4;
    unsigned int newHeight = height / 4;
    unsigned int newSize = newWidth * newHeight * sizeof(float);
    int kernelSize = 4;
    
    //allocating device memory for the input
    float *dInputData = NULL;
    float *dOutputData = NULL;
    checkCudaErrors(cudaMalloc((void **)&dInputData, size));
    checkCudaErrors(cudaMalloc((void **)&dOutputData, newSize));

    checkCudaErrors(cudaMemcpy(dInputData, hData, size, cudaMemcpyHostToDevice));

    // Threads per block
    int block_size = 16;
    // Blocks in each dimension
    int n = 512;
    int grid_size = (int)ceil(n / block_size);

    dim3 grid(grid_size, grid_size);
    dim3 threads(block_size, block_size);

    //printf("dim3 = %d x %d x 1", (newWidth / dimBlock.x), (newHeight / dimBlock.y));
    printf("\nwidth = %d, height = %d\n", width, height);
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Execute the kernel
    transformKernel<<<grid, threads>>>(dInputData, dOutputData, width, height, newWidth, newHeight, kernelSize);

    // Check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    // for output image
    float *hOutputData = (float *)malloc(newSize);
    
    checkCudaErrors(cudaMemcpy(hOutputData, dOutputData, newSize, cudaMemcpyDeviceToHost));
    
    //checkCudaErrors(cudaDeviceSynchronize());
    
    sdkSavePGM(outputFilename, hOutputData, newWidth, newHeight);
    printf("Saved '%s', %d x %d pixels\n", outputFilename, newWidth, newHeight);
    
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n",
         (width * height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&timer);
    
    // Write to file

    checkCudaErrors(cudaFree(dInputData));
    checkCudaErrors(cudaFree(dOutputData));
    
    free(hOutputData);
    free(hData);
    free(imagePath);

}
