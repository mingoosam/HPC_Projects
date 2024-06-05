
/*
*  File : PA2.cu
*  Author: Andy Varner
*  Created: 02.18.2024
*  References: 
*       For calculating the output size: https://stackoverflow.com/questions/53580088/calculate-the-output-size-in-convolution-layer

*  To compile this program --- nvcc -I/home/jovyan/cuda_sdk_samples/common/inc -o pa2 PA2.cu
*  On marge ------------------ nvcc -I/home/agkgd4/Documents/hpc_class/cuda-samples/Common -o pa2 PA2.cu
    
    *  To run -------------------- ./PA2 "lena/lena.pgm" "lena/lenaout.pgm" 16 4 4 "G"
                      args order : ./PA2, inputFilename, outputFilename,
                                   blockSize, edgeSize, strideSize, memType (G, S, T)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define MAX_WINDOW 64
#define SHARED_ARRAY_SIZE 256

const char *projectName = "PA2.cu";

// command line arguments
const char *imageFilename = NULL;
char *outputFilename = NULL;
//char formattedOutputFilename[100];
char cpuOutputFilename[] = "lenaout/cpuOut.pgm";
//char cpuFormattedOutputFilename[100];
const char *memType = NULL;

int blockSize = 0;
int edgeSize = 0; 
int strideSize = 0;
int padding = 0;

texture<float, 1, cudaReadModeElementType> tex;

__global__ void medianFilterGlobal(float *inputData, float *outputData, int width, int height, int newWidth, int newHeight, int edgeSize, int strideSize){
    
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    int start_input_x = x * strideSize;
    int start_input_y = y * strideSize;

    if (x < newWidth && y < newHeight){
        
        float kernel[MAX_WINDOW];

        // fill up kernel array
        for (int xx = 0; xx < edgeSize; xx++){
            for (int yy = 0; yy < edgeSize; yy++){
                int idx = (start_input_x + xx) + (start_input_y + yy) * height;
                kernel[xx * edgeSize + yy] = inputData[idx];
            }
        }
        // sort the array
        float temp;
        for (int i = 0; i < edgeSize*edgeSize; i++){
            for (int j = 0; j < edgeSize*edgeSize - i - 1; j++){
                if (kernel[j] > kernel[j+1]){
                    temp = kernel[j];
                    kernel[j] = kernel[j+1];
                    kernel[j+1] = temp;
                }
            }
        }
        // find the median and assign to output 
        float median = (kernel[7] + kernel[8]) / 2;
        outputData[y*newHeight + x] = median;
    }//end if
}


__global__ void medianFilterShared(float *inputData, float *outputData, int inputWidth, int inputHeight, int outputWidth, int outputHeight, int edgeSize, int strideSize){

    int gY = blockIdx.x * blockDim.x + threadIdx.x;
    int gX = blockIdx.y * blockDim.y + threadIdx.y;
    int localArea = edgeSize + (blockDim.x - 1) * strideSize;
    
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0){ 
        printf("local_area_size : %d\n\n", localArea);
    }   
    __syncthreads();  

    int start_input_i = blockIdx.x * (localArea - (edgeSize-strideSize));
    int start_input_j = blockIdx.y * (localArea - (edgeSize-strideSize));
    
    
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 84 && blockIdx.y == 84){
        printf("start_input_i : %d\n", start_input_i);
        printf("start_input_j : %d\n\n", start_input_j);
    }   
    __syncthreads();  


    // Copy the data to shared memory
    extern __shared__ float shmArr[];    
    int shmIdx = 0;
  
    if (threadIdx.x == 0 && threadIdx.y == 0){ 
            
        for(int ii = 0; ii < localArea; ii++){
            for(int jj = 0; jj < localArea; jj++){
                shmArr[shmIdx] = inputData[(start_input_i + ii) * inputWidth + start_input_j + jj];
                shmIdx++;
            }
        } 
        
    }   
   
    __syncthreads();  // "avoid the race condition" - all threads in thread block have to get to this point before we proceed (i.e. all the data is loaded)

      
    //collect values from shmArray instead of inputData 
    if (gX < outputWidth && gY < outputHeight){

        float kernel[MAX_WINDOW]; 
        //int xStart = threadIdx.x * strideSize;
        //int yStart = threadIdx.y * strideSize; 

        for (int xx = 0; xx < edgeSize; xx++){
            for (int yy = 0; yy < edgeSize; yy++){
                int idx = ( threadIdx.x * strideSize + xx )*localArea + ( threadIdx.y * strideSize + yy );
                kernel[xx * edgeSize + yy] = shmArr[idx];
            }
        }
       
        float temp;
        for (int i = 0; i < edgeSize*edgeSize; i++){
            for (int j = 0; j < edgeSize*edgeSize - i - 1; j++){
                if (kernel[j] > kernel[j+1]){
                    temp = kernel[j];
                    kernel[j] = kernel[j+1];
                    kernel[j+1] = temp;
                }
            }
        }
        // then find the median and assign to output 
        int kernel_length = (edgeSize*edgeSize);
        int idx1 = (kernel_length / 2);
        int idx2 = (kernel_length / 2) - 1;
        float median = (kernel[idx1] + kernel[idx2]) / 2; 

        outputData[gY*outputWidth + gX] = median; 

    }//end if    
   
}



__global__ void medianFilterTexture(float *outputData, int width, int height, int newWidth, int newHeight, int edgeSize, int strideSize){
    
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    int start_input_x = x * strideSize;
    int start_input_y = y * strideSize;

    if (x < newWidth && y < newHeight){
        
        float kernel[64];

        // fill up kernel array
        int kernel_len = 0;
        for (int xx = 0; xx < edgeSize; xx++){
            for (int yy = 0; yy < edgeSize; yy++){
                int idx = (start_input_x + xx) + (start_input_y + yy)*height;
                float texVal = tex1Dfetch(tex, idx);
                kernel[xx * edgeSize + yy] = texVal;
                kernel_len ++;
            }
        }
        // sort
        float temp;
        for (int i = 0; i < edgeSize*edgeSize; i++){
            for (int j = 0; j < edgeSize*edgeSize - i - 1; j++){
                if (kernel[j] > kernel[j+1]){
                    temp = kernel[j];
                    kernel[j] = kernel[j+1];
                    kernel[j+1] = temp;
                }
            }
        }

        // then find the median and assign to output 
        int idx1 = (kernel_len / 2);
        int idx2 = (kernel_len / 2) - 1;
        float median = (kernel[idx1] + kernel[idx2]) / 2; 

        outputData[y*newHeight + x] = median;
    }//end if
}

void run(int argc, char **argv);
unsigned int getOutputDim(unsigned int origWidth, unsigned int edgeSize, unsigned int strideSize);
int compare (const void *a, const void *b);
void cpuMedianFilter(float *inputImage, int kernelSize, int inputWidth, int inputHeight, int outputWidth, int outputHeight, int outputSize, int strideSize, char *outputFilename);
float median(float *arr, int n);
void copyTo2D(float *imageData, float **image2D, int width, int height);
void copyTo1D(float **image2D, float *imageData, int width, int height);
    
/*
 * Program Main
 */
int main(int argc, char **argv){

    printf("\n%s Starting...\n", projectName);    

    if (argc < 6){
        printf("Command line arg error: Please suppy the following in the order listed with no flags:\n"
               "inputFilename, outputFilename, blockSize, edgeSize, strideSize, memType\n");
        exit(EXIT_FAILURE);
    }
  
    imageFilename = argv[1];  //(A)
    outputFilename = argv[2];  //(B)
    //blockSize = static_cast<int>(argv[5]);
    //strideSize = static_cast<int>(argv[7]);
    blockSize = atoi(argv[3]); //(C)  memory block size for kernel
    edgeSize = atoi(argv[4]); //(D)
    strideSize = atoi(argv[5]);
    memType = argv[6];

    //snprintf(formattedOutputFilename, sizeof(formattedOutputFilename), "%s_%d_%d_%d_%s.pgm", outputFilename, blockSize, edgeSize, strideSize, memType);
    //snprintf(cpuFormattedOutputFilename, sizeof(cpuFormattedOutputFilename), "%s_%d_%d_%d_%s.pgm", cpuOutputFilename, blockSize, edgeSize, strideSize, memType);

    //printf("\nArgs:\n  -image filename = %s\n  -output filename = %s\n", imageFilename, formattedOutputFilename);
    printf("\nArgs:\n  -image filename = %s\n  -output filename = %s\n", imageFilename, outputFilename);
    printf("  -block size: %d\n  -edge size: %d\n  -stride size: %d\n", blockSize, edgeSize, strideSize);
    printf("  -memory type: %s\n\n", memType); 
  
    run(argc, argv);

    printf("\nFile execution complete\n\n");
}


void run(int argc, char **argv) {

    int devID = findCudaDevice(argc, (const char **)argv);
   
    FILE *fp;
    fp = fopen("execution_times.csv", "a");
    if (fp == NULL){
        printf("Couldn't open file for keeping track of execution time.\n");
    }

     //First timer: Load input data into data buffer
    clock_t start, end;
    double executionTime;

    start = clock();
     //Load image from disk
    
    float *hInputData = NULL;
    unsigned int inputWidth, inputHeight;
    
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);
    if (imagePath == NULL) {
        printf("Can't source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }
    
    if (sdkLoadPGM(imagePath, &hInputData, &inputWidth, &inputHeight)){
        printf("Loaded '%s', %d x %d pixels\n\n", imageFilename, inputWidth, inputHeight);
    }
   
    end = clock(); 
    executionTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    fprintf(fp, "Load time, %s, %s, %s, %s, %f\n", argv[3], argv[4], argv[5], argv[6], executionTime);


    start = clock();

    unsigned int inputSize = inputWidth * inputHeight * sizeof(float);

    // Get reduced size dimensions
    unsigned int outputWidth = getOutputDim(inputWidth, edgeSize, strideSize);
    unsigned int outputHeight = getOutputDim(inputWidth, edgeSize, strideSize);
    unsigned int outputSize = outputWidth * outputHeight * sizeof(float);
    
    // Allocating device memory
    float *dInputData = NULL;
    float *dOutputData = NULL;

    checkCudaErrors(cudaMalloc((void **)&dInputData, inputSize));
    checkCudaErrors(cudaMalloc((void **)&dOutputData, outputSize));

    checkCudaErrors(cudaMemcpy(dInputData, hInputData, inputSize, cudaMemcpyHostToDevice));

    // Threads per block
    // Blocks in each dimension
    int gridSize = (int)ceil(inputWidth / blockSize); // Ensure we have enough blocks to cover all elements in n (chosen because the input image size is 512x512. Prob. should make this dynamic)

    dim3 grid(gridSize, gridSize);
    dim3 threads(blockSize, blockSize);
   
    // Execute the kernel

    // **** global ***** //
    if (strcmp(memType,"G") == 0){ 

        printf("--- Using global memory ---\n");
        medianFilterGlobal<<<grid, threads>>>(dInputData, dOutputData, inputWidth, inputHeight, outputWidth, outputHeight, edgeSize, strideSize);

    }

    // **** shared ***** //
    else if (strcmp(memType,"S") == 0){

        printf("--- Using shared memory ---\n");

        unsigned int localAreaX = (threads.x - 1)*strideSize + edgeSize;
        unsigned int localAreaY = (threads.y - 1)*strideSize + edgeSize;

        //size_t shmSize = localAreaX*localAreaY*sizeof(float);
        size_t shmSize = localAreaX*localAreaY*sizeof(unsigned char);
        medianFilterShared<<<grid, threads, shmSize>>>(dInputData, dOutputData, inputWidth, inputHeight, outputWidth, outputHeight, edgeSize, strideSize);
        
        cudaDeviceSynchronize();
    }

    // ***** texture ***** //
    else if (strcmp(memType,"T") == 0){
        
        printf("--- Using texture memory ---\n");
        
        cudaBindTexture(NULL, tex, dInputData, inputSize);
        medianFilterTexture<<<grid, threads>>>(dOutputData, inputWidth, inputHeight, outputWidth, outputHeight, edgeSize, strideSize);

        // Cleanup
        cudaUnbindTexture(tex);
    }

    // Check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    // Allocate memory for output image on host
    float *hOutputData = (float *)malloc(outputSize);
     
    checkCudaErrors(cudaMemcpy(hOutputData, dOutputData, outputSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
   
    end = clock(); 
    executionTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    fprintf(fp, "copy-compute-copy time, %s, %s, %s, %s, %f\n", argv[3], argv[4], argv[5], argv[6], executionTime);

    // Write to file
    sdkSavePGM(outputFilename, hOutputData, outputWidth, outputHeight);
    //sdkSavePGM(formattedOutputFilename, hOutputData, outputWidth, outputHeight);
    printf("\nSaved '%s', %d x %d pixels\n", outputFilename, outputWidth, outputHeight);
    
    // Get gold standard cpu image

    start = clock();

    cpuMedianFilter(hInputData, edgeSize, inputWidth, inputHeight, outputWidth, outputHeight, outputSize, strideSize, cpuOutputFilename);
 
    end = clock(); 
    executionTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    fprintf(fp, "cpu time, %s, %s, %s, %s, %f\n", argv[3], argv[4], argv[5], argv[6], executionTime);

    checkCudaErrors(cudaFree(dInputData));
    checkCudaErrors(cudaFree(dOutputData));
    
    free(hOutputData);
    free(hInputData);
    free(imagePath);
    
    fclose(fp);

}

unsigned int getOutputDim(unsigned int origWidth, unsigned int edgeSize, unsigned int strideSize){

    return ((origWidth - edgeSize + 2*padding) / strideSize) + 1;
}

int compare (const void *a, const void *b) {
    return( *(float*)a - *(float*)b );
}

float median(float *arr, int n){

    qsort(arr, n, sizeof(arr), compare);

    return (n % 2 == 0) ? (arr[n/2] + arr[n/2 - 1]) / 2 : arr[n/2];

}

void copyTo2D(float *imageData, float **image2D, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            image2D[y][x] = imageData[y * width + x];
        }
    }
}

void copyTo1D(float **image2D, float *imageData, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            imageData[y * width + x] = image2D[y][x];
        }
    }
}

void cpuMedianFilter(float *inputImage, int kernelSize, int inputWidth, int inputHeight, int outputWidth, int outputHeight, int outputSize, int strideSize, char *outputFilename){


    // put the image in a 2d array
    float **inputImage2D = (float **)malloc(inputHeight * sizeof(float *));
    for (int i = 0; i < inputHeight; i++) {
        inputImage2D[i] = (float *)malloc(inputWidth * sizeof(float));
    }
    copyTo2D(inputImage, inputImage2D, inputWidth, inputHeight);

    //allocate memory for the new image
    float **cpuImage2D;
    cpuImage2D = (float **)malloc(outputHeight * sizeof(float *));
    for(int i = 0; i < outputHeight; i++){
        cpuImage2D[i] = (float *)malloc(outputWidth * sizeof(float));
    }

    //float kernel[];
    float *kernel = (float *)malloc(kernelSize * kernelSize * sizeof(float));

    // run median filter over 2d image
    for(int i = 0; i < outputHeight; i++){
        for(int j = 0; j < outputWidth; j++){
            for(int k = 0; k < kernelSize; k++){
                for(int l = 0; l < kernelSize; l++){
                    kernel[k * kernelSize + l] = inputImage2D[i * strideSize + k][j * strideSize + l];
                }
            }
            cpuImage2D[i][j] = median(kernel, kernelSize * kernelSize);
        }
    }

    float *cpuImage1D = (float *)malloc(outputSize);
    copyTo1D(cpuImage2D, cpuImage1D, outputWidth, outputHeight);       

    sdkSavePGM(outputFilename, cpuImage1D, outputWidth, outputHeight);
    printf("\nSaved cpu output '%s', %d x %d pixels\n", outputFilename, outputWidth, outputHeight);

    
    free(kernel);

    for (int i = 0; i < outputHeight; i++) {
        free(cpuImage2D[i]);
    }
    free(cpuImage2D);

    for (int i = 0; i < inputHeight; i++) {
        free(inputImage2D[i]);
    }
    free(inputImage2D);

    free(cpuImage1D);
}


