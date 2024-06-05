#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <string.h>


/*******************/
/* iDivUp FUNCTION */
/*******************/
/*
* https://stackoverflow.com/questions/16876480/2d-median-filtering-in-cuda-how-to-efficiently-copy-global-memory-to-shared-mem
*/
int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }


/*
 * Open a file for reading. If the file does not exist, print an error message and return 1.
 */
int open_file_read(char *filename, FILE **file){
    *file = fopen(filename, "rb");
    if(*file == NULL){
        printf("Error: Could not open file %s\n", filename);
        return 1;
    }
    return 0;
}

/*
 * Open a file for writing. If the file does not exist, print an error message and return 1.
 */
int open_file_write(char *filename, FILE **file){
    *file = fopen(filename, "wb");
    if(*file == NULL){
        printf("Error: Could not open file %s\n", filename);
        return 1;
    }
    return 0;
}

/*
 * Close a file. If the file does not exist, print an error message and return 1.
 */
int close_file(FILE *file){
    if(fclose(file) != 0){
        printf("Error: Could not close file\n");
        return 1;
    }
    return 0;
}

/*
 * Get the input filename from the command line. Filename is given after the -f flag.
 */
int get_input_filename(int argc, char *argv[], char **filename){
    int i;
    for(i = 0; i < argc; i++){
        if(strcmp(argv[i], "-f") == 0){
            *filename = argv[i+1];
            return 0;
        }
    }
    return 1;
}

/*
 * Get the output filename from the command line. Filename is given after the -o flag.
 */

int get_output_filename(int argc, char *argv[], char **filename){
    int i;
    for(i = 0; i < argc; i++){
        if(strcmp(argv[i], "-o") == 0){
            *filename = argv[i+1];
            return 0;
        }
    }
    return 1;
}


/*
 * Get the block dim. Given after -b flag.
 */

int get_block_dim(int argc, char *argv[], int *blockdim){
    int i;
    for(i = 0; i < argc; i++){
        if(strcmp(argv[i], "-b") == 0){
            *blockdim = atoi(argv[i+1]);
            return 0;
        }
    }
    return 1;
}

/*
 * Get the kernel size from the command line. Kernel size is given after the -k flag.
 */
int get_kernel_size(int argc, char *argv[], int *kernel_size){
    int i;
    for(i = 0; i < argc; i++){
        if(strcmp(argv[i], "-k") == 0){
            *kernel_size = atoi(argv[i+1]);
            return 0;
        }
    }
    return 1;
}

/*
 * Get the stride size from the command line. Stride size is given after the -s flag.
 */
int get_stride_size(int argc, char *argv[], int *stride){
    int i;
    for(i = 0; i < argc; i++){
        if(strcmp(argv[i], "-s") == 0){
            *stride = atoi(argv[i+1]);
            return 0;
        }
    }
    return 1;
}

/*
 * Get the memory type from the command line. Memory type is given after the -m flag.
 */
int get_memory_type(int argc, char *argv[], char *memory_type){
    int i;
    for(i = 0; i < argc; i++){
        if(strcmp(argv[i], "-m") == 0){
            *memory_type = *argv[i+1];
            return 0;
        }
    }
    return 1;
}




/*
 * Read the pgm file.
 */
int read_pgm(FILE *file, int *width, int *height, int *max_val, unsigned char **image){

    char str[100];
    if (fgets(str, 100, file) == NULL){
        printf("Error: Could not read file\n");
        return 1;
    }

    if (fgets(str, 100, file) == NULL){
        printf("Error: Could not read file\n");
        return 1;
    }

    if (fscanf(file, "%d %d", width, height) != 2){
        printf("Error: Could not read file\n");
        return 1;
    }

    if (fscanf(file, "%d", max_val) != 1){
        printf("Error: Could not read file\n");
        return 1;
    }
    
    fgetc(file); // Remove the newline character

    // Allocate memory for the rows
    // The value of the image pointer is changed to the address of the 2D array.
    *image = (unsigned char *)malloc(*height * *width * sizeof(unsigned char));

    if (*image == NULL){
        printf("Error: Could not allocate memory\n");
        return 1;
    }

    // Read the image
    for(int i = 0; i < *height * *width; i++){
        if(fscanf(file, "%c", &(*image)[i]) != 1){
            printf("Error: Could not read file\n");
            free(*image);
            return 1;
        }
    }

    return 0;

}

int write_pgm(FILE *file, int width, int height, int max_val, unsigned char *image){

    fprintf(file, "P5\n");
    fprintf(file, "# Image updated by Marshall Lindsay\n");
    fprintf(file, "%d %d\n", width, height);
    fprintf(file, "%d\n", max_val);

    for(int i = 0; i < height * width; i++){
        fprintf(file, "%c", image[i]);
    }

    return 0;
}



__global__ void median_filter_shared_mem(unsigned char *A, unsigned char *C, unsigned char *kernel, int width, int height, int new_width, int new_height, int kernel_size, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int local_area_size = kernel_size + (blockDim.x - 1) * stride;
    
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0){
        printf("local_area_size : %d\n\n", local_area_size);
    }
    __syncthreads();  

    int start_input_i = blockIdx.x * (local_area_size - (kernel_size-stride));
    int start_input_j = blockIdx.y * (local_area_size - (kernel_size-stride));
    
    
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 84 && blockIdx.y == 84){
        printf("start_input_i : %d\n", start_input_i);
        printf("start_input_j : %d\n\n", start_input_j);
    }
    __syncthreads();  


    // Copy the data to shared memory
    extern __shared__ unsigned char shared_mem[];        
    int shmIdx = 0;
    if (threadIdx.x == 0 && threadIdx.y == 0){
        
        for(int ii = 0; ii < local_area_size; ii++){
            for(int jj = 0; jj < local_area_size; jj++){
                shared_mem[shmIdx] = A[(start_input_i + ii) * width + start_input_j + jj];
                shmIdx++;
            }
        }
        
    }
    __syncthreads();  

    if (i < new_width && j < new_height){
        // Populate kernel space
        int kernel[8*8];
        int localY_start = threadIdx.y * stride;
        int localX_start = threadIdx.x * stride;
        
        for(int ii = 0, iii = localY_start; ii < kernel_size; ii++, iii++){
            for(int jj = 0, jjj = localX_start; jj < kernel_size; jj++, jjj++){
                kernel[ii * kernel_size + jj] = shared_mem[jjj * local_area_size + iii];
            }
        }
        
        
        // Sort the kernel space 
        int temp;
        int len_flattened = kernel_size * kernel_size;
        for(int ii = 0; ii < len_flattened; ii++){
            for(int jj = 0; jj < len_flattened - ii - 1; jj++){
                if(kernel[jj] > kernel[jj + 1]){
                    temp = kernel[jj];
                    kernel[jj] = kernel[jj + 1];
                    kernel[jj + 1] = temp;
                }
            }
        }
        
        // Select the median
        if(len_flattened % 2 == 0){
            // If the number of elements is even, return the average of the two middle elements
            C[i * new_width + j] =  (kernel[len_flattened/2] + kernel[len_flattened/2 - 1]) / 2;
        }else{
            // If the number of elements is odd, return the middle element
            C[i * new_width + j] =  kernel[len_flattened/2];
        }
        
    }
}

   
// Texture reference for 2D uchar texture
texture<unsigned char, 2, cudaReadModeElementType> tex;

__global__ void median_filter_texture_mem(unsigned char *A, unsigned char *C, unsigned char *kernel, int width, int height, int new_width, int new_height, int kernel_size, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    int start_input_i = i * stride;
    int start_input_j = j * stride;     
        
    if (i < new_width && j < new_height) {
    
        // Populate the kernel
        // Populate kernel space
        int kernel[8*8];
        for(int ii = 0; ii < kernel_size; ii++){
            for(int jj = 0; jj < kernel_size; jj++){
                unsigned char pixel = tex2D(tex, start_input_j + jj, start_input_i + ii);
                kernel[ii * kernel_size + jj] = pixel;
            }
        }
    
        // Sort the kernel space 
        int temp;
        int len_flattened = kernel_size * kernel_size;
        for(int ii = 0; ii < len_flattened; ii++){
            for(int jj = 0; jj < len_flattened - ii - 1; jj++){
                if(kernel[jj] > kernel[jj + 1]){
                    temp = kernel[jj];
                    kernel[jj] = kernel[jj + 1];
                    kernel[jj + 1] = temp;
                }
            }
        }

        
        // Select the median
        if(len_flattened % 2 == 0){
            // If the number of elements is even, return the average of the two middle elements
            C[i * new_width + j] =  (kernel[len_flattened/2] + kernel[len_flattened/2 - 1]) / 2;
        }else{
            // If the number of elements is odd, return the middle element
            C[i * new_width + j] =  kernel[len_flattened/2];
        }
    }
}



__global__ void median_filter_global_mem(unsigned char *A, unsigned char *C, unsigned char *kernel, int width, int height, int new_width, int new_height, int kernel_size, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    int start_input_i = i * stride;
    int start_input_j = j * stride;     
    
    if (i < new_width && j < new_height) {
    
        // Populate the kernel
        int kernel[8*8];
        for(int ii = 0; ii < kernel_size; ii++){
            for(int jj = 0; jj < kernel_size; jj++){
                kernel[ii * kernel_size + jj] = A[(start_input_i + ii) * width + start_input_j + jj];
            }
        }
    
        // Sort the kernel space 
        int temp;
        int len_flattened = kernel_size * kernel_size;
        for(int ii = 0; ii < len_flattened; ii++){
            for(int jj = 0; jj < len_flattened - ii - 1; jj++){
                if(kernel[jj] > kernel[jj + 1]){
                    temp = kernel[jj];
                    kernel[jj] = kernel[jj + 1];
                    kernel[jj + 1] = temp;
                }
            }
        }

        
        // Select the median
        if(len_flattened % 2 == 0){
            // If the number of elements is even, return the average of the two middle elements
            C[i * new_width + j] =  (kernel[len_flattened/2] + kernel[len_flattened/2 - 1]) / 2;
        }else{
            // If the number of elements is odd, return the middle element
            C[i * new_width + j] =  kernel[len_flattened/2];
        }
    }
}

void deleteEnd (char* myStr){

    //printf ("%s\n", myStr);
    char *del = &myStr[strlen(myStr)];

    while (del > myStr && *del != '/')
        del--;

    if (*del== '/')
        *del= '\0';

    return;
}



int main(int argc, char *argv[]){

    int kernel_size;
    int stride;
    char memory_type;
    int blockdim;
    int padding = 0;

    FILE *input_file;
    char *input_filename;

    FILE *output_file;
    char *output_filename;
    
    clock_t start, end;


    if(get_input_filename(argc, argv, &input_filename) != 0){
        printf("Error: No filename given\n");
        return 1;
    }

    if(get_kernel_size(argc, argv, &kernel_size) != 0){
        printf("Error: Kernel size not specified\n");
        return 1;
    }

    if(get_stride_size(argc, argv, &stride) != 0){
        printf("Error: Stride size not specified\n");
        return 1;
    }

    if(get_memory_type(argc, argv, &memory_type) != 0){
        printf("Error: Memory type not specified\n");
        return 1;
    }

    //Get the blockdim from the command line
    if(get_block_dim(argc, argv, &blockdim) != 0){
        printf("Error: No blockdim given\n");
        return 1;
    }
    printf("Kernel size: %d\n", kernel_size);
    printf("Stride : %d\n", stride);
    printf("Block dim : %d\n", blockdim);

    if(open_file_read(input_filename, &input_file) != 0){
        return 1;
    }


    // Header information from the pgm file
    int width, height, max_val;

    // Image will be stored as a 1D array
    unsigned char *image;

    // Read the pgm file
    // Passing the address of the width, height, max_val, and image pointers.
    read_pgm(input_file, &width, &height, &max_val, &image);


    // Close the file
     if(close_file(input_file) != 0){
         free(image);
         return 1;
    }

    // Size will be determined by the general formula
    // size = ((width_image - filter_size + 2 * padding) / stride ) + 1
    int new_width = ((width - kernel_size + (2 * padding)) / stride) + 1;
    int new_height = ((height - kernel_size + (2 * padding)) / stride) + 1;

    printf("New width: %d\n", new_width);
    printf("New height: %d\n", new_height);    
    
    unsigned char *h_A, *h_C;
    unsigned char *d_A, *d_C, *d_K;
    
    // Allocate host memory and initialize
    h_A = (unsigned char*) malloc(width * height * sizeof(unsigned char));
    h_C = (unsigned char*) malloc(new_width * new_height * sizeof(unsigned char));
    
    for (int i = 0; i < width * height; ++i) {
        h_A[i] = image[i];
    }
    free(image);
    
   
    // Allocate device memory
    cudaMalloc((void**)&d_A, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_C, new_width * new_height * sizeof(unsigned char));
    cudaMalloc((void**)&d_K, new_width * new_height * kernel_size * kernel_size * sizeof(unsigned char));

    // Create the block dimensions
    dim3 blockDim(blockdim, blockdim);
    dim3 gridDim(iDivUp(width, blockdim), iDivUp(height, blockdim));
    
    printf("blockDim : %d , %d , %d\n", blockDim.x, blockDim.y, blockDim.z);
    printf("gridDim : %d , %d , %d\n", gridDim.x, gridDim.y, gridDim.z);
    
    if (memory_type == 'G'){
        printf("Running with global memory\n");
        
        start = clock();
        // Copy the host array to the device
        cudaMemcpy(d_A, h_A, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
        // Execute the kernel
        median_filter_global_mem<<<gridDim, blockDim>>>(d_A, d_C, d_K, width, height, new_width, new_height, kernel_size, stride);

        // Copy the device array to the host
        cudaMemcpy(h_C, d_C, new_width * new_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        end = clock();
        
    }else if (memory_type == 'S'){
        printf("Running with shared memory\n");
        int local_area_size = kernel_size + (blockDim.x - 1) * stride;
    
        start = clock();
        // Copy the host array to the device
        cudaMemcpy(d_A, h_A, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
        // Execute the kernel
        int shared_mem_size = local_area_size * local_area_size * sizeof(unsigned char);
        median_filter_shared_mem<<<gridDim, blockDim, shared_mem_size>>>(d_A, d_C, d_K, width, height, new_width, new_height, kernel_size, stride);

        // Copy the device array to the host
        cudaMemcpy(h_C, d_C, new_width * new_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        end = clock();
        
        
    }else if (memory_type == 'T'){
        printf("Running with texture memory\n");
        
        // CUDA array
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
        cudaArray* cuArray;
        cudaMallocArray(&cuArray, &channelDesc, width, height);
        
        start = clock();
        // Copy the host array to texture memory
        cudaMemcpyToArray(cuArray, 0, 0, h_A, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
        
        // Bind the array to the texture reference
        cudaBindTextureToArray(tex, cuArray, channelDesc);
        
        //Execute the kernel
        median_filter_texture_mem<<<gridDim, blockDim>>>(d_A, d_C, d_K, width, height, new_width, new_height, kernel_size, stride);

        // Copy result back to host
        cudaMemcpy(h_C, d_C, new_width * new_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        end = clock();
        
        
    }else{
        printf("Memory type not supported");
        return 1;
    }    
    
        
    float time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time taken: %f\n", time_taken);    
    
    // Get the output filename from the command line
    if(get_output_filename(argc, argv, &output_filename) != 0){
        printf("Error: No filename given\n");
        return 1;
    }

    // Open the file for writing
    if(open_file_write(output_filename, &output_file) != 0){
        printf("Error: Could not open %s for writing\n", output_filename);
        return 1;
    }
    
    // Write the pgm file
    write_pgm(output_file, new_width, new_height, max_val, h_C);

    // Close the file
    if(close_file(output_file) != 0){
        return 1;
    }
    
    deleteEnd(output_filename);
    strcat(output_filename,"/output_time.txt");
    
    //printf("%s\n", output_filename);
    
    FILE *output_time_file;
    output_time_file = fopen(output_filename, "w");
    fprintf(output_time_file, "%f", time_taken);
    fclose(output_time_file);
    
    //fprintf(output_time_file, "%f", time_taken);
    //fclose(output_time_file);
    
    free(h_A);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_C);
    cudaFree(d_K);
    cudaUnbindTexture(tex);

    return 0;
}
