// This is Marshall Lindsay's code. I used it to understand the logic of median filter.
// to compile this program: gcc -o cpu.out cpu_median_filter.c
// to run: ./cpu.out -f "lena/lena.pgm" -o "lena/marshall.out" 16

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

//#include <cuda_runtime.h>
//#include <helper_functions.h>

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
 * Read the pgm file.
 */
int read_pgm(FILE *file, int *width, int *height, int *max_val, unsigned char ***image){

    char str[100];
    if (fgets(str, 100, file) == NULL){
        printf("Error: Could not read file\n");
        return 1;
    }else{
        printf("Magic number: %s\n", str);
    }

    if (fgets(str, 100, file) == NULL){
        printf("Error: Could not read file\n");
        return 1;
    }else{
        printf("Comment: %s\n", str);
    }

    if (fscanf(file, "%d %d", width, height) != 2){
        printf("Error: Could not read file\n");
        return 1;
    }else{
        printf("Width: %d\n", *width);
        printf("Height: %d\n", *height);
    }

    if (fscanf(file, "%d", max_val) != 1){
        printf("Error: Could not read file\n");
        return 1;
    }else{
        printf("Max value: %d\n", *max_val);
    }
    fgetc(file); // Remove the newline character

    // Allocate memory for the rows
    // The value of the image pointer is changed to the address of the 2D array.
    *image = (unsigned char **)malloc(*height * sizeof(unsigned char *));

    if (*image == NULL){
        printf("Error: Could not allocate memory\n");
        return 1;
    }

    // Allocate memory for each row
    for(int i = 0; i < *height; i++){
        // The value of the image pointer at each row is changed to the 
        // address of the 1D array. 
        (*image)[i] = (unsigned char *)malloc(*width * sizeof(unsigned char));

        if ((*image)[i] == NULL){
            printf("Error: Could not allocate memory\n");
            for (int j = 0; j < i; j++){
                free((*image)[j]);
            }
            free(*image);
            return 1;
        }
    }

    // Read the image
    for(int i = 0; i < *height; i++){
        for(int j = 0; j < *width; j++){
            // The value of the image pointer at each row and column is changed
            // to the pixel value.
            // Need to do the pointer magic to update the value of the image pointer.
            if(fscanf(file, "%c", &(*image)[i][j]) != 1){
                printf("Error: Could not read file\n");
                for(int j = 0; j < i; j++){
                    free((*image)[j]);
                }
                free(*image);
                return 1;
            }
        }
    }

    return 0;

}

/*
 * Write the pgm file.
 */

int write_pgm(FILE *file, int width, int height, int max_val, unsigned char **image){

    fprintf(file, "P5\n");
    fprintf(file, "# Image updated by Marshall Lindsay\n");
    fprintf(file, "%d %d\n", width, height);
    fprintf(file, "%d\n", max_val);

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            fprintf(file, "%c", image[i][j]);
        }
    }

    return 0;
}

/*
 * Helper function to free 2D array
 */

void free_2d_array(unsigned char **image, int height){
    printf("Freeing 2D array\n");
    for(int i = 0; i < height; i++){
        free(image[i]);
    }
    free(image);
}


/*
 * Bubble sort
 */

void bubble_sort(int *arr, int n){
    int temp;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n - i - 1; j++){
            if(arr[j] > arr[j+1]){
                temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

/*
 * helper for qsort
 */
int compare (const void *a, const void *b) {
    return( *(int*)a - *(int*)b );
}

/*
 * Median of an array
 */

int median(int *arr, int n){
    // Sort the array
    //bubble_sort(arr, n);
    int len = sizeof(arr) / sizeof(arr[0]);
    qsort(arr, len, sizeof(arr), compare);

    if(n % 2 == 0){
        // If the number of elements is even, return the average of the two middle elements
        return (arr[n/2] + arr[n/2 - 1]) / 2;
    }else{
        // If the number of elements is odd, return the middle element
        return arr[n/2];
    }
}

int main(int argc, char *argv[]){

    int i = 0;
    int kernel_size = 4;
    
    FILE *input_file;
    char *input_filename;

    FILE *output_file;
    char *output_filename;

    printf("%d\n",argc);
    if( argc == 6 ) {
    
       printf("The input file name is %s\nThe output file name is %s\nThe block-edge size is %d\n", argv[2],argv[4],atoi(argv[5]));
    } 
    else {
       printf("Please supply the input file name after -f flag, the output file name after -o flag, and the block-edge size as an int.\n");
    }
    
    clock_t start, end;


    if(get_input_filename(argc, argv, &input_filename) != 0){
        printf("Error: No filename given\n");
        return 1;
    }else{
        printf("Input filename: %s\n", input_filename);
    }

    if(open_file_read(input_filename, &input_file) != 0){
        return 1;
    }

    // Header information from the pgm file
    int width, height, max_val;

    // Image will be stored as a 2D array
    unsigned char **image;

    // Read the pgm file
    // Passing the address of the width, height, max_val, and image pointers.
    read_pgm(input_file, &width, &height, &max_val, &image);

    // Close the file
     if(close_file(input_file) != 0){
         free_2d_array(image, height);
         return 1;
    }
    
    
    printf("Image width: %d Image height: %d\n", width, height);

    // Print the image
    //for(int i = 0; i < height; i++){
    //    for(int j = 0; j < width; j++){
    //        printf("%d ", image[i][j]);
    //    }
    //    printf("\n");
    //}

    // Get the output filename from the command line
    if(get_output_filename(argc, argv, &output_filename) != 0){
        printf("Error: No filename given\n");
        free_2d_array(image, height);
        return 1;
    }

    // Open the file for writing
    if(open_file_write(output_filename, &output_file) != 0){
        printf("Error: Could not open %s for writing\n", output_filename);
        free_2d_array(image, height);
        return 1;
    }

    // Create a new 2D array to store the new image 
    // This will be 1/4 the size of the original image
    int new_width = width / 4;
    int new_height = height / 4;
    unsigned char **new_image;
    new_image = (unsigned char **)malloc(new_height * sizeof(unsigned char *));
    for(int i = 0; i < new_height; i++){
        new_image[i] = (unsigned char *)malloc(new_width * sizeof(unsigned char));
    }

    // Loop over the image and apply the 4x4 median filter
    int kernel[16];

    start = clock();
    for(int i = 0; i < new_height; i++){
        for(int j = 0; j < new_width; j++){
            for(int k = 0; k < kernel_size; k++){
                for(int l = 0; l < kernel_size; l++){
                    kernel[k * kernel_size + l] = image[i * kernel_size + k][j * kernel_size + l];
                }
            }
            /*
            int size = sizeof(kernel) / sizeof(kernel[0]);
            printf("size of kernel is %d\n", size);
            for (int i = 0; i < size; i++){
                printf("%d\n", kernel[i]);
            }
            if (j>1){
                exit(0);
            }
            */
            new_image[i][j] = median(kernel, kernel_size * kernel_size);
            //*(*(new_image + i) + j) = median(kernel, kernel_size * kernel_size);
        }
    }
    end = clock();
    printf("Time taken: %f\n", (double)(end - start) / CLOCKS_PER_SEC);

    /*
    for(i = 0; i < height; i ++){
        for(int j = 0; j < width; j++){
            printf("image[%d] : %u \n", i, image[i][j]);
        }
    }
    */
    // Write the pgm file
    write_pgm(output_file, new_width, new_height, max_val, new_image);

    // Close the file
    if(close_file(output_file) != 0){
        free_2d_array(new_image, new_height);
        free_2d_array(image, height);
        return 1;
    }

    // Free the 2D arrays
    free_2d_array(new_image, new_height);
    free_2d_array(image, height);
}

