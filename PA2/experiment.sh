#!/bin/bash

# Function to run the experiment with given parameters
run_experiment() {
    input_file="$1"
    output_file="$2"
    blocksize="$3"
    window_size="$4"
    stride_size="$5"
    memory_type="$6"

    ./PA2 "$input_file" "$output_file" "$blocksize" "$window_size" "$stride_size" "$memory_type"
}

# Set up input and output directories
input_dir="lena"
output_dir="lenaout"

# Define the ranges for parameters
block_sizes=(4 6 8 12 16 24 32)
window_sizes=(4 8)
stride_sizes=(1 2 4)
memory_types=("G" "S" "T")

# Loop through each combination of parameters
for blocksize in "${block_sizes[@]}"; do
    for window_size in "${window_sizes[@]}"; do
        for stride_size in "${stride_sizes[@]}"; do
            for memory_type in "${memory_types[@]}"; do
                # Run experiment for current parameter combination
                run_experiment "$input_dir/lena.pgm" "$output_dir/lenaout" "$blocksize" "$window_size" "$stride_size" "$memory_type"
            done
        done
    done
done

