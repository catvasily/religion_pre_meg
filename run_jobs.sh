#!/bin/bash
# Script to run a program with incrementing integer parameter

# Define the maximum value of N
N=19  # Change this to your desired upper limit
prog="python3 run_pipeline.py"

# Loop from 0 to N and run the program with the current value as an argument
for i in $(seq 0 $N); do
    $prog "$i"
done

