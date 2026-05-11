#!/bin/bash

# Run EvoPlay on GB1 dataset
python run_GB1.py --seed_file ../rand_seeds.txt --num_seeds 50 --use_gpu

# Run with GPU support
# python run_GB1.py --seed_file src/rndseed.txt --num_seeds 50
