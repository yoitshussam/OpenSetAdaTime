#!/bin/bash

# Define the DA methods you want to run
methods=("RAINCOAT" "OVANet" "PPOT" "DANCE" "UniJDOT")

# Define the dataset names
datasets=("RealWorld" "Pamap2" "mhealth")

# Iterate over each DA method
for method in "${methods[@]}"; do
    echo "============================================"
    echo "Starting experiments for DA Method: $method"
    echo "============================================"

    # Iterate over source datasets
    for source in "${datasets[@]}"; do
        # Iterate over target datasets
        for target in "${datasets[@]}"; do
            
            # Skip if source and target are the same
            if [ "$source" == "$target" ]; then
                continue
            fi

            echo "Running: Source=$source -> Target=$target"
            
            python main.py \
                --num_runs 5 \
                --source_dataset "$source" \
                --target_dataset "$target" \
                --data_path ../dataset \
                --da_method "$method" \
                --exp_name "closed_set"

        done
    done
done