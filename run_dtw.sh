#!/bin/bash

# File to store the progress of training steps
progress_file="progress.txt"
train_steps=1000
# Check if progress file exists and read the number of completed steps
completed_steps_embedder=0
completed_steps_supervisor=0
completed_steps_joint=0

if [ -f "$progress_file" ]; then
    # Read all three lines from the progress file
    IFS=$'\n' read -d '' -r completed_steps_embedder completed_steps_supervisor completed_steps_joint < "$progress_file"
fi

# Loop until the total number of training steps is reached
while [ $completed_steps_joint -lt $train_steps ] || [ $completed_steps_embedder -lt $train_steps ] || [ $completed_steps_supervisor -lt $train_steps ]; do
    echo "Running training script..."

    # Run the Python script
    python new_timegan.py

    # Check the exit code to see if the Python script completed successfully
    if [ $? -eq 0 ]; then
        # Update the progress file
        if [ -f "$progress_file" ]; then
            IFS=$'\n' read -d '' -r completed_steps_embedder completed_steps_supervisor completed_steps_joint < "$progress_file"
        fi
        
        echo "Current progress - Embedder: $completed_steps_embedder, Supervisor: $completed_steps_supervisor, Joint: $completed_steps_joint"
    else
        echo "Training was interrupted. Restarting..."
    fi
done

echo "Training completed for all steps: $train_steps"