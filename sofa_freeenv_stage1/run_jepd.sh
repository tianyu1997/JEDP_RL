#!/bin/bash
# Make sure the script is executable:
# chmod +x run_training.sh

# Path to the Python script
PYTHON_SCRIPT="NN_training.py"

# Number of runs
TOTAL_RUNS=150

# Starting load number
START_LOADNO=0

# Function to clear RAM (Linux example; may require root privileges)
clear_ram() {
    echo "Clearing system caches..."
    # Uncomment the next two lines if you have sudo rights and want to drop caches.
    # sudo sync
    # echo 3 | sudo tee /proc/sys/vm/drop_caches
    # Alternatively, call your custom RAM clear function here
}

# Loop through the runs
for ((i = 1; i <= TOTAL_RUNS; i++)); do
    LOADNO=$((START_LOADNO + i - 1))
    echo "Running phase $LOADNO"
    
    # Clear system RAM (if required)
    clear_ram

    # Run the Python script with reload flag to load the last checkpoint
    python3 $PYTHON_SCRIPT --loadno $LOADNO --reload

    # Check for errors
    if [ $? -ne 0 ]; then
        echo "Error occurred during execution at phase $LOADNO. Exiting..."
        exit 1
    fi

    echo "Phase $LOADNO completed successfully."
    sleep 5
done

echo "All phases completed successfully."