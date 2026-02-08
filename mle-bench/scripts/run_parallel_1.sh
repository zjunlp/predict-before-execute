#!/bin/bash


# Set common environment variables
export SUBMISSION_DIR=/home/submission
export LOGS_DIR=/home/logs
export CODE_DIR=/home/code
export AGENT_DIR=/home/agent

# Add current directory to ensure relative paths work correctly
export CURRENT_PROJECT_DIR=/disk1/zjs/mle-bench_aide

# Define task configuration (GPU device: competition set)
declare -A TASKS
TASKS[0]="$CURRENT_PROJECT_DIR/experiments/splits/ventilator-pressure-prediction.txt"
TASKS[1]="$CURRENT_PROJECT_DIR/experiments/splits/histopathologic-cancer-detection.txt"
TASKS[2]="$CURRENT_PROJECT_DIR/experiments/splits/histopathologic-cancer-detection.txt"
TASKS[3]="$CURRENT_PROJECT_DIR/experiments/splits/histopathologic-cancer-detection.txt"
TASKS[4]="$CURRENT_PROJECT_DIR/experiments/splits/histopathologic-cancer-detection.txt"
TASKS[5]="$CURRENT_PROJECT_DIR/experiments/splits/histopathologic-cancer-detection.txt"
TASKS[6]="$CURRENT_PROJECT_DIR/experiments/splits/lmsys-chatbot-arena.txt"
TASKS[7]="$CURRENT_PROJECT_DIR/experiments/splits/us-patent-phrase-to-phrase-matching.txt"

# Build the Docker image only once
echo "Building Docker image (this may take 15+ minutes)..."
start_time=$(date +%s)

IMAGE_ID=$(docker build -q --no-cache --platform=linux/amd64 -t aide agents/aide \
    --build-arg SUBMISSION_DIR=$SUBMISSION_DIR \
    --build-arg LOGS_DIR=$LOGS_DIR \
    --build-arg CODE_DIR=$CODE_DIR \
    --build-arg AGENT_DIR=$AGENT_DIR)

build_time=$(($(date +%s) - start_time))
echo "Docker build completed in ${build_time} seconds"
echo "Built image: $IMAGE_ID"

# Start all tasks in parallel
echo "Starting all tasks in parallel..."
pids=()
log_files=()

for gpu_device in "${!TASKS[@]}"; do
    competition_set="${TASKS[$gpu_device]}"
    competition_name=$(basename "$competition_set" .txt)
    # Use current project directory for logs
    log_file="/tmp/automind_main_gpu_${gpu_device}_${competition_name}.log"
    log_files+=("$log_file")
    
    echo "Starting GPU $gpu_device with competition: $competition_name from $CURRENT_PROJECT_DIR"
    
    # Run each task in the background and redirect output to log files
    (
        cd "$CURRENT_PROJECT_DIR"  # Ensure we're in the right directory
        echo "[GPU $gpu_device] Starting at $(date) from $(pwd)"
        python run_agent.py \
            --agent-id ForeAgent/DeepSeek-V3.2 \
            --competition-set "$competition_set" \
            --data-dir /disk1/zjs/mle-bench/data \
            --gpu-device $gpu_device
        echo "[GPU $gpu_device] Completed at $(date)"
    ) > "$log_file" 2>&1 &
    
    pids+=($!)
    sleep 1  # Slightly stagger the start times
done

echo "All tasks started. PIDs: ${pids[*]}"
echo "Log files: ${log_files[*]}"

# Monitor progress in real-time
monitor_tasks() {
    while true; do
        running_count=0
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                ((running_count++))
            fi
        done
        
        if [ $running_count -eq 0 ]; then
            break
        fi
        
        echo "[$(date)] Still running: $running_count/$((${#pids[@]})) tasks"
        sleep 60  # Check every minute
    done
}

# Monitor in the background
monitor_tasks &
monitor_pid=$!

# Wait for all tasks to complete
echo "Waiting for all tasks to complete..."
failed_tasks=()

for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    gpu_device=$(echo "${!TASKS[@]}" | cut -d' ' -f$((i+1)))
    
    if wait "$pid"; then
        echo "✓ GPU $gpu_device task completed successfully"
    else
        echo "✗ GPU $gpu_device task failed"
        failed_tasks+=("$gpu_device")
    fi
done

# Stop monitoring
kill $monitor_pid 2>/dev/null

# Display summary of results
echo "================================="
echo "All tasks completed!"
echo "Successful tasks: $((${#pids[@]} - ${#failed_tasks[@]}))/${#pids[@]}"

if [ ${#failed_tasks[@]} -gt 0 ]; then
    echo "Failed tasks on GPUs: ${failed_tasks[*]}"
    echo "Check log files for details:"
    for gpu in "${failed_tasks[@]}"; do
        echo "  - GPU $gpu: $(echo "${log_files[@]}" | tr ' ' '\n' | grep "gpu_${gpu}_")"
    done
fi

echo "All log files:"
for log_file in "${log_files[@]}"; do
    echo "  - $log_file"
done

# Clean up Docker image
echo "Cleaning up Docker image..."
docker rmi $IMAGE_ID

total_time=$(($(date +%s) - start_time))
echo "Total execution time: ${total_time} seconds ($((total_time/60)) minutes)"
echo "Script completed at $(date)"