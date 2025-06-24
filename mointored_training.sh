#!/bin/bash

# Configuration
PROJECT_DIR="/ssd/karen/Titan_OLMo_core"
SCRIPT_PATH="src/scripts/train/TITAN-OLMo2-1b.py"
RUN_NAME="restart_low_lr"  # Change this to your desired run name
MAX_RESTARTS=15  # Maximum number of restarts before giving up
RESTART_DELAY=30  # Seconds to wait before restarting
LOG_DIR="$HOME/titan_training_logs"
PHASE="full_model"  # Change to "memory_only" if needed

# Create log directory
mkdir -p "$LOG_DIR"

# Function to get current timestamp
get_timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Function to log messages
log_message() {
    echo "[$(get_timestamp)] $1" | tee -a "$LOG_DIR/monitor.log"
}

# Function to run the training script
run_training() {
    local attempt=$1
    local log_file="$LOG_DIR/training_attempt_${attempt}_$(date +%Y%m%d_%H%M%S).log"
    
    log_message "Starting training attempt #$attempt"
    log_message "Log file: $log_file"
    
    # Change to the project directory
    cd "$PROJECT_DIR"
    
    # Run the training script with the correct command structure
    python "$SCRIPT_PATH" train "$RUN_NAME" --phase="$PHASE" 2>&1 | tee "$log_file"
    
    # Return the exit code of the Python script (not tee)
    return ${PIPESTATUS[0]}
}

# Function to check if crash was due to OOM or other recoverable error
is_recoverable_error() {
    local log_file=$1
    
    # Check last 50 lines of log for common recoverable errors
    local last_lines=$(tail -n 50 "$log_file")
    
    # OOM errors
    if echo "$last_lines" | grep -q "out of memory\|OutOfMemoryError\|CUDA out of memory"; then
        log_message "Detected OOM error - this is recoverable"
        return 0
    fi
    
    # CUDA errors that might be recoverable
    if echo "$last_lines" | grep -q "CUDA error\|device-side assert\|illegal memory access"; then
        log_message "Detected CUDA error - attempting recovery"
        return 0
    fi
    
    # Connection/network errors
    if echo "$last_lines" | grep -q "ConnectionError\|TimeoutError\|Network is unreachable"; then
        log_message "Detected network error - this is recoverable"
        return 0
    fi
    
    # Wandb errors
    if echo "$last_lines" | grep -q "wandb.*error\|wandb.*timeout"; then
        log_message "Detected WandB error - this is recoverable"
        return 0
    fi
    
    # Checkpoint saving errors
    if echo "$last_lines" | grep -q "Error saving checkpoint\|Permission denied.*checkpoint"; then
        log_message "Detected checkpoint error - this is recoverable"
        return 0
    fi
    
    # If we get here, it might not be recoverable
    log_message "Could not identify error type - assuming recoverable"
    return 0  # Default to recoverable for now
}

# Function to cleanup CUDA memory
cleanup_cuda() {
    log_message "Cleaning up CUDA memory..."
    
    # Kill any remaining Python processes that might be holding GPU memory
    pkill -f "python.*TITAN-OLMo2-1b.py" 2>/dev/null || true
    
    # Wait a bit for processes to die
    sleep 5
    
    # Try to reset GPU if nvidia-smi is available
    if command -v nvidia-smi &> /dev/null; then
        log_message "GPU memory before cleanup:"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
        
        # Clear GPU memory cache (this might not work for all processes)
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        log_message "GPU memory after cleanup:"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
    fi
}

# Main monitoring loop
main() {
    log_message "=== Starting Titan Training Monitor ==="
    log_message "Project directory: $PROJECT_DIR"
    log_message "Script: $SCRIPT_PATH"
    log_message "Full command: python $SCRIPT_PATH train $RUN_NAME --phase=$PHASE"
    log_message "Run name: $RUN_NAME"
    log_message "Phase: $PHASE"
    log_message "Max restarts: $MAX_RESTARTS"
    log_message "Log directory: $LOG_DIR"
    
    local restart_count=0
    local consecutive_failures=0
    local last_restart_time=0
    
    while [ $restart_count -lt $MAX_RESTARTS ]; do
        local start_time=$(date +%s)
        
        # Run the training script
        run_training $((restart_count + 1))
        local exit_code=$?
        
        local end_time=$(date +%s)
        local runtime=$((end_time - start_time))
        
        if [ $exit_code -eq 0 ]; then
            log_message "Training completed successfully after ${runtime}s"
            break
        else
            restart_count=$((restart_count + 1))
            log_message "Training crashed with exit code $exit_code after ${runtime}s"
            
            # Find the most recent log file
            local latest_log=$(ls -t "$LOG_DIR"/training_attempt_*.log 2>/dev/null | head -n1)
            
            if [ -n "$latest_log" ]; then
                log_message "Checking crash reason in: $latest_log"
                
                if is_recoverable_error "$latest_log"; then
                    consecutive_failures=0
                else
                    consecutive_failures=$((consecutive_failures + 1))
                    log_message "Warning: Non-recoverable error detected (consecutive: $consecutive_failures)"
                fi
            fi
            
            # If we've had too many consecutive failures, increase delay
            local current_delay=$RESTART_DELAY
            if [ $consecutive_failures -gt 2 ]; then
                current_delay=$((RESTART_DELAY * consecutive_failures))
                log_message "Increasing restart delay to ${current_delay}s due to consecutive failures"
            fi
            
            # Check if we should continue
            if [ $restart_count -ge $MAX_RESTARTS ]; then
                log_message "Maximum restart attempts ($MAX_RESTARTS) reached. Giving up."
                break
            fi
            
            # Cleanup and wait before restart
            cleanup_cuda
            
            log_message "Waiting ${current_delay}s before restart attempt #$((restart_count + 1))..."
            sleep $current_delay
            
            # Track restart time
            last_restart_time=$(date +%s)
        fi
    done
    
    log_message "=== Training Monitor Finished ==="
    
    # Send a summary
    if [ $exit_code -eq 0 ]; then
        log_message "✅ Training completed successfully"
    else
        log_message "❌ Training failed after $restart_count restart attempts"
        log_message "Check logs in: $LOG_DIR"
    fi
}

# Handle Ctrl+C gracefully
trap 'log_message "Received interrupt signal. Cleaning up..."; cleanup_cuda; exit 1' INT TERM

# Run the main function
main "$@"