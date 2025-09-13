#!/bin/bash

# Learning Rate Scaler
# Save as /usr/local/bin/lr-scale and chmod +x

calculate_lr() {
    local current_lr=$1
    local current_batch=$2
    local new_batch=$3
    local method=$4
    
    if [ "$method" = "sqrt" ]; then
        # Square root scaling
        python3 -c "
import math
ratio = $new_batch / $current_batch
new_lr = $current_lr * math.sqrt(ratio)
print(f'New LR: {new_lr:.2e}')
print(f'Scale factor: {math.sqrt(ratio):.3f}x')
"
    else
        # Linear scaling
        python3 -c "
ratio = $new_batch / $current_batch
new_lr = $current_lr * ratio
print(f'New LR: {new_lr:.2e}')
print(f'Scale factor: {ratio:.3f}x')
"
    fi
}

echo "Learning Rate Scaler"
echo "======================"

while true; do
    echo
    read -p "Current LR (e.g., 1e-5): " current_lr
    read -p "Current batch size in tokens: " current_batch
    read -p "New batch size in tokens: " new_batch
    
    echo
    echo "Choose scaling method:"
    echo "1) Square root scaling (recommended)"
    echo "2) Linear scaling"
    read -p "Choice (1 or 2): " choice
    
    echo
    if [ "$choice" = "1" ]; then
        calculate_lr $current_lr $current_batch $new_batch "sqrt"
    else
        calculate_lr $current_lr $current_batch $new_batch "linear"
    fi
    
    echo
    read -p "Calculate another? (y/n): " again
    if [ "$again" != "y" ]; then
        break
    fi
done
