#!/usr/bin/env bash
# lora_rank_check.sh
# usage: ./lora_rank_check.sh <model_size> <hidden_size> <layers> <tokens> <your_rank>

set -e

# ---------- inputs ----------
model_size=$1          # e.g. 8B (only used for pretty print)
hidden=$2              # e.g. 4096
layers=$3              # e.g. 32
tokens=$4              # e.g. 500000000
user_rank=$5           # e.g. 64

# ---------- constants ----------
bits_per_token=1
bits_per_param=2

# ---------- maths ----------
# params_needed = tokens * bits_per_token / bits_per_param
params_needed=$(awk -v t="$tokens" 'BEGIN{printf "%.0f", t / 2}')

# min_rank = params_needed / (2 * hidden * layers * 2)
min_rank=$(awk -v p="$params_needed" -v h="$hidden" -v l="$layers" \
           'BEGIN{printf "%.0f", p / (8 * h * l)}')

# capacity of user’s rank (bits)
user_params=$(awk -v r="$user_rank" -v h="$hidden" -v l="$layers" \
              'BEGIN{printf "%.0f", 8 * r * h * l}')
user_bits=$(awk -v p="$user_params" 'BEGIN{printf "%.0f", p * 2}')

# ratio
ratio=$(awk -v u="$user_bits" -v t="$tokens" 'BEGIN{printf "%.2f", u / t}')

# ---------- report ----------
cat <<EOF
Model: ${model_size}  (${hidden} hidden, ${layers} layers)
Dataset: ${tokens} tokens  (~${bits_per_token} bit/token)

Minimum rank required: ${min_rank}
Your chosen rank:      ${user_rank}

Your LoRA capacity: ${user_bits} bits  (${ratio}× dataset size)
EOF

if (( user_bits >= tokens )); then
    echo "✅ Your rank is sufficient (or over-capacity)."
else
    under=$(awk -v t="$tokens" -v u="$user_bits" 'BEGIN{printf "%.1f", t/u}')
    echo "⚠️  Under-capacity by ~${under}×.  Consider rank >= ${min_rank}"
fi
