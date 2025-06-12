#!/bin/bash
set -e
set -x

# Load environment variables from .env file
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo ".env file not found" >&2
  exit 1
fi

# Trap Ctrl+C (SIGINT) and exit gracefully
trap cleanup SIGINT

function cleanup() {
  echo -e "\nBenchmark script interrupted. Exiting gracefully..."
  exit 1
}


CONCURRENT_REQUESTS=(1 2 4 8 16 32 64 128 256 512)
COMPLETED_REQUESTS=(16 32 64 128 256 512 512 512 512 1024)

# Loop through the parameter combinations
for ((i=0; i<${#COMPLETED_REQUESTS[@]}; i++)); do
  concurrent=${CONCURRENT_REQUESTS[$i]}
  completed=${COMPLETED_REQUESTS[$i]}
  
  echo "Running with concurrent-requests=${concurrent}, max-completed-requests=${completed}, dtype=${DTYPE}, tp=${TP_SIZE}, engine=${ENGINE}"

  python token_benchmark_ray.py \
    --model "$MODEL" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --dataset-path "$DATASET" \
    --mean-input-tokens 512 \
    --stddev-input-tokens 128 \
    --mean-output-tokens 4096 \
    --stddev-output-tokens 1024 \
    --max-num-completed-requests ${completed} \
    --timeout 1200 \
    --num-concurrent-requests ${concurrent} \
    --results-dir "$RESULTS_DIR" \
    --llm-api $LLM_API \
    --additional-sampling-params '{"max_tokens": 7192}' \
    --metadata dtype=${DTYPE},tp=${TP_SIZE},engine=${ENGINE} \
    # --prompt-type chat
  
  echo "Completed run ${i}"
  echo "-----------------------------------------"
done

echo "All benchmark runs completed"
