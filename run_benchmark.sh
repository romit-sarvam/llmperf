#!/bin/bash
set +xe

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

# Arrays for parameter values

CONCURRENT_REQUESTS=(1 2 4 8 16 32 64 128 256)
COMPLETED_REQUESTS=(16 32 64 128 256 512 1024 1024 1024)

# Loop through the parameter combinations
for i in {0..8}; do
  concurrent=${CONCURRENT_REQUESTS[$i]}
  completed=${COMPLETED_REQUESTS[$i]}
  
  echo "Running with concurrent-requests=${concurrent}, max-completed-requests=${completed}, dtype=${DTYPE}, tp=${TP_SIZE}, engine=${ENGINE}"

  python token_benchmark_ray.py \
    --model "$MODEL_PATH" \
    --mean-input-tokens 512 \
    --stddev-input-tokens 128 \
    --mean-output-tokens 4096 \
    --stddev-output-tokens 1024 \
    --max-num-completed-requests ${completed} \
    --timeout 1200 \
    --num-concurrent-requests ${concurrent} \
    --results-dir "tmp" \
    --llm-api openai \
    --additional-sampling-params '{"max_tokens": 7192}' \
    --metadata dtype=${DTYPE},tp=${TP_SIZE},engine=${ENGINE}
  
  echo "Completed run ${i} of 8"
  echo "-----------------------------------------"
done

echo "All benchmark runs completed"