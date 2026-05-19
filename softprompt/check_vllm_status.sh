#!/usr/bin/env bash

set -euo pipefail

BASE_PORT=8001
MODEL_NAME="Qwen3.5-35B"

echo "Checking vLLM services status..."
echo "================================"

check_service() {
    local port=$1
    local gpu_id=$((port - BASE_PORT))
    
    printf "GPU %d (Port %d): " ${gpu_id} ${port}
    
    if curl -s --connect-timeout 3 "http://localhost:${port}/health" >/dev/null 2>&1; then
        echo "✅ Running"
        
        # Check model info
        model_info=$(curl -s "http://localhost:${port}/v1/models" 2>/dev/null | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4 2>/dev/null || echo "Unknown")
        echo "    Model: ${model_info}"
    else
        echo "❌ Not responding"
    fi
}

for i in {0..7}; do
    port=$((BASE_PORT + i))
    check_service ${port}
done

echo ""
echo "Process status:"
ps aux | grep "vllm serve" | grep -v grep || echo "No vLLM processes found"

echo ""
echo "API endpoints:"
for i in {0..7}; do
    port=$((BASE_PORT + i))
    echo "  GPU ${i}: http://localhost:${port}/v1"
done
