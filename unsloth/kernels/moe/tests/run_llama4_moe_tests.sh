#!/bin/bash

set -euo pipefail

SEQLENS=(1024)  
DTYPES=(bfloat16)

for SEQLEN in "${SEQLENS[@]}"; do
    for DTYPE in "${DTYPES[@]}"; do

        ARGS=()
        ARGS+=(--seqlen "$SEQLEN" --dtype "$DTYPE")

        echo "Running with args: ${ARGS[*]}"
        if ! python -m tests.test_llama4_moe "${ARGS[@]}"; then
            echo "❌ Test failed with args: --seqlen=$SEQLEN --dtype=$DTYPE" >&2
        else
            echo "✅ Test passed with args: --seqlen=$SEQLEN --dtype=$DTYPE"
        fi

    done
done
