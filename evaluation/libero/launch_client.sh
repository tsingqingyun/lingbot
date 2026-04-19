#!/bin/bash

START=${START:-0}
END=${END:-10}
LIBERO_BENCHMARK=${LIBERO_BENCHMARK:-libero_10}
TEST_NUM=${TEST_NUM:-50}
PORT=${PORT:-29056}
OUT_DIR=${OUT_DIR:-outputs/libero}

python -m evaluation.libero.client \
    --libero-benchmark "${LIBERO_BENCHMARK}" \
    --port "${PORT}" \
    --test-num "${TEST_NUM}" \
    --task-range "${START}" "${END}" \
    --out-dir "${OUT_DIR}"
