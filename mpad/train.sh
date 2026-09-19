#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec python main.py -b configs/mpad.yaml -t True --no-test True --gpus "${GPUS:-0,}" "$@"
