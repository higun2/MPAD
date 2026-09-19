# Stage 2: MPAD

Train MPAD from this directory:

```bash
cd mpad
bash train.sh
```

Set the frozen stage 1 checkpoint and the train/validation/test directories in `configs/mpad.yaml`. The first-stage network is frozen and MPAD learns masked latent slice reconstruction. Training starts from scratch unless you pass a resume argument; use `-r logs/<run-directory>` to continue a run.

Inference lives here as well: `sample.py` for source-only synthesis and `evaluate.py` for paired evaluation with metrics. Both need a trained MPAD checkpoint and are run from this directory.

See the [root README](../README.md) for setup and the complete inference commands.
