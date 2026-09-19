# Stage 1: 3D autoencoder

Train the KL autoencoder from this directory:

```bash
cd vqgan
bash train.sh
```

Edit `configs/autoencoder_kl.yaml` to set the data paths and training parameters; paths are relative to this directory.

Configurations, logs and checkpoints are written under `logs/`. Point `model.params.first_stage_config.params.ckpt_path` in `../mpad/configs/mpad.yaml` at the selected checkpoint to hand it to stage 2.

See the [root README](../README.md) for setup and the data format.
