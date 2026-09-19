# MPAD &mdash; Official PyTorch implementation

This repository contains code for ECCV2026 paper **"Unified Multi-plane Autoregressive Diffusion for 3D Multi-Contrast MRI Synthesis"**.

Multi-Plane Autoregressive Diffusion (MPAD) compresses MRI volumes with a 3D KL-regularized autoencoder and synthesizes target contrasts with a 2D diffusion model that runs autoregressively across orthogonal latent planes.

## Repository layout

```text
MPAD/
├── environment.yaml        # One environment for both stages
├── requirements.txt
├── vqgan/                  # Stage 1: 3D autoencoder training
│   ├── configs/            # autoencoder_kl.yaml
│   ├── taming/
│   ├── main.py
│   └── train.sh
└── mpad/                   # Stage 2: latent diffusion training and inference
    ├── configs/mpad.yaml
    ├── ldm/
    ├── taming/
    ├── main.py
    ├── sample.py           # Source-only synthesis
    ├── evaluate.py         # Paired evaluation with metrics
    └── train.sh
```

The `vqgan` directory keeps its historical name from the upstream codebase. The paper's stage 1 is the **KL autoencoder** in `configs/autoencoder_kl.yaml`.

Each stage keeps its own `taming` package, so always run a stage's scripts from that stage's directory. `train.sh` does this for you; run `sample.py` and `evaluate.py` from `mpad/`. Do not `pip install` either stage as a package.

## Installation

```bash
conda env create -f environment.yaml
conda activate mpad
```

It installs Python 3.8, PyTorch 2.4.0 with CUDA 12.1, Lightning 1.4.2 and MONAI 1.2.0.

## Prepare data

Provide pre-aligned, preprocessed NIfTI volumes with subject-disjoint splits. Every filename must begin with the name of its enclosing subject directory:

```text
data/
├── train/subject001/subject001_t1.nii.gz
│                   subject001_t2.nii.gz
│                   subject001_pd.nii.gz
├── validation/subject002/...
└── test/subject003/...
```

| Dataset | Contrasts | Filename suffixes |
| --- | --- | --- |
| `adni` | `t1`, `t2`, `pd` | `_t1.nii.gz`, `_t2.nii.gz`, `_pd.nii.gz` |
| `ixi` | `t1`, `t2`, `pd` | `_T1.nii.gz`, `_T2.nii.gz`, `_PD.nii.gz` |

Set `data.params.<split>.params.dataset` explicitly for each split. Training and paired evaluation read every configured contrast, and all inputs must be aligned scalar 3D volumes.

## Stage 1: train the autoencoder

```bash
cd vqgan
bash train.sh
```

Edit `configs/autoencoder_kl.yaml` to set data paths and training parameters.

Configurations, logs and checkpoints are written under `logs/`. Copy the selected checkpoint to `MPAD/checkpoints/autoencoder.ckpt`, or point `model.params.first_stage_config.params.ckpt_path` in the stage 2 configuration at it.

## Stage 2: train MPAD

```bash
cd mpad
bash train.sh
```

Edit `configs/mpad.yaml` to set the stage 1 checkpoint and the train/validation/test directories. The first-stage network is frozen and MPAD learns masked latent slice reconstruction. Training starts from scratch unless you pass a resume argument; use `-r logs/<run-directory>` to continue a run.

## Synthesize a missing contrast

Run from `mpad/`. This entry point needs only the source volume:

```bash
python sample.py --config configs/mpad.yaml \
  --ckpt ../checkpoints/mpad.ckpt \
  --input ../data/test/subject003/subject003_t1.nii.gz \
  --source t1 --target t2 \
  --output ../outputs/subject003_t2.nii.gz
```

It follows the three-plane sequence and averages the resulting latents. It takes one source contrast per invocation at batch size one and supports T1, T2 and PD; run it again for another target. A full MPAD checkpoint already contains the autoencoder weights, so no separate stage 1 checkpoint is needed here.

## Paired evaluation

This route reads every contrast, including the target, and reports metrics:

```bash
python evaluate.py -b configs/mpad.yaml --ckpt ../checkpoints/mpad.ckpt \
  --source t1 --target t2 --ddim_eta 1.0 --max-subjects 100 \
  --outdir ../outputs/evaluation_t1_t2
```

It reports PSNR, SSIM and NMSE, and writes the three-plane average under `samples/`.

## Model Weights ##


|Dataset| Autoencoder | MPAD
 :---  |  :---  |  :---
IXI | [link](https://drive.google.com/file/d/120vwfHc2GFnoISa82XraoVBJA0QQw1Ci/view?usp=drive_link)  |[link](https://drive.google.com/file/d/1HyGu6mM6Y3gLxaZPrCW3UEeIpacREXZo/view?usp=drive_link)
ADNI| [link](https://drive.google.com/file/d/1BFEB1xZBWurrLh5VRStoPT3dXaFn_VIz/view?usp=sharing)  |[link](https://drive.google.com/file/d/12wEg_O2sCJ7m_PrW4ZCT4WmgBd_M-TXg/view?usp=sharing)<br>


## Acknowledgments

This implementation builds on CompVis latent-diffusion and taming-transformers; see [third-party notices](THIRD_PARTY_NOTICES.md).


