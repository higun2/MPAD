"""Source-only, three-plane MRI synthesis in the preprocessed 128-cubed grid."""
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/mpad.yaml')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--input', required=True, help='Pre-aligned source NIfTI volume')
    parser.add_argument('--source', choices=['t1', 't2', 'pd'], required=True)
    parser.add_argument('--target', choices=['t1', 't2', 'pd'], required=True)
    parser.add_argument('--output', required=True, help='Output .nii.gz in the preprocessed grid')
    parser.add_argument('--ddim-steps', type=int, default=10)
    parser.add_argument('--timesteps', type=int, default=10)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', choices=['adni', 'ixi'], default='adni')
    args = parser.parse_args()
    if args.source == args.target:
        parser.error('Source and target must differ.')
    if not 6 <= args.timesteps <= args.ddim_steps < 1000:
        parser.error('Require 6 <= timesteps <= ddim-steps < 1000 for three-plane sampling.')
    return args


def main():
    args = parse_args()
    import numpy as np
    import nibabel as nib
    import torch
    from omegaconf import OmegaConf
    from pytorch_lightning import seed_everything
    from ldm.util import instantiate_from_config
    from ldm.data.custom import build_transform, PROMPTS
    from ldm.models.diffusion.ddim import DDIMSampler

    if not torch.cuda.is_available():
        raise RuntimeError('The supplied MPAD sampler requires a CUDA GPU.')
    seed_everything(args.seed)
    config = OmegaConf.load(args.config)
    config.model.params.pop('ckpt_path', None)
    # Full diffusion checkpoints already contain the frozen first-stage weights.
    config.model.params.first_stage_config.params.pop('ckpt_path', None)
    model = instantiate_from_config(config.model)
    state = torch.load(args.ckpt, map_location='cpu')['state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    # CLIP is loaded independently from its pretrained source by the model.
    critical = [key for key in missing if not key.startswith('clip.')]
    if critical or unexpected:
        raise RuntimeError(f'Checkpoint mismatch: missing={critical}, unexpected={unexpected}')
    model = model.cuda().eval()
    percentile, clip = (99.95, True) if args.dataset == 'adni' else (99.75, False)
    item = build_transform([args.source], percentile, clip)({args.source: args.input})
    volume = item[args.source]
    affine = volume.affine.cpu().numpy()
    x = volume.as_tensor().unsqueeze(0).cuda()
    sampler = DDIMSampler(model)
    with torch.no_grad(), model.ema_scope():
        z = model.get_first_stage_encoding(model.first_stage_model.encode(x)).detach()
        z = z.clamp(-15, 15) * 0.2
        if tuple(z.shape[1:]) != (3, 32, 32, 32):
            raise ValueError(f'Expected latent shape (3,32,32,32), got {tuple(z.shape[1:])}')
        def text_features(modality):
            tokens = model.tokenizer(PROMPTS[modality]).to(model.device)
            return model.clip.encode_text(tokens).float()
        common = dict(S=args.ddim_steps, srcs=[z], target_prompt=text_features(args.target),
                      source_prompt=text_features(args.source), batch_size=1,
                      shape=[3, 32, 32, 32], verbose=False,
                      unconditional_guidance_scale=5.0, x0=z.clone(), eta=args.eta)
        a, _ = sampler.sample_ar2_p(**common, timesteps=args.timesteps)
        b, _ = sampler.sample_ar3_p(**common, timesteps=args.timesteps // 3, prior_feature=a)
        c, _ = sampler.sample_ar(**common, timesteps=args.timesteps // 3,
                                 prior_feature=(a + b) / 2)
        result = model.decode_first_stage((a + b + c) / 3 * 5.0).clamp(min=0)
        result = result / result.amax().clamp(min=1e-8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(result[0, 0].cpu().numpy().astype(np.float32), affine), str(output))
    print(f'Saved {output} (128 x 128 x 128, preprocessed RAI grid)')


if __name__ == '__main__':
    main()
