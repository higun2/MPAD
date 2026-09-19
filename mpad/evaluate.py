import argparse, os, sys, glob
import torch
import numpy as np
import nibabel as nib
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from main import DataModuleFromConfig
# import clip
import random
import torch.nn.functional as F
import open_clip



def seed_everything(see_num=1):
    os.environ["PL_GLOBAL_SEED"] = str(see_num)
    random.seed(see_num)
    np.random.seed(see_num)
    torch.manual_seed(see_num)
    torch.cuda.manual_seed(see_num)
    torch.cuda.manual_seed_all(see_num)
    os.environ["PL_SEED_WORKERS"] = f"{see_num}"
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# seed_everything()

@torch.no_grad()
def pick_two_most_similar(
    features,                    # list/tuple of length 3, each tensor shaped (1, 3, 32, 32, 32)
    metric: str = "cosine",      # "cosine" | "pearson" | "l2"
    channelwise: bool = False,   # whether to average the similarity per channel
    eps: float = 1e-8,
    return_similarity: bool = False
):
    assert len(features) == 3, "features must have length 3."
    # Check that all tensors share a device; otherwise compute on the first tensor's device
    device = features[0].device

    def _prep_vec(x):
        # x: (1, C, D, H, W)
        if channelwise:
            # (C, -1)
            v = x.view(x.shape[1], -1).to(device).float()
        else:
            # (-1,)
            v = x.reshape(-1).to(device).float()
        return v

    def _zscore(v):
        if channelwise:
            m = v.mean(dim=1, keepdim=True)
            s = v.std(dim=1, keepdim=True)
        else:
            m = v.mean()
            s = v.std()
        return (v - m) / (s + eps)

    def _cosine(a, b):
        if channelwise:
            a = F.normalize(a, p=2, dim=1)
            b = F.normalize(b, p=2, dim=1)
            # per-channel cosine similarity, then averaged
            sim_per_c = (a * b).sum(dim=1)       # (C,)
            return sim_per_c.mean()
        else:
            a = F.normalize(a, p=2, dim=0)
            b = F.normalize(b, p=2, dim=0)
            return (a * b).sum()

    def _pearson(a, b):
        return _cosine(_zscore(a), _zscore(b))  # equivalent to cosine similarity after z-scoring

    def _l2(a, b):
        # L2 distance after z-scoring; smaller is more similar
        return torch.norm(_zscore(a) - _zscore(b), p=2)

    vecs = [_prep_vec(f) for f in features]
    pairs = [(0,1), (0,2), (1,2)]

    scores = []
    for i, j in pairs:
        if metric == "cosine":
            s = _cosine(vecs[i], vecs[j])
        elif metric == "pearson":
            s = _pearson(vecs[i], vecs[j])
        elif metric == "l2":
            s = _l2(vecs[i], vecs[j])
        else:
            raise ValueError("metric must be one of 'cosine', 'pearson' or 'l2'.")
        scores.append(s)

    if metric == "l2":
        # the smallest distance is the most similar
        k = torch.argmin(torch.stack(scores)).item()
        best_score = scores[k].item()
    else:
        # the largest similarity is the most similar
        k = torch.argmax(torch.stack(scores)).item()
        best_score = scores[k].item()

    i, j = pairs[k]
    if return_similarity:
        return features[i], features[j], best_score
    else:
        return features[i], features[j]
    

def max_norm(x,max_value=1.0):
    return x / max_value

def fuse_by_softmax_cosine(
    gt,
    gens,
    temperature: float = 0.1,
    eps: float = 1e-8,
    mask_small_norm: bool = False,
    norm_thr: float = 1e-6,
    return_weights: bool = False,
):
    """
    gt:   (1, C, 32, 32, 32)
    gens: list of N generated feature tensors, each (1, C, 32, 32, 32)
    temperature: softmax temperature; lower values approach winner-take-all
    mask_small_norm: whether to exclude very low-norm voxels from the weights (-inf)
    Returns: fused (1, C, 32, 32, 32), optionally with weights (N, 32, 32, 32)
    """
    # (N, C, D, H, W)
    G = torch.stack([g.squeeze(0) for g in gens], dim=0)
    GT = gt.squeeze(0).unsqueeze(0).expand_as(G)

    # voxel-wise cosine sim: (N, D, H, W)
    sims = F.cosine_similarity(G, GT, dim=1, eps=eps)
    # sims = abs(sims)
    # Optionally mask very low-norm positions, where the weights are meaningless
    if mask_small_norm:
        gt_norm  = GT.norm(dim=1)     # (N, D, H, W); identical because GT is broadcast
        gen_norm = G.norm(dim=1)      # (N, D, H, W)
        mask = (gt_norm[0] < norm_thr) | (gen_norm < norm_thr)  # (N, D, H, W)
        sims = sims.masked_fill(mask, float('-inf'))

    # Softmax weights over the N axis, per voxel
    tau = max(temperature, 1e-8)
    # torch.softmax is numerically stable on its own; center the logits if needed
    weights = torch.softmax(sims / tau, dim=0)           # (N, D, H, W)

    # Weighted sum to produce the fused feature
    fused = (weights.unsqueeze(1) * G).sum(dim=0)        # (C, D, H, W)
    fused = fused.unsqueeze(0)                           # (1, C, D, H, W)

    return (fused, weights) if return_weights else fused

def voxelwise_cosine_torch(gt, gens, eps: float = 1e-8):
    """
    gt:  (1, 3, 32, 32, 32) reference tensor
    gens: list of 4 generated tensors, each (1, 3, 32, 32, 32)
    Returns: (4, 32, 32, 32), the voxel-wise cosine similarity volume per generated tensor
    """
    # (1, C, D, H, W) -> (C, D, H, W)
    gt = gt.squeeze(0)
    gens = [g.squeeze(0) for g in gens]               # each g: (3, D, H, W)
    gens_stacked = torch.stack(gens, dim=0)           # (4, 3, D, H, W)
    
    # Expand gt to the number of generated tensors: (4, 3, D, H, W)
    gt_expanded = gt.unsqueeze(0).expand_as(gens_stacked)
    
    # Cosine similarity over the channel dimension (dim=1) -> (4, D, H, W)
    sims = F.cosine_similarity(gens_stacked, gt_expanded, dim=1, eps=eps)
    return sims

def create_3d_gaussian_kernel(kernel_size=11, sigma=1.5, channels=1):
    """Build a 3D Gaussian kernel."""
    def gauss_1d(size, sigma):
        coords = torch.arange(size).float() - size // 2
        return torch.exp(-(coords**2) / (2 * sigma**2))

    g = gauss_1d(kernel_size, sigma)
    g = g / g.sum()
    g_3d = g[:, None, None] * g[None, :, None] * g[None, None, :]
    g_3d = g_3d.unsqueeze(0).unsqueeze(0)  # shape: (1,1,k,k,k)
    return g_3d.repeat(channels, 1, 1, 1, 1)

def ssim3D(x, y, kernel_size=11, sigma=1.5, data_range=1.0, C1=0.01**2, C2=0.03**2):
    """Compute 3D SSIM."""
    channels = x.size(1)
    window = create_3d_gaussian_kernel(kernel_size, sigma, channels).to(x.device)

    mu_x = F.conv3d(x, window, padding=kernel_size//2, groups=channels)
    mu_y = F.conv3d(y, window, padding=kernel_size//2, groups=channels)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv3d(x * x, window, padding=kernel_size//2, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv3d(y * y, window, padding=kernel_size//2, groups=channels) - mu_y_sq
    sigma_xy = F.conv3d(x * y, window, padding=kernel_size//2, groups=channels) - mu_xy

    C1 *= data_range ** 2
    C2 *= data_range ** 2

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    ssim_map = numerator / (denominator + 1e-8)
    return ssim_map.mean()

# Example usage
def calculate_ssim(x_samples_ddim, x_tgt, max_pixel=1.0):
    # Assumes x_samples_ddim and x_tgt are already torch tensors
    # Move to CPU if needed
    x_samples_ddim = x_samples_ddim.cpu()
    x_tgt = x_tgt.cpu()
    
    # Compute SSIM
    ssim_value = ssim3D(x_samples_ddim, x_tgt, data_range=max_pixel)
    
    return ssim_value.item()  # return as a scalar

def calculate_psnr(img1, img2, max_pixel=1.0):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def calculate_nmse(img1, img2):
    """Calculate NMSE (Normalized Mean Squared Error) between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    norm = torch.mean(img2 ** 2)
    if norm == 0:
        return float('inf')
    nmse = mse / norm
    return nmse.item()

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def save_nifti(img, path):
    img = img.squeeze(0)  # remove batch dimension, now it's (1, 192, 192, 160)
    if len(img.shape) != 4: return
    img = img.permute(1, 2, 3, 0)  # reorder dimensions to be compatible with nibabel

    img = img.numpy()

    os.makedirs(os.path.split(path)[0], exist_ok=True)

    nifti_img = nib.Nifti1Image(img, np.eye(4))  # you might want to replace np.eye(4) with the correct affine matrix
    nib.save(nifti_img, path)

def get_most_similar_codebook_entry(z_src: torch.Tensor, source_codebook: torch.Tensor, target_codebook: torch.Tensor, topk: int = 1):
    """
    z_src: (C, D, H, W) or (1, C, D, H, W)
    target_codebook: (K, C, D, H, W)
    return: (topk, C, D, H, W)
    """
    if z_src.dim() == 5:
        z_src = z_src.squeeze(0)  # (C, D, H, W)

    # Flatten: (C*D*H*W)
    z_src_flat = z_src.flatten().unsqueeze(0)  # (1, N)
    codebook_flat = source_codebook.view(target_codebook.shape[0], -1)  # (K, N)

    # Normalize
    # z_src_flat = F.normalize(z_src_flat, dim=1)
    # codebook_flat = F.normalize(codebook_flat, dim=1)

    # Cosine similarity: (K,)
    similarities = torch.matmul(codebook_flat, z_src_flat.T).squeeze(1)  # (K,)

    # top-k
    topk_vals, topk_idx = torch.topk(similarities, k=topk, largest=True)
    # print(target_codebook.shape)
    return target_codebook[topk_idx], topk_idx, topk_vals  # topk feature(s), index, and score

def match_feature_distribution(z_src: torch.Tensor, target_feat: torch.Tensor, mode='vector') -> torch.Tensor:
    """
    z_src, target_feat: (B, C, D, H, W)
    return: z_src matched to target_feat's distribution
    """

    src_c = z_src.clone().detach()
    tgt_c = target_feat.clone().detach()
        

    if mode == 'scalar':
        mu_s, std_s = src_c.mean(), src_c.std()
        mu_t, std_t = tgt_c.mean(), tgt_c.std()
    elif mode == 'vector':
        mu_s, std_s = src_c.mean(dim=(2,3,4), keepdim=True), src_c.std(dim=(2,3,4), keepdim=True)
        mu_t, std_t = tgt_c.mean(dim=(2,3,4), keepdim=True), tgt_c.std(dim=(2,3,4), keepdim=True)

    # z-score normalization, then scale to target stats
    normalized = (src_c - mu_s) / (std_s + 1e-8)
    return normalized * std_t + mu_t



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b",
        "--base",
        type=str,
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )

    parser.add_argument(
        "--source",
        type=str,
        nargs="+",
        default=["t1"],
        help="the source modality (select from t1, t1ce, t2, flair)",
    )

    parser.add_argument(
        "--target",
        type=str,
        nargs="?",
        default="t2",
        help="the target modality (select from t1, t1ce, t2, flair)",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=10,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=128,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=128,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--D",
        type=int,
        default=128,
        help="image depth, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10,
        help="timesteps to use for sampling",
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--max-subjects", type=int, default=0)
    opt, unknown = parser.parse_known_args()
    # Remaining arguments become OmegaConf dotlist overrides, so anything that is
    # not a key=value assignment is a typo rather than a silently ignored option.
    invalid = [arg for arg in unknown if "=" not in arg]
    if invalid:
        parser.error(
            "unrecognized arguments: {}. Configuration overrides must be written as "
            "key=value, for example data.params.test.params.data_path=../data/test".format(
                " ".join(invalid))
        )
    if opt.plms:
        parser.error("Autoregressive evaluation requires DDIM; PLMS is unsupported.")
    if len(opt.source) != 1 or "," in opt.source[0]:
        parser.error("This release supports one source contrast per invocation.")

    config = OmegaConf.merge(OmegaConf.load(opt.base), OmegaConf.from_dotlist(unknown))
    config.model.params.pop("ckpt_path", None)
    config.model.params.first_stage_config.params.pop("ckpt_path", None)
    config.data.params = {"batch_size": 1, "test": config.data.params.test}
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    data = data.datasets["test"]
    
    model = load_model_from_config(config, opt.ckpt)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    
    # clip_model, _ = clip.load("ViT-B/32")
    # clip_model = clip_model.to(model.device)
    # clip_model.eval()
    # tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    # clip_model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    # clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    clip_model, preprocess = open_clip.create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    clip_model = clip_model.to(model.device)
    clip_model.eval()
    
    
    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # modalities = ['t1', 't2', 'pd']

    sample_path = os.path.join(outpath, "samples")
    gt_path = os.path.join(outpath, "gt")
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples=list()
    psnr_list_1 = list()
    ssim_list_1 = list()
    nmse_list_1 = list()
    psnr_list_2 = list()
    ssim_list_2 = list()
    nmse_list_2 = list()
    psnr_list_3 = list()
    ssim_list_3 = list()
    nmse_list_3 = list()
    psnr_list_avg = list()
    ssim_list_avg = list()
    nmse_list_avg = list()
    psnr_list_fused = list()
    ssim_list_fused = list()
    nmse_list_fused = list()
    psnr_list_adapt = list()
    ssim_list_adapt = list()
    nmse_list_adapt = list()
    psnr_list_ar4 = list()
    ssim_list_ar4 = list()
    nmse_list_ar4 = list()
    opt.source = opt.source[0].split(',')  # First split the string inside the list
    
    max_list = list()
    
    # with torch.no_grad():
    with torch.no_grad(), model.ema_scope():
        for idx, batch in tqdm(enumerate(data), desc="Data",total=len(data)):
            if opt.max_subjects and idx >= opt.max_subjects:
                break
            # if idx != 99:
            #     continue
            subject_id = batch[0]["subject_id"]
            # x_src = batch[0][opt.source].unsqueeze(0).to(device)
            # Handle multiple source modalities
            z_srcs = []
            with torch.no_grad():
                for src in opt.source:
                    x_src = batch[0][src].unsqueeze(0).to(device)
                    x_src[x_src<0.0] = 0
                    # print(x_src.min(), x_src.max())
                    # x_src = torch.clamp(x_src,min=0.0,max=1.0)
                    z_src = model.first_stage_model.encode(x_src)
                    # z_tgtl, _, _ = model.first_stage_model.encode(x_src, opt.target)
                    z_src = model.get_first_stage_encoding(z_src).detach()
                    # print(z_src.min(), z_src.max())
                    z_src = torch.clamp(z_src, min=-15.0, max=15.0)
                    z_src = z_src*0.2
                    z_srcs.append(z_src)
                
            x_tgt = batch[0][opt.target].unsqueeze(0).to(device)
            x_tgt[x_tgt<0.0] = 0
            
            save_nifti(x_tgt.cpu(), os.path.join(gt_path, f"{subject_id}_{opt.target}.nii.gz"))
            # print(x_tgt.min(), x_tgt.max())
            # x_tgt[x_tgt>1.0] = 1.0
            x_tgt_max = x_tgt.max()
            max_list.append(x_tgt_max)
            # x_tgt = x_tgt / x_tgt.max()

            
            mask = x_tgt>0.0
            
            with torch.no_grad():
                z_tgt  = model.first_stage_model.encode(x_tgt)
                z_tgt = model.get_first_stage_encoding(z_tgt).detach()
                # print(z_tgt.min(), z_tgt.max())
                # x_tgt = torch.clamp(x_tgt, min=0.0, max=1.0)

                x_tar_prompt = batch[1][opt.target]
                x_tar_token = tokenizer(x_tar_prompt).to(device)
                x_tar_text_features = clip_model.encode_text(x_tar_token)
                x_tar_text_features = x_tar_text_features.float()
                
                x_src_prompt = batch[1][opt.source[0]]
                x_src_token = tokenizer(x_src_prompt).to(device)
                x_src_text_features = clip_model.encode_text(x_src_token)
                x_src_text_features = x_src_text_features.float()
            
            # x_tgt_recon = model.first_stage_model.decode(z_tgt)
            
            # psnr_value = calculate_psnr(x_tgt_recon.cpu(), x_tgt.cpu(), max_pixel=x_tgt.max())
            # print(f"PSNR for {subject_id} recon: {psnr_value:.2f} dB")
            # ssim_value_recon = calculate_ssim(x_tgt_recon.cpu(), x_tgt.cpu())
            # print(f"SSIM for {subject_id} recon: {ssim_value_recon:.2f}")  

            # z_tgtl = model.get_first_stage_encoding(z_tgtl).detach()

            # z_src = torch.cat([z_src, z_tgtl], dim=1)
        
            # x0 = z_src
            # x0 = torch.randn(z_src.shape, device=device)
            x0 = z_src.clone().detach()
            # print(torch.max(x0), torch.min(x0))
            # c = modalities.index(opt.target)
            # c = torch.nn.functional.one_hot(torch.tensor(c), num_classes=4).float()
            # c = c.unsqueeze(0).repeat(z_src.shape[0], 1).unsqueeze(1).to(device)
            shape = [3, opt.H//4, opt.W//4, opt.D//4]
            
            for j in range(1):
                samples_ddim, _ = sampler.sample_ar2_p(S=opt.ddim_steps,
                                                #  conditioning=c,
                                                    srcs = z_srcs,
                                                    target_prompt=x_tar_text_features,
                                                    source_prompt=x_src_text_features,
                                                    batch_size=opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    x0=x0,
                                                    eta=opt.ddim_eta,
                                                    timesteps=opt.timesteps,
                                                    # prior_feature=z_src_most_similar
                                                    )
                
                # print(samples_ddim.min(), samples_ddim.max())
                # samples_ddim = torch.clamp(samples_ddim, min=-3.0, max=3.0)
                samples_ddim *= 5.0
                
                # samples_ddim, _ = sampler.sample_ar_cons(S=opt.ddim_steps,
                #                                 #  conditioning=c,
                #                                     srcs = z_srcs,
                #                                     target_prompt=x_tar_text_features,
                #                                     source_prompt=x_src_text_features,
                #                                     batch_size=opt.n_samples,
                #                                     shape=shape,
                #                                     verbose=False,
                #                                     unconditional_guidance_scale=opt.scale,
                #                                     x0=samples_ddim,
                #                                     eta=opt.ddim_eta,
                #                                     timesteps=opt.timesteps)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                # x_tgt_recon = model.decode_first_stage(z_tgt)
                            # print(f"PSNR for {subject_id} recon: {psnr_value_recon:.2f} dB")
                # print(x_tgt_recon.min(), x_tgt_recon.max())
                # print(torch.max(x_tgt), torch.min(x_tgt))
                # print(torch.max(x_tgt_recon), torch.min(x_tgt_recon))
                # x_tgt_recon = (x_tgt_recon + 1.0)/2.0
                # x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0).detach().cpu()

                # x_samples_ddim = x_samples_ddim.detach().cpu()
                # x_samples_ddim = 0.5*(x_samples_ddim+1)
                
                # x_tgt = (x_tgt - x_tgt.min())/(x_tgt.max()-x_tgt.min())
                # x_samples_ddim = (x_samples_ddim - x_samples_ddim.min())/(x_samples_ddim.max()-x_samples_ddim.min())
                
                x_samples_ddim[x_samples_ddim<0.0] = 0
                # x_samples_ddim[x_samples_ddim>1.0] = 1.0
                x_samples_ddim = x_samples_ddim * mask
                # x_samples_ddim = x_samples_ddim / x_tgt_max.cpu()
                # x_samples_ddim = x_samples_ddim / x_samples_ddim.max()

                # Calculate PSNR
                # x_tgt_max_1 = x_tgt.max()
                # psnr_value = calculate_psnr(x_samples_ddim, x_tgt.cpu(), max_pixel=x_tgt_max_1)
                # print(f"PSNR for {subject_id}: {psnr_value:.4f} dB")
                # ssim_value = calculate_ssim(x_samples_ddim, x_tgt, max_pixel=x_tgt_max_1.item())
                # print(f"SSIM for {subject_id}: {ssim_value:.4f}")
                # nmse_value = calculate_nmse(x_samples_ddim, x_tgt.cpu())
                # print(f"NMSE for {subject_id}: {nmse_value:.4f}")
                # psnr_list_1.append(psnr_value)
                # ssim_list_1.append(ssim_value)
                # nmse_list_1.append(nmse_value)
                # print(f"Mean PSNR 1: {np.mean(psnr_list_1):.4f} dB")
                # print(f"Std  PSNR 1: {np.std(psnr_list_1):.4f} dB")
                # print(f"Mean SSIM 1: {np.mean(ssim_list_1):.4f}")
                # print(f"Std  SSIM 1: {np.std(ssim_list_1):.4f}")
                # print(f"Mean NMSE 1: {np.mean(nmse_list_1):.4f}")
                # print(f"Std  NMSE 1: {np.std(nmse_list_1):.4f}")
                
                x_samples_ddim_c = max_norm(x_samples_ddim, max_value=x_samples_ddim.max())
                # x_samples_ddim_c = max_norm(x_samples_ddim, max_value=x_tgt.max())
                x_tgt_c = max_norm(x_tgt, max_value=x_tgt.max())
                psnr_value = calculate_psnr(x_samples_ddim_c, x_tgt_c, max_pixel=1.0)
                print(f"PSNR for {subject_id}: {psnr_value:.4f} dB")
                # ssim_value = calculate_ssim(x_samples_ddim_c, x_tgt_c, max_pixel=1.0)
                # print(f"SSIM for {subject_id}: {ssim_value:.4f}")
                # nmse_value = calculate_nmse(x_samples_ddim_c, x_tgt_c)
                # print(f"NMSE for {subject_id}: {nmse_value:.4f}")
                # psnr_list_1.append(psnr_value)
                # ssim_list_1.append(ssim_value)
                # nmse_list_1.append(nmse_value)
                # print(f"Mean PSNR 1: {np.mean(psnr_list_1):.4f} dB")
                # print(f"Std  PSNR 1: {np.std(psnr_list_1):.4f} dB")
                # print(f"Mean SSIM 1: {np.mean(ssim_list_1):.4f}")
                # print(f"Std  SSIM 1: {np.std(ssim_list_1):.4f}")
                # print(f"Mean NMSE 1: {np.mean(nmse_list_1):.4f}")
                # print(f"Std  NMSE 1: {np.std(nmse_list_1):.4f}")
                
                x_samples_ddim = x_samples_ddim.cpu()
                # save_nifti(x_samples_ddim, os.path.join(sample_path, f"{subject_id}_{opt.source[0]}_to_{opt.target}_1.nii.gz"))


                samples_ddim2, _ = sampler.sample_ar3_p(S=opt.ddim_steps,
                                                #  conditioning=c,
                                                    srcs = z_srcs,
                                                    target_prompt=x_tar_text_features,
                                                    source_prompt=x_src_text_features,
                                                    batch_size=opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    x0=x0,
                                                    eta=opt.ddim_eta,
                                                    timesteps=opt.timesteps//3,
                                                    prior_feature=samples_ddim*0.2
                                                    )
                
                samples_ddim2 *= 5.0
                x_samples_ddim2 = model.decode_first_stage(samples_ddim2)
                # x_samples_ddim2 = x_samples_ddim2.detach().cpu()
                x_samples_ddim2[x_samples_ddim2<0.0] = 0
                x_samples_ddim2 = x_samples_ddim2 * mask
                # x_samples_ddim2 = x_samples_ddim2 / x_samples_ddim2.max()
                # x_samples_ddim2 = x_samples_ddim2 / x_tgt_max.cpu()
                
                
                # psnr_value2 = calculate_psnr(x_samples_ddim2, x_tgt.cpu(), max_pixel=x_tgt_max_1)
                # print(f"PSNR for {subject_id} AR2: {psnr_value2:.4f} dB")
                # ssim_value2 = calculate_ssim(x_samples_ddim2, x_tgt, max_pixel=x_tgt_max_1.item())
                # print(f"SSIM for {subject_id} AR2: {ssim_value2:.4f}")
                # nmse_value2 = calculate_nmse(x_samples_ddim2, x_tgt.cpu())
                # print(f"NMSE for {subject_id} AR2: {nmse_value2:.4f}")
                # psnr_list_2.append(psnr_value2)
                # ssim_list_2.append(ssim_value2)
                # nmse_list_2.append(nmse_value2)
                # print(f"Mean PSNR 2: {np.mean(psnr_list_2):.4f} dB")
                # print(f"Std  PSNR 2: {np.std(psnr_list_2):.4f} dB")
                # print(f"Mean SSIM 2: {np.mean(ssim_list_2):.4f}")
                # print(f"Std  SSIM 2: {np.std(ssim_list_2):.4f}")
                # print(f"Mean NMSE 2: {np.mean(nmse_list_2):.4f}")
                # print(f"Std  NMSE 2: {np.std(nmse_list_2):.4f}")
                
                x_samples_ddim2_c = max_norm(x_samples_ddim2, max_value=x_samples_ddim2.max())
                # x_samples_ddim2_c = max_norm(x_samples_ddim2, max_value=x_tgt.max())
                x_tgt_c = max_norm(x_tgt, max_value=x_tgt.max())
                psnr_value2 = calculate_psnr(x_samples_ddim2_c, x_tgt_c, max_pixel=1.0)
                print(f"PSNR for {subject_id} AR2: {psnr_value2:.4f} dB")
                # ssim_value2 = calculate_ssim(x_samples_ddim2_c, x_tgt_c, max_pixel=1.0)
                # print(f"SSIM for {subject_id} AR2: {ssim_value2:.4f}")
                # nmse_value2 = calculate_nmse(x_samples_ddim2_c, x_tgt_c)
                # print(f"NMSE for {subject_id} AR2: {nmse_value2:.4f}")
                # psnr_list_2.append(psnr_value2)
                # ssim_list_2.append(ssim_value2)
                # nmse_list_2.append(nmse_value2)
                # print(f"Mean PSNR 2: {np.mean(psnr_list_2):.4f} dB")
                # print(f"Std  PSNR 2: {np.std(psnr_list_2):.4f} dB")
                # print(f"Mean SSIM 2: {np.mean(ssim_list_2):.4f}")
                # print(f"Std  SSIM 2: {np.std(ssim_list_2):.4f}")
                # print(f"Mean NMSE 2: {np.mean(nmse_list_2):.4f}")
                # print(f"Std  NMSE 2: {np.std(nmse_list_2):.4f}")
                
                samples_ddim3, _ = sampler.sample_ar(S=opt.ddim_steps,
                                                #  conditioning=c,
                                                    srcs = z_srcs,
                                                    target_prompt=x_tar_text_features,
                                                    source_prompt=x_src_text_features,
                                                    batch_size=opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    x0=x0,
                                                    eta=opt.ddim_eta,
                                                    timesteps=opt.timesteps//3,
                                                    prior_feature=samples_ddim*0.1+samples_ddim2*0.1
                                                    )
                samples_ddim3 *= 5.0
                x_samples_ddim3 = model.decode_first_stage(samples_ddim3)
                # x_samples_ddim3 = x_samples_ddim3.detach().cpu()
                x_samples_ddim3[x_samples_ddim3<0.0] = 0
                x_samples_ddim3 = x_samples_ddim3 * mask


                x_samples_ddim3_c = max_norm(x_samples_ddim3, max_value=x_samples_ddim3.max())
                # x_samples_ddim3_c = max_norm(x_samples_ddim3, max_value=x_tgt.max())
                x_tgt_c = max_norm(x_tgt, max_value=x_tgt.max())
                psnr_value3 = calculate_psnr(x_samples_ddim3_c, x_tgt_c, max_pixel=1.0)
                print(f"PSNR for {subject_id} AR3: {psnr_value3:.4f} dB")  
                
                samples_avg_0 = (samples_ddim + samples_ddim2 + samples_ddim3) / 3
                ##############################
                samples_ddim4, _ = sampler.sample_ar2_p(S=opt.ddim_steps,
                                                #  conditioning=c,
                                                    srcs = z_srcs,
                                                    target_prompt=x_tar_text_features,
                                                    source_prompt=x_src_text_features,
                                                    batch_size=opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    x0=x0,
                                                    eta=opt.ddim_eta,
                                                    timesteps=opt.timesteps//3,
                                                    prior_feature=samples_avg_0 * 0.2
                                                    )
                samples_ddim4 *= 5.0
                x_samples_ddim4 = model.decode_first_stage(samples_ddim4)
                # x_samples_ddim3 = x_samples_ddim3.detach().cpu()
                x_samples_ddim4[x_samples_ddim4<0.0] = 0
                x_samples_ddim4 = x_samples_ddim4 * mask


                x_samples_ddim4_c = max_norm(x_samples_ddim4, max_value=x_samples_ddim4.max())
                # x_samples_ddim3_c = max_norm(x_samples_ddim3, max_value=x_tgt.max())
                x_tgt_c = max_norm(x_tgt, max_value=x_tgt.max())
                psnr_value_ar4 = calculate_psnr(x_samples_ddim4_c, x_tgt_c, max_pixel=1.0)
                print(f"PSNR for {subject_id} AR4: {psnr_value_ar4:.4f} dB")
                # ssim_value_ar4 = calculate_ssim(x_samples_ddim4_c, x_tgt_c, max_pixel=1.0)
                # print(f"SSIM for {subject_id} AR4: {ssim_value_ar4:.4f}")
                # nmse_value_ar4 = calculate_nmse(x_samples_ddim4_c, x_tgt_c)
                # print(f"NMSE for {subject_id} AR4: {nmse_value_ar4:.4f}")
                
                # psnr_list_ar4.append(psnr_value_ar4)
                # ssim_list_ar4.append(ssim_value_ar4)
                # nmse_list_ar4.append(nmse_value_ar4)
                # print(f"Mean PSNR AR4: {np.mean(psnr_list_ar4):.4f} dB")
                # print(f"Std  PSNR AR4: {np.std(psnr_list_ar4):.4f} dB")
                # print(f"Mean SSIM AR4: {np.mean(ssim_list_ar4):.4f}")
                # print(f"Std  SSIM AR4: {np.std(ssim_list_ar4):.4f}")
                # print(f"Mean NMSE AR4: {np.mean(nmse_list_ar4):.4f}")
                # print(f"Std  NMSE AR4: {np.std(nmse_list_ar4):.4f}")
                
                ##############################
                
                # samples_ddim_avg = (samples_ddim + samples_ddim2 + samples_ddim3 + samples_ddim4) / 4.0
                samples_ddim_avg = (samples_ddim + samples_ddim2 + samples_ddim3) / 3.0

                # samples_ddim_avg = (samples_ddim + samples_ddim2) / 2.0

                x_samples_ddim_avg = model.decode_first_stage(samples_ddim_avg)
                # x_samples_ddim_avg = x_samples_ddim_avg.detach().cpu()
                x_samples_ddim_avg[x_samples_ddim_avg<0.0] = 0
                x_samples_ddim_avg = x_samples_ddim_avg * mask
                                
                x_samples_ddim_avg = max_norm(x_samples_ddim_avg, max_value=x_samples_ddim_avg.max())
                # x_samples_ddim_adapt_c = max_norm(x_samples_ddim_adapt, max_value=x_tgt.max())
                x_tgt_c = max_norm(x_tgt, max_value=x_tgt.max())
                
                psnr_value_avg = calculate_psnr(x_samples_ddim_avg, x_tgt_c, max_pixel=1.0)
                print(f"PSNR for {subject_id} AVG: {psnr_value_avg:.4f} dB")
                ssim_value_avg = calculate_ssim(x_samples_ddim_avg, x_tgt_c, max_pixel=1.0)
                print(f"SSIM for {subject_id} AVG: {ssim_value_avg:.4f}")
                nmse_value_avg = calculate_nmse(x_samples_ddim_avg, x_tgt_c)
                print(f"NMSE for {subject_id} AVG: {nmse_value_avg:.4f}")
                psnr_list_avg.append(psnr_value_avg)
                ssim_list_avg.append(ssim_value_avg)
                nmse_list_avg.append(nmse_value_avg)
                print(f"Mean PSNR AVG: {np.mean(psnr_list_avg):.4f} dB")
                print(f"Std  PSNR AVG: {np.std(psnr_list_avg):.4f} dB")
                print(f"Mean SSIM AVG: {np.mean(ssim_list_avg):.4f}")
                print(f"Std  SSIM AVG: {np.std(ssim_list_avg):.4f}")
                print(f"Mean NMSE AVG: {np.mean(nmse_list_avg):.4f}")
                print(f"Std  NMSE AVG: {np.std(nmse_list_avg):.4f}")
                
                x_samples_ddim_avg_cpu = x_samples_ddim_avg.cpu().detach()
                save_nifti(x_samples_ddim_avg_cpu, os.path.join(sample_path, f"{subject_id}_{opt.source[0]}_to_{opt.target}.nii.gz"))

                
                # sims = voxelwise_cosine_torch(z_src*5, [z_tgt, samples_ddim,samples_ddim2,samples_ddim3,samples_ddim_avg])
                # print(sims[0,10,10,10])
                # print(sims[1,10,10,10])
                # print(sims[2,10,10,10])
                # print(sims[3,10,10,10])
                # print(sims[4,10,10,10])
                
                # sim_f1, sim_f2 = pick_two_most_similar([samples_ddim,samples_ddim2,samples_ddim3], metric="cosine", channelwise=True)

                fused, w = fuse_by_softmax_cosine(samples_ddim_avg, [samples_ddim, samples_ddim2,samples_ddim3,samples_ddim4],
                                  temperature=0.1,
                                  mask_small_norm=True,
                                  return_weights=True)
                
                # fused, w = fuse_by_softmax_cosine(z_src*5, [samples_ddim,samples_ddim2,samples_ddim3],
                #                   temperature=0.1,
                #                   mask_small_norm=True,
                #                   return_weights=True)
                
                x_samples_ddim_fused = model.decode_first_stage(fused)
                # x_samples_ddim_avg = x_samples_ddim_avg.detach().cpu()
                x_samples_ddim_fused[x_samples_ddim_fused<0.0] = 0
                x_samples_ddim_fused = x_samples_ddim_fused * mask
                                
                x_samples_ddim_fused = max_norm(x_samples_ddim_fused, max_value=x_samples_ddim_fused.max())
                # x_samples_ddim_adapt_c = max_norm(x_samples_ddim_adapt, max_value=x_tgt.max())
                x_tgt_c = max_norm(x_tgt, max_value=x_tgt.max())
                
                psnr_value_fused = calculate_psnr(x_samples_ddim_fused, x_tgt_c, max_pixel=1.0)
                print(f"PSNR for {subject_id} Fused: {psnr_value_fused:.4f} dB")
                ssim_value_fused = calculate_ssim(x_samples_ddim_fused, x_tgt_c, max_pixel=1.0)
                print(f"SSIM for {subject_id} Fused: {ssim_value_fused:.4f}")
                nmse_value_fused = calculate_nmse(x_samples_ddim_fused, x_tgt_c)
                print(f"NMSE for {subject_id} Fused: {nmse_value_fused:.4f}")
                
                psnr_list_fused.append(psnr_value_fused)
                ssim_list_fused.append(ssim_value_fused)
                nmse_list_fused.append(nmse_value_fused)
                print(f"Mean PSNR Fused: {np.mean(psnr_list_fused):.4f} dB")
                print(f"Std  PSNR Fused: {np.std(psnr_list_fused):.4f} dB")
                print(f"Mean SSIM Fused: {np.mean(ssim_list_fused):.4f}")
                print(f"Std  SSIM Fused: {np.std(ssim_list_fused):.4f}")
                print(f"Mean NMSE Fused: {np.mean(nmse_list_fused):.4f}")
                print(f"Std  NMSE Fused: {np.std(nmse_list_fused):.4f}")
                
            

    print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")
