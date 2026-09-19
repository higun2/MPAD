"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from torch.autograd import grad
from torch import nn
from einops import rearrange

def decalcomanie_groups(n, nrows, reverse=False):
    assert n % nrows == 0, "n must be divisible by nrows."
    row_len = n // nrows
    assert row_len % 2 == 0, "each row must have an even length."

    mid = n // 2
    half = row_len // 2
    rows = []
    for j in range(nrows):
        left_start = mid - (j + 1) * half
        left = torch.arange(left_start, left_start + half, dtype=torch.long)
        right_start = mid + j * half
        right = torch.arange(right_start, right_start + half, dtype=torch.long)
        rows.append(torch.cat([left, right], dim=0))

    groups = torch.stack(rows, dim=0)
    if reverse:
        groups = torch.flipud(groups)  # reverse the row order
    return groups

class Fusion(nn.Module):
    def __init__(self, in_channels, kernel_size=3, nhidden=64, norm_type='instance'):
        super().__init__()



        pw = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv3d(in_channels, nhidden, kernel_size=kernel_size, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv3d(nhidden, in_channels, kernel_size=kernel_size, padding=pw)
        self.mlp_beta = nn.Conv3d(nhidden, in_channels, kernel_size=kernel_size, padding=pw)
        self.act = nn.Tanh

    def forward(self, x):  # Added segmap parameter
        # Get modulation parameters
        actv = self.mlp_shared(x)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # print("#"*50)
        # print(x.min(),x.max())
        # print(gamma.min(),gamma.max())
        # print(beta.min(),beta.max())
        # print("#"*50)
        # Apply scale and bias
        out = x * (1 + 0.001*gamma) + beta * 0.001

        return out
    
def entropy_loss_soft_3d(x,
                         num_bins: int = 256,
                         vmin: float = 0.0,
                         vmax: float = 2.0,
                         sigma: float = None,
                         eps: float = 1e-12,
                         reduce: str = "mean"):
    """
    x: (B, C, D, H, W) float tensor with values in [vmin, vmax]
    num_bins: number of histogram bins
    vmin, vmax: lower and upper value bounds, used as given without normalization
    sigma: Gaussian width of the soft histogram in the value domain; defaults to half the bin spacing.
    reduce: "mean" to average over the batch, or "none"
    Returns: entropy loss; lower values mean a more peaked distribution
    """
    assert x.dim() == 5, "expected (B,C,D,H,W)"
    B = x.size(0)

    # Clamp to the value range, so outliers cannot derail training
    x = torch.clamp(x, vmin, vmax)

    # Bin centers, evenly spaced in the value domain
    device = x.device
    dtype = x.dtype
    bin_centers = torch.linspace(vmin, vmax, steps=num_bins, device=device, dtype=dtype)  # (K,)
    bin_centers = bin_centers.view(1, 1, num_bins)  # (1,1,K)

    # Default sigma: half the bin spacing is a stable choice
    if sigma is None:
        bin_width = (vmax - vmin) / (num_bins - 1)
        sigma = 0.5 * bin_width

    # Soft histogram by broadcasting (B, N, 1) against (1, 1, K)
    x_flat = x.view(B, -1, 1)  # (B, N, 1)
    dist2 = (x_flat - bin_centers) ** 2                 # (B, N, K)
    weights = torch.exp(-0.5 * dist2 / (sigma ** 2))    # (B, N, K)

    # Sum the weights per bin to form the histogram
    hist = weights.sum(dim=1) + eps                     # (B, K)
    p = hist / (hist.sum(dim=1, keepdim=True) + eps)    # probability distribution

    # Shannon entropy (nats)
    ent = -(p * (p + eps).log()).sum(dim=1)             # (B,)

    if reduce == "mean":
        return ent.mean()
    return ent  # per-sample



class HybridGatingBlock3D_Independent(nn.Module):
    def __init__(self, in_channels=3, inter_channels=16, shared_encoder=True):
        super().__init__()
        self.shared_encoder = shared_encoder

        self.conv_a = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        if not shared_encoder:
            self.conv_b = copy.deepcopy(self.conv_a)
            self.conv_c = copy.deepcopy(self.conv_a)

        self.fusion = nn.Sequential(
            nn.Conv3d(inter_channels * 3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.gate_conv = nn.Conv3d(32, 3, kernel_size=1)

    def forward(self, x_a, x_b, x_c):
        if self.shared_encoder:
            f_a = self.conv_a(x_a)
            f_b = self.conv_a(x_b)
            f_c = self.conv_a(x_c)
        else:
            f_a = self.conv_a(x_a)
            f_b = self.conv_b(x_b)
            f_c = self.conv_c(x_c)

        x_cat = torch.cat([f_a, f_b, f_c], dim=1)
        fused = self.fusion(x_cat)
        weights = F.softmax(self.gate_conv(fused), dim=1).unsqueeze(2)
        z_T = weights[:,0]*f_a + weights[:,1]*f_b + weights[:,2]*f_c
        return z_T, weights
    
    
class HybridGatingBlock3D(nn.Module):
    def __init__(self, in_channels=3, inter_channels=16, channel_wise=False):
        """
        Args:
            in_channels (int): Number of input channels per direction
            inter_channels (int): Internal feature channels
            channel_wise (bool): Whether to use per-channel gating weights
        """
        super().__init__()
        self.channel_wise = channel_wise

        # # Direction-specific feature extractors
        # self.conv_a = nn.Sequential(
        #     nn.Conv3d(in_channels, inter_channels, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv_b = nn.Sequential(
        #     nn.Conv3d(in_channels, inter_channels, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv_c = nn.Sequential(
        #     nn.Conv3d(in_channels, inter_channels, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True)
        # )

        # Shared fusion
        self.fusion = nn.Sequential(
            nn.Conv3d(in_channels * 3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Gating map prediction
        if self.channel_wise:
            self.gate_conv = nn.Conv3d(32, 3 * in_channels, kernel_size=1)  # [B, 3*C, D, H, W]
        else:
            self.gate_conv = nn.Conv3d(32, 3, kernel_size=1)  # [B, 3, D, H, W]

    def forward(self, x_a, x_b, x_c):
        B, C, D, H, W = x_a.shape

        # # 1. Extract per-direction features
        # f_a = self.conv_a(x_a)  # [B, C', D, H, W]
        # f_b = self.conv_b(x_b)
        # f_c = self.conv_c(x_c)

        # 2. Fuse context
        x_cat = torch.cat([x_a, x_b, x_c], dim=1)  # [B, 3C', D, H, W]
        fused = self.fusion(x_cat)                # [B, 32, D, H, W]

        # 3. Predict gating weights
        gate_raw = self.gate_conv(fused)

        if self.channel_wise:
            # [B, 3*C, D, H, W] → [B, 3, C, D, H, W]
            gates = gate_raw.view(B, 3, C, D, H, W)
            weights = F.softmax(gates, dim=1)  # softmax over directions
            # print("#"*50)
            # print(weights[0,0,0,10,10,10])
            # print(weights[0,1,0,10,10,10])
            # print(weights[0,2,0,10,10,10])
            # print("#"*50)
            # Weighted sum per channel
            z_T = (
                weights[:, 0] * x_a +  # [B, C, D, H, W]
                weights[:, 1] * x_b +
                weights[:, 2] * x_c
            )
        else:
            # [B, 3, D, H, W] → [B, 3, 1, D, H, W] for broadcast
            weights = F.softmax(gate_raw, dim=1).unsqueeze(2)
            # print(weights[0,0,0])
            # Weighted sum (same weight for all channels)
            z_T = (
                weights[:, 0] * x_a +
                weights[:, 1] * x_b +
                weights[:, 2] * x_c
            )

        return z_T, weights
    

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def sample_ar_adapt(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0_1=None,
               x0_2=None,
               x0_3=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               target_prompt=None,
               source_prompt=None,
               srcs=None,
               timesteps=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        
        
        # Initialize learnable parameters with same shape as x0_1 and initial value of 1/3
        
        # w3 = 1 - w1 - w2
        
        # cnn_1 = torch.nn.Conv3d(256, 256, 1, padding=0)
        # cnn_1 = cnn_1.to(self.model.device)
        cnn_1 = None
        # Initialize optimizer
        # optimizer = torch.optim.Adam([w1, w2, w3, *cnn_1.parameters()], lr=0.1)
        
        ##########################################################
        # w1 = torch.nn.Parameter(torch.ones_like(x0_1))
        # w2 = torch.nn.Parameter(torch.ones_like(x0_2))
        # w3 = torch.nn.Parameter(torch.ones_like(x0_3))
        # optimizer = torch.optim.Adam([w1, w2, w3], lr=0.03)
        
        # # Combine x0_1, x0_2, x0_3 with learnable weights
        # weights = torch.stack([w1, w2, w3], dim=0)
        # weights = F.softmax(weights, dim=0)
        # x0 = weights[0] * x0_1 + weights[1] * x0_2 + weights[2] * x0_3
        ##########################################################
        
        ##########################################################
        
        # gating_block = nn.Sequential(
        #     nn.Conv3d(9, 32, kernel_size=1, padding=0),
        #     # nn.Conv3d(32, 32, kernel_size=1, padding=0),
        #     # nn.ReLU(),
        #     nn.Conv3d(32, 3*3, kernel_size=1),
        #     # nn.ReLU()
        # )
        # gating_block = gating_block.to(self.model.device)
        
        gating_block = HybridGatingBlock3D(in_channels=3, channel_wise=True)
        # gating_block = HybridGatingBlock3D_Independent(in_channels=3, channel_wise=True)

        gating_block = gating_block.to(self.model.device)
        
        optimizer = torch.optim.SGD([*gating_block.parameters()], lr=0.4)
        x0 = x0_1.clone().detach()
        
        #############################################################
        
        
        # gating_block = nn.Sequential(
        #     nn.Conv3d(3*3, 16, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv3d(16, 3, kernel_size=1)
        # )
        
        # optimizer = torch.optim.Adam([*gating_block.parameters()], lr=0.1)
        # features = torch.cat([x0_1, x0_2, x0_3], dim=1)  # [B, 3C, D, H, W]
        # gating_output = gating_block(features)
        # gating_output = F.softmax(gating_output, dim=1)
        # x0 = gating_output[:, 0] * x0_1 + gating_output[:, 1] * x0_2 + gating_output[:, 2] * x0_3
        
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W, D = shape
        size = (batch_size, C, H, W, D)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling_ar_adapt(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    target_prompt=target_prompt,
                                                    source_prompt=source_prompt,
                                                    sources=srcs,
                                                    timesteps=timesteps,
                                                    optimizer=optimizer,
                                                    # weights=[w1, w2, w3],
                                                    x0_inputs=[x0_1, x0_2, x0_3],
                                                    cnn_1=cnn_1,
                                                    gating_block=gating_block
                                                    )
        return samples, intermediates
    
    def ddim_sampling_ar_adapt(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, target_prompt=None, source_prompt=None, sources=None,
                      optimizer=None, weights=None, x0_inputs=None, cnn_1=None, gating_block=None):
        device = self.model.betas.device
        b = shape[0]
        # img = torch.cat([x0.to(device), torch.randn((x0.shape[0], x0.shape[1]//2, *shape[2:])).to(device)], dim=1)
        # img = x0.to(device)

        # img = torch.randn(*shape[:]).to(device)
        img = sources[0].clone().detach()
        # timesteps = None
        # timesteps = 10
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            # timesteps = 100
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        x_start_list = []
        # Create tensor of indices 0-31
        indices = torch.arange(32)
        # Shuffle the indices
        # shuffled_indices = indices[torch.randperm(len(indices))]

        for i, step in enumerate(iterator):
            # if i < total_steps//3*2:
            # if i < total_steps//2:
            #     continue
            for j in range(1):
                shuffled_indices = indices[torch.randperm(len(indices))]
                g = shuffled_indices[:8]
                g2 = shuffled_indices[:16]
                # g = shuffled_indices[:]
                # g.sort()
                g = torch.sort(g)[0]
                g2 = torch.sort(g2)[0]

                noisy = torch.randn(*shape[:]).to(device)

                index = total_steps - i - 1
                # ts = torch.full((b,), step, device=device, dtype=torch.long)
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                # t_fix = torch.full((b,), 100, device=device, dtype=torch.long)
                
                ################################################################
                # features = torch.cat([x0_inputs[0], x0_inputs[1], x0_inputs[2]], dim=1)  # [B, 3C, D, H, W]
                # gating_output = gating_block(features)
                # gating_reshape = gating_output.reshape(b, 3, 3, gating_output.shape[2], gating_output.shape[3], gating_output.shape[4])
                # gating_output_s = F.softmax(gating_reshape, dim=1)
                # gating_output_s = F.sigmoid(gating_reshape)
                

                # # print(gating_output_s.shape)
                # # print(torch.sum(gating_output_s, dim=1))
                # # print(gating_output_s[:,0].min(), gating_output_s[:,0].max())
                # x0 = gating_output_s[:, 0] * x0_inputs[0] + gating_output_s[:, 1] * x0_inputs[1] + gating_output_s[:, 2] * x0_inputs[2]
                ################################################################
                x0, _ = gating_block(x0_inputs[0], x0_inputs[1], x0_inputs[2])
                
                ################################################################
                # weights_s = torch.stack(weights, dim=0)
                # weights_s = F.softmax(weights_s, dim=0)
                # # # print(weights_s[0])
                # x0 = weights_s[0] * x0_inputs[0] + weights_s[1] * x0_inputs[1] + weights_s[2] * x0_inputs[2]
                ################################ view ###################################
                # axis_to_mask = 2
                axis_to_mask = torch.randint(2, 5, (1,)).item()
                
                y = torch.zeros(img.shape[0], dtype=torch.long, device=device)  # 0: axial
                # if axis_to_mask == 2:
                #     y = torch.zeros(img.shape[0], dtype=torch.long, device=device)  # 0: axial
                if axis_to_mask == 3:
                    y = torch.ones(img.shape[0], dtype=torch.long, device=device)   # 1: coronal
                elif axis_to_mask == 4:
                    y = torch.full((img.shape[0],), 2, dtype=torch.long, device=device)  # 2: sagittal
        
        
                # Create a list of slice objects for all dimensions
                slices = [slice(None)] * 5  # 5 dimensions: [batch, channel, height, width, depth]
                slices[axis_to_mask] = g
                
                img1 = img.clone().detach()
                img_ar  = img1[tuple(slices)].clone().detach()
                img1[tuple(slices)] = noisy[tuple(slices)].clone().detach()
                src_ar = torch.cat([x0, img1], dim=1)

                img_ar1 = self.model.q_sample(img_ar, ts)
                outs = self.p_sample_ddim_cons(img_ar1, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        sources=src_ar,
                                        target_prompt=source_prompt,
                                        uncertainty_map=tuple(slices),
                                        # cnn_1=cnn_1,
                                        y=y)
                
                x_prev, pred_x0 = outs
                
                loss_1 = torch.norm(pred_x0 - img_ar)
                
                # axis_to_mask2 = torch.randint(2, 5, (1,)).item()
                # y2 = torch.zeros(img.shape[0], dtype=torch.long, device=device)  # 0: axial
                # if axis_to_mask2 == 3:
                #     y2 = torch.ones(img.shape[0], dtype=torch.long, device=device)   # 1: coronal
                # elif axis_to_mask2 == 4:
                #     y2 = torch.full((img.shape[0],), 2, dtype=torch.long, device=device)  # 2: sagittal
                
                slices2 = [slice(None)] * 5  # 5 dimensions: [batch, channel, height, width, depth]
                slices2[axis_to_mask] = g2
                
                img2 = img.clone().detach()
                # img_ar2  = img2[tuple(slices2)].clone().detach()
                img2[tuple(slices2)] = noisy[tuple(slices2)].clone().detach()
                src_ar2 = torch.cat([x0, img2], dim=1)
                
                # img_ar2 = self.model.q_sample(img_ar2, ts)
                outs2 = self.p_sample_ddim_cons(img_ar, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        sources=src_ar2,
                                        target_prompt=source_prompt,
                                        uncertainty_map=tuple(slices),
                                        y=y)
                
                x_prev2, pred_x02 = outs2  
                loss_2 = torch.norm(pred_x02 - img_ar)
                
                loss = loss_1
                # loss = 0.5*loss_1 + 0.5*loss_2

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                # print(weights[0])
                
                ##########################################################
                # weights_stack = torch.stack(weights, dim=0)
                # weights_soft = F.softmax(weights_stack, dim=0)
                # # print(weights_soft[0])
                # x0 = weights_soft[0] * x0_inputs[0] + weights_soft[1] * x0_inputs[1] + weights_soft[2] * x0_inputs[2]
                ##########################################################

                #################################### other view ##########################
                # src_ar2 = torch.cat([sources[0], img2.clone().detach()], dim=1)
                # temp = torch.randn(*shape[:]).to(device)
                # src_ar2 = torch.cat([sources[0], temp], dim=1)

                # axis_to_mask = 3
                # # Create a list of slice objects for all dimensions
                # slices2 = [slice(None)] * 5  # 5 dimensions: [batch, channel, height, width, depth]
                # slices2[axis_to_mask] = g

                # # uncertainty_map[uncertainty_map < 0.0] = 0.0
                # img_ar2 = self.model.q_sample(img_ar2, ts)
                # outs2 = self.p_sample_ddim(img_ar2, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                #                         quantize_denoised=quantize_denoised, temperature=temperature,
                #                         noise_dropout=noise_dropout, score_corrector=score_corrector,
                #                         corrector_kwargs=corrector_kwargs,
                #                         unconditional_guidance_scale=unconditional_guidance_scale,
                #                         unconditional_conditioning=unconditional_conditioning,
                #                         sources=src_ar2,
                #                         target_prompt=target_prompt,
                #                         uncertainty_map=tuple(slices2))

                # x_prev2, pred_x02 = outs2
                
                # img_src = self.model.q_sample(sources[0], t_fix)
                # cond_src = sources[0].clone().detach()
                # cond_src[:,:,g,:,:] = noisy[:,:,g,:,:].clone().detach()
                # img_mid = img.clone().detach()
                # img_mid[:,:,g,:,:] = pred_x0
                # tar_ar = torch.cat([img_mid, cond_src.clone().detach()], dim=1)
                # outs_y = self.p_sample_ddim(img_src, cond, t_fix, index=index, use_original_steps=ddim_use_original_steps,
                #                         quantize_denoised=quantize_denoised, temperature=temperature,
                #                         noise_dropout=noise_dropout, score_corrector=score_corrector,
                #                         corrector_kwargs=corrector_kwargs,
                #                         unconditional_guidance_scale=unconditional_guidance_scale,
                #                         unconditional_conditioning=unconditional_conditioning,
                #                         sources=tar_ar,
                #                         target_prompt=source_prompt,
                #                         uncertainty_map=g)
                
                # y_prev, pred_y0 = outs_y
                
                # def cosine_similarity_attention_shared(gt_feature: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
                #     """
                #     Compute shared-channel voxel-wise attention map between GT and feature.

                #     Args:
                #         gt_feature (torch.Tensor): (1, C, D, H, W)
                #         feature (torch.Tensor): (1, C, D, H, W)

                #     Returns:
                #         torch.Tensor: Attention map, shape (1, 1, D, H, W)
                #     """
                #     gt_norm = F.normalize(gt_feature, p=2, dim=1)
                #     feat_norm = F.normalize(feature, p=2, dim=1)
                #     sim = torch.sum(gt_norm * feat_norm, dim=1, keepdim=True)  # shape: (1, 1, D, H, W)
                #     attn = (sim + 1) / 2
                #     return attn

                # attn_pred_y0 = cosine_similarity_attention_shared(sources[0], pred_y0)  # shape: (1, 1, D, H, W)
                # u_list.append(attn_pred_y0)
                # pred_x0_list.append(pred_x0)
                
                # try:
                #     # Stack and apply softmax over feature axis
                #     attn_stack = torch.cat([u_list[-1], u_list[-2]], dim=0)  # shape: (2, 1, D, H, W)
                #     attn_soft = F.softmax(attn_stack, dim=0)         # softmax over A/B

                #     # Broadcast for multiplication
                #     attn_A_soft = attn_soft[0]  # (1, 1, D, H, W)
                #     attn_B_soft = attn_soft[1]

                #     # Expand to feature shape
                #     attn_A_expand = attn_A_soft.expand_as(pred_x0)
                #     attn_B_expand = attn_B_soft.expand_as(pred_x0)

                #     # Weighted output
                #     img_ar = attn_A_expand * pred_x0 + attn_B_expand * pred_x0_list[-2]
                # except:
                #     img_ar = pred_x0.clone().detach()
                
                # img_ar = x_prev.clone().detach()
                # if i > 350:
                #     img_ar1 = pred_x0_list[-1] *0.5 + pred_x0_list[-2] *0.5
                # else:
                #     img_ar1 = pred_x0.clone().detach()


        return x0[:, -x_prev.shape[1]:], intermediates
    
    def sample_ar_adapt_p(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0_1=None,
               x0_2=None,
               x0_3=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               target_prompt=None,
               source_prompt=None,
               srcs=None,
               timesteps=None,
               x_src=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        
        
        # Initialize learnable parameters with same shape as x0_1 and initial value of 1/3
        
        # w3 = 1 - w1 - w2
        
        # cnn_1 = torch.nn.Conv3d(256, 256, 1, padding=0)
        # cnn_1 = cnn_1.to(self.model.device)
        cnn_1 = None
        # Initialize optimizer
        # optimizer = torch.optim.Adam([w1, w2, w3, *cnn_1.parameters()], lr=0.1)
        
        ##########################################################
        # w1 = torch.nn.Parameter(torch.ones_like(x0_1))
        # w2 = torch.nn.Parameter(torch.ones_like(x0_2))
        # w3 = torch.nn.Parameter(torch.ones_like(x0_3))
        # optimizer = torch.optim.Adam([w1, w2, w3], lr=0.03)
        
        # # Combine x0_1, x0_2, x0_3 with learnable weights
        # weights = torch.stack([w1, w2, w3], dim=0)
        # weights = F.softmax(weights, dim=0)
        # x0 = weights[0] * x0_1 + weights[1] * x0_2 + weights[2] * x0_3
        ##########################################################
        
        ##########################################################
        
        # gating_block = nn.Sequential(
        #     nn.Conv3d(9, 32, kernel_size=1, padding=0),
        #     # nn.Conv3d(32, 32, kernel_size=1, padding=0),
        #     # nn.ReLU(),
        #     nn.Conv3d(32, 3*3, kernel_size=1),
        #     # nn.ReLU()
        # )
        # gating_block = gating_block.to(self.model.device)
        
        # gating_block = HybridGatingBlock3D(in_channels=3, channel_wise=True)
        gating_block = Fusion(in_channels=3)
        # gating_block = HybridGatingBlock3D_Independent(in_channels=3, shared_encoder=True)

        gating_block = gating_block.to(self.model.device)
        
        optimizer = torch.optim.SGD([*gating_block.parameters()], lr=0.05)
        x0 = x0_1.clone().detach()
        
        #############################################################
        
        # gating_block = nn.Sequential(
        #     nn.Conv3d(3*3, 16, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv3d(16, 3, kernel_size=1)
        # )
        
        # optimizer = torch.optim.Adam([*gating_block.parameters()], lr=0.1)
        # features = torch.cat([x0_1, x0_2, x0_3], dim=1)  # [B, 3C, D, H, W]
        # gating_output = gating_block(features)
        # gating_output = F.softmax(gating_output, dim=1)
        # x0 = gating_output[:, 0] * x0_1 + gating_output[:, 1] * x0_2 + gating_output[:, 2] * x0_3
        
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W, D = shape
        size = (batch_size, C, H, W, D)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling_ar_adapt_p(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    target_prompt=target_prompt,
                                                    source_prompt=source_prompt,
                                                    sources=srcs,
                                                    timesteps=timesteps,
                                                    optimizer=optimizer,
                                                    # weights=[w1, w2, w3],
                                                    x0_inputs=[x0_1, x0_2, x0_3],
                                                    cnn_1=cnn_1,
                                                    gating_block=gating_block,
                                                    x_src=x_src
                                                    )
        return samples, intermediates
    
    def ddim_sampling_ar_adapt_p(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, target_prompt=None, source_prompt=None, sources=None,
                      optimizer=None, weights=None, x0_inputs=None, cnn_1=None, gating_block=None,x_src=None):
        device = self.model.betas.device
        b = shape[0]
        # img = torch.cat([x0.to(device), torch.randn((x0.shape[0], x0.shape[1]//2, *shape[2:])).to(device)], dim=1)
        # img = x0.to(device)

        # img = torch.randn(*shape[:]).to(device)
        img = sources[0].clone().detach()
        # timesteps = None
        # timesteps = 10
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            # timesteps = 100
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        x_start_list = []
        # Create tensor of indices 0-31
        indices = torch.arange(32)
        # Shuffle the indices
        # shuffled_indices = indices[torch.randperm(len(indices))]

        for i, step in enumerate(iterator):
            # if i < total_steps//3*2:
            # if i < total_steps//2:
            #     continue
            for j in range(4):
                shuffled_indices = indices[torch.randperm(len(indices))]
                # g = shuffled_indices[:16]
                g = shuffled_indices[:]
                # g2 = shuffled_indices[:16]
                # g = shuffled_indices[:]
                # g.sort()
                g = torch.sort(g)[0]
                # g2 = torch.sort(g2)[0]

                noisy = torch.randn(*shape[:]).to(device)

                index = total_steps - i - 1
                # ts = torch.full((b,), step, device=device, dtype=torch.long)
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                # t_fix = torch.full((b,), 100, device=device, dtype=torch.long)
                
                ################################################################
                # features = torch.cat([x0_inputs[0], x0_inputs[1], x0_inputs[2]], dim=1)  # [B, 3C, D, H, W]
                # gating_output = gating_block(features)
                # gating_reshape = gating_output.reshape(b, 3, 3, gating_output.shape[2], gating_output.shape[3], gating_output.shape[4])
                # gating_output_s = F.softmax(gating_reshape, dim=1)
                # gating_output_s = F.sigmoid(gating_reshape)
                

                # # print(gating_output_s.shape)
                # # print(torch.sum(gating_output_s, dim=1))
                # # print(gating_output_s[:,0].min(), gating_output_s[:,0].max())
                # x0 = gating_output_s[:, 0] * x0_inputs[0] + gating_output_s[:, 1] * x0_inputs[1] + gating_output_s[:, 2] * x0_inputs[2]
                ################################################################
                # x0, _ = gating_block(x0_inputs[0], x0_inputs[1], x0_inputs[2])
                avg_x0 = (x0_inputs[0] + x0_inputs[1]+ x0_inputs[2]) / 3
                x0 = gating_block(avg_x0)
                
                ################################################################
                # weights_s = torch.stack(weights, dim=0)
                # weights_s = F.softmax(weights_s, dim=0)
                # # # print(weights_s[0])
                # x0 = weights_s[0] * x0_inputs[0] + weights_s[1] * x0_inputs[1] + weights_s[2] * x0_inputs[2]
                ################################ view ###################################
                # axis_to_mask = 2
                axis_to_mask = torch.randint(2, 5, (1,)).item()
                
                y = torch.zeros(img.shape[0], dtype=torch.long, device=device)  # 0: axial
                # if axis_to_mask == 2:
                #     y = torch.zeros(img.shape[0], dtype=torch.long, device=device)  # 0: axial
                if axis_to_mask == 3:
                    y = torch.ones(img.shape[0], dtype=torch.long, device=device)   # 1: coronal
                elif axis_to_mask == 4:
                    y = torch.full((img.shape[0],), 2, dtype=torch.long, device=device)  # 2: sagittal
        
        
                # Create a list of slice objects for all dimensions
                slices = [slice(None)] * 5  # 5 dimensions: [batch, channel, height, width, depth]
                slices[axis_to_mask] = g
                
                img1 = img.clone().detach()
                img_ar  = img1[tuple(slices)].clone().detach()
                img1[tuple(slices)] = noisy[tuple(slices)].clone().detach()
                src_ar = torch.cat([x0, img1], dim=1)
                

                img_ar1 = self.model.q_sample(img_ar, ts)
                if y[0] == 1:
                    img_ar1 = img_ar1.permute(0,1,3,2,4)
                elif y[0] == 2:
                    img_ar1 = img_ar1.permute(0,1,4,2,3)
                
                # img_ar1 = img_ar1.squeeze(0)
                # img_ar1 = img_ar1.permute(1,0,2,3)
                B, C, H, W, D = img_ar1.shape
                img_ar1 = rearrange(img_ar1, 'b c h w d -> (b h) c w d')

                outs = self.p_sample_ddim_cons(img_ar1, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        sources=src_ar,
                                        target_prompt=source_prompt,
                                        uncertainty_map=tuple(slices),
                                        # cnn_1=cnn_1,
                                        y=y)
                
                x_prev, pred_x0 = outs
                # print(pred_x0.shape)
                # pred_x0_rearr = rearrange(pred_x0, '(b h) c w d -> b c h w d', b=B, h=H)
                # pred_img = self.model.decode_first_stage_grad(pred_x0_rearr*5.0)
                # loss_1 = torch.norm(pred_img - x_src.clone().detach())

                
                # if y[0] == 1:
                #     img_ar = img_ar.permute(0,1,3,2,4)
                    
                # elif y[0] == 2:
                #     img_ar = img_ar.permute(0,1,4,2,3)
                    
                # # img_ar = img_ar.squeeze(0)
                # # img_ar = img_ar.permute(1,0,2,3)
                img_ar = rearrange(img_ar, 'b c h w d -> (b h) c w d')
                loss_1 = torch.norm(pred_x0 - img_ar)

                
                loss = loss_1
                # loss = loss_1 + dec_loss
                # loss = loss_1 + loss_2

                optimizer.zero_grad()
                # loss.backward(retain_graph=True)
                loss.backward()
                optimizer.step()
                # print(weights[0])
                
                ##########################################################
                # weights_stack = torch.stack(weights, dim=0)
                # weights_soft = F.softmax(weights_stack, dim=0)
                # # print(weights_soft[0])
                # x0 = weights_soft[0] * x0_inputs[0] + weights_soft[1] * x0_inputs[1] + weights_soft[2] * x0_inputs[2]
                ##########################################################

                #################################### other view ##########################
                # src_ar2 = torch.cat([sources[0], img2.clone().detach()], dim=1)
                # temp = torch.randn(*shape[:]).to(device)
                # src_ar2 = torch.cat([sources[0], temp], dim=1)

                # axis_to_mask = 3
                # # Create a list of slice objects for all dimensions
                # slices2 = [slice(None)] * 5  # 5 dimensions: [batch, channel, height, width, depth]
                # slices2[axis_to_mask] = g

                # # uncertainty_map[uncertainty_map < 0.0] = 0.0
                # img_ar2 = self.model.q_sample(img_ar2, ts)
                # outs2 = self.p_sample_ddim(img_ar2, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                #                         quantize_denoised=quantize_denoised, temperature=temperature,
                #                         noise_dropout=noise_dropout, score_corrector=score_corrector,
                #                         corrector_kwargs=corrector_kwargs,
                #                         unconditional_guidance_scale=unconditional_guidance_scale,
                #                         unconditional_conditioning=unconditional_conditioning,
                #                         sources=src_ar2,
                #                         target_prompt=target_prompt,
                #                         uncertainty_map=tuple(slices2))

                # x_prev2, pred_x02 = outs2
                
                # img_src = self.model.q_sample(sources[0], t_fix)
                # cond_src = sources[0].clone().detach()
                # cond_src[:,:,g,:,:] = noisy[:,:,g,:,:].clone().detach()
                # img_mid = img.clone().detach()
                # img_mid[:,:,g,:,:] = pred_x0
                # tar_ar = torch.cat([img_mid, cond_src.clone().detach()], dim=1)
                # outs_y = self.p_sample_ddim(img_src, cond, t_fix, index=index, use_original_steps=ddim_use_original_steps,
                #                         quantize_denoised=quantize_denoised, temperature=temperature,
                #                         noise_dropout=noise_dropout, score_corrector=score_corrector,
                #                         corrector_kwargs=corrector_kwargs,
                #                         unconditional_guidance_scale=unconditional_guidance_scale,
                #                         unconditional_conditioning=unconditional_conditioning,
                #                         sources=tar_ar,
                #                         target_prompt=source_prompt,
                #                         uncertainty_map=g)
                
                # y_prev, pred_y0 = outs_y
                
                # def cosine_similarity_attention_shared(gt_feature: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
                #     """
                #     Compute shared-channel voxel-wise attention map between GT and feature.

                #     Args:
                #         gt_feature (torch.Tensor): (1, C, D, H, W)
                #         feature (torch.Tensor): (1, C, D, H, W)

                #     Returns:
                #         torch.Tensor: Attention map, shape (1, 1, D, H, W)
                #     """
                #     gt_norm = F.normalize(gt_feature, p=2, dim=1)
                #     feat_norm = F.normalize(feature, p=2, dim=1)
                #     sim = torch.sum(gt_norm * feat_norm, dim=1, keepdim=True)  # shape: (1, 1, D, H, W)
                #     attn = (sim + 1) / 2
                #     return attn

                # attn_pred_y0 = cosine_similarity_attention_shared(sources[0], pred_y0)  # shape: (1, 1, D, H, W)
                # u_list.append(attn_pred_y0)
                # pred_x0_list.append(pred_x0)
                
                # try:
                #     # Stack and apply softmax over feature axis
                #     attn_stack = torch.cat([u_list[-1], u_list[-2]], dim=0)  # shape: (2, 1, D, H, W)
                #     attn_soft = F.softmax(attn_stack, dim=0)         # softmax over A/B

                #     # Broadcast for multiplication
                #     attn_A_soft = attn_soft[0]  # (1, 1, D, H, W)
                #     attn_B_soft = attn_soft[1]

                #     # Expand to feature shape
                #     attn_A_expand = attn_A_soft.expand_as(pred_x0)
                #     attn_B_expand = attn_B_soft.expand_as(pred_x0)

                #     # Weighted output
                #     img_ar = attn_A_expand * pred_x0 + attn_B_expand * pred_x0_list[-2]
                # except:
                #     img_ar = pred_x0.clone().detach()
                
                # img_ar = x_prev.clone().detach()
                # if i > 350:
                #     img_ar1 = pred_x0_list[-1] *0.5 + pred_x0_list[-2] *0.5
                # else:
                #     img_ar1 = pred_x0.clone().detach()


        return x0[:, -x_prev.shape[1]:], intermediates
    
    def sample_ar_cons(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               target_prompt=None,
               source_prompt=None,
               srcs=None,
               timesteps=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W, D = shape
        size = (batch_size, C, H, W, D)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling_ar_cons(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    target_prompt=target_prompt,
                                                    source_prompt=source_prompt,
                                                    sources=srcs,
                                                    timesteps=timesteps
                                                    )
        return samples, intermediates
    
    def ddim_sampling_ar_cons(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, target_prompt=None, source_prompt=None, sources=None):
        device = self.model.betas.device
        b = shape[0]
        # img = torch.cat([x0.to(device), torch.randn((x0.shape[0], x0.shape[1]//2, *shape[2:])).to(device)], dim=1)
        # img = x0.to(device)

        # img = torch.randn(*shape[:]).to(device)
        img = sources[0].clone().detach()

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        x_start_list = []
        # Create tensor of indices 0-31
        indices = torch.arange(32)
        # Shuffle the indices
        # shuffled_indices = indices[torch.randperm(len(indices))]

        # Reshape into 4 groups of 8
        # groups = shuffled_indices.reshape(4, 8)
        # groups = indices.reshape(4, 8)
        # first_half = indices[:16]  # 0-15
        # second_half = indices[16:]  # 31-16
        # # second_half.sorted(reverse=True)
        # # Create overlapping groups with stride 1 for each half
        # groups_first = torch.stack([first_half[i:i+8] for i in range(0,16-8+1,2)])
        # groups_second = torch.stack([second_half[i:i+8] for i in range(0,16-8+1,2)])
        # groups_second = groups_second.flip(dims=(0,))
        # # Concatenate both groups
        # groups = torch.cat([groups_first, groups_second], dim=0)

        # reversed_groups = groups.flip(dims=(0,))
        # groups = torch.cat([groups, reversed_groups], dim=0)
        # groups = groups.to(device)
        
        # for idx, g in tqdm(enumerate(groups),total=len(groups)):
        
        pred_x0_list = []
        u_list = []
        for i, step in enumerate(iterator):
            # if i < total_steps//3*2:
            if i < total_steps//2:
                continue
            for j in range(1):
                x1 = x0.clone().detach()
                x1.requires_grad = True
                shuffled_indices = indices[torch.randperm(len(indices))]
                g = shuffled_indices[:8]
                # g.sort()
                g = torch.sort(g)[0]


                noisy = torch.randn(*shape[:]).to(device)
                img1 = img.clone().detach()
                
                axis_to_mask = torch.randint(2, 5, (1,)).item()
                slices = [slice(None)] * 5  # 5 dimensions: [batch, channel, height, width, depth]
                slices[axis_to_mask] = g
                img_ar  = img1[tuple(slices)].clone().detach()
                img1[tuple(slices)] = noisy[tuple(slices)].clone().detach()
                
                # img_ar  = img1[:,:,g,:,:].clone().detach()
                # img1[:,:,g,:,:] = noisy[:,:,g,:,:].clone().detach()
                img1.requires_grad = True

                index = total_steps - i - 1
                # ts = torch.full((b,), step, device=device, dtype=torch.long)
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                # t_fix = torch.full((b,), 100, device=device, dtype=torch.long)
                

                src_ar = torch.cat([x1, img1], dim=1)
                ################################ view ###################################
                # axis_to_mask = 2
                # Create a list of slice objects for all dimensions
                # slices = [slice(None)] * 5  # 5 dimensions: [batch, channel, height, width, depth]
                # slices[axis_to_mask] = g

                img_ar1 = self.model.q_sample(img_ar, ts)
                outs = self.p_sample_ddim_cons(img_ar1, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        sources=src_ar,
                                        target_prompt=source_prompt,
                                        uncertainty_map=tuple(slices))
                
                x_prev, pred_x0 = outs
                
                norm = torch.norm(pred_x0 - img_ar)
                norm_grad = grad(outputs=norm, inputs=x1)[0] 
                # print(norm_grad.min(), norm_grad.max())
                # x0[:,:,g,:,:] = x1[:,:,g,:,:] - 10 * norm_grad[:,:,g,:,:]
                x0 = x1 - 10 * norm_grad
                pred_x0_list.append(pred_x0)
                #################################### other view ##########################
                # src_ar2 = torch.cat([sources[0], img2.clone().detach()], dim=1)
                # temp = torch.randn(*shape[:]).to(device)
                # src_ar2 = torch.cat([sources[0], temp], dim=1)

                # axis_to_mask = 3
                # # Create a list of slice objects for all dimensions
                # slices2 = [slice(None)] * 5  # 5 dimensions: [batch, channel, height, width, depth]
                # slices2[axis_to_mask] = g

                # # uncertainty_map[uncertainty_map < 0.0] = 0.0
                # img_ar2 = self.model.q_sample(img_ar2, ts)
                # outs2 = self.p_sample_ddim(img_ar2, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                #                         quantize_denoised=quantize_denoised, temperature=temperature,
                #                         noise_dropout=noise_dropout, score_corrector=score_corrector,
                #                         corrector_kwargs=corrector_kwargs,
                #                         unconditional_guidance_scale=unconditional_guidance_scale,
                #                         unconditional_conditioning=unconditional_conditioning,
                #                         sources=src_ar2,
                #                         target_prompt=target_prompt,
                #                         uncertainty_map=tuple(slices2))

                # x_prev2, pred_x02 = outs2
                
                # img_src = self.model.q_sample(sources[0], t_fix)
                # cond_src = sources[0].clone().detach()
                # cond_src[:,:,g,:,:] = noisy[:,:,g,:,:].clone().detach()
                # img_mid = img.clone().detach()
                # img_mid[:,:,g,:,:] = pred_x0
                # tar_ar = torch.cat([img_mid, cond_src.clone().detach()], dim=1)
                # outs_y = self.p_sample_ddim(img_src, cond, t_fix, index=index, use_original_steps=ddim_use_original_steps,
                #                         quantize_denoised=quantize_denoised, temperature=temperature,
                #                         noise_dropout=noise_dropout, score_corrector=score_corrector,
                #                         corrector_kwargs=corrector_kwargs,
                #                         unconditional_guidance_scale=unconditional_guidance_scale,
                #                         unconditional_conditioning=unconditional_conditioning,
                #                         sources=tar_ar,
                #                         target_prompt=source_prompt,
                #                         uncertainty_map=g)
                
                # y_prev, pred_y0 = outs_y
                
                # def cosine_similarity_attention_shared(gt_feature: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
                #     """
                #     Compute shared-channel voxel-wise attention map between GT and feature.

                #     Args:
                #         gt_feature (torch.Tensor): (1, C, D, H, W)
                #         feature (torch.Tensor): (1, C, D, H, W)

                #     Returns:
                #         torch.Tensor: Attention map, shape (1, 1, D, H, W)
                #     """
                #     gt_norm = F.normalize(gt_feature, p=2, dim=1)
                #     feat_norm = F.normalize(feature, p=2, dim=1)
                #     sim = torch.sum(gt_norm * feat_norm, dim=1, keepdim=True)  # shape: (1, 1, D, H, W)
                #     attn = (sim + 1) / 2
                #     return attn

                # attn_pred_y0 = cosine_similarity_attention_shared(sources[0], pred_y0)  # shape: (1, 1, D, H, W)
                # u_list.append(attn_pred_y0)
                # pred_x0_list.append(pred_x0)
                
                # try:
                #     # Stack and apply softmax over feature axis
                #     attn_stack = torch.cat([u_list[-1], u_list[-2]], dim=0)  # shape: (2, 1, D, H, W)
                #     attn_soft = F.softmax(attn_stack, dim=0)         # softmax over A/B

                #     # Broadcast for multiplication
                #     attn_A_soft = attn_soft[0]  # (1, 1, D, H, W)
                #     attn_B_soft = attn_soft[1]

                #     # Expand to feature shape
                #     attn_A_expand = attn_A_soft.expand_as(pred_x0)
                #     attn_B_expand = attn_B_soft.expand_as(pred_x0)

                #     # Weighted output
                #     img_ar = attn_A_expand * pred_x0 + attn_B_expand * pred_x0_list[-2]
                # except:
                #     img_ar = pred_x0.clone().detach()
                
                # img_ar = x_prev.clone().detach()
                # if i > 350:
                #     img_ar1 = pred_x0_list[-1] *0.5 + pred_x0_list[-2] *0.5
                # else:
                #     img_ar1 = pred_x0.clone().detach()


        return x0[:, -x_prev.shape[1]:], intermediates
    
    @torch.no_grad()
    def sample_ar(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               target_prompt=None,
               source_prompt=None,
               srcs=None,
               timesteps=None,
               prior_feature=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W, D = shape
        size = (batch_size, C, H, W, D)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling_ar(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    target_prompt=target_prompt,
                                                    source_prompt=source_prompt,
                                                    sources=srcs,
                                                    timesteps=timesteps,
                                                    prior_feature=prior_feature
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling_ar(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, target_prompt=None, source_prompt=None, sources=None, prior_feature=None):
        device = self.model.betas.device
        b = shape[0]
        # img = torch.cat([x0.to(device), torch.randn((x0.shape[0], x0.shape[1]//2, *shape[2:])).to(device)], dim=1)
        # img = x0.to(device)

        img = torch.randn(*shape[:]).to(device)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        x_start_list = []
        # Create tensor of indices 0-31
        indices = torch.arange(32)
        # Shuffle the indices
        shuffled_indices = indices[torch.randperm(len(indices))]

        # Reshape into 4 groups of 8
        groups = shuffled_indices.reshape(4, 8)
        # groups= shuffled_indices.reshape(32,1)
        # groups = indices.reshape(4,8)
        # groups = indices.reshape(1,32)

        
        # n = 32
        # nrows = 4
        # groups = decalcomanie_groups(n, nrows,reverse=True)

        # groups = shuffled_indices.reshape(8, 4)

        # groups = shuffled_indices.reshape(32, 1)
        # groups = indices.reshape(4,8)
        # groups = torch.cat([groups, groups[0:1]], dim=0)

        y = torch.zeros(img.shape[0], dtype=torch.long, device=device)   # 1: coronal
        ####causal#####
        # groups = indices.reshape(4, 8)
        # first_half = indices[:16]  # 0-15
        # second_half = indices[16:]  # 31-16
        # # second_half.sorted(reverse=True)
        # # Create overlapping groups with stride 1 for each half
        # groups_first = torch.stack([first_half[i:i+8] for i in range(0,16-8+1,2)])
        # groups_second = torch.stack([second_half[i:i+8] for i in range(0,16-8+1,2)])
        # groups_second = groups_second.flip(dims=(0,))
        # # Concatenate both groups
        # groups = torch.cat([groups_first, groups_second], dim=0)

        # reversed_groups = groups.flip(dims=(0,))
        # groups = torch.cat([groups, reversed_groups], dim=0)
        groups = groups.to(device)
        for idx, g in tqdm(enumerate(groups),total=len(groups)):
            # g.sort()
            g = torch.sort(g)[0]

            # print(g)
            noisy = torch.randn(*shape[:]).to(device)
            noisy2 = torch.randn(*shape[:]).to(device)
            img1 = img.clone().detach()
            img2 = img.clone().detach()
            img3 = img.clone().detach()

            img1[:,:,g,:,:] = noisy[:,:,g,:,:].clone().detach()
            img2[:,:,g,:,:] = noisy2[:,:,g,:,:].clone().detach()
            img_ar1  = noisy[:,:,g,:,:].clone().detach()
            # img_ar2 = noisy[:,:,:,g,:].clone().detach()
            # img_ar3 = noisy[:,:,:,:,g].clone().detach()
            if prior_feature is not None:
                img_ar1 = prior_feature[:,:,g,:,:].clone().detach()
            
            # img_ar1 = img_ar1.squeeze(0)
            # img_ar1 = img_ar1.permute(1,0,2,3)
            B, C, H, W, D = img_ar1.shape
            img_ar1 = rearrange(img_ar1, 'b c h w d -> (b h) c w d')

            pred_x0_list = []
            u_list = []
            for i, step in enumerate(iterator):
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                # t_fix = torch.full((b,), 100, device=device, dtype=torch.long)
                src_ar = torch.cat([sources[0], img1.clone().detach()], dim=1)
                # temp = torch.randn(*shape[:]).to(device)
                # print(temp.shape)
                # src_ar = torch.cat([sources[0], temp], dim=1)
                # src_ar = torch.cat([sources[0], img1.clone().detach()], dim=1)


                # uncertainty_map[uncertainty_map < 0.0] = 0.0

                ################################ view ###################################
                axis_to_mask = 2
                # Create a list of slice objects for all dimensions
                slices = [slice(None)] * 5  # 5 dimensions: [batch, channel, height, width, depth]
                slices[axis_to_mask] = g

                # img_ar1 = img_ar1.squeeze(2)
                img_ar1 = self.model.q_sample(img_ar1, ts)
                outs = self.p_sample_ddim(img_ar1, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        sources=src_ar,
                                        target_prompt=target_prompt,
                                        uncertainty_map=tuple(slices),
                                        y=y)
                
                x_prev, pred_x0 = outs
                pred_x0_list.append(pred_x0)
                img_ar1 = pred_x0.clone().detach()
                # img_ar2 = pred_x02.clone().detach()
            
            # img_ar1 = img_ar1.unsqueeze(2)
            # img_ar1 = img_ar1.permute(1,0,2,3)
            # img_ar1 = img_ar1.unsqueeze(0)
            img_ar1 = rearrange(img_ar1, '(b h) c w d -> b c h w d', b=B, h=H)
            img[tuple(slices)] = img_ar1.clone().detach()    
            # img[tuple(slices2)] = img_ar2.clone().detach()    

        return img[:, -x_prev.shape[1]:], intermediates
    
    @torch.no_grad()
    def sample_ar2(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               target_prompt=None,
               source_prompt=None,
               srcs=None,
               timesteps=None,
               prior_feature=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W, D = shape
        size = (batch_size, C, H, W, D)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling_ar2(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    target_prompt=target_prompt,
                                                    source_prompt=source_prompt,
                                                    sources=srcs,
                                                    timesteps=timesteps,
                                                    prior_feature=prior_feature
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling_ar2(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, target_prompt=None, source_prompt=None, sources=None, prior_feature=None):
        device = self.model.betas.device
        b = shape[0]
        # img = torch.cat([x0.to(device), torch.randn((x0.shape[0], x0.shape[1]//2, *shape[2:])).to(device)], dim=1)
        # img = x0.to(device)

        img = torch.randn(*shape[:]).to(device)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        x_start_list = []
        # Create tensor of indices 0-31
        indices = torch.arange(32)
        # Shuffle the indices
        shuffled_indices = indices[torch.randperm(len(indices))]

        # Reshape into 4 groups of 8
        groups = shuffled_indices.reshape(4, 8)
        # groups = torch.cat([groups, groups[0:1]], dim=0)

        ####causal##### 
        # groups = indices.reshape(4, 8)

        groups = groups.to(device)
        y = torch.ones(img.shape[0], dtype=torch.long, device=device)   # 1: coronal

        for idx, g in tqdm(enumerate(groups),total=len(groups)):
            # g.sort()
            g = torch.sort(g)[0]

            noisy = torch.randn(*shape[:]).to(device)
            noisy2 = torch.randn(*shape[:]).to(device)
            img1 = img.clone().detach()
            img2 = img.clone().detach()


            # img1[:,:,g,:,:] = noisy[:,:,g,:,:].clone().detach()
            img1[:,:,:,g,:] = noisy[:,:,:,g,:].clone().detach()
            img2[:,:,:,g,:] = noisy2[:,:,:,g,:].clone().detach()
            # img_ar1  = noisy[:,:,g,:,:].clone().detach()
            img_ar2 = noisy[:,:,:,g,:].clone().detach()
            # img_ar3 = noisy[:,:,:,:,g].clone().detach()
            if prior_feature is not None:
                img_ar2 = prior_feature[:,:,:,g,:].clone().detach()

            pred_x0_list = []
            u_list = []
            for i, step in enumerate(iterator):
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                # t_fix = torch.full((b,), 100, device=device, dtype=torch.long)

                src_ar = torch.cat([sources[0], img1.clone().detach()], dim=1)
                # temp = torch.randn(*shape[:]).to(device)
                # src_ar = torch.cat([sources[0], temp], dim=1)

                # uncertainty_map[uncertainty_map < 0.0] = 0.0

                ################################ view ###################################
                axis_to_mask = 3
                # Create a list of slice objects for all dimensions
                slices = [slice(None)] * 5  # 5 dimensions: [batch, channel, height, width, depth]
                slices[axis_to_mask] = g

                img_ar2 = self.model.q_sample(img_ar2, ts)
                outs = self.p_sample_ddim(img_ar2, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        sources=src_ar,
                                        target_prompt=target_prompt,
                                        uncertainty_map=tuple(slices),
                                        y=y)
                
                x_prev, pred_x0 = outs
                pred_x0_list.append(pred_x0)
                
                # src_ar2 = torch.cat([sources[0], img2.clone().detach()], dim=1)
                # outs2 = self.p_sample_ddim(img_ar2, cond, ts, index=index, use_original_steps=ddim_use_original_steps,  
                #                         quantize_denoised=quantize_denoised, temperature=temperature,
                #                         noise_dropout=noise_dropout, score_corrector=score_corrector,
                #                         corrector_kwargs=corrector_kwargs,
                #                         unconditional_guidance_scale=unconditional_guidance_scale,
                #                         unconditional_conditioning=unconditional_conditioning,
                #                         sources=src_ar2,
                #                         target_prompt=target_prompt,
                #                         uncertainty_map=tuple(slices),
                #                         y=y)
                
                # x_prev2, pred_x02 = outs2
                # pred_x0_list.append(pred_x02)
                    # temp = torch.randn(*shape[:]).to(device)
                    # src_ar2 = torch.cat([sources[0], temp], dim=1)

                # axis_to_mask = 3
                # # Create a list of slice objects for all dimensions
                # slices2 = [slice(None)] * 5  # 5 dimensions: [batch, channel, height, width, depth]
                # slices2[axis_to_mask] = g

                # # uncertainty_map[uncertainty_map < 0.0] = 0.0
                # img_ar2 = self.model.q_sample(img_ar2, ts)
                # outs2 = self.p_sample_ddim(img_ar2, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                #                         quantize_denoised=quantize_denoised, temperature=temperature,
                #                         noise_dropout=noise_dropout, score_corrector=score_corrector,
                #                         corrector_kwargs=corrector_kwargs,
                #                         unconditional_guidance_scale=unconditional_guidance_scale,
                #                         unconditional_conditioning=unconditional_conditioning,
                #                         sources=src_ar2,
                #                         target_prompt=target_prompt,
                #                         uncertainty_map=tuple(slices2))

                # x_prev2, pred_x02 = outs2
                
                # img_src = self.model.q_sample(sources[0], t_fix)
                # cond_src = sources[0].clone().detach()
                # cond_src[:,:,g,:,:] = noisy[:,:,g,:,:].clone().detach()
                # img_mid = img.clone().detach()
                # img_mid[:,:,g,:,:] = pred_x0
                # tar_ar = torch.cat([img_mid, cond_src.clone().detach()], dim=1)
                # outs_y = self.p_sample_ddim(img_src, cond, t_fix, index=index, use_original_steps=ddim_use_original_steps,
                #                         quantize_denoised=quantize_denoised, temperature=temperature,
                #                         noise_dropout=noise_dropout, score_corrector=score_corrector,
                #                         corrector_kwargs=corrector_kwargs,
                #                         unconditional_guidance_scale=unconditional_guidance_scale,
                #                         unconditional_conditioning=unconditional_conditioning,
                #                         sources=tar_ar,
                #                         target_prompt=source_prompt,
                #                         uncertainty_map=g)
                
                # y_prev, pred_y0 = outs_y
                
                # def cosine_similarity_attention_shared(gt_feature: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
                #     """
                #     Compute shared-channel voxel-wise attention map between GT and feature.

                #     Args:
                #         gt_feature (torch.Tensor): (1, C, D, H, W)
                #         feature (torch.Tensor): (1, C, D, H, W)

                #     Returns:
                #         torch.Tensor: Attention map, shape (1, 1, D, H, W)
                #     """
                #     gt_norm = F.normalize(gt_feature, p=2, dim=1)
                #     feat_norm = F.normalize(feature, p=2, dim=1)
                #     sim = torch.sum(gt_norm * feat_norm, dim=1, keepdim=True)  # shape: (1, 1, D, H, W)
                #     attn = (sim + 1) / 2
                #     return attn

                # attn_pred_y0 = cosine_similarity_attention_shared(sources[0], pred_y0)  # shape: (1, 1, D, H, W)
                # u_list.append(attn_pred_y0)
                # pred_x0_list.append(pred_x0)
                
                # try:
                #     # Stack and apply softmax over feature axis
                #     attn_stack = torch.cat([u_list[-1], u_list[-2]], dim=0)  # shape: (2, 1, D, H, W)
                #     attn_soft = F.softmax(attn_stack, dim=0)         # softmax over A/B

                #     # Broadcast for multiplication
                #     attn_A_soft = attn_soft[0]  # (1, 1, D, H, W)
                #     attn_B_soft = attn_soft[1]

                #     # Expand to feature shape
                #     attn_A_expand = attn_A_soft.expand_as(pred_x0)
                #     attn_B_expand = attn_B_soft.expand_as(pred_x0)

                #     # Weighted output
                #     img_ar = attn_A_expand * pred_x0 + attn_B_expand * pred_x0_list[-2]
                # except:
                #     img_ar = pred_x0.clone().detach()
                
                # img_ar = x_prev.clone().detach()
                # if i > 350:
                #     img_ar1 = pred_x0_list[-1] *0.5 + pred_x0_list[-2] *0.5
                # else:
                #     img_ar1 = pred_x0.clone().detach()
                # pred_x0 = torch.clamp(pred_x0, min=-5.0, max=5.0)
                # pred_x02 = torch.clamp(pred_x02, min=-5.0, max=5.0)
                # img_ar2 = (pred_x0.clone().detach()+pred_x02.clone().detach())/2
                img_ar2 = pred_x0.clone().detach()

                # img_ar2 = pred_x02.clone().detach()
                
            img[tuple(slices)] = img_ar2.clone().detach()    
            # img[tuple(slices2)] = img_ar2.clone().detach()    

        return img[:, -x_prev.shape[1]:], intermediates
    
    @torch.no_grad()
    def sample_ar3(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               target_prompt=None,
               source_prompt=None,
               srcs=None,
               timesteps=None,
               prior_feature=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W, D = shape
        size = (batch_size, C, H, W, D)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling_ar3(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    target_prompt=target_prompt,
                                                    source_prompt=source_prompt,
                                                    sources=srcs,
                                                    timesteps=timesteps,
                                                    prior_feature=prior_feature
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling_ar3(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, target_prompt=None, source_prompt=None, sources=None, prior_feature=None):
        device = self.model.betas.device
        b = shape[0]
        # img = torch.cat([x0.to(device), torch.randn((x0.shape[0], x0.shape[1]//2, *shape[2:])).to(device)], dim=1)
        # img = x0.to(device)

        img = torch.randn(*shape[:]).to(device)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        x_start_list = []
        # Create tensor of indices 0-31
        indices = torch.arange(32)
        # Shuffle the indices
        shuffled_indices = indices[torch.randperm(len(indices))]

        # Reshape into 4 groups of 8
        groups = shuffled_indices.reshape(4, 8)
        # groups = torch.cat([groups, groups[0:1]], dim=0)

        ####causal#####
        # groups = indices.reshape(4, 8)
        y = torch.full((img.shape[0],), 2, dtype=torch.long, device=device)  # 2: sagittal

        groups = groups.to(device)
        for idx, g in tqdm(enumerate(groups),total=len(groups)):
            # g.sort()
            g = torch.sort(g)[0]

            noisy = torch.randn(*shape[:]).to(device)
            noisy2 = torch.randn(*shape[:]).to(device)
            img1 = img.clone().detach()
            img2 = img.clone().detach()


            # img1[:,:,g,:,:] = noisy[:,:,g,:,:].clone().detach()
            img1[:,:,:,:,g] = noisy[:,:,:,:,g].clone().detach()
            img2[:,:,:,:,g] = noisy2[:,:,:,:,g].clone().detach()
            # img_ar1  = noisy[:,:,g,:,:].clone().detach()
            # img_ar2 = noisy[:,:,:,g,:].clone().detach()
            img_ar3 = noisy[:,:,:,:,g].clone().detach()
            if prior_feature is not None:
                img_ar3 = prior_feature[:,:,:,:,g].clone().detach()
            pred_x0_list = []
            u_list = []
            for i, step in enumerate(iterator):
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                # t_fix = torch.full((b,), 100, device=device, dtype=torch.long)

                src_ar = torch.cat([sources[0], img1.clone().detach()], dim=1)
                # temp = torch.randn(*shape[:]).to(device)
                # src_ar = torch.cat([sources[0], temp], dim=1)

                # uncertainty_map[uncertainty_map < 0.0] = 0.0

                ################################ view ###################################
                axis_to_mask = 4
                # Create a list of slice objects for all dimensions
                slices = [slice(None)] * 5  # 5 dimensions: [batch, channel, height, width, depth]
                slices[axis_to_mask] = g

                img_ar3 = self.model.q_sample(img_ar3, ts)
                outs = self.p_sample_ddim(img_ar3, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        sources=src_ar,
                                        target_prompt=target_prompt,
                                        uncertainty_map=tuple(slices),
                                        y=y)
                
                x_prev, pred_x0 = outs
                pred_x0_list.append(pred_x0)
                
                # src_ar2 = torch.cat([sources[0], img2.clone().detach()], dim=1)                         
                # outs2 = self.p_sample_ddim(img_ar3, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                #                         quantize_denoised=quantize_denoised, temperature=temperature,
                #                         noise_dropout=noise_dropout, score_corrector=score_corrector,
                #                         corrector_kwargs=corrector_kwargs,
                #                         unconditional_guidance_scale=unconditional_guidance_scale,
                #                         unconditional_conditioning=unconditional_conditioning,
                #                         sources=src_ar2,
                #                         target_prompt=target_prompt,
                #                         uncertainty_map=tuple(slices),
                #                         y=y)
                
                # x_prev2, pred_x02 = outs2
                # pred_x0_list.append(pred_x02)
                # temp = torch.randn(*shape[:]).to(device)
                # src_ar2 = torch.cat([sources[0], temp], dim=1)

                # axis_to_mask = 3
                # # Create a list of slice objects for all dimensions
                # slices2 = [slice(None)] * 5  # 5 dimensions: [batch, channel, height, width, depth]
                # slices2[axis_to_mask] = g

                # # uncertainty_map[uncertainty_map < 0.0] = 0.0
                # img_ar2 = self.model.q_sample(img_ar2, ts)
                # outs2 = self.p_sample_ddim(img_ar2, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                #                         quantize_denoised=quantize_denoised, temperature=temperature,
                #                         noise_dropout=noise_dropout, score_corrector=score_corrector,
                #                         corrector_kwargs=corrector_kwargs,
                #                         unconditional_guidance_scale=unconditional_guidance_scale,
                #                         unconditional_conditioning=unconditional_conditioning,
                #                         sources=src_ar2,
                #                         target_prompt=target_prompt,
                #                         uncertainty_map=tuple(slices2))

                # x_prev2, pred_x02 = outs2
                
                # img_src = self.model.q_sample(sources[0], t_fix)
                # cond_src = sources[0].clone().detach()
                # cond_src[:,:,g,:,:] = noisy[:,:,g,:,:].clone().detach()
                # img_mid = img.clone().detach()
                # img_mid[:,:,g,:,:] = pred_x0
                # tar_ar = torch.cat([img_mid, cond_src.clone().detach()], dim=1)
                # outs_y = self.p_sample_ddim(img_src, cond, t_fix, index=index, use_original_steps=ddim_use_original_steps,
                #                         quantize_denoised=quantize_denoised, temperature=temperature,
                #                         noise_dropout=noise_dropout, score_corrector=score_corrector,
                #                         corrector_kwargs=corrector_kwargs,
                #                         unconditional_guidance_scale=unconditional_guidance_scale,
                #                         unconditional_conditioning=unconditional_conditioning,
                #                         sources=tar_ar,
                #                         target_prompt=source_prompt,
                #                         uncertainty_map=g)
                
                # y_prev, pred_y0 = outs_y
                
                # def cosine_similarity_attention_shared(gt_feature: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
                #     """
                #     Compute shared-channel voxel-wise attention map between GT and feature.

                #     Args:
                #         gt_feature (torch.Tensor): (1, C, D, H, W)
                #         feature (torch.Tensor): (1, C, D, H, W)

                #     Returns:
                #         torch.Tensor: Attention map, shape (1, 1, D, H, W)
                #     """
                #     gt_norm = F.normalize(gt_feature, p=2, dim=1)
                #     feat_norm = F.normalize(feature, p=2, dim=1)
                #     sim = torch.sum(gt_norm * feat_norm, dim=1, keepdim=True)  # shape: (1, 1, D, H, W)
                #     attn = (sim + 1) / 2
                #     return attn

                # attn_pred_y0 = cosine_similarity_attention_shared(sources[0], pred_y0)  # shape: (1, 1, D, H, W)
                # u_list.append(attn_pred_y0)
                # pred_x0_list.append(pred_x0)
                
                # try:
                #     # Stack and apply softmax over feature axis
                #     attn_stack = torch.cat([u_list[-1], u_list[-2]], dim=0)  # shape: (2, 1, D, H, W)
                #     attn_soft = F.softmax(attn_stack, dim=0)         # softmax over A/B

                #     # Broadcast for multiplication
                #     attn_A_soft = attn_soft[0]  # (1, 1, D, H, W)
                #     attn_B_soft = attn_soft[1]

                #     # Expand to feature shape
                #     attn_A_expand = attn_A_soft.expand_as(pred_x0)
                #     attn_B_expand = attn_B_soft.expand_as(pred_x0)

                #     # Weighted output
                #     img_ar = attn_A_expand * pred_x0 + attn_B_expand * pred_x0_list[-2]
                # except:
                #     img_ar = pred_x0.clone().detach()
                
                # img_ar = x_prev.clone().detach()
                # if i > 350:
                #     img_ar1 = pred_x0_list[-1] *0.5 + pred_x0_list[-2] *0.5
                # else:
                #     img_ar1 = pred_x0.clone().detach()
                # pred_x0 = torch.clamp(pred_x0, min=-3.0, max=3.0)
                # pred_x02 = torch.clamp(pred_x02, min=-3.0, max=3.0)
                # img_ar3 = (pred_x0.clone().detach()+pred_x02.clone().detach())/2
                img_ar3 = pred_x0.clone().detach()
                # img_ar2 = pred_x02.clone().detach()
                
            img[tuple(slices)] = img_ar3.clone().detach()    
            # img[tuple(slices2)] = img_ar2.clone().detach()    

        return img[:, -x_prev.shape[1]:], intermediates
    
    
    @torch.no_grad()
    def sample_ar2_p(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               target_prompt=None,
               source_prompt=None,
               srcs=None,
               timesteps=None,
               prior_feature=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W, D = shape
        size = (batch_size, C, H, W, D)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling_ar2_p(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    target_prompt=target_prompt,
                                                    source_prompt=source_prompt,
                                                    sources=srcs,
                                                    timesteps=timesteps,
                                                    prior_feature=prior_feature
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling_ar2_p(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, target_prompt=None, source_prompt=None, sources=None, prior_feature=None):
        device = self.model.betas.device
        b = shape[0]
        # img = torch.cat([x0.to(device), torch.randn((x0.shape[0], x0.shape[1]//2, *shape[2:])).to(device)], dim=1)
        # img = x0.to(device)

        img = torch.randn(*shape[:]).to(device)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        x_start_list = []
        # Create tensor of indices 0-31
        indices = torch.arange(32)
        # Shuffle the indices
        shuffled_indices = indices[torch.randperm(len(indices))]

        # Reshape into 4 groups of 8
        # groups = shuffled_indices.reshape(8, 4)
        groups= shuffled_indices.reshape(4,8)
        # groups = indices.reshape(4,8)
        # groups = shuffled_indices.reshape(8, 4)
        # groups = indices.reshape(1,32)
        # groups = indices.reshape(1,32)

        # n = 32
        # nrows = 4
        # groups = decalcomanie_groups(n, nrows,reverse=True)

        # groups = shuffled_indices.reshape(32,1)
        ####causal##### 
        # groups = indices.reshape(4, 8)

        groups = groups.to(device)
        y = torch.ones(img.shape[0], dtype=torch.long, device=device)   # 1: coronal
        

        for idx, g in tqdm(enumerate(groups),total=len(groups)):
            # g.sort()
            g = torch.sort(g)[0]

            noisy = torch.randn(*shape[:]).to(device)
            noisy2 = torch.randn(*shape[:]).to(device)
            img1 = img.clone().detach()
            img2 = img.clone().detach()


            # img1[:,:,g,:,:] = noisy[:,:,g,:,:].clone().detach()
            img1[:,:,:,g,:] = noisy[:,:,:,g,:].clone().detach()
            # img2[:,:,:,g,:] = noisy2[:,:,:,g,:].clone().detach()
            # img_ar1  = noisy[:,:,g,:,:].clone().detach()
            img_ar2 = noisy[:,:,:,g,:].clone().detach()
            # img_ar3 = noisy[:,:,:,:,g].clone().detach()
            
            if prior_feature is not None:
                img_ar2 = prior_feature[:,:,:,g,:].clone().detach()

            img_ar2 = img_ar2.permute(0,1,3,2,4)
            B, C, H, W, D = img_ar2.shape
            img_ar2 = rearrange(img_ar2, 'b c h w d -> (b h) c w d')
            
            pred_x0_list = []
            u_list = []
            for i, step in enumerate(iterator):
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                # t_fix = torch.full((b,), 100, device=device, dtype=torch.long)

                src_ar = torch.cat([sources[0], img1.clone().detach()], dim=1)
                # temp = torch.randn(*shape[:]).to(device)
                # src_ar = torch.cat([sources[0], temp], dim=1)

                # uncertainty_map[uncertainty_map < 0.0] = 0.0

                ################################ view ###################################
                axis_to_mask = 3
                # Create a list of slice objects for all dimensions
                slices = [slice(None)] * 5  # 5 dimensions: [batch, channel, height, width, depth]
                slices[axis_to_mask] = g

                # img_ar2 = img_ar2.squeeze(2)
                img_ar2 = self.model.q_sample(img_ar2, ts)
                outs = self.p_sample_ddim(img_ar2, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        sources=src_ar,
                                        target_prompt=target_prompt,
                                        uncertainty_map=tuple(slices),
                                        y=y)
                
                x_prev, pred_x0 = outs
                pred_x0_list.append(pred_x0)
                
                # src_ar2 = torch.cat([sources[0], img2.clone().detach()], dim=1)
                # outs2 = self.p_sample_ddim(img_ar2, cond, ts, index=index, use_original_steps=ddim_use_original_steps,  
                #                         quantize_denoised=quantize_denoised, temperature=temperature,
                #                         noise_dropout=noise_dropout, score_corrector=score_corrector,
                #                         corrector_kwargs=corrector_kwargs,
                #                         unconditional_guidance_scale=unconditional_guidance_scale,
                #                         unconditional_conditioning=unconditional_conditioning,
                #                         sources=src_ar2,
                #                         target_prompt=target_prompt,
                #                         uncertainty_map=tuple(slices),
                #                         y=y)
                
                # x_prev2, pred_x02 = outs2
                # pred_x0_list.append(pred_x02)
                    # temp = torch.randn(*shape[:]).to(device)
                    # src_ar2 = torch.cat([sources[0], temp], dim=1)

                # axis_to_mask = 3
                # # Create a list of slice objects for all dimensions
                # slices2 = [slice(None)] * 5  # 5 dimensions: [batch, channel, height, width, depth]
                # slices2[axis_to_mask] = g

                # # uncertainty_map[uncertainty_map < 0.0] = 0.0
                # img_ar2 = self.model.q_sample(img_ar2, ts)
                # outs2 = self.p_sample_ddim(img_ar2, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                #                         quantize_denoised=quantize_denoised, temperature=temperature,
                #                         noise_dropout=noise_dropout, score_corrector=score_corrector,
                #                         corrector_kwargs=corrector_kwargs,
                #                         unconditional_guidance_scale=unconditional_guidance_scale,
                #                         unconditional_conditioning=unconditional_conditioning,
                #                         sources=src_ar2,
                #                         target_prompt=target_prompt,
                #                         uncertainty_map=tuple(slices2))

                # x_prev2, pred_x02 = outs2
                
                # img_src = self.model.q_sample(sources[0], t_fix)
                # cond_src = sources[0].clone().detach()
                # cond_src[:,:,g,:,:] = noisy[:,:,g,:,:].clone().detach()
                # img_mid = img.clone().detach()
                # img_mid[:,:,g,:,:] = pred_x0
                # tar_ar = torch.cat([img_mid, cond_src.clone().detach()], dim=1)
                # outs_y = self.p_sample_ddim(img_src, cond, t_fix, index=index, use_original_steps=ddim_use_original_steps,
                #                         quantize_denoised=quantize_denoised, temperature=temperature,
                #                         noise_dropout=noise_dropout, score_corrector=score_corrector,
                #                         corrector_kwargs=corrector_kwargs,
                #                         unconditional_guidance_scale=unconditional_guidance_scale,
                #                         unconditional_conditioning=unconditional_conditioning,
                #                         sources=tar_ar,
                #                         target_prompt=source_prompt,
                #                         uncertainty_map=g)
                
                # y_prev, pred_y0 = outs_y
                
                # def cosine_similarity_attention_shared(gt_feature: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
                #     """
                #     Compute shared-channel voxel-wise attention map between GT and feature.

                #     Args:
                #         gt_feature (torch.Tensor): (1, C, D, H, W)
                #         feature (torch.Tensor): (1, C, D, H, W)

                #     Returns:
                #         torch.Tensor: Attention map, shape (1, 1, D, H, W)
                #     """
                #     gt_norm = F.normalize(gt_feature, p=2, dim=1)
                #     feat_norm = F.normalize(feature, p=2, dim=1)
                #     sim = torch.sum(gt_norm * feat_norm, dim=1, keepdim=True)  # shape: (1, 1, D, H, W)
                #     attn = (sim + 1) / 2
                #     return attn

                # attn_pred_y0 = cosine_similarity_attention_shared(sources[0], pred_y0)  # shape: (1, 1, D, H, W)
                # u_list.append(attn_pred_y0)
                # pred_x0_list.append(pred_x0)
                
                # try:
                #     # Stack and apply softmax over feature axis
                #     attn_stack = torch.cat([u_list[-1], u_list[-2]], dim=0)  # shape: (2, 1, D, H, W)
                #     attn_soft = F.softmax(attn_stack, dim=0)         # softmax over A/B

                #     # Broadcast for multiplication
                #     attn_A_soft = attn_soft[0]  # (1, 1, D, H, W)
                #     attn_B_soft = attn_soft[1]

                #     # Expand to feature shape
                #     attn_A_expand = attn_A_soft.expand_as(pred_x0)
                #     attn_B_expand = attn_B_soft.expand_as(pred_x0)

                #     # Weighted output
                #     img_ar = attn_A_expand * pred_x0 + attn_B_expand * pred_x0_list[-2]
                # except:
                #     img_ar = pred_x0.clone().detach()
                
                # img_ar = x_prev.clone().detach()
                # if i > 350:
                #     img_ar1 = pred_x0_list[-1] *0.5 + pred_x0_list[-2] *0.5
                # else:
                #     img_ar1 = pred_x0.clone().detach()
                # pred_x0 = torch.clamp(pred_x0, min=-5.0, max=5.0)
                # pred_x02 = torch.clamp(pred_x02, min=-5.0, max=5.0)
                # img_ar2 = (pred_x0.clone().detach()+pred_x02.clone().detach())/2
                img_ar2 = pred_x0.clone().detach()

                # img_ar2 = pred_x02.clone().detach()
            
            img_ar2 = rearrange(img_ar2, '(b h) c w d -> b c h w d', b=B, h=H)

            img_ar2 = img_ar2.permute(0,1,3,2,4)
            img[tuple(slices)] = img_ar2.clone().detach()    
            # img[tuple(slices2)] = img_ar2.clone().detach()    

        return img[:, -x_prev.shape[1]:], intermediates
    
    @torch.no_grad()
    def sample_ar3_p(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               target_prompt=None,
               source_prompt=None,
               srcs=None,
               timesteps=None,
               prior_feature=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W, D = shape
        size = (batch_size, C, H, W, D)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling_ar3_p(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    target_prompt=target_prompt,
                                                    source_prompt=source_prompt,
                                                    sources=srcs,
                                                    timesteps=timesteps,
                                                    prior_feature=prior_feature
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling_ar3_p(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, target_prompt=None, source_prompt=None, sources=None, prior_feature=None):
        device = self.model.betas.device
        b = shape[0]
        # img = torch.cat([x0.to(device), torch.randn((x0.shape[0], x0.shape[1]//2, *shape[2:])).to(device)], dim=1)
        # img = x0.to(device)

        img = torch.randn(*shape[:]).to(device)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        x_start_list = []
        # Create tensor of indices 0-31
        indices = torch.arange(32)
        # Shuffle the indices
        shuffled_indices = indices[torch.randperm(len(indices))]
        
        # groups = indices.reshape(1,32)


        # Reshape into 4 groups of 8
        # groups = shuffled_indices.reshape(8, 4)
        # groups= shuffled_indices.reshape(32,1)
        groups = shuffled_indices.reshape(4, 8)
        # groups = indices.reshape(4,8)
        # n = 32
        # nrows = 4
        # groups = decalcomanie_groups(n, nrows,reverse=True)
        # groups = indices.reshape(1,32)

        ####causal#####
        # groups = indices.reshape(4, 8)
        y = torch.full((img.shape[0],), 2, dtype=torch.long, device=device)  # 2: sagittal

        groups = groups.to(device)
        for idx, g in tqdm(enumerate(groups),total=len(groups)):
            # g.sort()
            g = torch.sort(g)[0]

            noisy = torch.randn(*shape[:]).to(device)
            noisy2 = torch.randn(*shape[:]).to(device)
            img1 = img.clone().detach()
            img2 = img.clone().detach()


            # img1[:,:,g,:,:] = noisy[:,:,g,:,:].clone().detach()
            img1[:,:,:,:,g] = noisy[:,:,:,:,g].clone().detach()
            img2[:,:,:,:,g] = noisy2[:,:,:,:,g].clone().detach()
            # img_ar1  = noisy[:,:,g,:,:].clone().detach()
            # img_ar2 = noisy[:,:,:,g,:].clone().detach()
            img_ar3 = noisy[:,:,:,:,g].clone().detach()
            if prior_feature is not None:
                img_ar3 = prior_feature[:,:,:,:,g].clone().detach()
                
            img_ar3 = img_ar3.permute(0,1,4,2,3)
            B, C, H, W, D = img_ar3.shape
            img_ar3 = rearrange(img_ar3, 'b c h w d -> (b h) c w d')
            
            # img_ar3 = img_ar3.squeeze(0)
            # img_ar3 = img_ar3.permute(3,0,1,2)
            pred_x0_list = []
            u_list = []
            for i, step in enumerate(iterator):
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                # t_fix = torch.full((b,), 100, device=device, dtype=torch.long)

                src_ar = torch.cat([sources[0], img1.clone().detach()], dim=1)
                # temp = torch.randn(*shape[:]).to(device)
                # src_ar = torch.cat([sources[0], temp], dim=1)

                # uncertainty_map[uncertainty_map < 0.0] = 0.0

                ################################ view ###################################
                axis_to_mask = 4
                # Create a list of slice objects for all dimensions
                slices = [slice(None)] * 5  # 5 dimensions: [batch, channel, height, width, depth]
                slices[axis_to_mask] = g

                img_ar3 = self.model.q_sample(img_ar3, ts)
                outs = self.p_sample_ddim(img_ar3, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        sources=src_ar,
                                        target_prompt=target_prompt,
                                        uncertainty_map=tuple(slices),
                                        y=y)
                
                x_prev, pred_x0 = outs
                pred_x0_list.append(pred_x0)
                
                # src_ar2 = torch.cat([sources[0], img2.clone().detach()], dim=1)                         
                # outs2 = self.p_sample_ddim(img_ar3, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                #                         quantize_denoised=quantize_denoised, temperature=temperature,
                #                         noise_dropout=noise_dropout, score_corrector=score_corrector,
                #                         corrector_kwargs=corrector_kwargs,
                #                         unconditional_guidance_scale=unconditional_guidance_scale,
                #                         unconditional_conditioning=unconditional_conditioning,
                #                         sources=src_ar2,
                #                         target_prompt=target_prompt,
                #                         uncertainty_map=tuple(slices),
                #                         y=y)
                
                # x_prev2, pred_x02 = outs2
                # pred_x0_list.append(pred_x02)
                # temp = torch.randn(*shape[:]).to(device)
                # src_ar2 = torch.cat([sources[0], temp], dim=1)

                # axis_to_mask = 3
                # # Create a list of slice objects for all dimensions
                # slices2 = [slice(None)] * 5  # 5 dimensions: [batch, channel, height, width, depth]
                # slices2[axis_to_mask] = g

                # # uncertainty_map[uncertainty_map < 0.0] = 0.0
                # img_ar2 = self.model.q_sample(img_ar2, ts)
                # outs2 = self.p_sample_ddim(img_ar2, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                #                         quantize_denoised=quantize_denoised, temperature=temperature,
                #                         noise_dropout=noise_dropout, score_corrector=score_corrector,
                #                         corrector_kwargs=corrector_kwargs,
                #                         unconditional_guidance_scale=unconditional_guidance_scale,
                #                         unconditional_conditioning=unconditional_conditioning,
                #                         sources=src_ar2,
                #                         target_prompt=target_prompt,
                #                         uncertainty_map=tuple(slices2))

                # x_prev2, pred_x02 = outs2
                
                # img_src = self.model.q_sample(sources[0], t_fix)
                # cond_src = sources[0].clone().detach()
                # cond_src[:,:,g,:,:] = noisy[:,:,g,:,:].clone().detach()
                # img_mid = img.clone().detach()
                # img_mid[:,:,g,:,:] = pred_x0
                # tar_ar = torch.cat([img_mid, cond_src.clone().detach()], dim=1)
                # outs_y = self.p_sample_ddim(img_src, cond, t_fix, index=index, use_original_steps=ddim_use_original_steps,
                #                         quantize_denoised=quantize_denoised, temperature=temperature,
                #                         noise_dropout=noise_dropout, score_corrector=score_corrector,
                #                         corrector_kwargs=corrector_kwargs,
                #                         unconditional_guidance_scale=unconditional_guidance_scale,
                #                         unconditional_conditioning=unconditional_conditioning,
                #                         sources=tar_ar,
                #                         target_prompt=source_prompt,
                #                         uncertainty_map=g)
                
                # y_prev, pred_y0 = outs_y
                
                # def cosine_similarity_attention_shared(gt_feature: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
                #     """
                #     Compute shared-channel voxel-wise attention map between GT and feature.

                #     Args:
                #         gt_feature (torch.Tensor): (1, C, D, H, W)
                #         feature (torch.Tensor): (1, C, D, H, W)

                #     Returns:
                #         torch.Tensor: Attention map, shape (1, 1, D, H, W)
                #     """
                #     gt_norm = F.normalize(gt_feature, p=2, dim=1)
                #     feat_norm = F.normalize(feature, p=2, dim=1)
                #     sim = torch.sum(gt_norm * feat_norm, dim=1, keepdim=True)  # shape: (1, 1, D, H, W)
                #     attn = (sim + 1) / 2
                #     return attn

                # attn_pred_y0 = cosine_similarity_attention_shared(sources[0], pred_y0)  # shape: (1, 1, D, H, W)
                # u_list.append(attn_pred_y0)
                # pred_x0_list.append(pred_x0)
                
                # try:
                #     # Stack and apply softmax over feature axis
                #     attn_stack = torch.cat([u_list[-1], u_list[-2]], dim=0)  # shape: (2, 1, D, H, W)
                #     attn_soft = F.softmax(attn_stack, dim=0)         # softmax over A/B

                #     # Broadcast for multiplication
                #     attn_A_soft = attn_soft[0]  # (1, 1, D, H, W)
                #     attn_B_soft = attn_soft[1]

                #     # Expand to feature shape
                #     attn_A_expand = attn_A_soft.expand_as(pred_x0)
                #     attn_B_expand = attn_B_soft.expand_as(pred_x0)

                #     # Weighted output
                #     img_ar = attn_A_expand * pred_x0 + attn_B_expand * pred_x0_list[-2]
                # except:
                #     img_ar = pred_x0.clone().detach()
                
                # img_ar = x_prev.clone().detach()
                # if i > 350:
                #     img_ar1 = pred_x0_list[-1] *0.5 + pred_x0_list[-2] *0.5
                # else:
                #     img_ar1 = pred_x0.clone().detach()
                # pred_x0 = torch.clamp(pred_x0, min=-3.0, max=3.0)
                # pred_x02 = torch.clamp(pred_x02, min=-3.0, max=3.0)
                # img_ar3 = (pred_x0.clone().detach()+pred_x02.clone().detach())/2
                img_ar3 = pred_x0.clone().detach()
                # img_ar2 = pred_x02.clone().detach()
            
            # img_ar3 = img_ar3.permute(1,2,3,0)
            # img_ar3 = img_ar3.unsqueeze(0)
            img_ar3 = rearrange(img_ar3, '(b h) c w d -> b c h w d', b=B, h=H)
            img_ar3 = img_ar3.permute(0,1,3,4,2)
            img[tuple(slices)] = img_ar3.clone().detach()    
            # img[tuple(slices2)] = img_ar2.clone().detach()    

        return img[:, -x_prev.shape[1]:], intermediates
    
    
    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               target_prompt=None,
               srcs=None,
               timesteps=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W, D = shape
        size = (batch_size, C, H, W, D)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    target_prompt=target_prompt,
                                                    sources=srcs,
                                                    timesteps=timesteps
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, target_prompt=None, sources=None):
        device = self.model.betas.device
        b = shape[0]
        # img = torch.cat([x0.to(device), torch.randn((x0.shape[0], x0.shape[1]//2, *shape[2:])).to(device)], dim=1)
        # img = x0.to(device)

        img = torch.randn(*shape[:]).to(device)
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        x_start_list = []

        for i, step in enumerate(iterator):
            if (i+1) % 2 != 0:
                continue
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            # t_init = torch.clamp(step,0,total_steps-2)
            # random_step = torch.randint(t_init, total_steps, (1,), device=device)[0]
            # ts_r = torch.full((b,), random_step, device=device, dtype=torch.long)

            
            # uncertainty_list = []
            # if i != 0:    
            #     # ts_r = torch.full((b,), time_range[i-1], device=device, dtype=torch.long)
            #     for i in range(3):
            #         with torch.no_grad():
            #             x_noisy_u = self.model.q_sample(x_start=x0, t=ts_r, noise=None)
            #             model_output_1 = self.model.apply_model(x_noisy_u, ts_r, sources,target_prompt)
            #         o_1_recon = self.model.predict_start_from_noise(x_noisy_u, ts_r, noise=model_output_1)
            #         uncertainty_list.append(o_1_recon)
            #     u_map = torch.stack(uncertainty_list, dim=1).std(dim=1)
            #     u_map = (u_map - u_map.mean()) / u_map.std()
            #     uncertainty_map = (u_map - u_map.min()) / (u_map.max() - u_map.min() + 1e-7)
            # else:
            #     uncertainty_map = torch.ones_like(img)
            # print(step)
            uncertainty_cond = torch.rand_like(img)
            # uncertainty_cond = torch.ones_like(img)

            # uncertainty_map[uncertainty_map < 0.0] = 0.0
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      sources=sources,
                                      target_prompt=target_prompt,
                                      uncertainty_map=uncertainty_cond)
            
            
            x_prev, pred_x0 = outs
            x_start_list.append(pred_x0)
            # img = torch.cat([x0, x_prev], dim=1)
            img = x_prev.clone().detach()
            
        
        # count = 0
        for i, step in enumerate(iterator):
            # if step > 500:
            #     continue
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # if mask is not None:
            #     assert x0 is not None
            #     img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
            #     img = img_orig * mask + (1. - mask) * img
            if count == 0:
                img = self.model.q_sample(pred_x0, ts)
                count += 1

            # t_init = torch.clamp(step,0,total_steps-2)
            # random_step = torch.randint(t_init, total_steps, (1,), device=device)[0]
            # ts_r = torch.full((b,), random_step, device=device, dtype=torch.long)

            
            # uncertainty_list = []
            # if i != 0:    
            #     # ts_r = torch.full((b,), time_range[i-1], device=device, dtype=torch.long)
            #     for i in range(3):
            #         with torch.no_grad():
            #             x_noisy_u = self.model.q_sample(x_start=x0, t=ts_r, noise=None)
            #             model_output_1 = self.model.apply_model(x_noisy_u, ts_r, sources,target_prompt)
            #         o_1_recon = self.model.predict_start_from_noise(x_noisy_u, ts_r, noise=model_output_1)
            #         uncertainty_list.append(o_1_recon)
            #     u_map = torch.stack(uncertainty_list, dim=1).std(dim=1)
            #     u_map = (u_map - u_map.mean()) / u_map.std()
            #     uncertainty_map = (u_map - u_map.min()) / (u_map.max() - u_map.min() + 1e-7)
            # else:
            #     uncertainty_map = torch.ones_like(img)
            # print(step)
            uncertainty_cond = torch.rand_like(img)
            # uncertainty_cond = torch.ones_like(img)
            # if step > 100:
            #     uncertainty_cond = torch.rand_like(img)
            # else:
            # if step > 50:
            #     u_map = torch.stack(x_start_list[-3:], dim=1).std(dim=1)
            #     u_map = (u_map - u_map.mean()) / (u_map.std() + 1e-7)
            #     uncertainty_map = (u_map - u_map.min()) / (u_map.max() - u_map.min() + 1e-7)
            #     uncertainty_mask = uncertainty_map > uncertainty_map.mean()
            #     uncertainty_mask = uncertainty_mask * 1.0
                
            #     pseudo_label = torch.stack(x_start_list[-3:], dim=1).mean(dim=1)
            #     uncertainty_cond = (1-uncertainty_mask) * x_start_list[-1]  + uncertainty_mask * torch.rand_like(x_start_list[-1])
            # uncertainty_cond = torch.rand_like(img)
            # uncertainty_map = None
            # if i == 0:
            # uncertainty_map = torch.zeros_like(img)
            # uncertainty_map = torch.ones_like(img)
            # uncertainty_map[uncertainty_map > 0.0] = 1.0
            # uncertainty_map[uncertainty_map < 0.0] = 0.0
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      sources=sources,
                                      target_prompt=target_prompt,
                                      uncertainty_map=uncertainty_cond,
                                      )
            
            
            x_prev, pred_x0 = outs
            # pred_x0 = (1-uncertainty_mask) * x_start_list[-1] + uncertainty_mask * pred_x0
            x_start_list.append(pred_x0)
            # img = torch.cat([x0, x_prev], dim=1)
            img = x_prev.clone().detach()

            
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        # def tv_loss_height_only(x, reduction='mean'):
        #     dx = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        #     return dx.mean() if reduction == 'mean' else dx.sum()

        # def tv_denoise_inference(x, n_iter=20, lr=0.1, weight=0.1):
        #     """
        #     Minimizes TV loss via gradient descent at inference time
        #     Args:
        #         x: (1, C, D, H, W) tensor - the generated volume
        #         n_iter: number of optimization steps
        #         lr: learning rate for TV optimization
        #         weight: TV loss weight

        #     Returns:
        #         denoised tensor
        #     """
        #     x_refined = x.clone().detach().requires_grad_(True)
        #     optimizer = torch.optim.SGD([x_refined], lr=lr)
            
        #     for _ in range(n_iter):
        #         optimizer.zero_grad()
        #         tv_loss = tv_loss_height_only(x_refined, reduction='mean')
        #         loss = weight * tv_loss
        #         loss.backward()
        #         optimizer.step()
            
        #     return x_refined.detach()

        # img = tv_denoise_inference(img)
        
        return img[:, -x_prev.shape[1]:], intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, sources=None, target_prompt=None, uncertainty_map=None, y=None):
        b, *_, device = *x.shape, x.device
        
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, sources, target_prompt, y=y,u_map=uncertainty_map)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x[:,-e_t.shape[1]:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(e_t.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        
        return x_prev, pred_x0
    
    def p_sample_ddim_cons(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, sources=None, target_prompt=None, uncertainty_map=None, cnn_1=None, y=None):
        b, *_, device = *x.shape, x.device
        
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, sources, target_prompt, u_map=uncertainty_map, y=y)
            # e_t = self.model.apply_model(x, t, sources, target_prompt, u_map=uncertainty_map)

        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x[:,-e_t.shape[1]:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(e_t.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        
        return x_prev, pred_x0
