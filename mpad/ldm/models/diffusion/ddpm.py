"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
import random
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only
# from pytorch_lightning.utilities.rank_zero import rank_zero_only

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
# import clip
from open_clip import create_model_from_pretrained, get_tokenizer

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})
        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    # def get_input(self, batch, k):
    #     x = batch[k]
    #     return x.float()
    
    def get_input(self, batch, k):
        data = batch[0]
        x = data[k]

        return x.float()
    
    def get_input_prompt(self, batch, k):
        data_prompt = batch[1]
        return data_prompt[k]

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=False,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w d -> b n c h w d')
        denoise_grid = rearrange(denoise_grid, 'b n c h w d -> (b n) c h w d')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args, **kwargs):
        

        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        # self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None  

        self.modalities = first_stage_config.params.modalities

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True
        
        # self.clip, _ = clip.load("ViT-B/32")
        self.clip, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')



    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior, _, _ = self.first_stage_model.encode(x) 
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                            force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W, D
        denoise_grid = rearrange(denoise_row, 'n b c h w d -> b n c h w d')
        denoise_grid = rearrange(denoise_grid, 'b n c h w d -> (b n) c h w d')
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        return c

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    # @torch.no_grad()
    # def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
    #               cond_key=None, return_original_cond=False, bs=None):
    #     source = random.choice(self.modalities)
    #     target = random.choice(self.modalities)
    #     while source == target:
    #         target = random.choice(self.modalities)
            
    #     x_src = super().get_input(batch, source)
    #     x_tgt = super().get_input(batch, target)

    #     if bs is not None:
    #         x_src = x_src[:bs]
    #         x_tgt = x_tgt[:bs]
    #     x_src = x_src.to(self.device)
    #     x_tgt = x_tgt.to(self.device)
    #     z_src, _, _ = self.first_stage_model.encode(x_src)
    #     z_tgtl, _, _ = self.first_stage_model.encode(x_src, target)
    #     z_tgt, _, _ = self.first_stage_model.encode(x_tgt)

    #     z_src = self.get_first_stage_encoding(z_src).detach()
    #     z_tgt = self.get_first_stage_encoding(z_tgt).detach()
                    
    #     # class conditional one hot encoding
    #     c = self.modalities.index(target)
    #     c = torch.nn.functional.one_hot(torch.tensor(c), num_classes=len(self.modalities)).float()
    #     c = c.unsqueeze(0).repeat(z_src.shape[0], 1).unsqueeze(1).to(self.device)
        
    #     out = [torch.cat([z_src, z_tgtl], dim=1), z_tgt, c]
    #     if return_first_stage_outputs:
    #         xrec_src = self.decode_first_stage(z_src)
    #         xrec_tgt = self.decode_first_stage(z_tgt)
    #         out.extend([x_src, x_tgt, xrec_src, xrec_tgt, source, target])
    #     return out
    
    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        
        # source = random.choice(self.modalities)
        target = random.choice(self.modalities)
        sources = [m for m in self.modalities if m != target]
        # target = 't2'
        # sources = ['t1']

        x_srcs = []
        x_srcs_prompt = []
        for source in sources:
            x_src = super().get_input(batch, source)
            x_srcs.append(x_src)
            x_src_prompt = super().get_input_prompt(batch, source)
            # x_src_token = clip.tokenize(x_src_prompt).to(self.device)
            x_src_token = self.tokenizer(x_src_prompt).to(self.device)
            x_src_text_features = self.clip.encode_text(x_src_token)
            x_src_text_features = x_src_text_features.float()
            x_srcs_prompt.append(x_src_prompt)
            
        # x_src = super().get_input(batch, source)
        x_tgt = super().get_input(batch, target)
        x_tar_prompt = super().get_input_prompt(batch, target)
        # x_tar_token = clip.tokenize(x_tar_prompt).to(self.device)
        x_tar_token = self.tokenizer(x_tar_prompt).to(self.device)
        x_tar_text_features = self.clip.encode_text(x_tar_token)
        x_tar_text_features = x_tar_text_features.float()

        if bs is not None:
            # x_src = x_src[:bs]
            for i, x_src in enumerate(x_srcs):
                x_srcs[i] = x_src[:bs].to(self.device)
            x_tgt = x_tgt[:bs]
        
        # x_src = x_src.to(self.device)
        # x_srcs = x_srcs.to(self.device)
        x_tgt = x_tgt.to(self.device)
        
        z_srcs = []
        for j, x_src in enumerate(x_srcs):
            z_src = self.first_stage_model.encode(x_src)
            z_srcs.append(z_src)
            
        # z_tgtl, _, _ = self.first_stage_model.encode(x_src, target)
        z_tgt = self.first_stage_model.encode(x_tgt)

        # z_src = self.get_first_stage_encoding(z_src).detach()
        for j, z_src in enumerate(z_srcs):
            z_src_scaling = self.get_first_stage_encoding(z_src).detach()
            z_src_scaling = torch.clamp(z_src_scaling,min=-15.0,max=15.0)
            z_src_scaling = 0.2 * z_src_scaling
            z_srcs[j] = z_src_scaling.clone().detach()
        
        z_tgt = self.get_first_stage_encoding(z_tgt).detach()
        z_tgt_scaling = torch.clamp(z_tgt,min=-15.0,max=15.0)
        z_tgt = 0.2*z_tgt_scaling.clone().detach()
                    
        # class conditional one hot encoding
        # c = self.modalities.index(target)
        # c = torch.nn.functional.one_hot(torch.tensor(c), num_classes=len(self.modalities)).float()
        # c = c.unsqueeze(0).repeat(z_src.shape[0], 1).unsqueeze(1).to(self.device)
        
        # out = [torch.cat([z_src, z_tgtl], dim=1), z_tgt, c]
        
        selected_idx = torch.randint(0, len(z_srcs), (1,)).item()
        # print("#"*50)
        z_src = z_srcs[selected_idx]
        
        # ###concat all views###
        # z_tgt_1 = z_tgt.permute(0,1,3,4,2)
        # z_tgt_2 = z_tgt.permute(0,1,4,2,3)
        # z_src_1 = z_src.permute(0,1,3,4,2)
        # z_src_2 = z_src.permute(0,1,4,2,3)
        
        # z_tgt = torch.cat((z_tgt,z_tgt_1, z_tgt_2),dim=2)
        # z_src = torch.cat((z_src,z_src_1, z_src_2),dim=2)
        # #######################
        
        # Random masking indices
        mask_ratio = torch.rand(1).item() * 0.3 + 0.7  # Random ratio between 0.7 and 1.0
        num_masked = int(z_src.shape[2] * mask_ratio)  # Number of indices to mask
        mask_indices = torch.randperm(z_src.shape[2])[:num_masked]  # Randomly select indices to mask
        # print(mask_indices)
        # Create mask
        mask = torch.ones_like(z_src)
        mask[:, :, mask_indices, :, :] = 0  # Mask selected indices
        mask = mask.to(self.device)
        
        
        out = {
            "z_src": z_src,
            "z_tgt": z_tgt,
            "mask": mask,
            # "c": c,
            "x_srcs_prompt": x_srcs_prompt,
            "x_tgt_prompt": x_tar_text_features,
        }
        if return_first_stage_outputs:
            out = [z_srcs[0], z_tgt]
            xrec_src = self.decode_first_stage(z_srcs[0])
            xrec_tgt = self.decode_first_stage(z_tgt)
            out.extend([x_srcs[0], x_tgt, xrec_src, xrec_tgt, sources[0], target])
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w d c -> b c h w d').contiguous()

        z = 1. / self.scale_factor * z
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)
            
    
    def decode_first_stage_grad(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w d c -> b c h w d').contiguous()

        z = 1. / self.scale_factor * z
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w d c -> b c h w d').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):  
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        # x_src, x_tgt, c = self.get_input(batch, self.first_stage_key)
        # loss = self(x_src, x_tgt, c)
        inputs = self.get_input(batch, self.first_stage_key)
        loss = self(inputs)
        return loss

    # def forward(self, x_src, x_tgt, c, *args, **kwargs):
    #     t = torch.randint(0, self.num_timesteps, (x_tgt.shape[0],), device=self.device).long()
    #     if self.model.conditioning_key is not None:
    #         assert c is not None
    #         if self.cond_stage_trainable:
    #             c = self.get_learned_conditioning(c)
    #         if self.shorten_cond_schedule:  # TODO: drop this option
    #             tc = self.cond_ids[t].to(self.device)
    #             c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
    #     return self.p_losses(x_src, x_tgt, c, t, *args, **kwargs)
    
    def forward(self, inputs, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (inputs["z_tgt"].shape[0],), device=self.device).long()
        # if self.model.conditioning_key is not None:
        #     assert c is not None
        #     if self.cond_stage_trainable:
        #         c = self.get_learned_conditioning(c)
        #     if self.shorten_cond_schedule:  # TODO: drop this option
        #         tc = self.cond_ids[t].to(self.device)
        #         c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        # return self.p_losses(inputs, t, *args, **kwargs)
        return self.p_losses_uncertainty(inputs, t, *args, **kwargs)

    def _rescale_annotations(self, bboxes, crop_coordinates):  # TODO: move to dataset
        def rescale_bbox(bbox):
            x0 = clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)
            return x0, y0, w, h

        return [rescale_bbox(b) for b in bboxes]

    # def apply_model(self, x_noisy, t, cond, return_ids=False):
    #     if isinstance(cond, dict):
    #         # hybrid case, cond is exptected to be a dict
    #         pass
    #     else:
    #         if not isinstance(cond, list):
    #             cond = [cond]
    #         key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
    #         cond = {key: cond}

    #     if hasattr(self, "split_input_params"):
    #         assert len(cond) == 1  # todo can only deal with one conditioning atm
    #         assert not return_ids  
    #         ks = self.split_input_params["ks"]  # eg. (128, 128)
    #         stride = self.split_input_params["stride"]  # eg. (64, 64)

    #         fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

    #         z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
    #         # Reshape to img shape
    #         z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
    #         z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

    #         cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

    #         # apply model by loop over crops
    #         output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
    #         assert not isinstance(output_list[0],
    #                               tuple)  # todo cant deal with multiple model outputs check this never happens

    #         o = torch.stack(output_list, axis=-1)
    #         o = o * weighting
    #         # Reverse reshape to img shape
    #         o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
    #         # stitch crops together
    #         x_recon = fold(o) / normalization

    #     else:
    #         x_recon = self.model(x_noisy, t, **cond)

    #     if isinstance(x_recon, tuple) and not return_ids:
    #         return x_recon[0]
    #     else:
    #         return x_recon

    def get_random_masked_indices(self, mask, rand_num=8):
        # Get all positions where mask is 0
        masked_positions = torch.where(mask == 0)
        # Extract the height indices (dim=2) where masking occurred
        masked_height_indices = masked_positions[2]
        
        # Get unique height indices (among all B, C, W, D)
        unique_height_indices = torch.unique(masked_height_indices)
        # print(unique_height_indices)
        # Safety check
        if len(unique_height_indices) < rand_num:
            raise ValueError(f"Not enough masked height indices. Required: {rand_num}, Found: {len(unique_height_indices)}")

        # Randomly select 'rand_num' height indices
        selected = unique_height_indices[torch.randperm(len(unique_height_indices))[:rand_num]]

        return selected
    
    def get_random_masked_indices_per_batch(mask: torch.Tensor, rand_num=8, axis=2):
        """
        mask: (B, C, H, W, D); extract the indices of the slices masked along axis
        return: list of index tensors (len B)
        """
        B = mask.size(0)
        selected_indices = []

        for b in range(B):
            mask_b = mask[b]  # (C, H, W, D)

            # move axis to dim=0 for uniformity
            mask_b = mask_b.permute((axis - 1,) + tuple(i for i in range(3) if i != axis - 1))  # now (Slice, ...)
            slice_vals = mask_b.view(mask_b.size(0), -1).any(dim=1)  # shape: (slice,), True if any is masked

            masked_slices = torch.where(~slice_vals)[0]  # slice indices where mask == 0

            if len(masked_slices) < rand_num:
                raise ValueError(f"Not enough masked slices in batch {b}: found {len(masked_slices)}")

            selected = masked_slices[torch.randperm(len(masked_slices))[:rand_num]]
            selected_indices.append(selected)

        return selected_indices  # list of B tensors of indices

    
    def apply_model(self, x_noisy, t, z_srcs, x_tgt_prompt, u_map=None, y=None,return_ids=False):
        x_recon = self.model(x_noisy, t, z_srcs, x_tgt_prompt, u_map=u_map, y=y)
        # from thop import profile
        # macs, params = profile(self.model, inputs=(x_noisy, t, z_srcs, x_tgt_prompt, u_map, y))
        # print(f"MACs: {macs}, Params: {params}")
        # for name, param in self.model.named_parameters():
        #     print(f"{name:60s} {param.numel():,}")
        # total_params = sum(p.numel() for p in self.model.parameters())
        # print('total params: ', total_params)
        # total_params_ex_spade = sum(
        #     p.numel()
        #     for name, p in self.model.named_parameters()
        #     if "spade" not in name.lower()
        # )
        # print(f"Total (exclude 'spade'): {total_params_ex_spade:,}")
        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon
        
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    # def p_losses(self, x_src, x_tgt,  cond, t, noise=None):
    #     noise = default(noise, lambda: torch.randn_like(x_tgt))
    #     x_noisy = self.q_sample(x_start=x_tgt, t=t, noise=noise)
    #     x_noisy = torch.cat([x_src, x_noisy], dim=1)
    #     model_output = self.apply_model(x_noisy, t, cond)

    #     loss_dict = {}
    #     prefix = 'train' if self.training else 'val'

    #     if self.parameterization == "x0":
    #         target = x_tgt
    #     elif self.parameterization == "eps":
    #         target = noise
    #     else:
    #         raise NotImplementedError()

    #     loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
    #     loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

    #     self.logvar = self.logvar.to(self.device)
    #     logvar_t = self.logvar[t].to(self.device)
    #     logvar_t = repeat(logvar_t, 'b -> b c', c=loss_simple.shape[1])
    #     loss = loss_simple / torch.exp(logvar_t) + logvar_t
    #     # loss = loss_simple / torch.exp(self.logvar) + self.logvar
    #     if self.learn_logvar:
    #         loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
    #         loss_dict.update({'logvar': self.logvar.data.mean()})

    #     loss = self.l_simple_weight * loss.mean()

    #     loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
    #     lvlb_weights = repeat(self.lvlb_weights[t], 'b -> b c', c=loss_vlb.shape[1])
    #     loss_vlb = (lvlb_weights * loss_vlb).mean()
        
    #     loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
    #     loss += (self.original_elbo_weight * loss_vlb)
    #     loss_dict.update({f'{prefix}/loss': loss})

    #     return loss, loss_dict
    
    def p_losses(self, inputs, t, noise=None):
        x_tgt = inputs["z_tgt"]
        # x_srcs = inputs["z_srcs"]
        # x_srcs_prompt = inputs["x_srcs_prompt"]
        # x_tgt_prompt = inputs["x_tgt_prompt"]
        # z_src_idx = torch.randint(0, len(inputs["z_srcs"]), (1,)).item()
        # z_src = inputs["z_srcs"][z_src_idx]

        noise = default(noise, lambda: torch.randn_like(x_tgt))

        x_noisy = self.q_sample(x_start=x_tgt, t=t, noise=noise)
        # x_noisy = torch.cat([z_src, x_noisy], dim=1)
        
        model_output = self.apply_model(x_noisy, t, inputs["z_srcs"],inputs["x_tgt_prompt"])

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_tgt
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        self.logvar = self.logvar.to(self.device)
        logvar_t = self.logvar[t].to(self.device)
        logvar_t = repeat(logvar_t, 'b -> b c', c=loss_simple.shape[1])
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        lvlb_weights = repeat(self.lvlb_weights[t], 'b -> b c', c=loss_vlb.shape[1])
        loss_vlb = (lvlb_weights * loss_vlb).mean()
        
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict
    
    def p_losses_uncertainty(self, inputs, t, noise=None):
        x_tgt = inputs["z_tgt"]
        # x_srcs = inputs["z_srcs"]
        # x_srcs_prompt = inputs["x_srcs_prompt"]
        # x_tgt_prompt = inputs["x_tgt_prompt"]
        # z_src_idx = torch.randint(0, len(inputs["z_srcs"]), (1,)).item()
        # z_src = inputs["z_srcs"][z_src_idx]
        
        noise = default(noise, lambda: torch.randn_like(x_tgt))
        # t_1 = t + 1
        x_noisy = self.q_sample(x_start=x_tgt, t=t, noise=noise)
        # uncertainty_map = None
        # uncertainty_cond = torch.rand_like(x_noisy)
        
        # if t[0] > self.num_timesteps//2:
        #     uncertainty_mask = torch.ones_like(x_noisy)
        #     uncertainty_cond = torch.rand_like(x_noisy)
        # else:
        #     # if t[0] < self.num_timesteps//2:
        #     random_int = random.randint(1, 50)
        #     random_int_mul_4 = random_int * 4
        #     while t[0] + random_int_mul_4 >= self.num_timesteps:
        #         random_int = random.randint(1, 50)
        #         random_int_mul_4 = random_int * 4
        #     uncertainty_list = []
        #     # pick three random timesteps per batch element
        #     for i in range(3):
        #         with torch.no_grad():
        #             t_1 = t + random_int * (i+1)
        #             # t_1 = torch.clamp(t_1, 0, self.num_timesteps - 1)
        #             x_noisy_u = self.q_sample(x_start=x_tgt, t=t_1, noise=None)
        #             model_output_1 = self.apply_model(x_noisy_u, t_1, inputs["z_srcs"],inputs["x_tgt_prompt"])
        #         o_1_recon = self.predict_start_from_noise(x_noisy_u, t_1, noise=model_output_1)
        #         uncertainty_list.append(o_1_recon)
        #     u_map = torch.stack(uncertainty_list, dim=1).std(dim=1)
        #     u_map = (u_map - u_map.mean()) / (u_map.std() + 1e-7)
        #     uncertainty_map = (u_map - u_map.min()) / (u_map.max() - u_map.min() + 1e-7)
        #     uncertainty_mask = uncertainty_map > uncertainty_map.mean()
        #     uncertainty_mask = uncertainty_mask * 1.0
        #     uncertainty_cond = (1-uncertainty_mask) * x_tgt + uncertainty_mask * torch.rand_like(x_tgt)

        
        
        # model_output = self.apply_model(x_noisy, t, inputs["z_srcs"],inputs["x_tgt_prompt"],u_map=uncertainty_map)
        
        rand_number = 8
        random_indices = self.get_random_masked_indices(inputs["mask"],rand_num=rand_number)
        random_indices = torch.sort(random_indices)[0]
        # print("#"*50)
        # print(random_indices)
        # print("01:",torch.sort(torch.where(inputs["mask"][0, 0, :, 0, 0] == 0)[0])[0])
        # print("02: ",torch.sort(torch.where(inputs["mask"][1, 0, :, 0, 0] == 0)[0])[0])
        # print("#"*50)
        # random_indices = torch.tensor(random_indices,dtype=torch.long)
        # print(random_indices)
        # x_noisy_masked = x_noisy[:, :, random_indices, :, :]
        
        # Randomly select which axis to mask (2, 3, or 4)
        
        axis_to_mask = 2 # default
        #####axis choose####
        axis_to_mask = random.randint(2, 4)
        
        if axis_to_mask == 3:
            inputs["mask"] = inputs["mask"].permute(0,1,3,2,4)
        elif axis_to_mask == 4:
            inputs["mask"] = inputs["mask"].permute(0,1,3,4,2)
        ######################
        
        y = torch.zeros(inputs["mask"].shape[0], dtype=torch.long, device=self.device)  # 0: axial
        # if axis_to_mask == 2:
        #     y = torch.zeros(batch_size, dtype=torch.long, device=self.device)  # 0: axial
        if axis_to_mask == 3:
            y = torch.ones(inputs["mask"].shape[0], dtype=torch.long, device=self.device)   # 1: coronal
        elif axis_to_mask == 4:
            y = torch.full((inputs["mask"].shape[0],), 2, dtype=torch.long, device=self.device)  # 2: sagittal
        
        y = y.repeat(rand_number)
        # Create a list of slice objects for all dimensions
        slices = [slice(None)] * 5  # 5 dimensions: [batch, channel, height, width, depth]
        slices[axis_to_mask] = random_indices
        
        
        mask_x_tgt  = torch.randn_like(x_tgt)
        # conditioning = torch.cat([inputs["z_src"], mask_x_tgt], dim=1)

        cond_tgt = mask_x_tgt * (1-inputs["mask"]) + x_tgt * inputs["mask"]
        conditioning = torch.cat([inputs["z_src"], cond_tgt], dim=1)
        
        # Apply the masking
        x_noisy_masked = x_noisy[tuple(slices)]
        
        if y[0] == 1:
            x_noisy_masked = x_noisy_masked.permute(0,1,3,2,4)
        elif y[0] == 2:
            x_noisy_masked = x_noisy_masked.permute(0,1,4,2,3)
        
        #######3d to 2d permute#######
        # x_noisy_masked = x_noisy_masked.
        B, C, H, W, D = x_noisy_masked.shape
        x_noisy_masked = rearrange(x_noisy_masked, 'b c h w d -> (b h) c w d')
        ##############################
        # ######flatten######
        # x_noisy_masked = torch.squeeze(x_noisy_masked)
        
        # model_output = self.apply_model(x_noisy, t, inputs["z_src"],inputs["x_tgt_prompt"],u_map=uncertainty_cond)
        # model_output = self.apply_model(x_noisy_masked, t, conditioning, inputs["x_tgt_prompt"],u_map=random_indices)
        model_output = self.apply_model(x_noisy_masked, t, conditioning, inputs["x_tgt_prompt"],u_map=tuple(slices),y=y)
        model_output = rearrange(model_output, '(b h) c w d -> b c h w d', b=B, h=H)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            # target = x_tgt[:, :, random_indices, :, :]
            # target = x_tgt[tuple(slices)]
            # target = torch.squeeze(target)
            target = x_tgt[tuple(slices)]
            if y[0] == 1:
                target = target.permute(0,1,3,2,4)
            elif y[0] == 2:
                target = target.permute(0,1,4,2,3)
        elif self.parameterization == "eps":
            # target = noise[:, :, random_indices, :, :]
            # target = noise[tuple(slices)]
            # target = torch.squeeze(target)
            target = noise[tuple(slices)]
            if y[0] == 1:
                target = target.permute(0,1,3,2,4)
            elif y[0] == 2:
                target = target.permute(0,1,4,2,3)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        # loss_simple = self.get_loss(model_output * uncertainty_mask, target * uncertainty_mask, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        self.logvar = self.logvar.to(self.device)
        logvar_t = self.logvar[t].to(self.device)
        # logvar_t = repeat(logvar_t, 'b -> b c', c=loss_simple.shape[1])
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        # loss_vlb = self.get_loss(model_output * uncertainty_mask, target * uncertainty_mask, mean=False).mean(dim=(1, 2, 3))
        # lvlb_weights = repeat(self.lvlb_weights[t], 'b -> b c', c=loss_vlb.shape[1])
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x[:,model_out.shape[1]:], t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x[:,model_out.shape[1]:], t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like((model_mean.shape), device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        img = torch.cat([x0.to(self.device), torch.randn_like(x0).to(self.device)], dim=1)
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            x_prev, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            
            img = torch.cat([x0.to(self.device), x_prev], dim=1)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img[:, x_prev.shape[1]:], intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        img = torch.cat([x0.to(device), torch.randn_like(x0).to(device)], dim=1)

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            x_prev = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)

            img = torch.cat([x0.to(self.device), x_prev], dim=1)
            
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img[:, x_prev.shape[1]:], intermediates
        return img[:, x_prev.shape[1]:]

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_size, *self.shape)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    # @torch.no_grad()
    # def sample_log(self,cond,batch_size,ddim, ddim_steps,**kwargs):
    #     if ddim:
    #         ddim_sampler = DDIMSampler(self)
    #         shape = self.shape
    #         samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
    #                                                     shape,cond,verbose=False,**kwargs)

    #     else:
    #         samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
    #                                              return_intermediates=True,**kwargs)

    #     return samples, intermediates
    
    @torch.no_grad()
    def sample_log(self,batch_size,ddim, ddim_steps,**kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = self.shape
            samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,verbose=False,**kwargs)

        else:
            samples, intermediates = self.sample(cond=None, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates


    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=False,
                   plot_diffusion_rows=True, **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        z_src, z_tgt,  x_src, x_tgt, xrec_src, xrec_tgt, source, target = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x_src.shape[0], N)
        n_row = min(x_src.shape[0], n_row)
        log[f"src_{source}"] = x_src
        log[f"tgt_{target}"] = x_tgt
        log[f"recon_{source}"] = xrec_src
        log[f"recon_{target}"] = xrec_tgt

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z_src[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy[:,:3]))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W, D
            diffusion_grid = rearrange(diffusion_row, 'n b c h w d-> b n c h w d')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w d -> (b n) c h w d')
            log["diffusion_row"] = diffusion_grid

        self.shape = z_src.shape[1:]
        # if sample:
        #     # get denoise row
        #     with self.ema_scope("Plotting"):
        #         samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
        #                                                  ddim_steps=ddim_steps,eta=ddim_eta, x0=z_src[:N])
        #     x_samples = self.decode_first_stage(samples)
        #     log["samples"] = x_samples
        #     if plot_denoise_rows:
        #         denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
        #         log["denoise_row"] = denoise_grid

        #     if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
        #             self.first_stage_model, IdentityFirstStage):
        #         # also display when quantizing x0 while sampling
        #         with self.ema_scope("Plotting Quantized Denoised"):
        #             samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
        #                                                      ddim_steps=ddim_steps,eta=ddim_eta,
        #                                                      quantize_denoised=True, x0=z_src[:N])
        #         x_samples = self.decode_first_stage(samples.to(self.device))
        #         log["samples_x0_quantized"] = x_samples


        # if plot_progressive_rows:
        #     with self.ema_scope("Plotting Progressives"):
        #         img, progressives = self.progressive_denoising(c,
        #                                                        shape=(3, self.shape[1], self.shape[2], self.shape[3]),
        #                                                        x0=z_src[:N],
        #                                                        batch_size=N)
        #     prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
        #     log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

# class DiffusionWrapper(pl.LightningModule):
#     def __init__(self, diff_model_config, conditioning_key):
#         super().__init__()
#         self.diffusion_model = instantiate_from_config(diff_model_config)
#         self.conditioning_key = conditioning_key
#         assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

#     def forward(self, x, t, z_srcs, x_tgt_prompt, c_concat: list = None, c_crossattn: list = None):
#         if self.conditioning_key is None:
#             cc = torch.cat(c_crossattn, 1)
#             out = self.diffusion_model(x, t, z_srcs, x_tgt_prompt, context=cc)
#         elif self.conditioning_key == 'concat':
#             xc = torch.cat([x] + c_concat, dim=1)
#             out = self.diffusion_model(xc, t, z_srcs, x_tgt_prompt)
#         elif self.conditioning_key == 'crossattn':
#             cc = torch.cat(c_crossattn, 1)
#             out = self.diffusion_model(x, t, z_srcs, x_tgt_prompt, context=cc)
#         elif self.conditioning_key == 'hybrid':
#             xc = torch.cat([x] + c_concat, dim=1)
#             cc = torch.cat(c_crossattn, 1)
#             out = self.diffusion_model(xc, t, z_srcs, x_tgt_prompt, context=cc)
#         elif self.conditioning_key == 'adm':
#             cc = c_crossattn[0]
#             out = self.diffusion_model(x, t, z_srcs, x_tgt_prompt, y=cc)
#         else:
#             raise NotImplementedError()

#         return out

class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, z_srcs, x_tgt_prompt, u_map=None, y = None, c_concat: list = None, c_crossattn: list = None):
        model_kwargs = {}
        model_kwargs["z_srcs"] = z_srcs
        model_kwargs["x_tgt_prompt"] = x_tgt_prompt
        model_kwargs["u_map"] = u_map
        model_kwargs["y"] = y
        out = self.diffusion_model(x, t, **model_kwargs)
        return out


class Layout2ImgDiffusion(LatentDiffusion):
    # TODO: move all layout-specific hacks to this class
    def __init__(self, cond_stage_key, *args, **kwargs):
        assert cond_stage_key == 'coordinates_bbox', 'Layout2ImgDiffusion only for cond_stage_key="coordinates_bbox"'
        super().__init__(cond_stage_key=cond_stage_key, *args, **kwargs)

    def log_images(self, batch, N=8, *args, **kwargs):
        logs = super().log_images(batch=batch, N=N, *args, **kwargs)

        key = 'train' if self.training else 'validation'
        dset = self.trainer.datamodule.datasets[key]
        mapper = dset.conditional_builders[self.cond_stage_key]

        bbox_imgs = []
        map_fn = lambda catno: dset.get_textual_label(dset.get_category_id(catno))
        for tknzd_bbox in batch[self.cond_stage_key][:N]:
            bboximg = mapper.plot(tknzd_bbox.detach().cpu(), map_fn, (256, 256))
            bbox_imgs.append(bboximg)

        cond_img = torch.stack(bbox_imgs, dim=0)
        logs['bbox_image'] = cond_img
        return logs
