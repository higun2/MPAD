# Third-party notices

- **CompVis latent-diffusion**: https://github.com/CompVis/latent-diffusion . Stage 2 derives from this codebase. The supplied MIT license is preserved in `mpad/LICENSE.latent-diffusion`. Stage 1's Gaussian-distribution utility also originates here.
- **CompVis taming-transformers**: https://github.com/CompVis/taming-transformers . Modified stage 1 code, and the vector-quantizer utility retained under `mpad/taming/`, derive from this project. The upstream `License.txt` is preserved as `LICENSE.taming-transformers`.
- **OpenCLIP / BiomedCLIP**: https://github.com/mlfoundations/open_clip and https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 . Used for the contrast text conditioning in stage 2; pretrained assets are downloaded separately and are not bundled.
- **OpenAI CLIP**: https://github.com/openai/CLIP . Imported by the discrete-codebook `VQModel` route in `vqgan/taming/models/vqgan.py`. The stage 1 KL autoencoder used in the paper does not use it. Source and pretrained assets are not bundled.

Source-level attribution comments are retained. Third-party dependencies and pretrained models retain their respective terms. No license for the MPAD-specific additions is inferred from upstream licenses.
