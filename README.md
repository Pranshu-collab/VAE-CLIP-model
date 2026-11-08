# VAE-CLIP-model
Dual-branch variational autoencoders over frozen CLIP embeddings that map images and captions into a shared latent space, with normalizing-flow posteriors and a contrastive alignment objective.

## Overview
CLIP embeds images and text into a joint space, but that space is not designed for generation or smooth interpolation. This project learns a probabilistic latent space on top of CLIP: each modality gets its own VAE, flows make the posterior more expressive than a diagonal Gaussian, and a contrastive (InfoNCE) loss pulls paired image and text latents together.

The goals are to:

Compress and reconstruct CLIP image and text embeddings through a stochastic bottleneck.
Align paired image and caption latents so that cross-modal retrieval works in the learned space.
Study whether the latent space is smooth enough to interpolate in and generate from.
Architecture
image   -> CLIP ViT-B/32 image encoder (frozen) -> f_img (512-d, L2-normalized)
                -> VAE_img: encoder -> flow -> z_img -> decoder -> f_img_hat

caption -> CLIP ViT-B/32 text encoder (frozen)  -> f_txt (512-d, L2-normalized)
                -> VAE_txt: encoder -> flow -> z_txt -> decoder -> f_txt_hat

                  InfoNCE( z_img , z_txt )   (symmetric, in-batch negatives)
* Feature extractor: pretrained CLIP ViT-B/32, kept frozen. Image and text features are L2-normalized.
* Per-modality VAE: MLP encoder and decoder with reparameterized sampling.
* Normalizing flow: Flows applied to the posterior sample to increase expressiveness (flow type and length are configurable).
* Alignment: symmetric InfoNCE loss between paired image and text latents, temperature 0.07.
  
### Loss
L = L_rec(img) + L_rec(txt) + beta * (KL_img + KL_txt) + lambda * L_align

## Dataset
Flickr8k: about 8,000 images with five captions each. CLIP embeddings are extracted once per image-caption pair; the VAEs are trained on those embeddings.

## Training notes

With unit-norm 512-d CLIP features, the per-dimension reconstruction error is tiny compared with the KL term, which can push the posterior toward the prior (posterior collapse). The decoder then ignores z, and alignment cannot form. Mitigations used or tested here:
* Standardizing CLIP features per dimension before the VAE.
* KL warm-up (beta ramped from 0) and free bits per latent dimension.
* Computing the alignment loss on the posterior mean instead of the sampled z.
* Precomputing CLIP features once, since the encoders are frozen, so experiments run on stored tensors.
