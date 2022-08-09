import torch
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
from tensorfn import load_config

from model import UNet
from config import DiffusionConfig
# from diffusion import GaussianDiffusion, make_beta_schedule
import argparse

import pdb
st = pdb.set_trace


@torch.no_grad()
def p_sample_loop(self, model, noise, device, noise_fn=torch.randn, capture_every=1000, z=None):
    img = noise
    imgs = []

    for i in tqdm(reversed(range(self.num_timesteps)), total=self.num_timesteps):
        img = self.p_sample(
            model,
            img,
            torch.full((img.shape[0],), i, dtype=torch.int64).to(device),
            noise_fn=noise_fn,
            z=z,
        )

        if i % capture_every == 0:
            imgs.append(img)

    imgs.append(img)

    return imgs


if __name__ == "__main__":
    # conf = load_config(DiffusionConfig, "config/diffusion.conf", show=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/divae_cf.conf')
    parser.add_argument('--prefix', type=str, default='test_divae')
    parser.add_argument('--timestep_respacing', type=str, default="")
    parser.add_argument('--ckpt', type=str, default='diffusion.pt')
    parser.add_argument('--legacy', action='store_true')
    args = parser.parse_args()

    conf = load_config(DiffusionConfig, args.config, show=False)
    device = 'cuda'

    from torchvision import transforms
    from dataset import MultiResolutionDataset

    # NOTE: betas
    betas = conf.diffusion.beta_schedule.make()
    
    # NOTE: make Diffusion model
    # diffusion = GaussianDiffusion(betas).to("cuda")
    # NOTE: SpacedDiffusion
    from respace import SpacedDiffusion, space_timesteps
    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(len(betas), args.timestep_respacing),
        betas=betas,
        rescale_timesteps=False,
    ).to(device)

    # NOTE: load UNet
    ckpt = torch.load(args.ckpt)
    print(f"[checkpoint {args.ckpt} loaded]")
    model = conf.model.make()
    model.load_state_dict(ckpt["ema"])
    model = model.to(device)
    
    # NOTE: load VQGAN
    from utils_ext import get_vqgan_model
    from utils_ext import encode, decode, decode_z
    ae = get_vqgan_model()
    ae.to(device)

    # NOTE: load data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    test_set = MultiResolutionDataset(
        conf.dataset.path, transform, conf.dataset.resolution
    )
    n = 8
    n2 = n // 2
    real_imgs = torch.stack([test_set[i] for i in range(n)])
    real_imgs = real_imgs.to(device)
    w = real_imgs.shape[3]
    w2 = w // 2
    h = real_imgs.shape[2]
    h2 = h // 2
    ww2 = hh2 = 16 // 2
    ff = 16  # f

    noise = torch.randn([n, 3, 256, 256], device="cuda")

    prefix = args.prefix

    with torch.no_grad():

        # reconstruction
        zq, z, _ = encode(ae, real_imgs)
        x_vq = decode(ae, zq)
        imgs = p_sample_loop(diffusion, model, noise, "cuda", capture_every=100, z=z)

        imgs = imgs[1:][-1]
        imgs_all = torch.cat([real_imgs, imgs, x_vq], dim=0)
        save_image(imgs_all, f"{prefix}_recon.png", normalize=True, range=(-1, 1), nrow=n)
        del imgs_all

        exit(0)

        # swap right half
        real_swap = torch.cat([real_imgs[:,:,:,:w2], torch.cat([real_imgs[n2:,:,:,w2:], real_imgs[:n2,:,:,w2:]], dim=0)], dim=3)
        z_swap = torch.cat([z[:,:,:,:ww2], torch.cat([z[n2:,:,:,ww2:], z[:n2,:,:,ww2:]], dim=0)], dim=3)
        zq_swap = torch.cat([zq[:,:,:,:ww2], torch.cat([zq[n2:,:,:,ww2:], zq[:n2,:,:,ww2:]], dim=0)], dim=3)
        x_vq_swap = decode(ae, zq_swap)
        imgs = p_sample_loop(diffusion, model, noise, "cuda", capture_every=100, z=z_swap)
        imgs = imgs[1:][-1]
        imgs_all = torch.cat([real_swap, imgs, x_vq_swap], dim=0)
        save_image(imgs_all, f"{prefix}_swaplr.png", normalize=True, range=(-1, 1), nrow=n)
        del imgs_all, z_swap, zq_swap, x_vq_swap
        
        # swap top half
        real_swap = torch.cat([real_imgs[:,:,:h2,:], torch.cat([real_imgs[n2:,:,h2:,:], real_imgs[:n2,:,h2:,:]], dim=0)], dim=2)
        z_swap = torch.cat([z[:,:,:hh2,:], torch.cat([z[n2:,:,hh2:,:], z[:n2,:,hh2:,:]], dim=0)], dim=2)
        zq_swap = torch.cat([zq[:,:,:hh2,:], torch.cat([zq[n2:,:,hh2:,:], zq[:n2,:,hh2:,:]], dim=0)], dim=2)
        x_vq_swap = decode(ae, zq_swap)
        imgs = p_sample_loop(diffusion, model, noise, "cuda", capture_every=100, z=z_swap)
        imgs = imgs[1:][-1]
        imgs_all = torch.cat([real_swap, imgs, x_vq_swap], dim=0)
        save_image(imgs_all, f"{prefix}_swapud.png", normalize=True, range=(-1, 1), nrow=n)
        del imgs_all, z_swap, zq_swap, x_vq_swap

        # swap region
        z_swap = z.clone()
        z_swap[:n2,:,6:9,4:12] = z[n2:,:,6:9,4:12]
        z_swap[n2:,:,6:9,4:12] = z[:n2,:,6:9,4:12]
        zq_swap = zq.clone()
        zq_swap[:n2,:,6:9,4:12] = zq[n2:,:,6:9,4:12]
        zq_swap[n2:,:,6:9,4:12] = zq[:n2,:,6:9,4:12]
        real_swap = real_imgs.clone()
        real_swap[:n2,:,6*ff:9*ff,4*ff:12*ff] = real_imgs[n2:,:,6*ff:9*ff,4*ff:12*ff]
        real_swap[n2:,:,6*ff:9*ff,4*ff:12*ff] = real_imgs[:n2,:,6*ff:9*ff,4*ff:12*ff]
        x_vq_swap = decode(ae, zq_swap)
        imgs = p_sample_loop(diffusion, model, noise, "cuda", capture_every=100, z=z_swap)
        imgs = imgs[1:][-1]
        imgs_all = torch.cat([real_swap, imgs, x_vq_swap], dim=0)
        save_image(imgs_all, f"{prefix}_swapeye.png", normalize=True, range=(-1, 1), nrow=n)
        del imgs_all, z_swap, zq_swap, x_vq_swap
