import torch
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
from tensorfn import load_config

from model import UNet
from config import DiffusionConfig
from diffusion_debug import GaussianDiffusion, make_beta_schedule
import pdb
st = pdb.set_trace


@torch.no_grad()
def p_sample_loop(self, model, noise, device, noise_fn=torch.randn, capture_every=1000):
    img = noise
    imgs = []
    maxval = 5

    for i in tqdm(reversed(range(self.num_timesteps)), total=self.num_timesteps):
        img, mu, eps, x_0 = self.p_sample(
            model,
            img,
            torch.full((img.shape[0],), i, dtype=torch.int64).to(device),
            noise_fn=noise_fn,
        )
        if i == self.num_timesteps -1 or i % capture_every == 0:
            # imgs.append(torch.clamp(eps, -maxval, maxval) / maxval)
            # imgs.append(torch.clamp(x_0, -maxval, maxval))
            # imgs.append(torch.clamp(mu, -maxval, maxval) / maxval)
            # imgs.append(torch.clamp(img, -maxval, maxval) / maxval)
            imgs.append(torch.clamp(eps, -maxval, maxval) / maxval)
            imgs.append(torch.clamp(x_0, -maxval, maxval))
            imgs.append(torch.clamp(mu, -maxval, maxval))
            imgs.append(torch.clamp(img, -maxval, maxval))

    imgs.append(img)

    return imgs


if __name__ == "__main__":
    conf = load_config(DiffusionConfig, "config/ffhq_128_64.json", show=False)
    ckpt = torch.load(f"logs/ffhq_unet64/weight/diffusion_230000.pt")
    model = conf.model.make()
    model.load_state_dict(ckpt["ema"])
    model = model.to("cuda")
    betas = conf.diffusion.beta_schedule.make()
    diffusion = GaussianDiffusion(betas).to("cuda")
    noise = torch.randn([16, 3, 128, 128], device="cuda")
    imgs = p_sample_loop(diffusion, model, noise, "cuda", capture_every=100)
    imgs = imgs[:-1]
    num_capture = len(imgs) // 4

    imgs = torch.stack(imgs, dim=1)
    imgs = imgs.view(16 * num_capture * 4, 3, 128, 128)

    save_image(imgs, f"sample_debug3.png", normalize=True, range=(-1, 1), nrow=4*num_capture)
    print("eps, x_0, mu, x_t")
