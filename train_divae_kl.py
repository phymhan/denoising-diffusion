import os

import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from tensorfn import load_arg_config, load_wandb
from tensorfn import distributed as dist
from tensorfn.optim import lr_scheduler
from tqdm import tqdm

from model import UNet
from diffusion import GaussianDiffusion, make_beta_schedule
from dataset import MultiResolutionDataset
from config import DiffusionConfig
from torchvision.utils import save_image
from generate import p_sample_loop
# import wandb
import pdb
st = pdb.set_trace

from utils_ext import encode_kl, decode_kl
import random
import pickle

def sample_data(loader):
    loader_iter = iter(loader)
    epoch = 0

    while True:
        try:
            yield epoch, next(loader_iter)

        except StopIteration:
            epoch += 1
            loader_iter = iter(loader)

            yield epoch, next(loader_iter)


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def train(conf, loader, model, ema, ae, diffusion, optimizer, scheduler, device, wandb):
    log_dir = os.path.join(conf.logging.log_root, conf.logging.name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "weight"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "sample"), exist_ok=True)

    loader = sample_data(loader)
    batch_size = conf.training.dataloader.batch_size

    image_size = conf.dataset.resolution
    n_sample = min(16, conf.training.dataloader.batch_size)
    noise = torch.randn([n_sample, 3, image_size, image_size], device="cuda")
    _, img_fixed = next(loader)
    img_fixed = img_fixed.to(device)
    with torch.no_grad():
        zq_fixed, z_fixed, _ = encode_kl(ae, img_fixed)
        imgs_ae_fixed = decode_kl(ae, zq_fixed)
    del zq_fixed

    z_baseline = None
    if conf.training.classifier_free:
        if not os.path.exists('mean_image.pkl'):
            img_ = torch.zeros(3, image_size, image_size)
            for i in range(10000):
                epoch, img = next(loader)
                img_ += img.mean(dim=0)
            img_ /= 10000
            with open('mean_image.pkl', 'wb') as f:
                pickle.dump(img_, f)
        else:
            with open('mean_image.pkl', 'rb') as f:
                img_ = pickle.load(f)
        img_ = img_.unsqueeze(0).to(device)
        z_baseline, _, _ = encode_kl(ae, img_)
        z_baseline = z_baseline.repeat(batch_size, 1, 1, 1)

    if conf.distributed:
        model_module = model.module
    else:
        model_module = model

    pbar = range(conf.training.n_iter + 1)

    if dist.is_primary():
        pbar = tqdm(pbar, dynamic_ncols=True)

    for i in pbar:
        epoch, img = next(loader)
        img = img.to(device)
        time = torch.randint(
            0,
            conf.diffusion.beta_schedule["n_timestep"],
            (img.shape[0],),
            device=device,
        )
        with torch.no_grad():
            if conf.training.classifier_free and random.random() < 0.3:
                z = z_baseline
            else:
                zq, z, _ = encode_kl(ae, img)
        loss = diffusion.p_loss(model, img, time, z=z)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        scheduler.step()
        optimizer.step()

        accumulate(
            ema, model_module, 0 if i < conf.training.scheduler.warmup else 0.9999
        )

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_description(
                f"epoch: {epoch}; loss: {loss.item():.4f}; lr: {lr:.5f}"
            )

            if wandb is not None and i % conf.logging.log_every == 0:
                wandb.log({"epoch": epoch, "loss": loss.item(), "lr": lr}, step=i)

            if i % conf.logging.save_every == 0:
                if conf.distributed:
                    model_module = model.module

                else:
                    model_module = model

                torch.save(
                    {
                        "model": model_module.state_dict(),
                        "ema": ema.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "conf": conf,
                    },
                    os.path.join(log_dir, "weight", f"diffusion_{str(i).zfill(6)}.pt"),
                )

                if conf.training.classifier_free:
                    # sample one with z_fixed and one with baseline
                    imgs = p_sample_loop(diffusion, ema, noise, "cuda", capture_every=100, z=z_fixed)
                    img1 = imgs[1:][-1]

                    imgs = p_sample_loop(diffusion, ema, noise, "cuda", capture_every=100, z=z_baseline)
                    img2 = imgs[1:][-1]
                else:
                    # sample one with fixed noise and one with random noise
                    # sample with fixed noise
                    imgs = p_sample_loop(diffusion, ema, noise, "cuda", capture_every=100, z=z_fixed)
                    img1 = imgs[1:][-1]

                    # sample with random noise
                    noise2 = torch.randn([n_sample, 3, image_size, image_size], device="cuda")
                    imgs = p_sample_loop(diffusion, ema, noise2, "cuda", capture_every=100, z=z_fixed)
                    img2 = imgs[1:][-1]

                imgs_all = torch.cat((img_fixed, imgs_ae_fixed, img1, img2), dim=0)
                save_image(imgs_all, os.path.join(log_dir, "sample", f"{i:07d}_rec.png"),
                    normalize=True, range=(-1, 1), nrow=n_sample)


def main(conf):
    wandb = None
    if dist.is_primary() and conf.logging.wandb:
        wandb = load_wandb()
        # wandb.init(project="denoising diffusion")
        wandb.init(
            settings=wandb.Settings(start_method='fork'),
            project=conf.logging.wandb_project,
            name=conf.logging.name,
            entity=conf.logging.wandb_entity,
            resume=False,
            config=conf,
        )

    device = "cuda"
    beta_schedule = "linear"

    conf.distributed = dist.get_world_size() > 1

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    train_set = MultiResolutionDataset(
        conf.dataset.path, transform, conf.dataset.resolution
    )
    train_sampler = dist.data_sampler(
        train_set, shuffle=True, distributed=conf.distributed
    )
    train_loader = conf.training.dataloader.make(train_set, sampler=train_sampler)

    model = conf.model.make()
    model = model.to(device)
    ema = conf.model.make()
    ema = ema.to(device)

    if conf.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = conf.training.optimizer.make(model.parameters())
    scheduler = conf.training.scheduler.make(optimizer)

    if conf.ckpt is not None:
        ckpt = torch.load(conf.ckpt, map_location=lambda storage, loc: storage)

        if conf.distributed:
            model.module.load_state_dict(ckpt["model"])

        else:
            model.load_state_dict(ckpt["model"])

        ema.load_state_dict(ckpt["ema"])

    betas = conf.diffusion.beta_schedule.make()
    diffusion = GaussianDiffusion(betas).to(device)

    # Setup VQGAN
    from utils_ext import get_vqgan_model
    ae = get_vqgan_model(which_model='kl-f16')
    ae.to(device)

    train(
        conf, train_loader, model, ema, ae, diffusion, optimizer, scheduler, device, wandb
    )


if __name__ == "__main__":
    conf = load_arg_config(DiffusionConfig)

    dist.launch(
        main, conf.n_gpu, conf.n_machine, conf.machine_rank, conf.dist_url, args=(conf,)
    )
