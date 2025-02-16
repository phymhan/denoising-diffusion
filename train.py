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
from dataset import get_image_dataset
from config import DiffusionConfig
from torchvision.utils import save_image
from generate import p_sample_loop
import argparse
# import wandb
import pdb
st = pdb.set_trace

def sample_data(loader):
    loader_iter = iter(loader)
    epoch = 0

    while True:
        try:
            img = next(loader_iter)
            if isinstance(img, list):
                img = img[0]
            yield epoch, img

        except StopIteration:
            epoch += 1
            loader_iter = iter(loader)
            img = next(loader_iter)
            if isinstance(img, list):
                img = img[0]
            yield epoch, img


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def train(conf, loader, model, ema, diffusion, optimizer, scheduler, device, wandb):
    log_dir = os.path.join(conf.logging.log_root, conf.logging.name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "weight"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "sample"), exist_ok=True)

    image_size = conf.dataset.resolution
    noise = torch.randn([16, 3, image_size, image_size], device="cuda")

    loader = sample_data(loader)

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
        loss = diffusion.p_loss(model, img, time)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        scheduler.step()
        optimizer.step()

        accumulate(
            ema, model.module, 0 if i < conf.training.scheduler.warmup else 0.9999
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

                imgs = p_sample_loop(diffusion, ema, noise, "cuda", capture_every=10)
                imgs = imgs[1:]
                save_image(imgs[-1], os.path.join(log_dir, "sample", f"{i:07d}.png"),
                    normalize=True, range=(-1, 1), nrow=4)


def main(conf, args):
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

    # transform = transforms.Compose(
    #     [
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    #     ]
    # )
    # train_set = MultiResolutionDataset(
    #     conf.dataset.path, transform, conf.dataset.resolution
    # )
    # assert conf.dataset.resolution == args.size
    # assert conf.dataset.path == args.path
    train_set = get_image_dataset(args, args.dataset, args.path,
        random_crop=args.random_crop)

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

    train(
        conf, train_loader, model, ema, diffusion, optimizer, scheduler, device, wandb
    )


if __name__ == "__main__":
    defaults = dict(
        path=None,
        name=None,
        size=256,
        batch=16,
        iter=800000,
        dataset='multires',
        random_crop=False,
        crop_size=0,
    )
    conf, args = load_arg_config(DiffusionConfig, add_dict=defaults)
    conf.logging.name = args.name or conf.logging.name

    dist.launch(
        main, conf.n_gpu, conf.n_machine, conf.machine_rank, conf.dist_url, args=(conf, args,)
    )
