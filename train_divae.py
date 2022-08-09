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
from einops import rearrange
# import wandb
import pdb
st = pdb.set_trace

# from utils_ext import encode, decode, decode_z
import random


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


def train(conf, args, loader, val_loader, model, ema, ae, diffusion, optimizer, scheduler, device, wandb):
    log_dir = os.path.join(conf.logging.log_root, conf.logging.name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "weight"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "sample"), exist_ok=True)

    if args.which_vqgan.startswith('kl'):
        from utils_ext import encode_kl as encode_fn
        from utils_ext import decode_kl as decode_fn
    else:
        from utils_ext import encode as encode_fn
        from utils_ext import decode as decode_fn

    loader = sample_data(loader)
    batch_size = conf.training.dataloader.batch_size

    if args.random_crop:
        image_size = args.crop_size
    else:
        # image_size = conf.dataset.resolution
        image_size = args.size
    image_size_val = args.size
    n_sample = min(args.n_sample, conf.training.dataloader.batch_size)
    
    noise = torch.randn([n_sample, 3, image_size, image_size], device="cuda")
    noise_val = torch.randn([n_sample, 3, image_size_val, image_size_val], device="cuda")

    _, img_fixed = next(loader)
    img_fixed = img_fixed.to(device)

    #----- full size images from val_loader -----#
    val_loader = sample_data(val_loader)
    _, img_fixed_val = next(val_loader)
    img_fixed_val = img_fixed_val.to(device)
    with torch.no_grad():
        zq_fixed, z_fixed, _ = encode_fn(ae, img_fixed)
        if args.quantize:
            z_fixed = zq_fixed
        imgs_ae_fixed = decode_fn(ae, zq_fixed)
        zq_fixed_val, z_fixed_val, _ = encode_fn(ae, img_fixed_val)
        imgs_ae_fixed_val = decode_fn(ae, zq_fixed_val)
    del zq_fixed, zq_fixed_val

    zq_baseline = None
    mask_token = -1
    mask_embed = None
    if conf.training.classifier_free:
        if args.bert:
            vocab_size, embed_dim = ae.quantize.embedding.weight.data.shape
            mask_token = vocab_size
            mask_embed = ae.quantize.embedding.weight.data.mean(0).detach().clone().to(device)
            zq_baseline = mask_embed.repeat(n_sample, 16, 16, 1).permute(0, 3, 1, 2)  # NOTE: hardcoded
        else:
            ind_baseline = torch.zeros(n_sample*16*16, device=device).long()  # NOTE: hardcoded
            zq_baseline = ae.quantize.embedding(ind_baseline)
            zq_baseline = zq_baseline.reshape(n_sample, 16, 16, -1).permute(0, 3, 1, 2)  # NOTE: hardcoded

    if conf.distributed:
        model_module = model.module
    else:
        model_module = model

    pbar = range(conf.training.n_iter + 1)

    if dist.is_primary():
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True)

    for idx in pbar:
        i = idx + args.start_iter

        epoch, img = next(loader)
        img = img.to(device)
        time = torch.randint(
            0,
            conf.diffusion.beta_schedule["n_timestep"],
            (img.shape[0],),
            device=device,
        )
        with torch.no_grad():
            if conf.training.classifier_free:
                if args.bert:
                    zq, z, ind = encode_fn(ae, img)
                    zq = rearrange(zq, 'b c h w -> b h w c')
                    mask = torch.rand(*zq.shape[:3], device=device) < 0.15  # NOTE: hardcoded
                    zq[mask] = mask_embed
                    zq = rearrange(zq, 'b h w c -> b c h w')
                    z = zq
                else:
                    z = zq_baseline if random.random() < 0.15 else z  # NOTE: hardcoded
            else:
                zq, z, ind = encode_fn(ae, img)
                if args.quantize:
                    z = zq
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

                if args.random_crop:
                    # NOTE: sample full size image
                    imgs = p_sample_loop(diffusion, ema, noise_val, "cuda", capture_every=100, z=z_fixed_val)
                    img0 = imgs[1:][-1]
                    imgs_all = torch.cat((img_fixed_val, imgs_ae_fixed_val, img0), dim=0)
                    save_image(imgs_all, os.path.join(log_dir, "sample", f"{i:07d}_val.png"),
                        normalize=True, range=(-1, 1), nrow=n_sample)

                if conf.training.classifier_free:
                    # sample one with z_fixed and one with baseline
                    imgs = p_sample_loop(diffusion, ema, noise, "cuda", capture_every=100, z=z_fixed)
                    img1 = imgs[1:][-1]

                    imgs = p_sample_loop(diffusion, ema, noise, "cuda", capture_every=100, z=zq_baseline)
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

    train_set = get_image_dataset(args, args.dataset, args.path,
        random_crop=args.random_crop)
    val_set = get_image_dataset(args, args.dataset, args.path,
        random_crop=False)

    train_sampler = dist.data_sampler(
        train_set, shuffle=True, distributed=conf.distributed
    )
    train_loader = conf.training.dataloader.make(train_set, sampler=train_sampler)

    val_sampler = dist.data_sampler(
        val_set, shuffle=False, distributed=conf.distributed
    )
    val_loader = conf.training.dataloader.make(val_set, sampler=val_sampler)

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

    if args.resume:
        if conf.ckpt is not None:
            ckpt = torch.load(conf.ckpt, map_location=lambda storage, loc: storage)

            if conf.distributed:
                model.module.load_state_dict(ckpt["model"])

            else:
                model.load_state_dict(ckpt["model"])

            ema.load_state_dict(ckpt["ema"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])

            print(f"resume from iteration {args.start_iter}!!!")

    betas = conf.diffusion.beta_schedule.make()
    diffusion = GaussianDiffusion(betas).to(device)

    # Setup VQGAN
    from utils_ext import get_vqgan_model
    ae = get_vqgan_model(which_model=args.which_vqgan)
    ae.to(device)

    train(
        conf, args, train_loader, val_loader, model, ema, ae, diffusion, optimizer, scheduler, device, wandb
    )


if __name__ == "__main__":
    defaults = dict(
        path=None,
        name=None,
        size=256,
        batch=8,
        iter=800000,
        dataset='multires',
        random_crop=False,
        crop_size=0,
        sample_cache=None,
        resume=False,
        start_iter=0,
        which_vqgan='vq-f16',
        n_sample=16,
        bert=False,
        quantize=True,
    )
    conf, args = load_arg_config(DiffusionConfig, add_dict=defaults)
    conf.logging.name = args.name or conf.logging.name
    conf.training.dataloader.batch_size = args.batch

    if args.bert:
        assert args.quantize

    dist.launch(
        main, conf.n_gpu, conf.n_machine, conf.machine_rank, conf.dist_url, args=(conf, args,)
    )
