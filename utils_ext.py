import os
import sys
import shutil
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from glob import glob
import torch
import torch.nn.functional as F
try:
    import wandb
except ImportError:
    wandb = None


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# modified from
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.\
# ipynb#scrollTo=RI1Y8bSImD7N

# test using a knn monitor
def knn_monitor(net, memory_data_loader, test_data_loader, device='cuda', k=200, t=0.1, hide_progress=False,
                targets=None):
    if not targets:
        targets = memory_data_loader.dataset.targets
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in memory_data_loader:
            feature = net(data.to(device=device, non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        for data, target in test_data_loader:
            data, target = data.to(device=device, non_blocking=True), target.to(device=device, non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
    return total_top1 / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


import matplotlib.pyplot as plt

def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.requires_grad = on_or_off

def image2tensor(image):
    image = torch.FloatTensor(image).permute(2,0,1).unsqueeze(0)/255.
    return (image-0.5)/0.5

def tensor2image(tensor):
    # tensor = tensor.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy()
    tensor = torch.clamp(tensor, -1, 1).detach().squeeze().permute(1,2,0).cpu().numpy()
    return tensor*0.5 + 0.5

def imshow(img, size=5, cmap='jet'):
    plt.figure(figsize=(size,size))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()

def imsave(name, img, size=5, cmap='jet'):
    plt.imsave(name, img, cmap=cmap)


# ========== Convenience functions for logging ==========
class Config(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class logging_file(object):
    def __init__(self, path, mode='a+', time_stamp=True, **kwargs):
        self.path = path
        self.mode = mode
        if time_stamp:
            # self.path = self.path + '_' + time.strftime('%Y%m%d_%H%M%S')
            # self.write(f'{time.strftime("%Y%m%d_%H%M%S")}\n')
            self.write(f'{datetime.now()}\n')
    
    def write(self, line_to_print):
        with open(self.path, self.mode) as f:
            f.write(line_to_print)

    def flush(self):
        pass

    def close(self):
        pass

    def __del__(self):
        pass


import logging
def log(output, flush=True):
    logging.info(output)
    if flush:
        print(output)


def setup_wandb_run_id(log_dir, resume=False):
    # NOTE: if resume, use the existing wandb run id, otherwise create a new one
    os.makedirs(log_dir, exist_ok=True)
    file_path = Path(log_dir) / 'wandb_run_id.txt'
    if resume:
        assert file_path.exists(), 'wandb_run_id.txt does not exist'
        with open(file_path, 'r') as f:
            run_id = f.readlines()[-1].strip()  # resume from the last run
    else:
        run_id = wandb.util.generate_id()
        with open(file_path, 'a+') as f:
            f.write(run_id + '\n')
    return run_id


def setup_wandb(args):
    # try:
    #     import wandb
    if wandb is not None:
        name = Path(args.log_dir).name
        resume = getattr(args, 'resume', False)
        run_id = setup_wandb_run_id(args.log_dir, resume)
        args.wandb_run_id = run_id
        run = wandb.init(
            project=args.wandb_project,
            name=name,
            id=run_id,
            config=args,
            resume=True if resume else "allow",
        )
        return run
    else:
        log_str = "Failed to set up wandb - aborting"
        log(log_str, level="error")
        raise RuntimeError(log_str)


def get_hostname():
    try:
        import socket
        return socket.gethostname()
    except:
        return 'unknown'

def print_args(parser, args, is_dict=False):
    # args = deepcopy(args)
    if not is_dict and hasattr(args, 'parser'):
        delattr(args, 'parser')
    name = getattr(args, 'name', Path(args.log_dir).name if hasattr(args, 'log_dir') else 'unknown')
    datetime_now = datetime.now()
    message = f"Name: {name} Time: {datetime_now}\n"
    message += f"{os.getenv('USER')}@{get_hostname()}:\n"
    if os.getenv('CUDA_VISIBLE_DEVICES'):
        message += f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}\n"
    message += '--------------- Arguments ---------------\n'
    args_vars = args if is_dict else vars(args)
    for k, v in sorted(args_vars.items()):
        comment = ''
        default = None if parser is None else parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '------------------ End ------------------'
    # print(message)  # suppress messages to std out

    # save to the disk
    log_dir = Path(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir / 'src', exist_ok=True)
    file_name = log_dir / 'args.txt'
    with open(file_name, 'a+') as f:
        f.write(message)
        f.write('\n\n')

    # save command to disk
    file_name = log_dir / 'cmd.txt'
    with open(file_name, 'a+') as f:
        f.write(f'Time: {datetime_now}\n')
        if os.getenv('CUDA_VISIBLE_DEVICES'):
            f.write('CUDA_VISIBLE_DEVICES=%s ' % os.getenv('CUDA_VISIBLE_DEVICES'))
        f.write('deepspeed ' if getattr(args, 'deepspeed', False) else 'python3 ')
        f.write(' '.join(sys.argv))
        f.write('\n\n')

    # backup train code
    shutil.copyfile(sys.argv[0], log_dir / 'src' / f'{os.path.basename(sys.argv[0])}.txt')


def get_last_checkpoint(ckpt_dir, ckpt_ext='.pt', latest=None):
    assert ckpt_ext.startswith('.')
    if latest is None:
        ckpt_path = sorted(glob(os.path.join(ckpt_dir, '*'+ckpt_ext)), key=os.path.getmtime, reverse=True)[0]
    else:
        if not latest.endswith(ckpt_ext):
            latest += ckpt_ext
        ckpt_path = Path(ckpt_dir) / latest
    return ckpt_path

import ast
def str2list(s, sep=','):
    assert isinstance(s, str)
    s = s.strip()
    if s.endswith(('.npy', '.npz')):
        s = np.load(s)
    elif s.startswith('[') and s.endswith(']'):
        s = ast.literal_eval(s)
    else:
        s = list(filter(None, s.split(sep)))
    return s

import importlib
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_vqgan_model(which_model='vq-f16'):
    from taming.models.vqgan import VQModel
    from omegaconf import OmegaConf

    if which_model == 'kl-f16':
        vqgan_config_path = '/research/cbim/medical/lh599/code/latent-diffusion/configs/autoencoder/autoencoder_kl_16x16x16.yaml'
        vqgan_model_path = '/research/cbim/medical/lh599/code/latent-diffusion/pretrained/kl-f16.ckpt'
        config = OmegaConf.load(vqgan_config_path)
        # config.model.params['ddconfig']['resolution'] = 256  # NOTE: yeah
        vqgan = instantiate_from_config(config.model)
        state = torch.load(vqgan_model_path, map_location='cpu')['state_dict']
        vqgan.load_state_dict(state, strict=True)  # NOTE: set False if resoluton is changed
    elif which_model == 'kl-f32':
        vqgan_config_path = '/research/cbim/medical/lh599/code/latent-diffusion/configs/autoencoder/autoencoder_kl_8x8x64.yaml'
        vqgan_model_path = '/research/cbim/medical/lh599/code/latent-diffusion/pretrained/kl-f32.ckpt'
        config = OmegaConf.load(vqgan_config_path)
        # config.model.params['ddconfig']['resolution'] = 256  # NOTE: yeah
        vqgan = instantiate_from_config(config.model)
        state = torch.load(vqgan_model_path, map_location='cpu')['state_dict']
        vqgan.load_state_dict(state, strict=True)  # NOTE: set False if resoluton is changed
    elif which_model == 'vq-f16':
        vqgan_config_path = '/research/cbim/medical/lh599/code/DALLE/pretrained/vqgan.1024.config.yml'
        vqgan_model_path = '/research/cbim/medical/lh599/code/DALLE/pretrained/vqgan.1024.model.ckpt'
        config = OmegaConf.load(vqgan_config_path)
        config.model.params['ddconfig']['resolution'] = 256
        vqgan = VQModel(**config.model.params)
        state = torch.load(vqgan_model_path, map_location='cpu')['state_dict']
        vqgan.load_state_dict(state, strict=True)  # NOTE: set False if resoluton is changed
    else:
        raise NotImplementedError

    return vqgan


def encode(vqgan, x):
    h = vqgan.encoder(x)
    z = vqgan.quant_conv(h)
    z_q, _, [_, _, indices] = vqgan.quantize(z)
    return z_q, z, indices

def decode(vqgan, z_q):
    quant = vqgan.post_quant_conv(z_q)
    x = vqgan.decoder(quant)
    return x

def decode_z(vqgan, z):
    z_q, _, info = vqgan.quantize(z)
    quant = vqgan.post_quant_conv(z_q)
    x = vqgan.decoder(quant)
    return x

def encode_kl(vqgan, x):
    post = vqgan.encode(x)
    z = post.sample()
    z_q = z
    return z_q, z, post

def decode_kl(vqgan, z_q):
    x = vqgan.decode(z_q)
    return x
