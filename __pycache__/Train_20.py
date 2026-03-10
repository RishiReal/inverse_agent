
import os
import argparse
import json
import pdb
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import types
import sys

# create a fake MPI that does nothing
mpi_stub = types.SimpleNamespace(
    COMM_WORLD=types.SimpleNamespace(
        Get_rank=lambda: 0,
        Get_size=lambda: 1,
        Barrier=lambda: None,
    )
)
batch_size = 16
res = 2
sys.modules["mpi4py"] = types.SimpleNamespace(MPI=mpi_stub)

import improved_diffusion.dist_util as dist_util
from improved_diffusion.image_datasets import load_data
from improved_diffusion.unet import *
from improved_diffusion.nn    import *
from improved_diffusion.gaussian_diffusion import *

from matplotlib import pyplot as plt
from glob import glob
from imageio import imread
from skimage.transform import resize as imresize

th.manual_seed(0)
np.random.seed(0)
gpu_id = 1
device = f'cuda:{gpu_id}' if th.cuda.is_available() else 'cpu'

def uniform_sample_timesteps(steps, batch_size):
    indices_np = np.random.choice(steps, size=(batch_size,))
    indices = th.from_numpy(indices_np).long() # .to(device)
    return indices
def update_ema(target_params, source_params, ema_rate=0.99):
    # update target in-place
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(ema_rate).add_(src, alpha=1 - ema_rate)

def params_to_state_dict(target_params, model):
    state_dict = model.state_dict()
    for i, (name, _value) in enumerate(model.named_parameters()):
        assert name in state_dict
        state_dict[name] = target_params[i]
    return state_dict

def get_dataset(dataset_type, base_dir, start_index=0, num_images=None, resolution=64):
    """Get dataset class"""
    DATASET_MAPPING = dict(heatmap=HeatmapDataset)
    if dataset_type in DATASET_MAPPING:
        dataset = DATASET_MAPPING[dataset_type](base_dir, start_index=start_index, num_images=num_images, resolution=resolution)
    else:
        raise NotImplementedError(f'dataset: {dataset_type} is not implemented.')
    return dataset

class Data(Dataset):
    def __init__(self, base_dir, path='', resolution=64, start_index=0, num_images=None):
        self.resolution = resolution
        self.base_dir = base_dir
        self.path = self.base_dir + path
        self.images = sorted(glob(self.path))
        self.start_index = start_index
        self.num_images = num_images
        if num_images is not None:
            self.images = self.images[start_index:start_index + num_images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im_path = self.images[index]
        im = imread(im_path)
        im = imresize(im, (self.resolution, self.resolution))[:, :, :3]

        im = th.Tensor(im).permute(2, 0, 1)
        return im, index
    
class HeatmapDataset(Data):
    def __init__(self, base_dir, resolution=64, start_index=0, num_images=None):
        # reuse Data's init to collect files
        super().__init__(
            base_dir,
            path='T*.npy', # your heatmaps folder
            resolution=resolution,
            start_index=start_index,
            num_images=num_images
        )

    def __getitem__(self, index):
        file_path = self.images[index]
        im = np.load(file_path)  # load 2D array
        # resize if needed
        if self.resolution != im.shape[0]:
            im = imresize(im, (self.resolution, self.resolution), anti_aliasing=True)
        # add channel dimension
        tensor = th.tensor(im, dtype=th.float32).unsqueeze(0)
        return tensor, index

def create_unet_model(
        image_size=64,
        num_channels=64, # 128, #192,
        enc_channels=64,
        num_res_blocks=res, # 3,
        channel_mult="",
        num_heads=1,
        num_head_channels=64,
        num_heads_upsample=-1,
        attention_resolutions="32,16,8",
        dropout=0.1,

        use_scale_shift_norm=True, # True, # False??
        #resblock_updown=True,
        model_desc='unet_model',
        emb_dim=64 ## timesteps

    ):
        # everything else False

    if channel_mult == "":
        if image_size == 64:
            channel_mult = (1, 2, 3) # (1, 2, 3, 4)
        elif image_size == 128:
            channel_mult = (1, 2, 3, 4)
        elif image_size < 64: # eg 35
            channel_mult = (1, 2)
    elif len(channel_mult) > 0: # passed in comma-delimited series of numbers
        channel_mult = channel_mult.split(',')
        channel_mult = [int(n) for n in channel_mult]
        channel_mult = tuple(channel_mult)
        print('channel_mult')
        print(channel_mult)

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    unet = UNetModel(
        in_channels=1,
        model_channels=num_channels,
        out_channels=1, 
        num_res_blocks=num_res_blocks,

        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_heads=num_heads,

        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        
    )
    return unet

def unet_model_defaults():
    return dict(
        image_size=64,
        num_channels=64, # 128, #192,
        enc_channels=64,
        num_res_blocks=res, # 2,
        channel_mult="",
        num_heads=1,
        num_heads_upsample=-1,
        attention_resolutions="32,16,8",
        dropout=0.1,
        use_scale_shift_norm=True, 

    )

def create_diffusion_model(model_desc='unet_model', **model_kwargs):
    if model_desc == 'unet_model':
        model = create_unet_model(**model_kwargs)
    return model

from improved_diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType
)

def create_gaussian_diffusion(steps=1000, noise_schedule="cosine", ):
    betas = get_named_beta_schedule(noise_schedule, steps)
    gd = GaussianDiffusion(betas=betas,model_mean_type=ModelMeanType.START_X,model_var_type=ModelVarType.FIXED_SMALL,loss_type=LossType.MSE)
    return gd

# create model
from torch.utils.data import DataLoader, Dataset
model_kwargs = unet_model_defaults()

print(model_kwargs)
model = create_diffusion_model(**model_kwargs)
model.to(device)

def diffusion_defaults():
    return dict(
        steps=1000,
        noise_schedule="cosine",
    )
# create diffusion
diffusion_kwargs = diffusion_defaults()

gd = create_gaussian_diffusion(**diffusion_kwargs)


# load data
data_dir = '/home/nandan/Projects/Decomposing Heat Equation/Heat Equation FD/Dataset_with_20_samples/' 

dataset = 'heatmap'  
image_size = 64
batch_size = batch_size
num_images = 20

def load_data(
    *,
    base_dir,
    dataset_type,
    batch_size,
    image_size,
    deterministic=False,
    random_crop=False,
    random_flip=False,
    num_images=None
):  

    dataset = get_dataset(dataset_type, base_dir=base_dir, num_images=num_images, resolution=image_size)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True
        )
    while True:
        yield from loader
        
data = load_data(
    base_dir=data_dir,
    dataset_type=dataset,
    batch_size=batch_size,
    image_size=image_size,
    num_images=num_images
)


batch, idx = next(data)
x = np.loadtxt('/home/nandan/Projects/Decomposing Heat Equation/Heat Equation FD/x.txt')
y = np.loadtxt('/home/nandan/Projects/Decomposing Heat Equation/Heat Equation FD/y.txt')
X,Y = np.meshgrid(x,y)
print("Batch shape:", batch.shape)
print("Index shape:", idx.shape)


default_im = '/home/nandan/Projects/Decomposing Heat Equation/Heat Equation FD/Dataset_with_10_samples/T9.npy'
model_desc = 'unet_model'
save_desc = f'resblock_{res}_batch_{batch_size}_{model_desc}_{dataset}_{num_images}'


num_its = 10
epoch_block = 5000

start_epoch=0
batch_size=16

image_size=64
use_dist=False
latent_orthog=False
ema_rate=0.9999


lr = 1e-3
p_uncond=0.0    
downweight=False
Loss = []

optimizer = th.optim.Adam(model.parameters(),
                            lr=lr,weight_decay = 1e-8)
total_epochs = epoch_block * num_its
save_dir = 'logs_' + save_desc # f'{save_desc}_params/'
os.makedirs(save_dir, exist_ok=True)
print(f'Saving model ckpts to {save_dir}')
free = p_uncond > 0


for epoch in range(start_epoch, start_epoch + total_epochs):
    batch, cond = next(data)
    batch = batch.to(device)
    model_kwargs = {}

    t = uniform_sample_timesteps(gd.num_timesteps, len(batch)).to(device)
    loss = gd.training_losses(model, batch, t, model_kwargs=model_kwargs)['loss']
    loss = loss.mean() 
    Loss.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    imgs_save_dir = f'gen_imgs_{save_desc}'
    if epoch % 1000 == 0:
       
        print('loss = ',loss)
        th.save(model.state_dict(), os.path.join(save_dir, f'model_{epoch}.pt'))

    if epoch % 10 == 0:
        plt.figure()
        plt.semilogy(Loss)
        np.save(save_dir+"/Loss",np.array(Loss))
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training over Heat Equation')
        plt.grid()
        plt.savefig(os.path.join(save_dir, f'loss_plot.png'))
        plt.close()