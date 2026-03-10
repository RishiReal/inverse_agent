import os
import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset
from glob import glob
from PIL import Image
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

sys.modules["mpi4py"] = types.SimpleNamespace(MPI=mpi_stub)

from improved_diffusion.unet import UNetModel
from improved_diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType,
    get_named_beta_schedule,
)

from matplotlib import pyplot as plt

th.manual_seed(0)
np.random.seed(0)

#configuration
gpu_id = 1
device = f'cuda:{gpu_id}' if th.cuda.is_available() else 'cpu'
data_dir = '/Users/rishi/research/Normalization'
image_size = 16
batch_size = 16
num_images = None
res = 2
num_its = 10
epoch_block = 5000
start_epoch = 0
lr = 1e-3
checkpoint = 1000

class ArteryDataset(Dataset):
    def __init__(self, base_dir, resolution=16, start_index=0, num_images=None):
        self.resolution = resolution
        self.images = sorted(glob(os.path.join(base_dir, '*.npy')))
        if num_images is not None:
            self.images = self.images[start_index:start_index + num_images]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        im = np.load(self.images[index]).astype(np.float32)
        im = np.squeeze(im)

        #alr normalized
        im_uint8 = ((im + 1) * 127.5).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(im_uint8, mode='RGB')

        # this is just for just in case; resize image if not native 16x16
        if self.resolution != pil_image.size[0]:
            pil_image = pil_image.resize(
                (self.resolution, self.resolution), resampl=Image.BICUBIC
            )

        arr = np.array(pil_image).astype(np.float32) / 127.5 - 1

        tensor = th.tensor(np.transpose(arr, [2, 0, 1]))
        return tensor, index

def load_data(base_dir,dataset_type,batch_size,image_size,deterministic=False):
    dataset = ArteryDataset(base_dir, resolution=image_size, num_images=num_images)
    print(f'Dataset size: {len(dataset)} images')
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers = 4,
        drop_last = True,
    )
    while True:
        yield from loader

def create_unet_model(image_size = 16,num_channels = 64, num_res_blocks = 2, 
                      dropout = 0.1, use_scale_shift_norm=True):

    if image_size <= 16:
        channel_mult = (1,2)
    elif image_size == 64:
        channel_mult = (1,2,3)
    elif image_size == 128:
        channel_mult = (1, 2, 3, 4)
    else:
        channel_mult = (1, 2)

    attention_ds = [image_size // 8] if image_size >= 8 else []

    return UNetModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=3, 
        num_res_blocks=num_res_blocks,

        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_heads=1,

        num_heads_upsample=-1,
        use_scale_shift_norm=use_scale_shift_norm,
    )

def create_gaussian_diffusion(steps=1000, noise_schedule="cosine"):
    betas = get_named_beta_schedule(noise_schedule, steps)
    return GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )

def uniform_sample_timesteps(steps, batch_size):
    indices_np = np.random.choice(steps, size=(batch_size,))
    return th.from_numpy(indices_np).long()

#create and setup model
model = create_unet_model(image_size=image_size, num_res_blocks=res).to(device)
gd = create_gaussian_diffusion()
data = load_data(data_dir, batch_size=batch_size, image_size=image_size, num_images=num_images)

optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
save_desc = f'resblock_{res}_batch_{batch_size}_unet_artery_{image_size}px'
save_dir = f'logs_{save_desc}'
os.makedirs(save_dir, exist_ok=True)
print(f'Saving checkpoints to {save_dir}')

# delete this later
batch, idx = next(data)
print(f'Batch shape: {batch.shape}')  
print(f'Batch range: [{batch.min():.3f}, {batch.max():.3f}]') 

total_epochs = epoch_block *num_its
Loss = []

#train
for epoch in range(start_epoch, start_epoch + total_epochs):
    batch, _ = next(data)
    batch = batch.to(device)

    t = uniform_sample_timesteps(gd.num_timesteps, len(batch)).to(device)
    loss = gd.training_losses(model, batch, t, model_kwargs={})['loss'].mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    Loss.append(loss.item())

    #checkpoint

    if epoch % checkpoint == 0:
        ckpt_path = os.path.join(save_dir, f'model_{epoch}.pt')
        th.save(model.state_dict(), ckpt_path)
        print(f'[{epoch}/{total_epochs}] loss={loss.item():.4f}  →  saved {ckpt_path}')

    #loss
    if epoch % 10 == 0:
        np.save(os.path.join(save_dir, 'Loss.npy'), np.array(Loss))
        plt.figure()
        plt.semilogy(Loss)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training over Heat Equation')
        plt.grid()
        plt.savefig(os.path.join(save_dir, f'loss_plot.png'))
        plt.close()

