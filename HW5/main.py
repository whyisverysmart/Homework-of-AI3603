from model import Unet,GaussianDiffusion
from trainer import Trainer
import torch
torch.backends.cudnn.benchmark = True
# torch.manual_seed(4096)

# if torch.cuda.is_available():
#   torch.cuda.manual_seed(4096)


path = './Data/resized'
IMG_SIZE = 64             # Size of images, do not change this if you do not know why you need to change
batch_size = 128
train_num_steps = 40000        # total training steps
lr = 1e-3
grad_steps = 1            # gradient accumulation steps, the equivalent batch size for updating equals to batch_size * grad_steps = 16 * 1
ema_decay = 0.997           # exponential moving average decay

channels = 128             # Numbers of channels of the first layer of CNN
dim_mults = (1, 2, 4)        # The model size will be (channels, 2 * channels, 4 * channels, 4 * channels, 2 * channels, channels)

timesteps = 2000            # Number of steps (adding noise)
beta_schedule = 'linear'


print("batch_size: ",batch_size)
print("train_num_steps: ",train_num_steps)
print("lr: ",lr)
print("grad_steps: ",grad_steps)
print("ema_decay: ",ema_decay)
print("channels: ",channels)
print("dim_mults: ",dim_mults)
print("timesteps: ",timesteps)

model = Unet(
    dim = channels,
    dim_mults = dim_mults
)

diffusion = GaussianDiffusion(
    model,
    image_size = IMG_SIZE,
    timesteps = timesteps,
    beta_schedule = beta_schedule
)

trainer = Trainer(
    diffusion,
    path,
    train_batch_size = batch_size,
    train_lr = lr,
    train_num_steps = train_num_steps,
    gradient_accumulate_every = grad_steps,
    ema_decay = ema_decay,
    save_and_sample_every = 8000,
    results_folder="./model_results"
)

print(trainer.device)

# Train
trainer.train()


# Require changing along with hyperparameters
ckpt = f'./model_results/model-{train_num_steps//8000}.pt'
# ckpt = "./model_results/10000_7e-5_1500/model-4.pt"
# ckpt = f'./model_results/body/model-5.pt'
trainer.load(ckpt)
# Random generation
trainer.inference(output_path="./random_gen")
print("Random generation (inference) done")
# Fusion generation
# for i in range(50):
#     trainer.inference2(lamda=0.5,index1=9000+i,index2=8888+i,output_path="./fusion_gen/fusion",source_path='./fusion_gen/source')
# print("Fusion generation (inference2) done")