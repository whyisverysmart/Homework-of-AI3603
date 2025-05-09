import math
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision import transforms as T, utils
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from PIL import Image
from tqdm.auto import tqdm


## The forward process functions (Noise scheduler)
def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))



## The backward process functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

## The backward process model U-Net

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        resnet_block_groups = 8,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time):
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)



## The Diffusion Process model
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        beta_schedule = 'linear',
        auto_normalize = True
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model #Unet

        self.channels = self.model.channels

        self.image_size = image_size


        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        
        # calculate beta and other precalculated parameters
        betas = beta_schedule_fn(timesteps)
                                            
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = timesteps # default num sampling timesteps to number of timesteps at training

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()

        register_buffer('loss_weight', maybe_clipped_snr / snr)

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )


    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, clip_x_start = False, rederive_pred_noise = False):
        """
        comments:
        Perform predictions for the reverse process in a diffusion model.

        Returns:
        - pred_noise (torch.Tensor): The predicted noise component at timestep t, representing the model's 
                                     learned approximation of the noise added during the forward diffusion process.
        - x_start (torch.Tensor): The reconstructed clean sample based on the predicted noise.

        1. U-Net (self.model) in Diffusion:
           U-Net learns to estimate the noise component added 
           during the forward process. By doing so, it enables the reverse process to iteratively denoise 
           the noisy input and reconstruct the original data.

        2. Reconstructing x_start:
           - x_start is derived using the reverse process formula:
             x_0 ≈ (x_t - sqrt(1 - alpha_t) * ε) / sqrt(alpha_t).
           - The method self.predict_start_from_noise implements this calculation.
        """
        model_output = self.model(x, t)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        pred_noise = model_output
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        x_start = maybe_clip(x_start)

        if clip_x_start and rederive_pred_noise:
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised = True):
        noise, x_start = self.model_predictions(x, t)

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int):
        """
        comments:
        Perform a single step of the reverse diffusion process, generating a sample x_t-1
        from x_t at the given timestep t.

        Process Overview:
        1. The function first computes the model's predicted mean and log variance at the given time step t using the p_mean_variance method.
        2. The denoised image is generated by adding the scaled noise to the predicted mean, 
        based on the model's predicted variance.

        Returns:
        - pred_img: The denoised image predicted by the model at time step t. This is the output of
        the reverse diffusion process for this step.
        - x_start: The estimated original image sample, which can be used for further processing
        or reference in the model's denoising steps.
        """
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        """
        comments:
        This function performs the full sampling loop for generating an image using the reverse diffusion process. 
        It iteratively applies the p_sample function over all time steps, starting from random noise and progressively 
        denoising it through the reverse diffusion steps. 
        The loop starts with a random tensor and applies the reverse diffusion process (denoising) step by step,
        producing an image. Optionally, it can return the intermediate images at each time step.

        Process Overview:
        1. Initialize the input image as random noise of the specified shape.
        2. Iterate through the time steps in reverse order, starting from the final diffusion step (t = num_timesteps) and
        progressively applying the reverse denoising process (using p_sample).
        3. For each time step, p_sample is called to obtain a denoised image, and the intermediate images are stored.

        Returns:
        - ret: The generated image after the full reverse diffusion process.
        """
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None
        
        ###########################################
        ## TODO: plot the sampling process ##
        ###########################################
        # for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
        #     img, x_start = self.p_sample(img, t)
        #     imgs.append(img)
        
        with tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps) as pbar:
            for i, t in enumerate(reversed(range(0, self.num_timesteps))):
                img, x_start = self.p_sample(img, t)
                imgs.append(img)
                if (i+1) % 500 == 0:
                    pbar.update(500)
        
        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret
    
    

    @torch.no_grad()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop
        return sample_fn((batch_size, channels, image_size, image_size), return_all_timesteps = return_all_timesteps)
    
    def sample2(self, batch_size = 16, lamda=0.5,index1=1,index2=2,return_all_timesteps = False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop2
        img1 = Image.open(f'./Data/faces/cat_{index1:0{5}}.jpg')
        img2 = Image.open(f'./Data/faces/cat_{index2:0{5}}.jpg')
        img1 = T.ToTensor()(img1)
        img2 = T.ToTensor()(img2)
        return sample_fn((batch_size, channels, image_size, image_size), img1.unsqueeze(0), img2.unsqueeze(0), lamda,return_all_timesteps = return_all_timesteps)



    def p_sample_loop2(self, shape, img1, img2, lamda,return_all_timesteps = False):
        """
        comments:
        This function performs the reverse diffusion sampling loop to generate an image by combining 
        two input images (`img1` and `img2`) through a weighted interpolation, and then applying the 
        reverse diffusion process iteratively.
        It starts by generating noisy versions of `img1` and `img2` at random time steps, and then 
        interpolates between these noisy images based on the weight `lamda`. The loop then applies 
        reverse denoising steps (`p_sample`) to progressively denoise the image over multiple timesteps.

        Process Overview:
        1. Two noisy images are generated from `img1` and `img2` at random time steps using the `q_sample` function.
        2. The two noisy images are then interpolated using `lamda` to form the initial image (`img`).
        3. The image is normalized before starting the reverse diffusion loop.
        4. The reverse diffusion process is applied iteratively using the `p_sample` function, and intermediate images
        are collected.
        5. Optionally, return all intermediate images (if `return_all_timesteps` is True), or just the final generated image.
        6. After the loop, the generated image is unnormalized (if necessary) and returned.

        Returns:
        - ret: The generated image after the reverse diffusion process.
        """
        batch, device = shape[0], self.betas.device

        def q_sample(x_start, t, noise=None):
            noise = default(noise, lambda: torch.randn_like(x_start))

            return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
            )

        img1_start = q_sample(img1.to(device),torch.randint(0, 100,(1,),device=device))
        img2_start = q_sample(img2.to(device),torch.randint(0, 100,(1,),device=device))
        # img = torch.randn(shape, device = device)
        img = (1-lamda)*img1_start + lamda*img2_start
        img = self.normalize(img)

        imgs = [img]
        x_start = None
        
        ###########################################
        ## TODO: plot the sampling process ##
        ###########################################
        # for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
        #     img, x_start = self.p_sample(img, t)
        #     imgs.append(img)
        
        with tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps) as pbar:
            for i, t in enumerate(reversed(range(0, self.num_timesteps))):
                img, x_start = self.p_sample(img, t)
                imgs.append(img)
                if (i+1) % 500 == 0:
                    pbar.update(500)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret
    @property
    def loss_fn(self):
        return F.mse_loss


    def q_sample(self, x_start, t, noise=None):
        """
        comments:
        This function generates a noisy version of the input image (`x_start`) at a given time step `t` 
        by adding noise to it. The amount of noise added depends on the model's parameters at that time step. 
        It uses the cumulative product of the diffusion steps (through `sqrt_alphas_cumprod` and `sqrt_one_minus_alphas_cumprod`) 
        to control the strength of the noise added based on the current time step `t`. If no custom noise is provided, 
        standard Gaussian noise is generated and added to the input.

        Process Overview:
        1. If no custom noise is provided, Gaussian noise is generated with the same shape as `x_start`.
        2. The noise is scaled and added to the original image based on the model's parameters (`sqrt_alphas_cumprod` 
        and `sqrt_one_minus_alphas_cumprod`) at the given time step `t`.
        3. The function returns the noisy image, which is a weighted combination of the original image and the noise.
        
        Arguments:
        - x_start: The original image or signal, which will be perturbed by noise.
        - t: The current time step, representing how much noise should be added to `x_start` based on the diffusion process.
        - noise: (Optional) Custom noise to add to the input. If not provided, random Gaussian noise is generated.

        Returns:
        - A noisy version of the input `x_start` at time step `t`.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )


    def p_losses(self, x_start, t, noise = None):
        """
        comments:
        This function computes the loss for a single timestep in the reverse diffusion process. 
        The loss is based on the difference between the predicted noise and the true noise added to the original image 
        (`x_start`) during the forward diffusion process.
        The function first generates a noisy version of the input image (`x_start`) at time step `t` using the `q_sample` 
        function. It then computes the model's output by passing the noisy image through the model. The loss is calculated 
        by comparing the model's output to the true noise and applying a weighting factor that depends on the current time 
        step (`t`).

        Process Overview:
        1. A noisy version of the original image (`x_start`) is generated using the `q_sample` function at the specified 
        time step `t` by adding Gaussian noise.
        2. The noisy image is passed through the model to predict the noise.
        3. The loss between the predicted noise and the true noise is computed using the model's output.
        4. The loss is weighted by a factor (`loss_weight`) that may vary with the time step `t`.

        Arguments:
        - x_start: The original image.
        - t: The current time step at which the loss should be computed. It controls the amount of noise in the image.
        - noise: (Optional) The true noise that was added to `x_start` to obtain the noisy image. If not provided, random 
        Gaussian noise is generated.

        Returns:
        - The average loss for this timestep, which measures how well the model predicted the noise added to `x_start` 
        at time step `t`.
        """
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # predict and take gradient step

        model_out = self.model(x, t)

        loss = self.loss_fn(model_out, noise, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)