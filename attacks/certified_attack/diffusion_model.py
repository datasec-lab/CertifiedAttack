import torch
from attacks.certified_attack.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




def denoise(noise,img,sigma,diffusion_model):
    img=img*2-1
    img=img+noise
    img=torch.clip(img,-1.0,1.0)
    sampled_images = diffusion_model.denoise_for_RS(img, sigma, return_all_timesteps=False)
    sampled_images=sampled_images/2+0.5
    return sampled_images

if __name__ == '__main__':
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=64,
        timesteps=500,
        sampling_timesteps=50  # number of steps
    )

    pre_train = torch.load('/home/hah22011/projects/denoising-diffusion-pytorch/results/model-50.pt')
    diffusion.load_state_dict(pre_train['model'])

    trainer = Trainer(
        diffusion,
        './cifar10_images',
        train_batch_size=64,
        train_lr=8e-5,
        train_num_steps=700000,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        calculate_fid=True  # whether to calculate fid during training
    )

    img=np.load('./sample1.np.npy')
    img = np.transpose(img, (1, 2, 0))

    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    img_ = Image.fromarray(img)
    img_.save('adv_example.png')

    resized_image = np.asarray(img_.resize((64,64)))
    resized_image = np.transpose(resized_image, (2, 0, 1))
    x=torch.from_numpy(resized_image).cuda().unsqueeze(0)/255*2-1
    sigma=0.25
    x=x+sigma*torch.randn_like(x).cuda()
    x=x.clamp_(-1,1)

    sampled_images = diffusion.denoise_for_RS(x,sigma,return_all_timesteps=True)
    image_start=sampled_images[0].squeeze(0)/2+0.5
    image_end=sampled_images[-1].squeeze(0)/2+0.5

    image_start_np = image_start.mul(255).add(0.5).clamp_(0, 255).byte()
    image_start_np = image_start_np.permute(1, 2, 0).cpu().numpy()
    img_start = Image.fromarray(image_start_np)
    # img_start.show()
    img_start.save('output_start.png')

    image_end_np = image_end.mul(255).add(0.5).clamp_(0, 255).byte()
    image_end_np = image_end_np.permute(1, 2, 0).cpu().numpy()
    img_end = Image.fromarray(image_end_np)
    # img_end.show()
    img_end.save('output_end.png')