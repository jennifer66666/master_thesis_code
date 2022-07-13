import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
from projector import *
def generate(args, g_ema, g_ema2, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
           sample_z = torch.randn(args.sample, args.latent, device=device)

           sample, _ = g_ema([sample_z[-1]], truncation=args.truncation, truncation_latent=mean_latent)
           
           utils.save_image(
            sample,
            'transfer100/'+f'{str(i).zfill(6)}.png',
            nrow=1,
            normalize=True,
            range=(-1, 1),
            )

           result_file = {
            "img": sample,
            "latent": sample_z
            }

           torch.save(result_file, 'transfer100/'+f'{str(i).zfill(6)}.pt')
           # swap style ###################################
           sample_swap, _ = g_ema2([sample_z[-1]], truncation=args.truncation, truncation_latent=mean_latent)

           utils.save_image(
            sample_swap,
            'transfer100/'+f'{str(i).zfill(6)}'+'_swap.png',
            nrow=1,
            normalize=True,
            range=(-1, 1),
            )
if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--sample', type=int, default=10)
    parser.add_argument('--pics', type=int, default=1000)
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="ckpt_downloaded/550000.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument("--input_is_latent", action="store_true")

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint['g_ema'],strict=False)

    g_ema2 = Generator(
        args.size, args.latent, args.n_mlp).to(device)
    g_ema2.load_state_dict(checkpoint['g_ema'],strict=False)

    g_temp = Generator(
        args.size, args.latent, args.n_mlp).to(device)
    checkpoint2 = torch.load("ckpt_generated/checkpoint_kaggle_loc4/580000.pt")
    g_temp.load_state_dict(checkpoint2["g_ema"], strict=False)  

    for i in range(3):#args.transfer_part = 3
        g_ema2.convs[13-2-2*i]= g_temp.convs[13-2-2*i]
        g_ema2.convs[13-3-2*i]= g_temp.convs[13-3-2*i]
        g_ema2.to_rgbs[8-3-i]= g_temp.to_rgbs[8-3-i]

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, g_ema2, device, mean_latent)
    #################################################
    # project
    n_mean_latent = 10000
    resize = min(args.size, 256)
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    
    files_ori = os.listdir('transfer100')
    files_ori = [i for i in files_ori if i[-5]=='p']
    files_ori.sort()
    g_ema = g_temp
    for idx,imgfile in enumerate(files_ori):
        imgs = []
        imgfile = 'transfer100/'+imgfile
        img = transform(Image.open(imgfile).convert("RGB"))
        imgs.append(img)
        imgs = torch.stack(imgs, 0).to(device)

        with torch.no_grad():
            noise_sample = torch.randn(n_mean_latent, 512, device=device)
            latent_out = g_ema.style(noise_sample)
            latent_mean = latent_out.mean(0)
            latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

        percept = lpips.PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
        )

        noises_single = g_ema.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

        # if args.w_plus:
        #     latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
    
        if not args.input_is_latent:
            latent_in = torch.randn(1,512, device=device)
        latent_in.requires_grad = True

        for noise in noises:
            noise.requires_grad = True

        optimizer = optim.Adam([latent_in] + noises, lr=0.1)

        pbar = tqdm(range(1000))
        latent_path = []

        for i in pbar:
            t = i / 1000
            lr = get_lr(t, 0.1)
            optimizer.param_groups[0]["lr"] = lr
            noise_strength = latent_std * 0.05 * max(0, 1 - t / 0.75) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())
            
            if args.input_is_latent:
                img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)
            else:
                img_gen, _ = g_ema([latent_n[-1]], input_is_latent=False, noise=noises)
            batch, channel, height, width = img_gen.shape

            if height > 256:
                factor = height // 256

                img_gen = img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                img_gen = img_gen.mean([3, 5])

            p_loss = percept(img_gen, imgs).sum()
            n_loss = noise_regularize(noises)
            mse_loss = F.mse_loss(img_gen, imgs)

            loss = (1-0)*p_loss + 1e5 * n_loss + 0 * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_normalize_(noises)

            if (i + 1) % 100 == 0:
                latent_path.append(latent_in.detach().clone())

            pbar.set_description(
                (
                    f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                    f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                )
            )
        if args.input_is_latent:
            img_gen, _ = g_ema([latent_path[-1]], input_is_latent=args.input_is_latent, noise=noises)
        else:
            img_gen, _ = g_ema([latent_path[-1][-1]], input_is_latent=args.input_is_latent, noise=noises)

        utils.save_image(
                img_gen,
                'transfer100/'+f'{str(idx).zfill(6)}'+'_project.png',
                nrow=1,
                normalize=True,
                range=(-1, 1),
                )
        ori_image = transform(Image.open('transfer100/'+f'{str(idx).zfill(6)}'+'.png').convert("RGB")).unsqueeze(0)
        swap_image = transform(Image.open('transfer100/'+f'{str(idx).zfill(6)}'+'_swap.png').convert("RGB")).unsqueeze(0)
        pro_image = transform(Image.open('transfer100/'+f'{str(idx).zfill(6)}'+'_project.png').convert("RGB")).unsqueeze(0)
        utils.save_image(
                torch.cat([ori_image, swap_image,pro_image]),
                'transfer100/'+f'{str(idx).zfill(6)}'+'_concate.png',
                nrow=1,
                normalize=True,
                range=(-1, 1),
                )


