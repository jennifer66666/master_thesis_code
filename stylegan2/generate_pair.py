import argparse

import torch
import os
from torchvision import utils
from model import Generator

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "--truncation",
        type=float,
        default=0.7,
        help="truncation ratio"
    )
    parser.add_argument(
        "--ckpt1",
        type=str,
        default="ckpt_downloaded/550000.pt",
        help="path to the original model checkpoint",
    )
    parser.add_argument(
        "--ckpt2",
        type=str,
        default="face2met_10k.pt",
        help="path to the finetuned model checkpoint",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    ) 
    parser.add_argument(
        "--transfer_part",
        type=int,
        default=0,
    ) 
    parser.add_argument(
        "--single_file",
        type=str,
        default=None
    ) 
    parser.add_argument(
        "--swap_style_from",
        type=int,
        default=4
    ) 
    parser.add_argument(
        "--another_style",
        type=str,
        default=None
    ) 
    parser.add_argument(
    "--input_dir",
    type=str,
    default='projected_real'
    )
    parser.add_argument("--input_is_latent", action="store_true")
    parser.add_argument("--noise", action="store_true")
    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema1 = Generator(
        args.size, args.latent, args.n_mlp).to(device)
    checkpoint1 = torch.load(args.ckpt1, map_location="cpu")

    g_ema1.load_state_dict(checkpoint1["g_ema"], strict=False) 

    g_ema2 = Generator(
        args.size, args.latent, args.n_mlp).to(device)
    checkpoint2 = torch.load(args.ckpt2, map_location="cpu")

    if args.transfer_part > 0:
        g_temp = Generator(
        args.size, args.latent, args.n_mlp).to(device)
        g_temp.load_state_dict(checkpoint2["g_ema"], strict=False)  
        g_ema2.load_state_dict(checkpoint1["g_ema"], strict=False)  
        for i in range(args.transfer_part):
            g_ema2.convs[13-2-2*i]= g_temp.convs[13-2-2*i]
            g_ema2.convs[13-3-2*i]= g_temp.convs[13-3-2*i]
            g_ema2.to_rgbs[8-3-i]= g_temp.to_rgbs[8-3-i]
    else:
        g_ema2.load_state_dict(checkpoint2["g_ema"], strict=False)  
    
    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema1.mean_latent(4096)
    else:
        mean_latent = None

    with torch.no_grad():
        g_ema1.eval()
        g_ema2.eval()
        file_names = [i[:6] for i in os.listdir(args.input_dir) if i.split('.')[-1]=='png']
        if args.single_file: 
            file_names = [args.single_file.split('/')[-1][:6]]
        for pic in file_names:
            if args.single_file:
                projector_pt = args.single_file
                latents = torch.load(projector_pt)['images_input_source/550000real_generated/'+pic+'.png']["latent"].to(device)
            else:
                projector_pt = args.input_dir + '/' +pic+'.pt'
                latents = torch.load(projector_pt)['images_input_source/550000real_generated/'+pic+'.png']["latent"].to(device)
            noise = None
            if args.noise:
                noise = torch.load(projector_pt)['images_input_source/550000real_generated/'+pic+'.png']["noise"]
                noise = [i.to(device) for i in noise]
            sample1, _ = g_ema1(
                [latents], truncation=args.truncation, truncation_latent=mean_latent, input_is_latent=args.input_is_latent
            )
            if args.another_style:
                another_style = torch.load(args.another_style)["images_input_source/550000real_generated/"+args.another_style.split('/')[-1][:6]+".png"]["latent"].to(device)
                sample2, _ = g_ema2(
                    [latents], truncation=args.truncation, truncation_latent=mean_latent, 
                    input_is_latent=args.input_is_latent,another_style=[another_style],
                    swap_style_from=args.swap_style_from, noise = noise
                )
            else:
                sample2,_ = g_ema2(
                    [latents], truncation=args.truncation, truncation_latent=mean_latent,
                    input_is_latent=args.input_is_latent, noise=noise
                )
            outputname = pic+"0619.png"
            utils.save_image(
                #torch.cat([sample1, sample2]),
                sample2,
                outputname,
                nrow=args.sample,
                normalize=True,
                range=(-1, 1),
            )

