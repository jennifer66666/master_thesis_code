import os
import cv2
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from tqdm import tqdm
from model_swap import Encoder, Generator
from model import Generator as StyleGAN
from torchvision import utils

def load_image(path, size):
    image = image2tensor(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))

    w, h = image.shape[-2:]
    if w != h:
        crop_size = min(w, h)
        left = (w - crop_size)//2
        right = left + crop_size
        top = (h - crop_size)//2
        bottom = top + crop_size
        image = image[:,:,left:right, top:bottom]

    if image.shape[-1] != size:
        image = torch.nn.functional.interpolate(image, (size, size), mode="bilinear", align_corners=True)
    
    return image

def image2tensor(image):
    image = torch.FloatTensor(image).permute(2,0,1).unsqueeze(0)/255.
    return (image-0.5)/0.5

def tensor2image(tensor):
    tensor = tensor.clamp(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy()
    return tensor*0.5 + 0.5

def imshow(img, size=5, cmap='jet'):
    plt.figure(figsize=(size,size))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()

def horizontal_concat(imgs):
    return torch.cat([img.unsqueeze(0) for img in imgs], 3) 

if __name__ == "__main__":
    device = 'cuda:0'
    image_size = 256
    torch.set_grad_enabled(False)
    ae_model_path = 'ckpt_downloaded/naverwebtoon_500k_ffhq_2k_finetune.pt'
        
    encoder = Encoder(32).to(device)
    generator = Generator(32).to(device)
    ckpt = torch.load(ae_model_path, map_location=device)
    encoder.load_state_dict(ckpt["e_ema"])
    generator.load_state_dict(ckpt["g_ema"])

    encoder.eval()
    generator.eval()

    print(f'[SwapAE model loaded] {ae_model_path}')

    #stylegan_model_path = 'ckpt_downloaded/stylegan2-naverwebtoon-800k.pt'
    stylegan_model_path = 'ckpt_generated/checkpoint_loc4anime/600000.pt'
    stylegan_ckpt = torch.load(stylegan_model_path, map_location=device)

    latent_dim = stylegan_ckpt['args'].latent

    stylegan = StyleGAN(image_size, latent_dim, 8).to(device)
    stylegan.load_state_dict(stylegan_ckpt["g_ema"], strict=False)
    stylegan.eval()
    print(f'[StyleGAN2 generator loaded] {stylegan_model_path}\n')

    truncation = 0.7
    trunc = stylegan.mean_latent(4096).detach().clone()

    num_samples = 8

    latent = stylegan.get_latent(torch.randn(num_samples, latent_dim, device=device))
    imgs_gen, _ = stylegan([latent],
                            truncation=truncation,
                            truncation_latent=trunc,
                            input_is_latent=True,
                            randomize_noise=True)

    # print("StyleGAN2 generated images:")
    # imshow(tensor2image(horizontal_concat(imgs_gen)), size=20)

    # structures, textures = encoder(imgs_gen)
    # recon_results = generator(structures, textures)

    # print("SwapAE reconstructions:")    
    # imshow(tensor2image(horizontal_concat(recon_results)), size=20)

    # print("Swapping results:")    
    # swap_results = generator(structures, textures[0].unsqueeze(0).repeat(num_samples,1))
    # imshow(tensor2image(horizontal_concat(swap_results)), size=20)
    # utils.save_image(
    #             imgs_gen,
    #             'swap.png',
    #             normalize=True,
    #             range=(-1, 1),
    #         )   
    ######################################################################
    path = 'images_input_source/550000real_generated'
    for image_name in sorted(os.listdir(path)):
            if os.path.splitext(image_name)[-1].lower() not in [".jpg", ".png", ".bmp", ".tiff"]:
                continue
            print(os.path.join(path, image_name))
            test_image_path = os.path.join(path, image_name)


            test_image = load_image(test_image_path, 256)

            num_styles = 1

            latent = stylegan.get_latent(torch.randn(num_styles, latent_dim, device=device))
            imgs_gen, _ = stylegan([latent],
                                    truncation=truncation,
                                    truncation_latent=trunc,
                                    input_is_latent=True,
                                    randomize_noise=True)
            # 这里stylegan是拿来生成动漫图的
            # autoendoder的encoder和generator是
            inputs = torch.cat([test_image.to(device), imgs_gen])

            results = horizontal_concat(inputs.cpu())
            #cv2.imwrite('see.jpg', cv2.cvtColor(255*tensor2image(results), cv2.COLOR_BGR2RGB))
            # 生成第一排：一张真图+五张动漫图

            structures, target_textures = encoder(inputs)

            structure = structures[0].unsqueeze(0).repeat(len(target_textures),1,1,1)
            source_texture = target_textures[0].unsqueeze(0).repeat(len(target_textures),1)

            # 第一列下面两张像真人脸又有点动漫风格，是因为用了encoder在真人脸上提取动漫风格
            #for swap_loc in [3, 5]:
            for swap_loc in [3]:
                textures = [source_texture for _ in range(swap_loc)] + [target_textures for _ in range(len(generator.layers) - swap_loc)]        
                fake_imgs = generator(structure, textures, noises=0)

                #results = torch.cat([results, horizontal_concat(fake_imgs).cpu()], dim=2)
                results = horizontal_concat(fake_imgs).cpu()
                    
            #imshow(tensor2image(results), 23)

            cv2.imwrite("output/"+image_name, cv2.cvtColor(255*tensor2image(results), cv2.COLOR_BGR2RGB))