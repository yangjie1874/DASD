import os

import cv2
import random
import einops
import numpy as np
import torch

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from utils.utils import makedir
from transformers import AutoImageProcessor

# Reference images path of target domain
ref_img_path = 'datasets/scab2023/images/'
assert os.path.exists(
    ref_img_path), f"The folder '{ref_img_path}' does not exist. Please run 'generate_Terraref_testimg.py' to get the reference images first."
# Layout images path
layout_path = 'datasets/random_layout2/'
assert os.path.exists(
    layout_path), f"The folder '{layout_path}' does not exist. Please run 'random_generate_layout_images.py' to get the layout images first."

output_path = 'output/wheat2023/'

# Names of reference images that containing the detection target: wheat heads
names_img_with_wheat = "datasets/2023/name.txt"
# Names of reference images that NOT containing the detection target: wheat heads


n_img = 8  # The number of images that need to be generated
seed = 0  #21
batch_size = 4  #4
img_resolution = 416
configs = 'configs/controlnet/DODA_wheat_cldm_kl_4.yaml'
#configs = 'configs/latent-diffusion/DODA_wheat_ldm_kl_4.yaml'
weight = "logs/val-ldm-cldmnw/allimagescldm-v1.ckpt"

layout_img_path = layout_path + 'img/'
label_path = layout_path + 'bounding_boxes.txt'

seed_everything(seed)


def process(control, reference_img, ddim_steps=50, guess_mode=False, strength=1.75, scale=1, eta=0.0):
    with torch.no_grad():
        B, H, W, C = control.shape

        control = torch.from_numpy(control.copy()).float().cuda() / 255.0
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        reference = torch.from_numpy(reference_img.copy()).float().cuda()

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning(reference)]}
        un_cond = {"c_concat": None if guess_mode else [control],
                   "c_crossattn": [model.get_learned_conditioning(torch.zeros((B, 3, 224, 224)).cuda())]}
        shape = (3, H // 4, W // 4)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
                    [strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, B,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0,
                                                                                                           255).astype(
            np.uint8)

        results = [x_samples[i] for i in range(B)]
    return results


output_img_path = output_path + 'img/'
output_ctr_path = output_path + 'ctr/'

model = create_model(configs).cpu()
model.load_state_dict(load_state_dict(weight, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)
maepath = 'vit-mae-base'
img_processor = AutoImageProcessor.from_pretrained(maepath)


makedir(output_img_path)
makedir(output_ctr_path)

# Get the labels
label_dic = {}
imgnames_list = []
with open(label_path) as f:
    label_lines = f.readlines()
    try:
        label_lines = label_lines[:n_img]
    except:
        print('The number of images to be generated is greater than the number of layout images')
        print('please use "random_generate_layout_images.py" to generate more layout images')
    for label_line in label_lines:
        label_line = label_line.rstrip()
        label_line = label_line.split(' ')
        img_name = label_line[0].split('/')[-1]
        if len(label_line) > 1:
            label_dic[img_name] = label_line[1:]
        else:
            label_dic[img_name] = 'no_box'
        imgnames_list.append(img_name)

# Get reference img names that there is/isn't wheat in the img
with open(names_img_with_wheat) as f:
    imgs_with_wheat = f.readlines()


# read img by batch
for i in range(0, len(imgnames_list), batch_size):
    batch_imgnames = imgnames_list[i:i + batch_size]
    label_lst = [label_dic[imgname] for imgname in batch_imgnames]

    # read and process reference imgs
    ref_imgs = []
    for bboxes in label_lst:
        if bboxes != 'no_box':
            ref_img_name = random.choice(imgs_with_wheat)

        ref_img_name = ref_img_name[:-1]
        ref_img = \
        img_processor(cv2.cvtColor(cv2.imread(ref_img_path + ref_img_name), cv2.COLOR_BGR2RGB))['pixel_values'][0]
        ref_imgs.append(ref_img)

    control_imgs = [cv2.resize(cv2.imread(layout_img_path + imgname), (img_resolution, img_resolution)) for imgname in
                    batch_imgnames]

    # Stack images in batch dimensions
    ref_imgs = np.stack(ref_imgs, axis=0)
    control_imgs = np.stack(control_imgs, axis=0)

    out_imgs = process(control_imgs, ref_imgs)
    for n, out_img in enumerate(out_imgs):
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_img_path + batch_imgnames[n], out_img)
        bboxes = label_lst[n]
        if bboxes != 'no_box':
            for box in bboxes:
                box = box.split(',')
                ctr_img = cv2.rectangle(out_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (208, 146, 0),
                                        2)
        else:
            ctr_img = out_img
        cv2.imwrite(output_ctr_path + batch_imgnames[n], ctr_img)
