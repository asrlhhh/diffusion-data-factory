from semdiffusers import SemanticEditPipeline
import torch
import cv2
import numpy as np
from PIL import Image
from utils import *
import os
from tqdm import tqdm
import pickle

# random select a set of seeds
# for each seed random select x number of editing prompt 
# compare editing prompt vs latent interpolation 

device='cuda'
num_images_per_prompt = 1
guidance_scale = 7
edit_threshold_min = 0.95
edit_threshold_max = 0.99
edit_guidance_scale = 5
#edit_guidance_scale_min = 5
#edit_guidance_scale_max = 7
sample_size = 3
samples_num = 20
initial_prompt = 'a professional portrait of a person in the center of the frame, 4k, Canon 5D, ZEISS lens, 85mm, high quality, detailed.'

# read the selected seeds from the pickle file
with open('selected_seeds.pkl', 'rb') as f:
    selected_seeds = pickle.load(f)

data_folder = "display_cooldowns"
# create folder if not exist
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

selected_seeds = selected_seeds[:5]

# read attributes from the json file
attributes_prompts = read_attributes_json("attributes.json")
attributes_setups = read_setups_json("setup.json")

attributes_full = merge_prompts_setups(attributes_prompts,attributes_setups)
attributes = list(attributes_full.keys())
attributes_lst = attributes[:5]

pipe = SemanticEditPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
).to(device)

gen = torch.Generator(device=device)

for selected_seed,attribute in zip(selected_seeds,attributes_lst):
    seed_list = []
    # create sub folder for each seed
    seed_folder = os.path.join(data_folder, str(selected_seed))
    if not os.path.exists(seed_folder):
        os.makedirs(seed_folder)
    gen.manual_seed(selected_seed)
    out = pipe(prompt=initial_prompt, generator=gen, num_images_per_prompt=num_images_per_prompt, guidance_scale=0)
    images = out.images
    origin_image = images[0]
    origin_image = pil_to_numpy(origin_image)
    

    edit_prompt = attributes_full[attribute]['prompt']

    attribute_warmup = attributes_full[attribute]['warmups']

    seed_list.append((origin_image,attribute_warmup))

    for i in range(attribute_warmup+5,45,3):
        edit_cooldown_steps = [i]
        gen.manual_seed(selected_seed)
        out = pipe(prompt = initial_prompt, generator = gen, 
                   num_images_per_prompt = num_images_per_prompt, guidance_scale=guidance_scale,
                   editing_prompt= [edit_prompt],
                   reverse_editing_direction=[False], # Direction of guidance i.e. increase all concepts
                   edit_warmup_steps= [attribute_warmup], # Warmup period for each concept
                   edit_cooldown_steps = edit_cooldown_steps,
                   edit_guidance_scale=[edit_guidance_scale], # Guidance scale for each concept
                   edit_threshold=[0.95], # Threshold for each concept. Threshold equals the percentile of the latent space that will be discarded. I.e. threshold=0.99 uses 1% of the latent dimensions
                   edit_momentum_scale=0.3, # Momentum scale that will be added to the latent guidance
                   edit_mom_beta=0.6, # Momentum beta
                   edit_weights=[1] # Weights of the individual concepts against each other
                  )
        images = out.images
        image = images[0]
        image = pil_to_numpy(image)
        seed_list.append((image,i))
    
    # save the seed list to the seed folder
    for i,(image,steps) in enumerate(seed_list):
        image_name = str(i).zfill(4) + "_" + str(steps) + ".png"
        image_path = os.path.join(seed_folder,image_name)
        cv2.imwrite(image_path,image)


