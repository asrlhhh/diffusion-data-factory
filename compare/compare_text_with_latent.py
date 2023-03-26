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
edit_guidance_scale_min = 5
edit_guidance_scale_max = 7
edit_warmup_steps = 20
sample_size = 3
samples_num = 20
initial_prompt = 'a professional portrait of a person in the center of the frame, 4k, Canon 5D, ZEISS lens, 85mm, high quality, detailed.'

# read the selected seeds from the pickle file
with open('selected_seeds.pkl', 'rb') as f:
    selected_seeds = pickle.load(f)

selected_seeds = sorted(selected_seeds)
selected_seed = selected_seeds[2]

print("selected seed: ",selected_seed)

# read attributes from the json file
attributes_prompts = read_attributes_json("attributes.json")
attributes_setups = read_setups_json("setup.json")

attributes_full = merge_prompts_setups(attributes_prompts,attributes_setups)
# randomly pick 5 attributes from the dictionary
#selected_attributes = random.sample(attributes_full.keys(), 5)
#selected_attributes = [[x] for x in selected_attributes]
attributes = list(attributes_full.keys())
attributes_lst = generate_random_sample_2d_list(attributes, samples_num, sample_size)
attributes_lst = attributes_lst[:5]

pipe = SemanticEditPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
).to(device)

gen = torch.Generator(device=device)

gen.manual_seed(selected_seed)
out = pipe(prompt=initial_prompt, generator=gen, num_images_per_prompt=num_images_per_prompt, guidance_scale=guidance_scale)
images = out.images
origin_image = images[0]
origin_image = pil_to_numpy(origin_image)

latent_images = []
latent_images.append(origin_image)

prompt_images = []
prompt_images.append(origin_image)

for attr in tqdm(attributes_lst):
    direction_lst = generate_random_bool_list(len(attr))
    print("attributes: ",attr)
    print("directions: ",direction_lst)
    guidance_lst = generate_random_float_list(len(attr), edit_guidance_scale_min, edit_guidance_scale_max)
    threshold_lst = generate_random_float_list(len(attr), edit_threshold_min, edit_threshold_max)
    
    warmup_steps = []
    cooldown_steps = []
    attribute_prompts = []

    for attribute in attr:
        warmup_steps.append(attributes_full[attribute]['warmups'])
        cooldown_steps.append(attributes_full[attribute]['cooldowns'])
        attribute_prompts.append(attributes_full[attribute]['prompt'])

    gen.manual_seed(selected_seed)
    out = pipe(prompt = initial_prompt, generator = gen, 
                   num_images_per_prompt = num_images_per_prompt, guidance_scale=guidance_scale,
                   editing_prompt= attribute_prompts,
                   reverse_editing_direction=direction_lst, # Direction of guidance i.e. increase all concepts
                   edit_warmup_steps= warmup_steps, # Warmup period for each concept
                   edit_cooldown_steps= cooldown_steps,
                   edit_guidance_scale=guidance_lst, # Guidance scale for each concept
                   edit_threshold=threshold_lst, # Threshold for each concept. Threshold equals the percentile of the latent space that will be discarded. I.e. threshold=0.99 uses 1% of the latent dimensions
                   edit_momentum_scale=0.3, # Momentum scale that will be added to the latent guidance
                   edit_mom_beta=0.6, # Momentum beta
                   edit_weights=[1]*len(attr) # Weights of the individual concepts against each other
                  )
    images = out.images
    image = images[0]
    image = pil_to_numpy(image)
    latent_images.append(image)

    attached_prompt = attach_prompt(attr,direction_lst)
    augmented_prompt = initial_prompt.split(", ")[0] + ", " + attached_prompt + ", ".join(initial_prompt.split(", ")[1:])

    gen.manual_seed(selected_seed)
    out = pipe(prompt=augmented_prompt, generator=gen, num_images_per_prompt=num_images_per_prompt, guidance_scale=guidance_scale)
    images = out.images
    image = images[0]
    image = pil_to_numpy(image)
    prompt_images.append(image)

# stack the latent images and prompt images horizontally first
latent_images = np.hstack(latent_images)
prompt_images = np.hstack(prompt_images)

# stack the latent images and prompt images vertically
final_image = np.vstack((latent_images,prompt_images))

# save the final image
cv2.imwrite("compare_text_with_latent.png", final_image)









