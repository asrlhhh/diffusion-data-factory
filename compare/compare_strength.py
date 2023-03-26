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

display_dict = {
    4: "Gray_Hair",
    5: "Brown_Hair",
    6: "Chubby",
    15: "Chubby",
    18: "Double_Chin",
    25: "Smiling",
    27: "High_Cheekbones",
    38: "Mouth_Slightly_Open",
    42: "Big_Lips",
    43: "Wavy_Hair",
    55: "Sideburns",
    63: "Young",
    64: "Straight_Hair",
    77: "Wearing_Hat",
    88: "Heavy_Makeup",
    99: "Rosy_Cheeks",
    17: "Gray_Hair"}

device='cuda'
num_images_per_prompt = 1
guidance_scale = 7
edit_threshold_min = 0.95
edit_threshold_max = 0.99
#edit_guidance_scale_min = 5
#edit_guidance_scale_max = 7
edit_guidance_scale_range = [1,10]
edit_warmup_steps = 0
sample_size = 3
samples_num = 20
initial_prompt = 'a professional portrait of a person in the center of the frame, 4k, Canon 5D, ZEISS lens, 85mm, high quality, detailed.'

# read the selected seeds from the pickle file
with open('selected_seeds.pkl', 'rb') as f:
    selected_seeds = pickle.load(f)

data_folder = "display_strength_final"
# create folder if not exist
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
selected_seeds = sorted(selected_seeds)
selected_seed = 42

# read attributes from the json file
attributes_prompts = read_attributes_json("attributes.json")
attributes_setups = read_setups_json("setup.json")

attributes_full = merge_prompts_setups(attributes_prompts,attributes_setups)
# randomly pick 5 attributes from the dictionary
#selected_attributes = random.sample(attributes_full.keys(), 5)
#selected_attributes = [[x] for x in selected_attributes]
attributes = list(attributes_full.keys())
attributes_lst = ["Big_Lips"]

pipe = SemanticEditPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
).to(device)

gen = torch.Generator(device=device)

for selected_seed, attribute in display_dict.items():
    print(attribute)
    seed_list = []
    # create sub folder for each seed
    seed_folder = os.path.join(data_folder, attribute+"_"+str(selected_seed))
    if not os.path.exists(seed_folder):
        os.makedirs(seed_folder)
    gen.manual_seed(selected_seed)
    out = pipe(prompt=initial_prompt, generator=gen, num_images_per_prompt=num_images_per_prompt, guidance_scale=guidance_scale)
    images = out.images
    origin_image = images[0]
    origin_image = pil_to_numpy(origin_image)
    seed_list.append((origin_image,0))
    image_name = str(0).zfill(4) + ".png"
    image_path = os.path.join(seed_folder,image_name)
    cv2.imwrite(image_path,origin_image)

    edit_prompt = attributes_full[attribute]['prompt']
    edit_warmup_steps = attributes_full[attribute]['warmups']
    edit_cooldown_steps = attributes_full[attribute]['cooldowns']

    for ind,i in enumerate(np.arange(edit_guidance_scale_range[0],edit_guidance_scale_range[1],0.3)):
        edit_guidance_scale = i
        gen.manual_seed(selected_seed)
        out = pipe(prompt = initial_prompt, generator = gen, 
                   num_images_per_prompt = num_images_per_prompt, guidance_scale=guidance_scale,
                   editing_prompt= [edit_prompt],
                   reverse_editing_direction=[False], # Direction of guidance i.e. increase all concepts
                   edit_warmup_steps= [edit_warmup_steps], # Warmup period for each concept
                   edit_guidance_scale=[edit_guidance_scale], # Guidance scale for each concept
                   edit_cooldown_steps= [edit_cooldown_steps],
                   edit_threshold=[0.95], # Threshold for each concept. Threshold equals the percentile of the latent space that will be discarded. I.e. threshold=0.99 uses 1% of the latent dimensions
                   edit_momentum_scale=0.3, # Momentum scale that will be added to the latent guidance
                   edit_mom_beta=0.6, # Momentum beta
                   edit_weights=[1] # Weights of the individual concepts against each other
                  )
        images = out.images
        image = images[0]
        image = pil_to_numpy(image)

        image_name = str(ind+1).zfill(4) + ".png"
        image_path = os.path.join(seed_folder,image_name)
        cv2.imwrite(image_path,image)


