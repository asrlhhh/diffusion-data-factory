from semdiffusers import SemanticEditPipeline
import torch
import cv2
import numpy as np
from PIL import Image
from utils import *
import os
from tqdm import tqdm
import pandas as pd

device='cuda'
edit_threshold_min = 0.95
edit_threshold_max = 0.99
edit_guidance_scale_min = 5
edit_guidance_scale_max = 7
edit_warmup_steps = 20
num_images_per_prompt = 1
guidance_scale = 7
sample_size = 3
samples_num = 10

data_folder = "data"
annotation = "annotation.csv"


if not os.path.exists(data_folder):
    os.makedirs(data_folder)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    # Convert the PIL Image object to a NumPy array
    image_np = np.array(image)

    # Convert RGB to BGR (OpenCV uses BGR format)
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    return image_np

def save_image(image_np: np.ndarray, output_path: str) -> None:
    # Save the NumPy array as an image file using OpenCV
    cv2.imwrite(output_path, image_np)
    

pipe = SemanticEditPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
).to(device)

gen = torch.Generator(device=device)

with open("selected_seeds.json", "r") as file:
    selected_seeds = json.load(file)

attributes_prompts = read_attributes_json("attributes.json")
attributes_setups = read_setups_json("setup.json")
attributes_full = merge_prompts_setups(attributes_prompts,attributes_setups)
image_cnt = 0
annotate_data = {"path":[],"labels":[]}
selected_seeds = selected_seeds[:1]
for single_seed in tqdm(selected_seeds):
    seed = single_seed['seed']
    initial_prompt = single_seed['init prompt']
    negative_prompt = single_seed['negative prompt']
    labels = single_seed['label']
    gen.manual_seed(seed)
    attributes = list(attributes_full.keys())
    attributes_lst = generate_random_sample_2d_list(attributes, samples_num, sample_size)
    out = pipe(prompt=initial_prompt, negative_prompt=negative_prompt, generator=gen, num_images_per_prompt=num_images_per_prompt, guidance_scale=guidance_scale)
    images = out.images
    image = images[0]
    tmp = pil_to_numpy(image)
    png_name = str(image_cnt).zfill(6)+".png"
    save_image(tmp, os.path.join(data_folder,png_name))
    annotate_data["path"].append(os.path.join(data_folder,png_name))
    annotate_data["labels"].append(labels)
    image_cnt = image_cnt + 1
    for attribute_ind,single_attribute in enumerate(attributes_lst):
        direction_lst = generate_random_bool_list(len(single_attribute))
        guidance_lst = generate_random_float_list(len(single_attribute), edit_guidance_scale_min, edit_guidance_scale_max)
        threshold_lst = generate_random_float_list(len(single_attribute), edit_threshold_min, edit_threshold_max)
        warmup_steps = []
        cooldown_steps = []
        attribute_prompts = []
        for attribute in single_attribute:
            warmup_steps.append(attributes_full[attribute]['warmups'])
            cooldown_steps.append(attributes_full[attribute]['cooldowns'])
            attribute_prompts.append(attributes_full[attribute]['prompt'])
        labels_from_latent = {}
        for attr, direction in zip(single_attribute,direction_lst):
            if attr in labels:
                if direction:
                    labels_from_latent[attr] = -1
                else:
                    labels_from_latent[attr] = 1
        gen.manual_seed(seed)
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
                   edit_weights=[1]*len(attributes) # Weights of the individual concepts against each other
                  )
        images = out.images
        update_labels = labels.copy()
        update_labels.update(labels_from_latent)
        image = images[0]
        tmp = pil_to_numpy(image)
        if_face = has_single_high_quality_face(tmp, min_confidence=0.9)
        if if_face:
            png_name = str(image_cnt).zfill(6)+".png"
            save_image(tmp, os.path.join(data_folder,png_name))
            annotate_data["path"].append(os.path.join(data_folder,png_name))
            annotate_data["labels"].append(update_labels)
            image_cnt = image_cnt + 1
        
        
flattened_annotations = {}
flattened_annotations["path"] = []
for key in annotate_data["labels"][0].keys():
    flattened_annotations[key] = []

for i in range(len(annotate_data["path"])):    
    flattened_annotations["path"].append(annotate_data["path"][i])
    for key in annotate_data["labels"][i].keys():
        flattened_annotations[key].append(annotate_data["labels"][i][key])

df = pd.DataFrame(flattened_annotations)
df.to_csv(annotation, index=False)