from semdiffusers import SemanticEditPipeline
import torch
import cv2
import numpy as np
from PIL import Image
from utils import *
import os
from tqdm import tqdm
import pickle
from initiate_random_prompt import randomize_prompt, get_labels
import json

device='cuda'
num_images_per_prompt = 1
guidance_scale = 7

dst_file = "seed.csv"

data_folder = "face_seeds"
selected_folder = "selected"
filtered_folder = "filtered"
selected_folder_full_path = os.path.join(data_folder,selected_folder)
filtered_folder_full_path = os.path.join(data_folder,filtered_folder)

if not os.path.exists(data_folder):
    os.makedirs(data_folder)
    
if not os.path.exists(selected_folder_full_path):
    os.makedirs(selected_folder_full_path)
    
if not os.path.exists(filtered_folder_full_path):
    os.makedirs(filtered_folder_full_path)


pipe = SemanticEditPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
).to(device)

gen = torch.Generator(device=device)

filtered_seeds = []#{"seed":[],"init prompt":[],"label":[]}

for seed in tqdm(range(20)):
    gen.manual_seed(seed)
    initial_prompt, negative_prompts, prompt_map = randomize_prompt()
    negative_prompts = ", ".join(negative_prompts)
    labels = get_labels(prompt_map)
    out = pipe(prompt=initial_prompt, negative_prompt=negative_prompts, generator=gen, num_images_per_prompt=num_images_per_prompt, guidance_scale=guidance_scale)
    images = out.images
    image = images[0]
    image = pil_to_numpy(image)
    if_face = has_single_high_quality_face(image, min_confidence=0.9)
    if if_face:
        filtered_seeds.append({"seed":seed,"init prompt":initial_prompt,"negative prompt":negative_prompts, "label":labels})
        image_path = str(seed).zfill(4)+".png"
        image_path_full = os.path.join(selected_folder_full_path,image_path)
        save_image(image, image_path_full)
    else:
        image_path = str(seed).zfill(4)+".png"
        image_path_full = os.path.join(filtered_folder_full_path,image_path)
        save_image(image, image_path_full)

with open("selected_seeds.json", "w") as file:
    json.dump(filtered_seeds, file)
    