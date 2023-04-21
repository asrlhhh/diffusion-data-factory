from semdiffusers import SemanticStableDiffusionImg2ImgPipeline
import cv2
import numpy as np
from PIL import Image

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
    
pipe = SemanticStableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",).to("cuda")
image_compare, _ , cond_embeddings, coordinate, original_size = pipe.invert("output/Bald/1_156766.jpg",7.5)
image_original = cv2.imread("output/Bald/1_156766.jpg")
image_gt, image_rec = image_compare
x1, y1, x2, y2 = coordinate
w, h = original_size
save_image(image_gt, "original.png")
save_image(image_rec, "recon.png")
#out = pipe(49,editing_prompt= ["Smiling, smile, laughing"],reverse_editing_direction=[False],edit_warmup_steps=[20],edit_cooldown_steps= [35], edit_guidance_scale=[5.0]) #edit_warmup_steps=[20], edit_cooldown_steps= [30]
out = pipe(45)#,editing_prompt= ["Smiling, smile, laughing"],reverse_editing_direction=[False])
images = out.images
image = images[0]
image = pil_to_numpy(image)
image = image[y1:y2+1,x1:x2+1,:]
image = cv2.resize(image,(w, h))
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
vis = np.concatenate((image_original, image), axis=1)

save_image(vis, "edited23.png")