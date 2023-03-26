from FacePrompt import *
from FacePrompt.prompt import prompt_template
from FacePrompt.prompt_map import prompt_map, negative_prompt_map
from FacePrompt.prompt_map import hair_color_map, pronoun_map, mutual_exclusive_map, hair_style_map
from FacePrompt.base import PromptContainer
import random
from transformers import GPT2TokenizerFast

token_counter = GPT2TokenizerFast.from_pretrained("gpt2")
token_limit = 70

def handle_mutual_exlusion(prompt_map_random):
    # handle the mutual exclusive map
    for key in prompt_map_random:
        if key in mutual_exclusive_map and mutual_exclusive_map[key][0]==prompt_map_random[key]:
            for mutual_key in mutual_exclusive_map[key][1]:
                prompt_map_random[mutual_key] = 0
    return prompt_map_random

def randomize_prompt():
    # randomly select a key value from each key of the prompt_map
    # make sure that there are half the keys that have non -1 values
    # and return the prompt
    prompt_map_random = {}
    # randomly select half of the keys
    prompt_map_keys = list(prompt_map.keys())
    random.shuffle(prompt_map_keys)
    prompt_map_keys = prompt_map_keys[:len(prompt_map_keys)//3*2]
    for key in prompt_map_keys:
        # randomly select a non -1 value
        prompt_map_random[key] = random.choice([x for i,x in enumerate(prompt_map[key]) if x != -1])
    # for the keys that are not selected, set the value to -1 if the value is in prompt_map[key]
    # otherwise set the value to arbitrary value
    for key in prompt_map:
        if key not in prompt_map_random:
            if -1 in prompt_map[key]:
                prompt_map_random[key] = -1
            else:
                prompt_map_random[key] = random.choice(list(prompt_map[key].keys()))
    # reorder the keys of prompt_map_random as the same order of prompt_map
    prompt_map_random = {key:prompt_map_random[key] for key in prompt_map}
    prompt_map_random = handle_mutual_exlusion(prompt_map_random)
    
    negative_prompts = []
    prompt_map_random_val = {}
    for key in prompt_map_random:
        if prompt_map_random[key] == -1:
            prompt_map_random_val[key] = prompt_map[key][prompt_map_random[key]]
            negative_prompts.append(negative_prompt_map[key])
        elif prompt_map_random[key] == 0:
            prompt_map_random_val[key] = ""
            prompt_map_random[key] = -1
        else:
            prompt_map_random_val[key] = prompt_map[key][prompt_map_random[key]]
    prompt_final = ""
    prompt_template_keys = list(prompt_template.keys())
    prompt_template_keys.remove("base")
    random.shuffle(prompt_template_keys)
    prompt_template_keys = ["base"] + prompt_template_keys
    new_prompt_map_random = {}
    for i,prompt_key in enumerate(prompt_template_keys):
        container = PromptContainer(prompt_template[prompt_key])
        current_prompt_keys = container.get_all_variables()
        for key in current_prompt_keys:
            new_prompt_map_random[key] = prompt_map_random[key]
        tmp_prompt = container.populate(prompt_map_random_val)
        if len(token_counter(prompt_final+tmp_prompt)['input_ids']) > token_limit:
            break
        prompt_final += tmp_prompt
    if prompt_final[-1:] == ",":
        prompt_final = prompt_final[:-1] + "."
    prompt_final = prompt_final + " high quality, detailed."
    return prompt_final, negative_prompts, prompt_map_random

def get_labels(prompt_map_random_val):
    # get the labels for the prompt
    labels = {}
    for key in prompt_map_random_val:
        if key == "hair_color":
            hair_map = hair_color_map[prompt_map_random_val[key]]
            labels.update(hair_map)
        elif key == "Wavy_Hair":
            hair_map = hair_style_map[prompt_map_random_val[key]]
            labels.update(hair_map)
        else:
            labels[key] = prompt_map_random_val[key]
    return labels

if __name__ == "__main__":
    prompt, negative_prompts, prompt_map_random_val = randomize_prompt()
    labels = get_labels(prompt_map_random_val)

