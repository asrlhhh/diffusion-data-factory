prompt_map = {
    "Smiling": {-1:"", 1:"smiling"},"Young": {1:"young", -1:"old"},
    "Male": {1:"male", -1:"female"},"Wavy_Hair": {1:"curly", 2:"straight"},
    "hair_color": {1:"blonde", 2:"black", 3:"brown", 4:"gray"},
    "Receding_Hairline": {-1:"", 1:"receding"},"Bangs": {-1:"", 1:"hair over forehead"},
    "Bald": {-1:"", 1:"bald"},"Big_Lips": {1:"thick lips", -1:"thin lips"},
    "Mouth_Slightly_Open": {-1:"closed mouth", 1:"wide open mouth"},
    "Rosy_Cheeks": {-1:"", 1:"flushing"},"Chubby": {-1:"", 1:"chubby face"},
    "Oval_Face": {-1:"", 1:"oval face"},"Double_Chin": {-1:"", 1:"double chins"},
    "High_Cheekbones": {-1:"", 1:"high and defined cheekbone"},"Bushy_Eyebrows": {-1:"", 1:"bushy"},
    "Arched_Eyebrows": {-1:"", 1:"arched"},"Narrow_Eyes": {-1:"big eyes", 1:"narrow eyes"},
    "Bags_Under_Eyes": {-1:"", 1:"bags under eyes"},"Eye_glasses": {-1:"", 1:"eye glasses"},
    "Pale_Skin": {-1:"", 1:"pale skin"},"Heavy_Makeup": {-1:"", 1:"heavy makeups"},
    "Big_Nose": {1:"big", -1:"small"},"Pointy_Nose": {1:"pointy", -1:"flat"},
    "Mustache": {-1:"", 1:"mustache"},"Sideburns": {-1:"", 1:"sideburns"},
    "Wearing_Earrings": {-1:"", 1:"earrings"},"Wearing_Hat": {-1:"", 1:"hat"},
    "Wearing_Lipstick": {-1:"", 1:"lipstick"},"Wearing_Necklace": {-1:"", 1:"necklace"},
    "Wearing_Necktie": {-1:"", 1:"necktie"}
}

negative_prompt_map = {
    "Smiling": "similing", "Young": "young", "Male": "male", "Receding_Hairline": "receding hairline",
    "Bangs": "bangs", "Bald": "bald", "Big_Lips": "big lips", "Mouth_Slightly_Open": "mouth open",
    "Rosy_Cheeks": "flushing cheeks", "Chubby": "chubby", "Oval_Face": "oval face", "Double_Chin": "double chin",
    "High_Cheekbones": "high cheekbones", "Bushy_Eyebrows": "bushy eyebrows", "Arched_Eyebrows": "arched eyebrows",
    "Narrow_Eyes": "narrow eyes", "Bags_Under_Eyes": "bags under eyes", "Eye_glasses": "eye glasses",
    "Pale_Skin": "pale skin", "Heavy_Makeup": "heavy makeup", "Big_Nose": "big nose", "Pointy_Nose": "pointy nose",
    "Mustache": "mustache", "Sideburns": "sideburns", "Wearing_Earrings": "earrings", "Wearing_Hat": "hat",
    "Wearing_Lipstick": "lipstick", "Wearing_Necklace": "necklace", "Wearing_Necktie": "necktie"}

# 1. random select from prompt_map
# 2. loop in the selection: set value to 0 (value is "", label is -1, but don't add negative prompt)

mutual_exclusive_map = {"Male":(-1,["Mustache","Sideburns"]),
                   "Receding_Hairline":(1,["Bangs","Bald"]),
                    "Wavy_Hair":(1,["Bald"]),"Wavy_Hair":(2,["Bald"]),
                    "hair_color":(1,["Bald"]),"hair_color":(2,["Bald"]),
                    "hair_color":(3,["Bald"]),"hair_color":(4,["Bald"]),
                   "Bangs":(1,["Receding_Hairline","Bald"]),
                        "Bald":(1,["hair_color","Wavy_Hair"])
                   }

hair_style_map = {-1:{"Wavy_Hair":-1,"Straight_Hair":-1},
                  0: {"Wavy_Hair":-1,"Straight_Hair":-1},
                 1: {"Wavy_Hair":1,"Straight_Hair":-1},
                 2: {"Wavy_Hair":-1,"Straight_Hair":1}}

hair_color_map = {-1:{"Black_Hair":-1, "Blond_Hair":-1, "Brown_Hair":-1, "Gray_Hair":-1}, 
                    0:{"Black_Hair":-1, "Blond_Hair":-1, "Brown_Hair":-1, "Gray_Hair":-1},
                  1:{"Black_Hair":-1, "Blond_Hair":1, "Brown_Hair":-1, "Gray_Hair":-1}, 
                  2:{"Black_Hair":1, "Blond_Hair":-1, "Brown_Hair":-1, "Gray_Hair":-1}, 
                  3:{"Black_Hair":-1, "Blond_Hair":-1, "Brown_Hair":1, "Gray_Hair":-1},
                  4:{"Black_Hair":-1, "Blond_Hair":-1, "Brown_Hair":-1, "Gray_Hair":1}}

pronoun_map = {1:"he", -1:"she"}