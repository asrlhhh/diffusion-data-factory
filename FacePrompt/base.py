import re

pattern = r'\{(\w+)\}'

class DefaultDict(dict):
    def __missing__(self, key):
        return f"{{{key}}}"

class PromptContainer:
    def __init__(self, prompt=''):
        self.prompt = prompt

    def preload(self, preload_map):
        self.prompt = self.prompt.format_map(DefaultDict(preload_map))

    def get_all_variables(self):
        matches = re.findall(pattern, self.prompt)
        return matches
    
    def populate(self, prompt_map):

        matches = re.findall(pattern, self.prompt)
        matches = [(x,prompt_map[x]) for x in matches]
        # find the number of matches that are not empty
        num_matches = len([x for x in matches if x[1]!=""])
        if num_matches == 0:
            return ""
        
        else:
            # find the first and last match
            first_match = None
            last_match = None
            for i, x in enumerate(matches):
                if x[1] != "":
                    if first_match is None:
                        first_match = i
                    last_match = i

            for i, x in enumerate(matches):
                if x[1] != "":
                    if x[0] == "Male":
                        tmpp_match = {x[0]:x[1]}
                        self.prompt = self.prompt.format_map(DefaultDict(tmpp_match))
                    elif first_match == i:
                        # replace the first match with the value
                        tmpp_match = {x[0]:" "+x[1]}
                        #print(tmpp_match)
                        #print(self.prompt)
                        self.prompt = self.prompt.format_map(DefaultDict(tmpp_match))
                    elif last_match == i:
                        # replace the last match with the value
                        tmpp_match = {x[0]:", and "+x[1]}
                        self.prompt = self.prompt.format_map(DefaultDict(tmpp_match))
                    else:
                        # replace the middle match with the value
                        tmpp_match = {x[0]:", "+x[1]}
                        self.prompt = self.prompt.format_map(DefaultDict(tmpp_match))
                else:
                    # replace the empty match with empty string
                    tmpp_match = {x[0]:""}
                    #print(tmpp_match)
                    #print(self.prompt)
                    self.prompt = self.prompt.format_map(DefaultDict(tmpp_match))

            return self.prompt
            
