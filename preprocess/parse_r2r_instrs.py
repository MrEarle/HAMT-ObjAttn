from transformers import AutoTokenizer
import json
from tqdm import tqdm
import os

def get_tokenizer():
    cfg_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def parse_r2r_file(split: str, path: str, dest_path: str, tokenizer: AutoTokenizer):
    with open(path, 'r') as f:
        data = json.load(f)

    for path_dict in tqdm(data, desc=f'Parsing split {split}', unit='path'):
        instr_encs = []
        for instr in path_dict['instructions']:
            instr_enc = tokenizer.encode(instr)
            instr_encs.append(instr_enc)

        path_dict['instr_encodings'] = instr_encs

    with open(dest_path, 'w') as f:
        json.dump(data, f)


base_path = '/home/mrearle/datasets/Matterport3DSimulator/semantically_richer_instructions'
dest_base_path = '/home/mrearle/repos/VLN-HAMT/datasets/R2R/annotations'
splts = [ 'train', 'val_unseen' ]

tokenizer = get_tokenizer()
for split in splts:
    path = os.path.join(base_path, f'R2R_craft_{split}.json')
    dest_path = os.path.join(dest_base_path, f'R2R_craft_{split}.json')
    parse_r2r_file(split, path, dest_path, tokenizer)

