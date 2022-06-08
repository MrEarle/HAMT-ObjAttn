import sys

from tqdm import tqdm

mattersim_path = "/home/mrearle/repos/VLN-HAMT/mattersim_build"
if mattersim_path not in sys.path:
    sys.path.append(mattersim_path)

vis_path = "/home/mrearle/repos/VLN-HAMT/visualization_src"
if vis_path not in sys.path:
    sys.path.append(vis_path)


from vis_utils.model import parse_args, get_model, evaluate

# # Args


args = parse_args(
    output_dir='/home/mrearle/repos/VLN-HAMT/datasets/trained_models/test/',
)
args.visualization_mode


# # Init model

resume_file = '/home/mrearle/repos/VLN-HAMT/datasets/R2R/trained_models/vitbase-finetune-objs/ckpts/best_val_unseen'
print(f'Loading model from {resume_file}')
agent, val_envs, (attns, self_attns), model_ins, reset_attns = get_model(args, resume_file)
print('Model loaded')


# # Eval

import jsonlines

print('Starting evaluation...', flush=True)
dest_file = '/home/mrearle/repos/VLN-HAMT/datasets/vis_data.jsonl'
open(dest_file, 'w').close()


pbar = tqdm()
num_processed = 0
def eval_callback(*_):
    global attns, self_attns, model_ins, reset_attns, pbar, num_processed

    data = [
        {
            'attns': attn,
            'self_attns': self_attn,
            'model_ins': model_in,
        } for attn, self_attn, model_in in zip(attns, self_attns, model_ins)
    ]

    with jsonlines.open(dest_file, mode='a') as writer:
        writer.write_all(data)

    num_processed += len(data)
    attns, self_attns, model_ins = reset_attns()

    pbar.update(1)
    pbar.set_description(f'Number of runs done: {num_processed}')

evaluate(val_envs, agent, num_iters=None, callback=eval_callback)


import json
with open('/home/mrearle/repos/VLN-HAMT/vis_data.json') as f:
    json.dump(all_data, f)

