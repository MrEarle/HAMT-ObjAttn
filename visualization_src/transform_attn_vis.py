import jsonlines
import h5py
from tqdm import tqdm
import numpy as np

with jsonlines.open('/home/mrearle/repos/VLN-HAMT/datasets/vis_data.jsonl') as reader:
    with h5py.File('/home/mrearle/repos/VLN-HAMT/datasets/vis_data.h5', 'w') as writer:
        for obj in tqdm(reader):
            model_ins = obj['model_ins']
            if 'instr_id' not in model_ins:
                continue
            attns = obj['attns']
            self_attns = obj['self_attns']

            ds = writer.create_group(model_ins['instr_id'])
            ds.attrs['instr_id'] = model_ins['instr_id']
            ds.attrs['txt_instr'] = model_ins['txt_instr']

            ds.create_dataset('instr', data=model_ins['instr'])


            indices = [k for k in obj['attns'].keys()]
            indices.sort()

            for i in indices:
                step = ds.create_group(str(i))
                step.create_dataset('attn', data=np.array(list(attns[i].values())))
                step.create_dataset('self_attn', data=np.array(list(self_attns[i].values())))
                step.attrs['scan'] = model_ins[i]['scan']
                step.attrs['viewpoint'] = model_ins[i]['viewpoint']
                step.attrs['agent_heading'] = model_ins[i]['agent_heading']

                step.attrs['views'] =  model_ins[i]['views']
                step.attrs['objs'] = model_ins[i]['objs']
                step.create_dataset('obj_orients', data=model_ins[i]['obj_orients'])
                step.create_dataset('view_orients', data=model_ins[i]['view_orients'])
