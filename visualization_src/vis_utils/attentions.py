from collections import defaultdict
from bertviz import head_view
from IPython.core.display import display, HTML, Javascript
import re

def text_obj_attn(model_ins, attns, tokenizer, batch_idx, step):
    obj_ids = model_ins[batch_idx][step]['objs']
    txt_ids = [tokenizer.decode([word]) for word in model_ins[batch_idx]['instr'].tolist()]
    obj_count = len(obj_ids)
    txt_count = len(txt_ids)
    obj_text_attn = [attn[:, -20:-20 + obj_count, :txt_count].unsqueeze(0) for attn in attns[batch_idx][step].values()]

    head_view(
        cross_attention=obj_text_attn,
        encoder_tokens=txt_ids,
        decoder_tokens=obj_ids,    
    )

def view_obj_attn(obj_ids, view_ids, attn):
    view_ids += ['STOP']
    obj_count = len(obj_ids)
    view_count = len(view_ids)

    view_start_idx = -(20 + 37)
    view_end_idx = -(20 + 37) + view_count
    obj_start_idx = -20
    obj_end_idx = -20 + obj_count

    view_range = slice(view_start_idx, view_end_idx)
    obj_range = slice(obj_start_idx, obj_end_idx or None)

    obj_view_attn = attn[:, :, view_range, obj_range].unsqueeze(1)

    obj_count = defaultdict(int)
    obj_names = []
    for obj in obj_ids:
        obj_names.append(f'{obj}_{obj_count[obj]}')
        obj_count[obj] += 1

    view_names = [i[:6] for i in view_ids]

    print(obj_view_attn)

    html = head_view(
        cross_attention=obj_view_attn,
        encoder_tokens=obj_names,
        decoder_tokens=view_names,
        # html_action='return'
    )

    
    # display(HTML('<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>'))
    # display(HTML(f'<div style="background-color: #f5f5f5; padding: 10px;">{html.data}</div>'))