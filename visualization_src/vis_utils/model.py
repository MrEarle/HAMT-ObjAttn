from collections import defaultdict
import os
import sys
from typing import List

r2r_path = "/home/mrearle/repos/VLN-HAMT/finetune_src"
if r2r_path not in sys.path:
    sys.path.append(r2r_path)

from models.model_HAMT import VLNBertCMT
from models.vilmodel_cmt import BertAttention, BertXAttention, LXRTXLayer, LxmertEncoder, NavCMT
from r2r.main import build_dataset
from r2r.agent_cmt import Seq2SeqCMTAgent
from r2r.parser import parse_args as parse_args_r2r, postprocess_args as postprocess_args_r2r

def parse_args(
    output_dir,
    rootdir='/home/mrearle/repos/VLN-HAMT/datasets',
    dataset='r2r',
    ob_type='pano',
    world_size=1,
    seed=0,
    num_l_layers=9,
    num_x_layters=4,
    hist_enc_pano=True,
    hist_pano_num_layers=2,
    fix_lang_embedding=True,
    fix_hist_embedding=True,
    features='vitbase',
    feedback='sample',
    max_action_len=15,
    max_instr_len=60,
    image_feat_size=768,
    angle_feat_size=4,
    lr=1e-5,
    iters=300000,
    log_every=2000,
    batch_size=8,
    optim='adamW',
    ml_weight=0.2,
    feat_dropout=0.4,
    dropout=0.5,
    include_objects=True,
):
    kwargs = locals()
    args = parse_args_r2r()
    for k, v in kwargs.items():
        setattr(args, k, v)
    args = postprocess_args_r2r(args)
    setattr(args, 'visualization_mode', True)
    setattr(args, 'output_attentions', True)
    return args


def get_hooks(batch_size):
    # batch_size x { [step]: { layer_id: attn_vals } }
    attentions = [defaultdict(dict) for _ in range(batch_size)]
    self_attentions = [defaultdict(dict) for _ in range(batch_size)]
    # batch_size x { instr: instr_tokens, [step]: { obj: obj_tokens, views: view_ids } }
    model_ins = [defaultdict(dict) for _ in range(batch_size)]

    step = 0
    def get_bert_x_attn_hook(layer_id):
        def hook(module: BertXAttention, ins, outs):
            nonlocal step
            _, scores = outs
            for i in range(batch_size):
                run = attentions[i]
                run[step][layer_id] = scores[i].detach().cpu().tolist()
        return hook

    def get_bert_self_attn_hook(layer_id):
        def hook(module: BertAttention, ins, outs):
            nonlocal step
            _, scores = outs
            for i in range(batch_size):
                run = self_attentions[i]
                run[step][layer_id] = scores[i].detach().cpu().tolist()
        return hook

    def get_vln_bert_cmt_hook():
        def hook(module: VLNBertCMT, ins, outs):
            nonlocal step, model_ins
            in_args = module.forward_args
            vis_ids = module.vis_ids
            if in_args['mode'] == 'visual':
                for i in range(batch_size):
                    batch = model_ins[i]
                    b_step = batch[step]
                    batch['instr'] = vis_ids['txt_ids'][i]
                    batch['txt_instr'] = vis_ids['instruction'][i]
                    batch['instr_id'] = vis_ids['instr_id'][i]

                    b_step['scan'] = vis_ids['scan'][i]
                    b_step['viewpoint'] = vis_ids['viewpoint'][i]
                    b_step['agent_heading'] = vis_ids['heading'][i]

                    b_step['views'] = vis_ids['view_ids'][i]
                    b_step['objs'] = [s.decode() for s in vis_ids['obj_ids'][i]]
                    b_step['obj_orients'] = vis_ids['obj_angles'][i]
                    b_step['view_orients'] = vis_ids['view_angles'][i]
                step += 1

        return hook

    def reset():
        nonlocal step
        nonlocal attentions
        nonlocal self_attentions
        nonlocal model_ins
        step = 0
        attentions = [defaultdict(dict) for _ in range(batch_size)]
        self_attentions = [defaultdict(dict) for _ in range(batch_size)]
        model_ins = [defaultdict(dict) for _ in range(batch_size)]

        return attentions, self_attentions, model_ins

    return (
        [attentions, self_attentions],
        model_ins,
        reset,
        get_bert_x_attn_hook,
        get_bert_self_attn_hook,
        get_vln_bert_cmt_hook
    )

def attach_hooks(agent: Seq2SeqCMTAgent, args):
    (
        attentions,
        model_ins,
        reset,
        get_bert_x_attn_hook,
        get_bert_self_attn_hook,
        get_vln_bert_cmt_hook
    ) = get_hooks(args.batch_size)

    vln_bert_cmt: VLNBertCMT = agent.vln_bert
    vln_bert_cmt_hook = get_vln_bert_cmt_hook()
    vln_bert_cmt.register_forward_hook(vln_bert_cmt_hook)

    nav_cmt: NavCMT = vln_bert_cmt.vln_bert
    lxmert_encoder: LxmertEncoder = nav_cmt.encoder
    x_layers: List[LXRTXLayer] = lxmert_encoder.x_layers
    for i, x_layer in enumerate(x_layers):
        bert_x_attn: BertXAttention = x_layer.visual_attention
        hook = get_bert_x_attn_hook(i)
        bert_x_attn.register_forward_hook(hook)

        bert_vis_self_attn: BertAttention = x_layer.visn_self_att
        hook = get_bert_self_attn_hook(i)
        bert_vis_self_attn.register_forward_hook(hook)


    return attentions, model_ins, reset

def get_model(args, resume_file):
    train_env, val_envs, _ = build_dataset(args, rank=0)

    agent_class = Seq2SeqCMTAgent
    listner = agent_class(args, train_env, rank=0)

    attentions, model_ins, reset_hooks = attach_hooks(listner, args)

    # resume file
    start_iter = listner.load(os.path.join(resume_file))
    print("\nLOAD the model from {}, iteration {}".format(args.resume_file, start_iter))

    return listner, val_envs, attentions, model_ins, reset_hooks

def evaluate(val_envs, listner: Seq2SeqCMTAgent, num_iters=None, callback=None):
    # first evaluation
    loss_str = "validation before training"
    for env_name, env in val_envs.items():
        if env_name != 'val_unseen':
            continue
        listner.env = env
        # Get validation distance from goal under test evaluation conditions
        listner.test_cb(use_dropout=False, feedback='argmax', iters=num_iters, callback=callback)
        preds = listner.get_results()
        
        score_summary, _ = env.eval_metrics(preds)
        loss_str += ", %s " % env_name
        for metric, val in score_summary.items():
            loss_str += ', %s: %.2f' % (metric, val)
    print(loss_str)
