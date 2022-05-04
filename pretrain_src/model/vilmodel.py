import json
import logging
import math
import os
import sys
from io import open
from typing import Callable, List, Tuple
import numpy as np
import copy

import torch
from torch import nn
from torch import Tensor, dtype

from transformers import BertPreTrainedModel
from transformers.modeling_utils import get_parameter_device

logger = logging.getLogger(__name__)

BertLayerNorm = torch.nn.LayerNorm


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}



class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, 
                                         None if head_mask is None else head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOutAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim =config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class BertXAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output

class LXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Lang self-att and FFN layer
        self.lang_self_att = BertAttention(config)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)

        # Visn self-att and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

        # The cross attention layer
        self.visual_attention = BertXAttention(config)

    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Cross Attention
        lang_att_output = self.visual_attention(lang_input, visn_input, ctx_att_mask=visn_attention_mask)
        visn_att_output = self.visual_attention(visn_input, lang_input, ctx_att_mask=lang_attention_mask)
        return lang_att_output, visn_att_output

    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Self Attention
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)
        return lang_att_output, visn_att_output

    def output_fc(self, lang_input, visn_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask,
                      visn_feats, visn_attention_mask):
        lang_att_output = lang_feats
        visn_att_output = visn_feats

        lang_att_output, visn_att_output = self.cross_att(lang_att_output, lang_attention_mask,
                                                          visn_att_output, visn_attention_mask)
        lang_att_output, visn_att_output = self.self_att(lang_att_output, lang_attention_mask,
                                                         visn_att_output, visn_attention_mask)
        lang_output, visn_output = self.output_fc(lang_att_output[0], visn_att_output[0])

        return lang_output, visn_output

class LxmertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_l_layers = config.num_l_layers
        self.num_r_layers = config.num_r_layers
        self.num_h_layers = config.num_h_layers
        self.num_x_layers = config.num_x_layers
        self.update_lang_bert = config.update_lang_bert

        # Using self.layer instead of self.l_layers to support loading BERT weights.
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )
        self.h_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_h_layers)]
        ) if self.num_h_layers > 0 else None
        self.r_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_r_layers)]
        ) if self.num_r_layers > 0 else None
        self.x_layers = nn.ModuleList(
            [LXRTXLayer(config) for _ in range(self.num_x_layers)]
        )

        if config.include_objects:
            self.num_o_layers = config.num_o_layers
            self.o_layers = nn.ModuleList(
                [BertLayer(config) for _ in range(self.num_o_layers)]
            ) if self.num_o_layers > 0 else None

    def forward(self, txt_embeds, extended_txt_masks, hist_embeds,
                extended_hist_masks, img_embeds=None, extended_img_masks=None,
                obj_embeds=None, extended_obj_masks=None):
        # text encoding
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]

        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()

        # image encoding
        if img_embeds is not None:
            if self.r_layers is not None:
                for layer_module in self.r_layers:
                    temp_output = layer_module(img_embeds, extended_img_masks)
                    img_embeds = temp_output[0]
        img_max_len = 0 if img_embeds is None else img_embeds.size(1)
            

        # history encoding
        if self.h_layers is not None:
            for layer_module in self.h_layers:
                temp_output = layer_module(hist_embeds, extended_hist_masks)
                hist_embeds = temp_output[0]
        hist_max_len = hist_embeds.size(1)

        # object encoding
        if obj_embeds is not None:
            # extended_obj_head_masks: mask padding for non included objects
            if self.o_layers is not None:
                for layer_module in self.o_layers:
                    temp_output = layer_module(obj_embeds, extended_obj_masks)
                    obj_embeds = temp_output[0]
        
        # cross-modal encoding
        if img_embeds is None:
            hist_img_embeds = hist_embeds
            extended_hist_img_masks = extended_hist_masks
        else:
            hist_img_embeds = torch.cat([hist_embeds, img_embeds], 1)
            extended_hist_img_masks = torch.cat([extended_hist_masks, extended_img_masks], -1)

        if obj_embeds is not None:
            hist_img_embeds = torch.cat([hist_img_embeds, obj_embeds], 1)
            extended_hist_img_masks = torch.cat([extended_hist_img_masks, extended_obj_masks], -1)

        for layer_module in self.x_layers:
            txt_embeds, hist_img_embeds = layer_module(
                txt_embeds, extended_txt_masks, 
                hist_img_embeds, extended_hist_img_masks)

        hist_embeds = hist_img_embeds[:, :hist_max_len]
        if img_embeds is not None:
            if obj_embeds is None:
                img_embeds = hist_img_embeds[:, hist_max_len:]
            else:
                img_embeds = hist_img_embeds[:, hist_max_len:hist_max_len+img_max_len]

        if obj_embeds is not None:
            if img_embeds is None:
                obj_embeds = hist_img_embeds[:, hist_max_len:]
            else:
                obj_embeds = hist_img_embeds[:, hist_max_len+img_max_len:]

        return txt_embeds, hist_embeds, img_embeds, obj_embeds



class ImageEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
        self.ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        # 0: non-navigable, 1: navigable, 2: stop
        self.nav_type_embedding = nn.Embedding(3, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, ang_feat, type_embeddings, nav_types=None):
        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_ang = self.ang_layer_norm(self.ang_linear(ang_feat))
        embeddings = transformed_im + transformed_ang + type_embeddings
        if nav_types is not None:
            nav_embeddings = self.nav_type_embedding(nav_types)
            embeddings = embeddings + nav_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class HistoryEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
        self.ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        if config.num_h_pano_layers > 0:
            self.pano_img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
            self.pano_img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            self.pano_ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
            self.pano_ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            pano_encoder_config = copy.copy(config)
            pano_encoder_config.num_hidden_layers = config.num_h_pano_layers
            self.pano_encoder = BertEncoder(pano_encoder_config)
        else:
            self.pano_encoder = None

        self.position_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        # special type embedding for history
        self.type_embedding = nn.Embedding(1, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    @property
    def device(self):
        return get_parameter_device(self)

    def forward(self, img_feats, ang_feats, pano_img_feats, pano_ang_feats, 
                pos_ids=None, batch_size=None):
        type_ids = torch.zeros((batch_size, 1)).long().to(self.device)
        type_embeddings = self.type_embedding(type_ids)

        cls_embeddings = self.dropout(self.layer_norm(
            self.cls_token.expand(batch_size, -1, -1) + type_embeddings))

        if img_feats is not None:
            embeddings = self.img_layer_norm(self.img_linear(img_feats)) + \
                         self.ang_layer_norm(self.ang_linear(ang_feats)) + \
                         type_embeddings

            if self.pano_encoder is not None:
                batch_size, num_steps, num_pano, _ = pano_img_feats.size()
                pano_img_feats = pano_img_feats.view(batch_size*num_steps, num_pano, -1)
                pano_ang_feats = pano_ang_feats.view(batch_size*num_steps, num_pano, -1)
                pano_embeddings = self.pano_img_layer_norm(self.pano_img_linear(pano_img_feats)) + \
                                  self.pano_ang_layer_norm(self.pano_ang_linear(pano_ang_feats))
                # assume pano all exists
                ext_pano_masks = torch.zeros(batch_size*num_steps, num_pano, dtype=torch.float).to(self.device).unsqueeze(1).unsqueeze(2)
                pano_embeddings = self.pano_encoder(pano_embeddings, ext_pano_masks)[0]
                
                pano_embeddings = pano_embeddings.view(batch_size, num_steps, num_pano, -1)
                pano_embeddings = torch.mean(pano_embeddings, 2)
                
                embeddings = embeddings + pano_embeddings

            if pos_ids is not None:
                embeddings = embeddings + self.position_embeddings(pos_ids)
                embeddings = self.layer_norm(embeddings)
                embeddings = self.dropout(embeddings)
        else:
            embeddings = None

        return cls_embeddings, embeddings


class NavPreTrainedModel(BertPreTrainedModel):
    r""" Modification of LXMERT Model """
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.img_embeddings = ImageEmbeddings(config)

        if config.include_objects:
            self.obj_embeddings = ImageEmbeddings(config)

        # share image encoding
        self.hist_embeddings = HistoryEmbeddings(config)

        self.encoder = LxmertEncoder(config)

        self.init_weights()

    def forward(self,
        txt_ids, txt_masks, 
        hist_img_feats, hist_ang_feats, hist_pano_img_feats, hist_pano_ang_feats, hist_masks,
        ob_img_feats, ob_ang_feats, ob_nav_types, ob_masks,
        obj_img_feats, obj_ang_feats, obj_masks,
    ):
        batch_size = txt_ids.size(0)

        # text embedding
        extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2)
        extended_txt_masks = extended_txt_masks.to(dtype=self.dtype)
        extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0
        txt_token_type_ids = torch.zeros_like(txt_ids)
        txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)

        # history embedding
        extended_hist_masks = hist_masks.unsqueeze(1).unsqueeze(2)
        extended_hist_masks = extended_hist_masks.to(dtype=self.dtype)
        extended_hist_masks = (1.0 - extended_hist_masks) * -10000.0
        if hist_img_feats is not None:
            hist_max_len = hist_img_feats.size(1)
            hist_step_ids = torch.arange(hist_max_len).expand((1, -1)).to(self.device)
        else:
            hist_step_ids = None
        hist_cls_embeds, hist_vp_embeds = self.hist_embeddings(hist_img_feats, hist_ang_feats, 
            hist_pano_img_feats, hist_pano_ang_feats, hist_step_ids,
            batch_size=batch_size)
        if hist_vp_embeds is None:
            hist_embeds = hist_cls_embeds
        else:
            hist_embeds = torch.cat([hist_cls_embeds, hist_vp_embeds], dim=1)

        # image embedding
        if ob_img_feats is not None:
            ob_token_type_ids = torch.ones(batch_size, 1).long().to(self.device)
            ob_embeds = self.img_embeddings(ob_img_feats, ob_ang_feats, 
                self.embeddings.token_type_embeddings(ob_token_type_ids), 
                nav_types=ob_nav_types)
            extended_ob_masks = ob_masks.unsqueeze(1).unsqueeze(2)
            extended_ob_masks = extended_ob_masks.to(dtype=self.dtype)
            extended_ob_masks = (1.0 - extended_ob_masks) * -10000.0
        else:
            ob_embeds, extended_ob_masks = None, None

        if obj_img_feats is not None:
            # Object type = 3
            obj_token_type_ids = torch.ones(batch_size, 1, dtype=torch.long, device=self.device) * 3
            obj_embeds = self.obj_embeddings(
                obj_img_feats,
                obj_ang_feats,
                self.embeddings.token_type_embeddings(obj_token_type_ids)
            )
            extended_obj_head_masks = obj_masks.unsqueeze(1).unsqueeze(2)
            extended_obj_head_masks = extended_obj_head_masks.to(dtype=self.dtype)
            extended_obj_head_masks = (1.0 - extended_obj_head_masks) * -10000.0
        else:
            obj_embeds, extended_obj_head_masks = None, None

        # multi-modal encoding
        txt_embeds, hist_embeds, ob_embeds, obj_embeds = self.encoder(
            txt_embeds, extended_txt_masks, 
            hist_embeds, extended_hist_masks,
            ob_embeds, extended_ob_masks,
            obj_embeds, extended_obj_head_masks
        )

        return txt_embeds, hist_embeds, ob_embeds, obj_embeds

    def forward_itm(self, txt_ids, txt_masks, 
                hist_img_feats, hist_ang_feats, hist_pano_img_feats, hist_pano_ang_feats, hist_masks,
                num_neg_trajs=4):
        batch_size, hist_max_len, _ = hist_img_feats.size()

        # text encoding
        extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2)
        extended_txt_masks = extended_txt_masks.to(dtype=self.dtype)
        extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0
        txt_token_type_ids = torch.zeros_like(txt_ids)
        txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
        for layer_module in self.encoder.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]
        # copy txt_embeds
        batch_size = txt_embeds.size(0)
        txt_embeds = txt_embeds.repeat(1+num_neg_trajs, 1, 1)
        extended_txt_masks = extended_txt_masks.repeat(1+num_neg_trajs, 1, 1, 1)

        # history encoding
        extended_hist_masks = hist_masks.unsqueeze(1).unsqueeze(2)
        extended_hist_masks = extended_hist_masks.to(dtype=self.dtype)
        extended_hist_masks = (1.0 - extended_hist_masks) * -10000.0

        hist_cls_embeds, hist_vp_embeds_no_pos = self.hist_embeddings(hist_img_feats, hist_ang_feats, 
            hist_pano_img_feats, hist_pano_ang_feats, pos_ids=None,
            batch_size=batch_size)
        hist_max_len = hist_img_feats.size(1)
        hist_step_ids = torch.arange(hist_max_len).expand((1, -1)).to(self.device)
        hist_vp_embeds = self.hist_embeddings.dropout(self.hist_embeddings.layer_norm(
            hist_vp_embeds_no_pos + self.hist_embeddings.position_embeddings(hist_step_ids)))
        hist_embeds = torch.cat([hist_cls_embeds, hist_vp_embeds], dim=1)
        
        if self.encoder.h_layers is not None:
            for layer_module in self.encoder.h_layers:
                temp_output = layer_module(hist_embeds, extended_hist_masks)
                hist_embeds = temp_output[0]

        neg_hist_embeds, neg_hist_masks = [], []
        # random negs in batch
        K = num_neg_trajs // 2
        if batch_size > 1:
            neg_idxs = []
            for i in range(batch_size):
                neg_idxs.append(np.random.choice(np.arange(0, i).tolist() + np.arange(i+1, batch_size).tolist(), K))
            neg_idxs = torch.from_numpy(np.stack(neg_idxs, 0)).to(self.device)

            for k in range(K):
                neg_hist_embeds.append(hist_embeds[neg_idxs[:, k]])
                neg_hist_masks.append(extended_hist_masks[neg_idxs[:, k]])
        else:
            K = num_neg_trajs

        # shuffled negs
        hist_lens = torch.sum(hist_masks, 1) - 1
        for _ in range(K):
            shuffled_pos_ids = []
            for i in range(batch_size):
                shuffled_idxs = torch.randperm(hist_lens[i])
                shuffled_idxs = torch.cat([shuffled_idxs, torch.arange(hist_lens[i], hist_max_len, dtype=torch.long)], 0).to(self.device)
                shuffled_pos_ids.append(shuffled_idxs)
            shuffled_pos_ids = torch.stack(shuffled_pos_ids, 0)
            shuffled_hist_embeds = torch.cat([hist_cls_embeds, \
                self.hist_embeddings.dropout(self.hist_embeddings.layer_norm(
                hist_vp_embeds_no_pos + self.hist_embeddings.position_embeddings(shuffled_pos_ids)))], dim=1)

            if self.encoder.h_layers is not None:
                for layer_module in self.encoder.h_layers:
                    temp_output = layer_module(shuffled_hist_embeds, extended_hist_masks)
                    shuffled_hist_embeds = temp_output[0]
            neg_hist_embeds.append(shuffled_hist_embeds)
            neg_hist_masks.append(extended_hist_masks)
        
        pos_neg_hist_embeds = torch.cat([hist_embeds] + neg_hist_embeds, 0)
        pos_neg_hist_masks = torch.cat([extended_hist_masks] + neg_hist_masks, 0)
            
        # multi-modal encoding
        for layer_module in self.encoder.x_layers:
            txt_embeds, pos_neg_hist_embeds = layer_module(
                txt_embeds, extended_txt_masks, 
                pos_neg_hist_embeds, pos_neg_hist_masks)

        fused_embeds = txt_embeds[:, 0] * pos_neg_hist_embeds[:, 0]
        fused_embeds = torch.stack(torch.split(fused_embeds, batch_size), 1)
        return fused_embeds

