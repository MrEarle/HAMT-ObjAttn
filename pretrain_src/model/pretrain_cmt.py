from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel

from .vilmodel import BertLayerNorm, BertOnlyMLMHead
from .vilmodel import NavPreTrainedModel


class NextActionPrediction(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

class NextActionRegression(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 3))

    def forward(self, x):
        return self.net(x)

class SpatialRelRegression(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size*2, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 2))

    def forward(self, x):
        return self.net(x)

class RegionClassification(nn.Module):
    " for MRC(-kl)"
    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output

class ItmPrediction(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

class ObjectClassification(nn.Module):
    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            BertLayerNorm(hidden_size, eps=1e-12),
            nn.Linear(hidden_size, label_dim)
        )

    def forward(self, x):
        return self.net(x)

class RoomClassification(nn.Module):
    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            BertLayerNorm(hidden_size, eps=1e-12),
            nn.Linear(hidden_size, label_dim)
        )

    def forward(self, x):
        return self.net(x.mean(dim=1))


class MultiStepNavCMTPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.bert = NavPreTrainedModel(config)

        if 'mlm' in config.pretrain_tasks:
            self.mlm_head = BertOnlyMLMHead(self.config)
        if 'sap' in config.pretrain_tasks:
            self.next_action = NextActionPrediction(self.config.hidden_size, self.config.pred_head_dropout_prob)
        if 'sar' in config.pretrain_tasks:
            self.regress_action = NextActionRegression(self.config.hidden_size, self.config.pred_head_dropout_prob)
        if 'sprel' in config.pretrain_tasks:
            self.sprel_head = SpatialRelRegression(self.config.hidden_size, self.config.pred_head_dropout_prob)
        if 'mrc' in config.pretrain_tasks:
            self.image_classifier = RegionClassification(self.config.hidden_size, self.config.image_prob_size)
        if 'itm' in config.pretrain_tasks:
            self.itm_head = ItmPrediction(self.config.hidden_size)
        if 'obj' in config.pretrain_tasks:
            self.obj_head = ObjectClassification(self.config.hidden_size, self.config.obj_num_labels)
        if 'room' in config.pretrain_tasks:
            self.room_head = RoomClassification(self.config.hidden_size, self.config.room_num_labels)
        
        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        if 'mlm' in self.config.pretrain_tasks:
            self._tie_or_clone_weights(self.mlm_head.predictions.decoder,
                self.bert.embeddings.word_embeddings)

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        if task.startswith('mlm'):
            return self.forward_mlm(batch['txt_ids'], batch['txt_masks'], 
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['txt_labels'], compute_loss)
        elif task.startswith('sap'):
            return self.forward_sap(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['ob_img_fts'], batch['ob_ang_fts'], 
                                    batch['ob_nav_types'], batch['ob_masks'],
                                    batch['obj_img_fts'], batch['obj_ang_fts'], batch['obj_head_masks'],
                                    batch['ob_action_viewindex'], compute_loss)
        elif task.startswith('sar'):
            return self.forward_sar(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['ob_img_fts'], batch['ob_ang_fts'], 
                                    batch['ob_nav_types'], batch['ob_masks'],
                                    batch['obj_img_fts'], batch['obj_ang_fts'], batch['obj_head_masks'],
                                    batch['ob_action_angles'], batch['ob_progress'], compute_loss)
        elif task.startswith('sprel'):
            return self.forward_sprel(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['ob_img_fts'], batch['ob_ang_fts'], 
                                    batch['ob_nav_types'], batch['ob_masks'],
                                    batch['obj_img_fts'], batch['obj_ang_fts'], batch['obj_head_masks'],
                                    batch['sp_anchor_idxs'], batch['sp_targets'], 
                                    compute_loss)
        elif task.startswith('mrc'):
            return self.forward_mrc(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['hist_mrc_masks'], batch['hist_img_probs'], compute_loss)
        elif task.startswith('itm'):
            return self.forward_itm(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'], 4, compute_loss)
        elif task.startswith('obj'):
            return self.forward_obj(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['obj_img_fts'], batch['obj_ang_fts'], batch['obj_head_masks'],
                                    batch['obj_labels'], compute_loss)
        elif task.startswith('room'):
            return self.forward_room(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['obj_img_fts'], batch['obj_ang_fts'], batch['obj_head_masks'],
                                    batch['room_labels'], compute_loss)
        else:
            raise ValueError('invalid task')

    def forward_mlm(self, txt_ids, txt_masks, 
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks, 
                    txt_labels, compute_loss):
        txt_embeds, _, _, _ = self.bert(txt_ids, txt_masks, 
            hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
            None, None, None, None, None, None, None)

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(txt_embeds, txt_labels != -1)
        prediction_scores = self.mlm_head(masked_output)

        if compute_loss:
            mask_loss = F.cross_entropy(prediction_scores, 
                                        txt_labels[txt_labels != -1], 
                                        reduction='none')
            return mask_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        '''get only the masked region (don't compute unnecessary hiddens)'''
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def forward_sap(self, txt_ids, txt_masks, 
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks, 
                    ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks,
                    obj_img_fts, obj_ang_fts, obj_masks,
                    act_labels, compute_loss): #! Added params
        txt_embeds, hist_embeds, ob_embeds, obj_embeds = self.bert(txt_ids, txt_masks, 
            hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
            ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks, obj_img_fts, obj_ang_fts, obj_masks)
        
        # combine text and visual to predict next action
        prediction_scores = self.next_action(ob_embeds * txt_embeds[:, :1]).squeeze(-1)
        prediction_scores.masked_fill_(ob_nav_types == 0, -float('inf'))

        if compute_loss:
            act_loss = F.cross_entropy(prediction_scores, act_labels, reduction='none')
            return act_loss
        else:
            return prediction_scores

    def forward_sar(self, txt_ids, txt_masks, 
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks, 
                    ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks, 
                    obj_img_fts, obj_ang_fts, obj_masks,
                    ob_act_angles, ob_progress, compute_loss): #! Added params
        txt_embeds, hist_embeds, ob_embeds, obj_embeds = self.bert(txt_ids, txt_masks, 
            hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
            ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks, obj_img_fts, obj_ang_fts, obj_masks)

        prediction_scores = self.regress_action(txt_embeds[:, 0])   # [CLS] token

        if compute_loss:
            act_targets = torch.cat([ob_act_angles, ob_progress.unsqueeze(1)], dim=1)
            act_loss = F.mse_loss(prediction_scores, act_targets, reduction='none')
            return act_loss
        else:
            return prediction_scores

    def forward_sprel(self, txt_ids, txt_masks, 
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks, 
                    ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks,
                    obj_img_fts, obj_ang_fts, obj_masks,  #! Added params
                    sp_anchor_idxs, sp_targets, compute_loss):
        txt_embeds, hist_embeds, ob_embeds, _ = self.bert(txt_ids, txt_masks, 
            hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
            ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks, obj_img_fts, obj_ang_fts, obj_masks)

        # img_embeds: (batch, views, dim), sp_anchor_idxs: (batch)
        anchor_ob_embeds = torch.gather(ob_embeds, 1, 
            sp_anchor_idxs.unsqueeze(1).unsqueeze(2).repeat(1, 36, ob_embeds.size(-1)))
        # (batch, 1, dim)
        cat_ob_embeds = torch.cat([anchor_ob_embeds, ob_embeds[:, :-1]], -1)
        
        prediction_scores = self.sprel_head(cat_ob_embeds) # (batch, 36, 2)

        if compute_loss:
            sprel_loss = F.mse_loss(prediction_scores, sp_targets, reduction='none')
            return sprel_loss
        else:
            return prediction_scores

    def forward_mrc(self, txt_ids, txt_masks, 
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks, 
                    hist_mrc_masks, hist_img_probs, compute_loss=True):
        txt_embeds, hist_embeds, _, _ = self.bert(txt_ids, txt_masks, 
            hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
            None, None, None, None, None, None, None)

        # only compute masked regions for better efficient=cy
        hist_embeds = hist_embeds[:, 1:] # remove global embedding
        masked_output = self._compute_masked_hidden(hist_embeds, hist_mrc_masks)
        prediction_soft_labels = self.image_classifier(masked_output)

        hist_mrc_targets = self._compute_masked_hidden(hist_img_probs, hist_mrc_masks)

        if compute_loss:
            prediction_soft_labels = F.log_softmax(prediction_soft_labels, dim=-1)
            mrc_loss = F.kl_div(prediction_soft_labels, hist_mrc_targets, reduction='none').sum(dim=1)
            return mrc_loss
        else:
            return prediction_soft_labels, hist_mrc_targets

    def forward_itm(self, txt_ids, txt_masks, 
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks, 
                    num_neg_trajs, compute_loss):
        # (batch_size, 1+num_negs, dim)
        fused_embeds = self.bert.forward_itm(
            txt_ids, txt_masks, 
            hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
            num_neg_trajs=num_neg_trajs)

        prediction_scores = self.itm_head(fused_embeds).squeeze(2) # (batch, 1+num_negs, 1)
        # The first is positive
        itm_targets = torch.zeros(fused_embeds.size(0), dtype=torch.long).to(self.device)

        if compute_loss:
            sprel_loss = F.cross_entropy(prediction_scores, itm_targets, reduction='none')
            return sprel_loss
        else:
            return prediction_scores, itm_targets

    def forward_obj(self, txt_ids, txt_masks,
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks, 
                    obj_img_fts, obj_ang_fts, obj_masks,
                    obj_labels, compute_loss):
        _, _, _, obj_embeds = self.bert(txt_ids, txt_masks, 
            hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
            None, None, None, None, obj_img_fts, obj_ang_fts, obj_masks)

        # obj embeds: [batch, num_objs, dim]
        flattened_obj_embeds = obj_embeds.reshape((-1, obj_embeds.size(-1)))
        
        # obj labels: [batch, num_objs]
        flattened_obj_labels = obj_labels.reshape((-1, ))

        # prediction_scores: [batch * num_objs, label_dim]
        prediction_scores = self.obj_head(flattened_obj_embeds)

        if compute_loss:
            mask_loss = F.cross_entropy(prediction_scores, 
                                        flattened_obj_labels, 
                                        reduction='none')
            return mask_loss
        else:
            return prediction_scores

    def forward_room(self, txt_ids, txt_masks,
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks, 
                    obj_img_fts, obj_ang_fts, obj_masks,
                    room_labels, compute_loss):
        _, _, _, obj_embeds = self.bert(txt_ids, txt_masks, 
            hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
            None, None, None, None, obj_img_fts, obj_ang_fts, obj_masks)

        # obj embeds: [batch, num_objs, dim]
        # room labels: [batch, 1]
        # prediction_scores: [batch, label_dim]
        prediction_scores = self.room_head(obj_embeds)

        if compute_loss:
            mask_loss = F.cross_entropy(prediction_scores, 
                                        room_labels.squeeze(), 
                                        reduction='none')
            return mask_loss
        else:
            return prediction_scores