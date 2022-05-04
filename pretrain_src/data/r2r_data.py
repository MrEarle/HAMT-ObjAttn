'''
R2R-style dataset
'''
import os
import json
from typing import List, Tuple
import jsonlines
import numpy as np
import h5py
import math

import networkx as nx

from utils.logger import LOGGER


def angle_feature(heading, elevation, angle_feat_size):
    return np.array([math.sin(heading), math.cos(heading),
                     math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
                    dtype=np.float32)

def get_point_angle_feature(angle_feat_size, baseViewId=0):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    for ix in range(36):
        if ix == 0:
            heading = 0
            elevation = math.radians(-30)
        elif ix % 12 == 0:
            heading = 0
            elevation += math.radians(30)
        else:
            heading += math.radians(30)
        feature[ix, :] = angle_feature(heading - base_heading, elevation, angle_feat_size)
    return feature

def get_all_point_angle_feature(angle_feat_size):
    return [get_point_angle_feature(angle_feat_size, baseViewId) for baseViewId in range(36)]

def get_point_rel_angles(baseViewId=0):
    rel_angles = np.zeros((36, 2), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    base_elevation = (baseViewId // 12 - 1) * math.radians(30)
    for ix in range(36):
        if ix == 0:
            heading = 0
            elevation = math.radians(-30)
        elif ix % 12 == 0:
            heading = 0
            elevation += math.radians(30)
        else:
            heading += math.radians(30)
        rel_angles[ix, 0] = heading - base_heading
        rel_angles[ix, 1] = elevation #- base_elevation
    return rel_angles

def get_all_point_rel_angles():
    return [get_point_rel_angles(baseViewId) for baseViewId in range(36)]

def load_nav_graphs(connectivity_dir):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    scans = [x.strip() for x in open(os.path.join(connectivity_dir, 'scans.txt')).readlines()]
    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G

    shortest_distances = {}
    for scan, G in graphs.items():  # compute all shortest paths
        shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
    return graphs, shortest_distances

def softmax(logits, dim=1):
    # logits: (n, d)
    tmp = np.exp(logits)
    return tmp / np.sum(tmp, axis=dim, keepdims=True)


class MultiStepNavData(object):
    def __init__(
        self, traj_files, img_ft_file, scanvp_cands_file,  connectivity_dir,
        image_prob_size=1000, image_feat_size=2048, angle_feat_size=4,
        max_txt_len=80, max_act_len=100,
        hist_enc_pano=True, val_sample_num=None,
        in_memory=False, ob_cand_pano_view=False,
        obj_feat_file=None, obj_sample_size=20,
        obj_label_file=None, room_label_file=None,
        room_annotation_file = None
    ):
        self.traj_files = traj_files
        self.img_ft_file = img_ft_file


        self.obj_feat_file = obj_feat_file and h5py.File(obj_feat_file, 'r')
        self.obj_sample_size = obj_sample_size

        if obj_feat_file:
            with open(obj_label_file) as f:
                self.obj_labels = json.load(f)
                self.obj_labels = { name: i for i, name in enumerate(self.obj_labels) }
                
            with open(room_label_file) as f:
                self.room_labels = json.load(f)
                self.room_labels = { name: i for i, name in enumerate(self.room_labels) }

            with open(room_annotation_file) as f:
                self.room_annotations = json.load(f)

        if self.obj_feat_file:
            LOGGER.info(f'Loading object features from {obj_feat_file} for {traj_files}')

        self.image_feat_size = image_feat_size
        self.image_prob_size = image_prob_size
        self.angle_feat_size = angle_feat_size
        self.max_txt_len = max_txt_len
        self.max_act_len = min(30, max_act_len) # due to memory issue
        self.hist_enc_pano = hist_enc_pano
        self.ob_cand_pano_view = ob_cand_pano_view

        self.in_memory = in_memory
        if self.in_memory:
            self._feature_store = {}

        self.scanvp_cands = json.load(open(scanvp_cands_file))

        self.graphs, self.shortest_distances = load_nav_graphs(connectivity_dir)

        self.angle_features = get_all_point_angle_feature(angle_feat_size)
        self.rel_angles = get_all_point_rel_angles()

        self.traj_data = []
        self.traj_refer, self.traj_step_refer = [], []
        n = 0
        for traj_file in self.traj_files:
            with open(traj_file, 'r') as f:
                for item in jsonlines.Reader(f):
                    self.traj_data.append(item)
                    path_len = min(len(item['path']), self.max_act_len-1)
                    for j in range(len(item['instr_encodings'])):
                        self.traj_refer.append((n, j, path_len))
                        self.traj_step_refer.extend([(n, j, t) for t in range(path_len)])
                    n += 1

        if val_sample_num:
            # cannot evaluate all the samples as it takes too much time
            # sample K data for validation
            sel_idxs = np.random.permutation(len(self.traj_refer))[:val_sample_num]
            self.traj_refer = [self.traj_refer[sidx] for sidx in sel_idxs]
            sel_idxs = np.random.permutation(len(self.traj_step_refer))[:val_sample_num]
            self.traj_step_refer = [self.traj_step_refer[sidx] for sidx in sel_idxs]

    def get_input(
        self, i_path, j_instr, t_cur, 
        return_ob=False, return_hist_img_probs=False,
        return_ob_action=False, return_ob_progress=False,
        ob_cand_pano_view=None, return_objs=False,
    ):
        traj_data = self.traj_data[i_path]
        scan = traj_data['scan']
        path = traj_data['path'][:self.max_act_len-1]
        path_viewindex = traj_data['path_viewindex']
        action_viewindex = traj_data['action_viewindex']
        abs_pos_angles = traj_data['abs_pos_angles']
        rel_act_angles = traj_data['rel_act_angles']

        instr_id = traj_data['instr_ids'][j_instr]
        instr_encoding = traj_data['instr_encodings'][j_instr][:self.max_txt_len]
        
        hist_inputs = self.get_history_feature(
            scan, path, path_viewindex, rel_act_angles, t_cur,
            return_img_probs=return_hist_img_probs
        )

        outs = {
            'instr_id': instr_id,
            'instr_encoding': instr_encoding,
            'hist_img_fts': hist_inputs[0],
            'hist_ang_fts': hist_inputs[1],
            'hist_lens': t_cur,
        }
        if self.hist_enc_pano:
            outs['hist_pano_img_fts'] = hist_inputs[2]
            outs['hist_pano_ang_fts'] = hist_inputs[3]
        if return_hist_img_probs:
            outs['hist_img_probs'] = hist_inputs[4]

        if return_ob:
            if ob_cand_pano_view is None: 
                ob_cand_pano_view = self.ob_cand_pano_view
            if ob_cand_pano_view:
                ob_img_feats, ob_ang_feats, ob_nav_types, gt_label, gt_angle = \
                    self.get_ob_cand_pano_view(scan, path, path_viewindex, action_viewindex, rel_act_angles, t_cur)
            else:
                ob_img_feats, ob_ang_feats, ob_nav_types, gt_label, gt_angle = \
                    self.get_ob_pano_view(scan, path, path_viewindex, action_viewindex, rel_act_angles, t_cur)

            outs.update({
                'ob_img_fts': ob_img_feats,
                'ob_ang_fts': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
            })
            if return_ob_action:
                outs['ob_action_viewindex'] = gt_label
                outs['ob_action_angles'] = gt_angle
            if return_ob_progress:
                outs['ob_progress'] = self.get_progress(scan, path[0], path[t_cur], path[-1] if 'guide_path' not in traj_data else traj_data['guide_path'][-1])
        
        if return_objs:
            obj_img_fts, obj_ang_fts, names, obj_masks = self.get_obj_feats(scan, path[t_cur], path_viewindex[t_cur])
            outs['obj_img_fts'] = obj_img_fts
            outs['obj_ang_fts'] = obj_ang_fts
            outs['obj_head_masks'] = obj_masks
            outs['obj_labels'] = names

            outs['room_labels'] = self.get_viewpoint_room(scan, path[t_cur])

        #! Aca salen los datos de un paso de la trajecotria (creo) en preentrenamiento
        return outs

    def get_obj_feats(self, scan: str, viewpoint: str, viewindex: int):
        feats, orients, names, mask = self.get_obj_feature(scan, viewpoint, self.obj_sample_size)


        if len(orients) > 0:
            base_heading = (viewindex % 12) * math.radians(30)
            orients[:, 0] -= base_heading
        
        heading = orients[:, 0].squeeze()
        elevation = orients[:, 1].squeeze()
        angle_feats = np.array([
                np.sin(heading), np.cos(heading),
                np.sin(elevation), np.cos(elevation)
                ] * (self.angle_feat_size // 4),
            dtype=np.float32)
        angle_feats = angle_feats.transpose((1, 0))
        
        return feats, angle_feats, names, mask

    def get_ob_pano_view(self, scan, path, path_viewindex, action_viewindex, rel_act_angles, t_cur):
        ob_img_feats = self.get_image_feature(scan, path[t_cur], pad_stop_token=True)[:, :self.image_feat_size]
        ob_ang_feats = self.get_angle_feature(path_viewindex[t_cur], pad_stop_token=True)
        ob_nav_types = np.zeros((ob_img_feats.shape[0], ), dtype=np.int64)
        ob_nav_types[-1] = 2   # 2 for [STOP]
        ob_nav_cands = self.scanvp_cands['%s_%s'%(scan, path[t_cur])]
        ob_nav_viewindexes = np.array([v[0] for v in ob_nav_cands.values()])
        ob_nav_types[ob_nav_viewindexes] = 1

        if action_viewindex[t_cur] != -1:
            gt_label = action_viewindex[t_cur]
            gt_angle = rel_act_angles[t_cur]
        else: # stop
            gt_label = ob_img_feats.shape[0] - 1
            gt_angle = np.zeros((2, ), dtype=np.float32)

        return ob_img_feats, ob_ang_feats, ob_nav_types, gt_label, gt_angle

    def get_ob_cand_pano_view(self, scan, path, path_viewindex, action_viewindex, rel_act_angles, t_cur):
        # 36 pano views
        ob_img_feats = self.get_image_feature(scan, path[t_cur], pad_stop_token=False)[:, :self.image_feat_size]
        ob_ang_feats = self.get_angle_feature(path_viewindex[t_cur], pad_stop_token=False)

        # cand views
        ob_nav_cands = self.scanvp_cands['%s_%s'%(scan, path[t_cur])]
        cand_img_feats, cand_ang_feats = [], []
        non_cand_viewidxs = np.ones((36, ), dtype=np.bool)
        gt_label = None
        for k, v in ob_nav_cands.items():
            if t_cur < len(path) - 1 and k == path[t_cur + 1]: # non-stop
                gt_label = len(cand_img_feats)
            non_cand_viewidxs[v[0]] = False
            cand_img_feats.append(ob_img_feats[v[0]])
            tmp_angle = self.rel_angles[path_viewindex[t_cur]][v[0]]
            cand_ang_feats.append(
                angle_feature(tmp_angle[0] + v[2], tmp_angle[1] + v[3], self.angle_feat_size)
            )
        cand_img_feats = np.stack(cand_img_feats, 0)
        cand_ang_feats = np.stack(cand_ang_feats, 0)
            
        # non cand views
        non_cand_img_feats = ob_img_feats[non_cand_viewidxs]
        non_cand_ang_feats = ob_ang_feats[non_cand_viewidxs]

        # combine
        ob_nav_types = np.array(
            [1] * len(cand_img_feats) + [2] + [0] * len(non_cand_img_feats)
        )
        ob_img_feats = np.concatenate([cand_img_feats, np.zeros((1, self.image_feat_size), dtype=np.float32), non_cand_img_feats], 0)
        ob_ang_feats = np.concatenate([cand_ang_feats, np.zeros((1, self.angle_feat_size), dtype=np.float32), non_cand_ang_feats], 0)

        if gt_label is None:    # stop action
            gt_label = len(cand_img_feats)
            gt_angle = np.zeros((2, ), dtype=np.float32)
        else:
            gt_angle = rel_act_angles[t_cur]
                
        return ob_img_feats, ob_ang_feats, ob_nav_types, gt_label, gt_angle


    def get_history_feature(
        self, scan, path, path_viewindex, rel_act_angles, t_cur, 
        return_img_probs=False
    ):
        # get history features before the step t_cur
        image_feats, angle_feats, image_probs = [], [], []
        pano_image_feats, pano_angle_feats = [], []

        for t in range(0, t_cur):
            vp = path[t]
            viewidx = path_viewindex[t]
            heading, elevation = rel_act_angles[t]

            if t == len(path) - 1: # STOP Action
                angle_feats.append(np.zeros((self.angle_feat_size, ), dtype=np.float32))
            else:
                angle_feats.append(angle_feature(heading, elevation, self.angle_feat_size))

            vp_fts = self.get_image_feature(scan, vp, pad_stop_token=False)

            image_feats.append(vp_fts[viewidx, :self.image_feat_size])

            if self.hist_enc_pano:
                pano_image_feats.append(vp_fts[:, :self.image_feat_size])
                pano_angle_feats.append(self.angle_features[viewidx])

            if return_img_probs:
                image_probs.append(vp_fts[viewidx, self.image_feat_size:])

        if t_cur > 0:
            image_feats = np.stack(image_feats, 0)
            angle_feats = np.stack(angle_feats)

            if self.hist_enc_pano:
                pano_image_feats = np.stack(pano_image_feats, 0)
                pano_angle_feats = np.stack(pano_angle_feats, 0)

            if return_img_probs:
                image_probs = np.stack(image_probs, 0)
                image_probs = softmax(image_probs)
        else:
            image_feats = np.zeros((0, self.image_feat_size), dtype=np.float32)
            angle_feats = np.zeros((0, self.angle_feat_size), dtype=np.float32)
            if self.hist_enc_pano:
                pano_image_feats = np.zeros((0, 36, self.image_feat_size), dtype=np.float32)
                pano_angle_feats = np.zeros((0, 36, self.angle_feat_size), dtype=np.float32)
            image_probs = np.zeros((0, self.image_prob_size), dtype=np.float32)
        
        if return_img_probs:
            return image_feats, angle_feats, pano_image_feats, pano_angle_feats, image_probs
        
        return image_feats, angle_feats, pano_image_feats, pano_angle_feats

    def get_obj_feature(self, scan, viewpoint, sample_size) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        roi = np.zeros((sample_size, self.image_feat_size), dtype=np.float32)
        orients = np.zeros((sample_size, 2), dtype=np.float32)
        names = np.zeros((sample_size, ), dtype=np.int64)
        mask = np.zeros((sample_size, ), dtype=np.bool)

        ds_path = f'{scan}/{viewpoint}'
        if ds_path in self.obj_feat_file:
            # If viewpoint isnt in the file, there are zero objects in it
            obj_store = self.obj_feat_file[ds_path]
            num_objs = len(obj_store['names'])

            real_sample_size = min(sample_size, num_objs)
            sample_indices: np.ndarray = np.random.choice(
                np.arange(num_objs, dtype=np.int32),
                real_sample_size,
                replace=False,
            )
            sample_indices.sort()
            
            roi[:real_sample_size] = obj_store["features"][sample_indices]
            orients[:real_sample_size] = obj_store["orients"][sample_indices]
            mask[:real_sample_size] = np.ones((real_sample_size, ), dtype=np.bool)

            str_names = obj_store["names"][sample_indices]
            int_names = [self.obj_labels[name.decode('utf-8')] for name in str_names]
            names[:real_sample_size] = int_names

        return roi, orients, names, mask

    def get_viewpoint_room(self, scan, viewpoint):
        room = self.room_annotations[scan][viewpoint]
        room_idx = self.room_labels[room]

        return np.array([room_idx], dtype=np.int64)

    def get_image_feature(self, scan, viewpoint, pad_stop_token=False):
        key = '%s_%s' % (scan, viewpoint)
        if self.in_memory and key in self._feature_store:
            fts = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                fts = f[key][...].astype(np.float32)
            if self.in_memory:
                self._feature_store[key] = fts

        if pad_stop_token:
            fts = np.vstack([fts, np.zeros((1, fts.shape[-1]), dtype=fts.dtype)])
        return fts

    def get_angle_feature(self, viewindex, pad_stop_token=False):
        fts = self.angle_features[viewindex]
        if pad_stop_token:
            fts = np.vstack([fts, np.zeros((1, fts.shape[-1]), dtype=fts.dtype)])
        return fts

    def get_progress(self, scan, start_vp, cur_vp, end_vp):
        if cur_vp == end_vp:
            return 1
        elif start_vp == cur_vp:
            return 0
        else:
            total_dist = self.shortest_distances[scan][start_vp][end_vp]
            remained_dist = self.shortest_distances[scan][cur_vp][end_vp]
            return 1 - remained_dist / max(total_dist, 0.1)

