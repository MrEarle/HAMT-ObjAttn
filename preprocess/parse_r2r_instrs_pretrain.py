from transformers import AutoTokenizer
import json
from tqdm import tqdm
import os, math, sys
import numpy as np

matterport_build_path = "/home/mrearle/datasets/Matterport3DSimulator/build"
if matterport_build_path not in sys.path:
    sys.path.append(matterport_build_path)

import MatterSim

scan_dir = '/home/mrearle/datasets/Matterport3DSimulator/data_v2/v1/scans'
connectivity_dir = '/home/mrearle/repos/VLN-HAMT/datasets/R2R/connectivity'
WIDTH = 640
HEIGHT = 480
VFOV = 60


def get_tokenizer():
    cfg_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def get_sim():
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
    sim.initialize()
    return sim

def get_first_viewindex(scan, viewpointId, heading):
    sim = get_sim()
    
    sim.newEpisode([scan], [viewpointId], [heading], [0])
    state = sim.getState()[0]

    return state.viewIndex

def get_robot_orient(scan, viewpointId, heading, elevation):
    sim = get_sim()
    
    sim.newEpisode([scan], [viewpointId], [heading], [elevation])
    state = sim.getState()[0]

    return state.heading, state.elevation


def _loc_distance(loc):
    return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)

def make_candidate(scanId, viewpointId, base_heading=0, return_robot_orient=False):
    sim = get_sim()
    adj_dict = {}
    for ix in range(36):
        if ix == 0:
            sim.newEpisode([scanId], [viewpointId], [base_heading], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        # Heading and elevation for the viewpoint center
        heading = state.heading - base_heading
        elevation = state.elevation

        # get adjacent locations
        for j, loc in enumerate(state.navigableLocations[1:]):
            # if a loc is visible from multiple view, use the closest
            # view (in angular distance) as its representation
            distance = _loc_distance(loc)

            # Heading and elevation for for the loc
            loc_heading = heading + loc.rel_heading
            loc_elevation = elevation + loc.rel_elevation
            
            if (loc.viewpointId not in adj_dict or distance < adj_dict[loc.viewpointId]['distance']):
                adj_dict[loc.viewpointId] = {
                    'heading': loc_heading,
                    'elevation': loc_elevation,
                    'scanId': scanId,
                    'viewpointId': loc.viewpointId, # Next viewpoint id
                    'pointId': ix,
                    'distance': distance,
                    'rel_heading': loc.rel_heading,
                    'rel_elevation': loc.rel_elevation,
                }

    return adj_dict

def get_path_viewindex(scan, path, initial_heading):
    path_viewindex = [get_first_viewindex(scan, path[0], initial_heading)]
    for viewpointId, nextViewpointId in zip(path[:-1], path[1:]):
        candidate = make_candidate(scan, viewpointId, 0)
        assert nextViewpointId in candidate
        path_viewindex.append(candidate[nextViewpointId]['pointId'])

    action_viewindex = path_viewindex[1:] + [-1]
    return path_viewindex, action_viewindex

def get_abs_pos_angles(path_viewindex):
    return [
        [
            (i % 12) * math.radians(30),
            ((i // 12) - 1) * math.radians(30)
        ] for i in path_viewindex
    ]

def get_rel_act_angles(scan, path, initial_heading):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
    sim.setRestrictedNavigation(False)
    sim.initialize()

    sim.newEpisode([scan], [path[0]], [initial_heading], [0])
    heading = initial_heading
    elevation = 0
    rel_act_angles = []
    for viewpointId in path[1:]:
        state = sim.getState()[0]

        min_dist = float('inf')
        closest = None
        for loc in state.navigableLocations[1:]:
            distance = _loc_distance(loc)
            if loc.viewpointId == viewpointId and distance < min_dist:
                min_dist = distance
                closest = loc
        if closest is None:
            raise Exception('No closest viewpoint found')

        rel_heading = closest.rel_heading
        rel_elevation = closest.rel_elevation
        rel_act_angles.append([rel_heading, rel_elevation])
        heading += rel_heading
        elevation += rel_elevation
        sim.newEpisode([scan], [viewpointId], [heading], [elevation])

    rel_act_angles.append([90, 90])
    
    return rel_act_angles

def parse_r2r_file(split: str, path: str, dest_path: str, tokenizer: AutoTokenizer):
    with open(path, 'r') as f:
        data = json.load(f)

    for path_dict in tqdm(data, desc=f'Parsing split {split}', unit='path'):
        instr_encs = []
        instr_ids = []
        for i, instr in enumerate(path_dict['instructions']):
            instr_enc = tokenizer.encode(instr)
            instr_encs.append(instr_enc)

            instr_ids.append(f'{path_dict["path_id"]}_{i}')

        path_viewindex, action_viewindex = get_path_viewindex(
            path_dict['scan'],
            path_dict['path'],
            path_dict['heading']
        )

        abs_pos_angles = get_abs_pos_angles(path_viewindex)
        rel_act_angles = get_rel_act_angles(path_dict['scan'], path_dict['path'], path_dict['heading'])

        path_dict['instr_encodings'] = instr_encs
        path_dict['instr_ids'] = instr_ids
        path_dict['path_viewindex'] = path_viewindex
        path_dict['action_viewindex'] = action_viewindex
        path_dict['abs_pos_angles'] = abs_pos_angles
        path_dict['rel_act_angles'] = rel_act_angles

    with open(dest_path, 'w') as f:
        for path_dict in data:
            json.dump(path_dict, f)
            f.write('\n')


base_path = '/home/mrearle/datasets/Matterport3DSimulator/semantically_richer_instructions'
dest_base_path = '/home/mrearle/repos/VLN-HAMT/datasets/R2R/annotations/pretrain'
splts = [ 'train', 'val_unseen' ]

tokenizer = get_tokenizer()
for split in splts:
    path = os.path.join(base_path, f'R2R_craft_{split}.json')
    dest_path = os.path.join(dest_base_path, f'R2R_craft_{split}.jsonl')
    parse_r2r_file(split, path, dest_path, tokenizer)
