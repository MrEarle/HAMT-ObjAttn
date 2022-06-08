#!/usr/bin/env python3

''' Script to precompute image features using a Pytorch ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''

import argparse
import json
import math
import os
from collections import defaultdict
import sys

import h5py
import MatterSim
import numpy as np
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm
from torchvision import transforms

resnet_path = '/home/mrearle/repos/VLN-HAMT/preprocess'
sys.path.append(resnet_path)
from resnet import get_model_pooled

FEATURE_SIZE = 2048
ORIENT_SIZE = 2
NAME_SIZE = 16
BATCH_SIZE = 32

WIDTH=640
HEIGHT=480

global update_pbars

def count_scan_objs(ds: h5py.Dataset):
    objs = [len(ds[vp]['names']) for vp in ds.keys()]
    return sum(objs)

def get_progress_bar_updater(h5file):
    pbars = { scan: tqdm(
        desc=f'Scan {scan}',
        total=count_scan_objs(h5file[scan]),
        unit='viewpoints'
    ) for scan in h5file.keys() }
    update_by = { scan: 0 for scan in pbars }
    iteration = 0
    def update(scan):
        update_by[scan] += 1

        if iteration % 100 == 0:
            for scan, pbar in pbars.items():
                if update_by[scan] != 0:
                    pbar.update(update_by[scan])
                    update_by[scan] = 0
    return update

preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def transform_img(im):
    """Prep opencv 3 channel image for the network"""
    im_orig = im.astype(np.float32, copy=True)
    blob = np.zeros((1, im.shape[0], im.shape[1], 3), dtype=np.float32)
    blob[0, :, :, :] = im_orig
    blob = blob.transpose((0, 3, 1, 2)) / 255.0
    blob = torch.from_numpy(blob)
    blob = preprocess(blob)
    return blob

def build_feature_extractor():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model_pooled().to(device)
    model.eval()

    return model, transform_img, device

def build_simulator(connectivity_dir, scan_dir, vfov):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(vfov)
    sim.initialize()
    return sim

def visualize_img(scan, viewpoint, heading, elevation, vfov, connectivity_dir, scan_dir):
    HEIGHT = 224
    WIDTH = 224

    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(vfov)
    sim.initialize()
    sim.newEpisode([scan], [viewpoint], [heading], [elevation])
    state = sim.getState()
    im = state[0].rgb
    im = np.array(im, copy=True)
    return im

def build_ds(file: h5py.File, vfov_dict, args):
    print('Preparing dataset...', flush=True)
    scans = defaultdict(lambda *_: defaultdict(lambda *_: 0))

    print('Computing size of dataset...', flush=True)
    for objs in tqdm(vfov_dict.values(), unit='vfovs'):
        for obj in objs:
            scans[obj['scan']][obj['viewpoint']] += 1

    print('Creating dataset...', flush=True)
    feat_size = FEATURE_SIZE
    if args.out_image_logits:
        feat_size += LOGIT_SIZE

    for scan, vps in tqdm(scans.items(), unit='scans'):
        for vp, num in vps.items():
            file.create_dataset(f'{scan}/{vp}/features', (num, feat_size), dtype=np.float32, data=np.zeros((num, feat_size)))
            file.create_dataset(f'{scan}/{vp}/orients', (num, ORIENT_SIZE), dtype=np.float32)
            file.create_dataset(f'{scan}/{vp}/names', (num,), dtype=f'|S{NAME_SIZE}')

            file[f'{scan}/{vp}'].attrs['num_data'] = 0

def add_feature_to_ds(file: h5py.File, scan, view, feature, orient, name):
    """Add feature to dataset
    
    Arguments:
        file {h5py.File} -- h5py file
        scan {str} -- scan id
        view {str} -- viewpoint id
        features {np.ndarray} -- features. Shape: (FEATURE_SIZE,)
        orients {np.ndarray} -- orientation. Shape: (ORIENT_SIZE,)
    """
    global update_pbars
    pth = f'{scan}/{view}'
    f_path = f'{pth}/features'
    o_path = f'{pth}/orients'
    n_path = f'{pth}/names'

    idx = file[pth].attrs['num_data']

    file[f_path][idx] = feature
    file[o_path][idx] = orient
    file[n_path][idx] = name

    file[pth].attrs['num_data'] += 1

    update_pbars(scan)

def add_to_ds(h5file, viewpoints, fts, orients, names, args):
    feats = fts

    for i in range(len(viewpoints)):
        add_feature_to_ds(h5file, *viewpoints[i], feats[i], orients[i], names[i])

def process_features(vfov, model, img_transforms, device, h5file, obj_dict_list, args):
    images = []
    viewpoints = []
    orients = []
    names = []
    for obj_dict in obj_dict_list:
        # Gather up to BATCHSIZE imgs from the simulator
        scan_id = obj_dict['scan']
        viewpoint_id = obj_dict['viewpoint']

        # Get objects from simulator
        # sim.newEpisode([scan_id], [viewpoint_id], [obj_dict['heading']], [obj_dict['elevation']])
        # state = sim.getState()[0]
        # image = np.array(state.rgb, copy=True) # in BGR channel
        image = visualize_img(scan_id, viewpoint_id, obj_dict['heading'], obj_dict['elevation'], vfov, args.connectivity_dir, args.scan_dir)


        images.append(image)
        viewpoints.append((scan_id, viewpoint_id))
        orients.append([obj_dict['heading'], obj_dict['elevation']])
        names.append(obj_dict['category'])

        if len(images) == BATCH_SIZE:
            tensor_images = torch.concat([img_transforms(x).to(device) for x in images], 0)
            b_fts = model(tensor_images)
            fts = b_fts.data.cpu().numpy()

            add_to_ds(h5file, viewpoints, fts, np.array(orients), np.array(names, dtype=f'|S{NAME_SIZE}'), args)

            images = []
            viewpoints = []
            orients = []
            names = []

    
    if len(images) > 0:
        tensor_images = torch.concat([img_transforms(x).to(device) for x in images], 0)
        b_fts = model(tensor_images)
        fts = b_fts.data.cpu().numpy()

        add_to_ds(h5file, viewpoints, fts, np.array(orients), np.array(names, dtype=f'|S{NAME_SIZE}'), args)


def build_feature_file(args):
    global update_pbars
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    print('loading object metadata...', flush=True)
    with open(args.object_metadata_file, 'r') as f:
        object_dict = json.load(f)

    # Set up the simulator

    # Set up PyTorch CNN model
    print('loading model...', flush=True)
    torch.set_grad_enabled(False)
    model, img_transforms, device = build_feature_extractor()

    with h5py.File(args.output_file, 'w') as outf:
        build_ds(outf, object_dict, args)
        update_pbars = get_progress_bar_updater(outf)
        print('building dataset...', flush=True)
        for vfov, obj_dict_list in object_dict.items():
            # sim = build_simulator(args.connectivity_dir, args.scan_dir, float(vfov))
            process_features(float(vfov), model, img_transforms, device, outf, obj_dict_list, args)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='vit_base_patch16_224')
    parser.add_argument('--checkpoint_file', default=None)
    parser.add_argument('--connectivity_dir', default='../connectivity')
    parser.add_argument('--scan_dir', default='../data/v1/scans')
    parser.add_argument('--out_image_logits', action='store_true', default=False)
    parser.add_argument('--output_file')
    parser.add_argument('--object_metadata_file')
    args = parser.parse_args()

    build_feature_file(args)


