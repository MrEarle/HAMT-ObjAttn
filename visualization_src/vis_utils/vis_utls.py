IALAB_MEMBER = True
IALAB_USER = "mrearle"


import sys
from typing import Tuple
import json


if IALAB_MEMBER:
    matterport_build_path = f"/home/{IALAB_USER}/datasets/Matterport3DSimulator/build"
    metadata_script_path = f"/home/{IALAB_USER}/repos/360-visualization/metadata_parser"
else:
    matterport_build_path = f"/Matterport3DSimulator/build"  # Path to simulator
    metadata_script_path = f"/360-visualization/metadata_parser"  # Path to metadata parser of this repository


if matterport_build_path not in sys.path:
    sys.path.append(matterport_build_path)

if metadata_script_path not in sys.path:
    sys.path.append(metadata_script_path)


import sys
import MatterSim
import numpy as np
import networkx as nx

scan_dir = '/home/mrearle/datasets/Matterport3DSimulator/data_v2/v1/scans'
connectivity_dir = '/home/mrearle/repos/VLN-HAMT/datasets/R2R/connectivity'

def visualize_panorama_img(scan, viewpoint, heading, elevation):
    WIDTH = 80
    HEIGHT = 480
    pano_img = np.zeros((HEIGHT, WIDTH * 36, 3), np.uint8)
    VFOV = np.radians(55)
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(VFOV)
    sim.initialize()
    for n_angle, angle in enumerate(range(-175, 180, 10)):
        sim.newEpisode([scan], [viewpoint], [heading + np.radians(angle)], [elevation])
        state = sim.getState()
        im = state[0].rgb
        im = np.array(im)
        pano_img[:, WIDTH * n_angle : WIDTH * (n_angle + 1), :] = im[..., ::-1]
    return pano_img


def get_panorama(scan, viewpoint, viewpoint_heading):
    # Get panorama image
    images = []
    for viewpoint_elevation in (np.pi / 2 * x for x in range(-1, 2)):
        im = visualize_panorama_img(scan, viewpoint, viewpoint_heading, viewpoint_elevation)
        images.append(im)

    return np.concatenate(images[::-1])

def orient_to_coord(
    heading: float,
    elevation: float,
    agent_heading: float,
    agent_elevation: float,
    img_height: int,
    img_width: int
    ) -> Tuple[int, int]:
    heading -= agent_heading
    elevation -= agent_elevation

    while heading > np.pi:
        heading -= 2 * np.pi
    while heading < -np.pi:
        heading += 2 * np.pi

    while elevation > np.pi:
        heading -= 2 * np.pi
    while elevation < -np.pi:
        elevation += 2 * np.pi

    
    first_coord = (heading / (2 * np.pi) + 0.5) * img_width  # img.shape[1]
    if first_coord < 0:
        first_coord += img_width
    second_coord = (0.5 - elevation / (np.pi / 1.1)) * img_height  # img.shape[0]
    return int(first_coord), int(second_coord)

def load_nav_graph(graph_path):
    with open(graph_path) as f:
        G = nx.Graph()
        positions = {}
        heights = {}
        data = json.load(f)
        for i, item in enumerate(data):
            if item["included"]:
                for j, conn in enumerate(item["unobstructed"]):
                    if conn and data[j]["included"]:
                        positions[item["image_id"]] = np.array([item["pose"][3], item["pose"][7], item["pose"][11]])
                        heights[item["image_id"]] = float(item["height"])
                        assert data[j]["unobstructed"][i], "Graph should be undirected"
                        G.add_edge(item["image_id"], data[j]["image_id"])
        nx.set_node_attributes(G, values=positions, name="position")
        nx.set_node_attributes(G, values=heights, name="height")
    return G

def get_neighbour_viewpoints_coords(scan, viewpointId):
    # /home/mrearle/repos/VLN-HAMT/datasets/R2R/connectivity/XcA2TqTSSAj_connectivity.json
    graph_path = f"{connectivity_dir}/{scan}_connectivity.json"
    graph: nx.Graph = load_nav_graph(graph_path)

    curr_node = graph.nodes[viewpointId] 
    # TODO: Obtener coord de curr node. Usar esto para arreglar posicionamiento de viewpoints.

    viewpoints = {}
    for reachable in graph.neighbors(viewpointId):
        node = graph.nodes[reachable]
        x = node["position"][0] - curr_node["position"][0]
        y = node["position"][1] - curr_node["position"][1]
        z = node["position"][2] - curr_node["position"][2]
        dist = np.sqrt(x ** 2 + y ** 2)

        heading = (np.pi / 2) - np.arctan2(y, x)
        heading -= (2 * np.pi) * np.floor((heading + np.pi) / (2 * np.pi))
        elevation = np.arctan2(z - node["height"], dist)
        viewpoints[reachable] = (heading, elevation)

    return viewpoints