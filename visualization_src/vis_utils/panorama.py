from collections import defaultdict
from typing import Any, List, Tuple

from .vis_utls import get_neighbour_viewpoints_coords, orient_to_coord
from .vis_utls import get_panorama
import matplotlib.pyplot as plt
import numpy as np

BBOX_STYLE = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.1)

def plot_view(
    scan: str,
    viewpoint: str,
    agent_heading: float,
    reachable_viewpoints: List[Tuple[str, Tuple[float, float]]],
    objects: List[Tuple[str, Tuple[float, float]]],
    instruction: str,
):
    panorama = get_panorama(scan, viewpoint, agent_heading)
    img_height, img_width = panorama.shape[:2]

    fig = plt.figure(figsize=(18, 10))
    view_ax = fig.add_axes((0.1, 0.1, 0.9, 0.9))

    # Setup panorama image
    view_ax.imshow(panorama)
    view_ax.set_xticks(np.linspace(0, img_width - 1, 5), [-180, -90, 0, 90, 180])
    view_ax.set_xlabel(f"relative heading from the agent")
    view_ax.set_yticks(np.linspace(0, img_height - 1, 5), [-180, -90, 0, 90, 180])

    # Show objects
    obj_idxs = defaultdict(lambda: 0)
    for name, [heading, elevation] in objects:
        x, y = orient_to_coord(heading, elevation, 0, 0, img_height, img_width)

        obj_idx = obj_idxs[name]
        label = f"{name} {obj_idx}"
        obj_idxs[name] += 1

        view_ax.annotate(label, (x + 15, y + 15), bbox=BBOX_STYLE, color="black")
        view_ax.plot(x, y, marker="v", linewidth=3)

    reachable_viewpoints = get_neighbour_viewpoints_coords(scan, viewpoint)
    for name, (heading, elevation) in reachable_viewpoints.items():
        x, y = orient_to_coord(heading, elevation, agent_heading, 0, img_height, img_width)
        label = name[:6]
        print(label, heading, elevation, x, y)

        view_ax.annotate(label, (x, y), bbox=BBOX_STYLE, color="black")
        view_ax.plot(x, y, marker="o", linewidth=1, markersize=10)


    # Add text
    text = f"Instruction:\n{insert_newlines(instruction)}"
    fig.text(0.1, 0.1, text, fontsize=16, wrap=True, verticalalignment="top")

    plt.show()


def insert_newlines(text, every=100):
    words = text.split(" ")
    lines = []
    to_insert = []
    curr_len = 0
    for w in words:
        to_insert.append(w)
        curr_len += len(w)
        if curr_len > every:
            lines.append(" ".join(to_insert))
            to_insert = []
            curr_len = 0

    if to_insert:
        lines.append(" ".join(to_insert))
    return "\n".join(lines)