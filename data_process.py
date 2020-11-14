from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset
from l5kit.dataset import AgentDataset
import numpy as np

# Declare global variables
zarr_dt = ChunkedDataset("/home/majoradi/Documents/l5_sample/sample.zarr")
zarr_dt.open()
AGENTS = zarr_dt.agents
FRAMES = zarr_dt.frames
SCENES = zarr_dt.scenes
datasetSize = 5
cols = ["frame_id", "object_id", "object_type", "posx", "posy", "posz", "velx", "vely", "length", "width", "height",
        "heading"]
dataset_dict = {}
dataset_list = []
for col in cols:
    dataset_dict[col] = 0


# ADDING AGENT INFORMATION
def addAgentInformation():
    for idx in range(datasetSize):
        dataset_dict["posx"], dataset_dict["posy"] = AGENTS[idx][0]
        dataset_dict["length"], dataset_dict["width"], dataset_dict["height"] = AGENTS[idx][1]
        dataset_dict["heading"] = AGENTS[idx][2]
        dataset_dict["velx"], dataset_dict["vely"] = AGENTS[idx][3]
        dataset_dict["object_id"] = AGENTS[idx][4]  # Not sure
        label_probabilities = np.array(AGENTS[idx][5])
        obj_type = np.where(label_probabilities == 1)[0] + 1  # Not sure
        dataset_dict["object_type"] = obj_type[0]
        dataset_list.append(dataset_dict)
    print(dataset_dict)
    return dataset_list

# ADDING FRAME INFORMATION TO THE LIST
def addFrameInformation():
    return dataset_list
