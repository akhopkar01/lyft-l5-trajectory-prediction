#%%
from torch.utils.data import DataLoader
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset


from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data


import numpy as np
import os

#%%
def loadDataset(TYPE):
    # Load config file
    cfg = load_config_data("../cfg/agent_motion_config.yaml")
    print(cfg)

    '''
    set env variable for data   (PATH to the main DATASET folder)
    ../lyft-motion-prediction-autonomous-vehicles/
        +-arial_map
        --scenes
            --sample.zarr
            --test.zarr
            --train.zarr
            --validate.zarr
        +-semantic_map
        --meta.json
    '''

    os.environ["L5KIT_DATA_FOLDER"] = "/media/kartik/62A60BCB35EBD08A/lyft-motion-prediction-autonomous-vehicles"
    dm = LocalDataManager(None)

    '''
    Load the dataset for training and validation.
        For training:
            TYPE: "train_data_loader" 
        For validation:
            TYPE: "val_data_loader"
    '''

    dt_cfg = cfg[TYPE]
    rasterizer = build_rasterizer(cfg, dm)
    dt_zarr = ChunkedDataset(dm.require(dt_cfg["key"])).open()
    dt_dataset = AgentDataset(cfg, dt_zarr, rasterizer)
    dt_dataloader = DataLoader(dt_dataset, shuffle=dt_cfg["shuffle"],
                                            batch_size=dt_cfg["batch_size"],
                                            num_workers=dt_cfg["num_workers"])

    return dt_zarr, dt_dataset, dt_dataloader



#%%

# ADDING AGENT INFORMATION
def addAgentInformation(AGENTS, FRAMES, SCENES):
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

#%%
'''
Function to generate the Training Dataset
'''
def generateTrainDataset(train_zarr):

    SCENES = train_zarr.scenes
    FRAMES = train_zarr.frames
    AGENTS = train_zarr.agents

    # print("SCENES:\n",SCENES[0])
    # print("FRAMES:\n",FRAMES[0])
    print("AGENTS:\n",AGENTS[0])

    train_dataset_list = addAgentInformation(AGENTS, FRAMES, SCENES)

    print("Train dataset list:\n", train_dataset_list)


'''
Function to generate the Validation Dataset
'''
def generateValDataset(val_zarr):

    AGENTS = val_zarr.agents
    FRAMES = val_zarr.frames
    SCENES = val_zarr.scenes

    # print("SCENES:\n", SCENES[0])
    # print("FRAMES:\n", FRAMES[0])
    print("AGENTS:\n", AGENTS[0])

    val_dataset_list = addAgentInformation(AGENTS, FRAMES, SCENES)

#%%



#%%

if __name__ == '__main__':

    datasetSize = 5
    cols = ["frame_id", "object_id", "object_type", "posx", "posy", "posz", "velx", "vely", "length", "width", "height",
            "heading"]
    dataset_dict = {}
    dataset_list = []
    for col in cols:
        dataset_dict[col] = 0

    train_zarr, train_dataset, train_dataloader = loadDataset("train_data_loader")

    val_zarr, val_dataset, val_dataloader = loadDataset("val_data_loader")

    generateTrainDataset(train_zarr)


