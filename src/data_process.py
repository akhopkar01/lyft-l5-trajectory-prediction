#%%
from torch.utils.data import DataLoader
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset


from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.data.labels import PERCEPTION_LABELS

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
def addAgentInformation(AGENTS):
    '''
    :param AGENTS: All the agents in dataset in zarr format
    :return: list of dictionaries (each containing all information in a frame).
    '''

    '''
    
    dataset_dict = { posx : x coordinate of agent , 
                     posy : y coordinate of agent ,
                     length : length of agent  ,
                     width : width of agent , 
                     height : height of agent ,
                     heading : heading of agent , 
                     velx : x-axis velocity of agent , 
                     vely : y-axis velocity of agent ,
                     object_id : object id of agent to keep track of agent, 
                     object_type : type of object (car,bus,bicycle,... etc) }
                     
    dataset_list = [dataset_dict, dataset_dict, ... ] // len: datasetSize
    
    '''
    agent_dict = {}
    agent_list = []
    for idx in range(datasetSize):
        agent_dict["posx"], agent_dict["posy"] = AGENTS[idx][0]
        agent_dict["length"], agent_dict["width"], agent_dict["height"] = AGENTS[idx][1]
        agent_dict["heading"] = AGENTS[idx][2]
        agent_dict["velx"], agent_dict["vely"] = AGENTS[idx][3]
        agent_dict["object_id"] = AGENTS[idx][4]
        label_probabilities = np.array(AGENTS[idx][5])
        obj_type = np.where(label_probabilities == 1)[0][0]
        agent_dict["object_type"] = PERCEPTION_LABELS[obj_type]
        agent_list.append(agent_dict)

    return agent_list

# ADDING FRAME INFORMATION TO THE LIST
def addFrameInformation(AGENTS, FRAMES, SCENES):
    '''
    :param FRAMES: All the frames in dataset in zarr format
    :return: list of dictionaries (each containing all information in a frame).
    '''

    '''

    dataset_dict = { posx : x coordinate of agent , 
                     posy : y coordinate of agent ,
                     length : length of agent  ,
                     width : width of agent , 
                     height : height of agent ,
                     heading : heading of agent , 
                     velx : x-axis velocity of agent , 
                     vely : y-axis velocity of agent ,
                     object_id : object id of agent to keep track of agent, 
                     object_type : type of object (car,bus,bicycle,... etc) }

    dataset_list = [dataset_dict, dataset_dict, ... ] // len: datasetSize

    '''
    agent_dict = {}
    cols = ["frame_id", "object_id", "object_type", "posx", "posy", "posz", "velx", "vely", "length", "width", "height",
            "heading"]
    for col in cols:
        agent_dict[col] = 0
    agent_list = []
    for idx in range(datasetSize):
        agent_dict["posx"], agent_dict["posy"] = AGENTS[idx][0]
        agent_dict["length"], agent_dict["width"], agent_dict["height"] = AGENTS[idx][1]
        agent_dict["heading"] = AGENTS[idx][2]
        agent_dict["velx"], agent_dict["vely"] = AGENTS[idx][3]
        agent_dict["object_id"] = AGENTS[idx][4]
        label_probabilities = np.array(AGENTS[idx][5])
        obj_type = np.where(label_probabilities == 1)[0][0]
        agent_dict["object_type"] = PERCEPTION_LABELS[obj_type]
        agent_list.append(agent_dict)

    return agent_list

#%%
'''
Function to generate the Training Dataset
'''
def generateTrainDataset(train_zarr):

    SCENES = train_zarr.scenes
    FRAMES = train_zarr.frames
    AGENTS = train_zarr.agents

    # print("SCENES:\n",SCENES.size)
    print("FRAMES:\n",FRAMES[0])
    print("AGENTS:\n",AGENTS.size)

    train_dataset_list = addAgentInformation(AGENTS)

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

    val_dataset_list = addAgentInformation(AGENTS)

#%%



#%%

if __name__ == '__main__':

    datasetSize = 5
    cols = ["frame_id", "object_id", "object_type", "posx", "posy", "posz", "velx", "vely", "length", "width", "height",
            "heading"]

    # for col in cols:
    #     dataset_dict[col] = 0

    train_zarr, train_dataset, train_dataloader = loadDataset("train_data_loader")
    print("TRAIN DATASET:\n", train_dataset)

    val_zarr, val_dataset, val_dataloader = loadDataset("val_data_loader")

    generateTrainDataset(train_zarr)


