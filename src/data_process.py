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
    print("Configuration for {} : \n".format(TYPE))
    for key,val in cfg.items():
        print(key,val )


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


# ADDING AGENT INFORMATION TO THE LIST
def addAgentInformation(AGENTS, FRAMES, SCENES, type='train'):
    '''
#     :param AGENTS: All the agents in dataset in zarr format
#     :param FRAMES: All the frames in dataset in zarr format
#     :param SCENES: All the scenes in dataset in zarr format
#     :return: list of dictionaries (each containing all information in a frame).
#     '''

    '''

    agent_dict = { posx : x coordinate of agent , 
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

    scene_size = len(SCENES)
    test_file = open("../DATASET/prediction_test/SceneTest.txt", "w")
    if type == 'train':
        print("Generating Training data...")
    elif type == 'test':
        print("Generating Training data...")

    for scene_id in range(scene_size):
        print("Formatting Scene {}".format(scene_id))
        '''
        Total scenes in sample dataset: 100
        Total scenes in Training Dataset:99 [Scene0,1,2...98.txt]
        Total scenes in Testing Dataset:1 [Scene99.txt]
        '''

        train_file = open("../DATASET/prediction_train/Scene{}.txt".format(scene_id), "w")

        object_scene = []
        frame_start_idx, frame_end_idx = SCENES[scene_id][0]
        for frame_id in range(frame_start_idx, frame_end_idx):
            object_frame = []
            agent_start_id, agent_end_id = FRAMES[frame_id][1]
            for idx in range(agent_start_id, agent_end_id):
                agent_track_id = AGENTS[idx][4]
                label_probabilities = np.array(AGENTS[idx][5])
                object_type = np.where(label_probabilities == 1)[0]
                # print(object_type[0])
                if object_type:
                    if object_type[0] < 3:
                        continue
                    agent_dict["frame_id"] = frame_id
                    agent_dict["object_id"] = agent_track_id
                    agent_dict["object_type"] = object_type[0]
                    agent_dict["posx"], agent_dict["posy"] = AGENTS[idx][0]
                    agent_dict["length"], agent_dict["width"], agent_dict["height"] = AGENTS[idx][1]
                    agent_dict["heading"] = AGENTS[idx][2]
                    agent_dict["velx"], agent_dict["vely"] = AGENTS[idx][3]
                    # embedding = (i, agent_track_id, object_type[0], x, y, velx, vely, length, width, height, heading)

                    # Save each line in a scene.
                    if type == 'train':
                        saveData(train_file, agent_dict)
                    elif type == 'test':
                        saveData(test_file,agent_dict)

    return agent_list
'''
Function to save the generated Text files.
'''
def saveData(file, agent_dict):
    line = [str(agent_dict[key]) + " " for key in agent_dict]
    line.append("\n")
    file.writelines(line)

#%%
'''
Function to generate the Training Dataset
'''
def generateDataset(train_zarr):

    SCENES = train_zarr.scenes
    FRAMES = train_zarr.frames
    AGENTS = train_zarr.agents

    # print("SCENES:\n",SCENES.size)
    # print("FRAMES:\n",FRAMES[0])
    # print("AGENTS:\n",AGENTS.size)
    total_scenes = len(SCENES)
    trainRatio = 0.7
    testRatio = 1 - trainRatio
    trainSceneSize = int(total_scenes*trainRatio)
    testSceneSize = int(total_scenes * testRatio)

    addAgentInformation(AGENTS,FRAMES,SCENES[0:trainSceneSize],type='train')
    addAgentInformation(AGENTS, FRAMES, SCENES[testSceneSize:total_scenes], type='test')


#%%

if __name__ == '__main__':

    datasetSize = 5

    train_zarr, train_dataset, train_dataloader = loadDataset("train_data_loader")
    # val_zarr, val_dataset, val_dataloader = loadDataset("val_data_loader")

    print("TRAIN DATASET:\n", train_dataset)
    # print("VALIDATION DATASET:\n", val_dataset)

    generateDataset(train_zarr)



