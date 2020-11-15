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
print("Scene Size : {}, Frame Size : {}, Agent size : {} ".format(len(SCENES), len(FRAMES), len(AGENTS)))
print(SCENES[0][0])
print(FRAMES[0][1])
print(AGENTS[200])
cols = ["frame_id", "object_id", "object_type", "posx", "posy", "posz", "velx", "vely", "length", "width", "height",
        "heading"]
dataset_dict = {}
dataset_list = []
for col in cols:
    dataset_dict[col] = 0

# ADDING FRAME INFORMATION TO THE LIST
def addObjectInformation():
    all_object_list = []
    # scene_size = len(SCENES)
    scene_size = 2
    for scene_id in range(scene_size):
        object_scene = []
        frame_start_idx, frame_end_idx = SCENES[scene_id][0]
        for frame_id in range(frame_start_idx, frame_end_idx):
            object_frame = []
            dataset_dict["frame_id"] = frame_id
            agent_start_id, agent_end_id = FRAMES[frame_id][1]
            for idx in range(agent_start_id, agent_end_id):
                agent_track_id = AGENTS[idx][4]
                label_probabilities = np.array(AGENTS[idx][5])
                object_type = np.where(label_probabilities == 1)[0]
                # print(object_type[0])
                if object_type:
                    if object_type[0] < 3:
                        continue
                    dataset_dict["object_id"] = agent_track_id
                    dataset_dict["object_type"] = object_type[0]
                    dataset_dict["posx"], dataset_dict["posy"] = AGENTS[idx][0]
                    dataset_dict["length"], dataset_dict["width"], dataset_dict["height"] = AGENTS[idx][1]
                    dataset_dict["heading"] = AGENTS[idx][2]
                    dataset_dict["velx"], dataset_dict["vely"] = AGENTS[idx][3]

                    object_frame.append(dataset_dict)
            object_scene.append(object_frame)
        all_object_list.append(object_scene)

    return all_object_list

# start_id = 0
# end_id = 10
#
# all_ids = []
# for frame_id in range(start_id, end_id):
#     frames = []
#     agent_start_id, agent_end_id = FRAMES[frame_id][1]
#     for agent_id in range(agent_start_id, agent_end_id):
#         agent_track_id = AGENTS[agent_id][4]
#         label_probabilities = np.array(AGENTS[agent_id][5])
#         object_type = np.where(label_probabilities == 1)[0]
#         # print(object_type[0])
#         if object_type[0] < 3:
#             continue
#         frames.append(agent_track_id)
#         # print(agent_track_id)
#     all_ids.append(frames)
#
#
# print(len(all_ids))
# print(all_ids[9])
#
# a_start_id, a_end_id = FRAMES[1][1]
# for idx in range(a_start_id, a_end_id):
#     label_probabilities = np.array(AGENTS[idx][5])
#     print(label_probabilities)

abl = addObjectInformation()
print(len(abl))
print(abl[0][0][0])



