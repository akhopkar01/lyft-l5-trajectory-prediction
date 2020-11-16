from l5kit.data import ChunkedDataset
import numpy as np
import time

# Declare global variables
zarr_dt = ChunkedDataset("/home/majoradi/Documents/l5_sample/sample.zarr")
zarr_dt.open()
AGENTS = zarr_dt.agents
FRAMES = zarr_dt.frames
SCENES = zarr_dt.scenes
datasetSize = 5
print("Scene Size : {}, Frame Size : {}, Agent size : {} ".format(len(SCENES), len(FRAMES), len(AGENTS)))
# print(SCENES[0][0])
# print(FRAMES[0][1])
print("Formatting dataset ..")
# print(AGENTS[200])
cols = ["frame_id", "object_id", "object_type", "posx", "posy", "posz", "velx", "vely", "length", "width", "height",
        "heading"]
dataset_dict = {}
dataset_list = []
for col in cols:
    dataset_dict[col] = 0

# ADDING FRAME INFORMATION TO THE LIST
def addObjectInformation():
    # all_object_list = []
    scene_size = len(SCENES)
    # scene_size = 2
    for scene_id in range(scene_size):
        print("Formatting Scene {}".format(scene_id))
        file = open("./data/Scene{}.txt".format(scene_id), "w")
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
                    dataset_dict["frame_id"] = frame_id
                    dataset_dict["object_id"] = agent_track_id
                    dataset_dict["object_type"] = object_type[0]
                    dataset_dict["posx"], dataset_dict["posy"] = AGENTS[idx][0]
                    dataset_dict["length"], dataset_dict["width"], dataset_dict["height"] = AGENTS[idx][1]
                    dataset_dict["heading"] = AGENTS[idx][2]
                    dataset_dict["velx"], dataset_dict["vely"] = AGENTS[idx][3]
                    # embedding = (i, agent_track_id, object_type[0], x, y, velx, vely, length, width, height, heading)
                    line = [str(dataset_dict[key]) + " " for key in dataset_dict]
                    line.append("\n")
                    file.writelines(line)
        #             object_frame.append(dataset_dict)
        #     object_scene.append(object_frame)
        # all_object_list.append(object_scene)

    return

xy = addObjectInformation()
# start_id = 0
# end_id = 2
# start_time = time.time()
# all_ids = []
# for frame_id in range(start_id, end_id):
#     frames = []
#     agent_start_id, agent_end_id = FRAMES[frame_id][1]
#
#     for idx in range(agent_start_id, agent_end_id):
#         agent_track_id = AGENTS[idx][4]
#         label_probabilities = np.array(AGENTS[idx][5])
#         object_type = np.where(label_probabilities == 1)[0]
#         # print(object_type[0])
#         if object_type:
#             if object_type[0] < 3:
#                 continue
#             dataset_dict["frame_id"] = frame_id
#             dataset_dict["object_id"] = agent_track_id
#             dataset_dict["object_type"] = object_type[0]
#             dataset_dict["posx"], dataset_dict["posy"] = AGENTS[idx][0]
#             dataset_dict["length"], dataset_dict["width"], dataset_dict["height"] = AGENTS[idx][1]
#             dataset_dict["heading"] = AGENTS[idx][2]
#             dataset_dict["velx"], dataset_dict["vely"] = AGENTS[idx][3]
#             print(dataset_dict)
#             frames.append(dataset_dict)
#         # print(agent_track_id)
#     all_ids.append(frames)
#
#
# print(len(frames))
# print(frames)
# print(len(all_ids))
# print(all_ids[0])
# time_taken = time.time() - start_time
# print("Time: ", time_taken)
#
# a_start_id, a_end_id = FRAMES[1][1]
# for idx in range(a_start_id, a_end_id):
#     label_probabilities = np.array(AGENTS[idx][5])
#     print(label_probabilities)

# abl = addObjectInformation()
# print(len(abl))
# print(abl[0][0][0])
# for scene_id in range(2):
#     file = open("./data/Scene{}.txt".format(scene_id), "w")
#     for i in range(3):
#         framesii = []
#         for idx in range(10):
#             agent_track_id = AGENTS[idx][4]
#             label_probabilities = np.array(AGENTS[idx][5])
#             object_type = np.where(label_probabilities == 1)[0]
#             # print(object_type[0])
#             if object_type:
#                 if object_type[0] < 3:
#                     continue
#                 # dataset_dict["frame_id"] = frame_id
#                 # dataset_dict["object_id"] = agent_track_id
#                 # dataset_dict["object_type"] = object_type[0]
#                 # dataset_dict["posx"], dataset_dict["posy"] = AGENTS[idx][0]
#                 # dataset_dict["length"], dataset_dict["width"], dataset_dict["height"] = AGENTS[idx][1]
#                 # dataset_dict["heading"] = AGENTS[idx][2]
#                 # dataset_dict["velx"], dataset_dict["vely"] = AGENTS[idx][3]
#                 # print(dataset_dict)
#                 x, y = AGENTS[idx][0]
#                 length, width, height = AGENTS[idx][1]
#                 heading = AGENTS[idx][2]
#                 velx, vely = AGENTS[idx][3]
#                 embedding = (i, agent_track_id, object_type[0], x, y, velx, vely, length, width, height, heading)
#                 line = [str(element)+" " for element in embedding]
#                 line.append("\n")
#                 # for elements in embedding:
#                 #     file.write(str(elements))
#                 file.writelines(line)
                # framesii.append(embedding)

# print(framesii)