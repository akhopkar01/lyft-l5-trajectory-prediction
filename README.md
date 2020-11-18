# Interaction Aware Trajectory Prediction - Lyft L5 Dataset
[![License](https://img.shields.io/badge/License-MIT%20-green.svg)](https://github.com/akhopkar01/lyft-l5-trajectory-prediction/blob/master/LICENSE)
[![WebPage](https://img.shields.io/badge/WebPage-IATP%20-blue.svg)](https://sites.google.com/view/motion-prediction-lyftl5)


This is an experimental repository to implement the paper on [GRIP++](https://arxiv.org/pdf/1907.07792.pdf) by Xin Li, Xiaowen Ying, Mooi Choo Chuah and is not used for any commercial purpose. The implementation in this repository is done for [Lyft Level 5 dataset](https://arxiv.org/pdf/2006.14480.pdf) - the largest self-driving dataset available currently.

![Implementation Workflow](https://github.com/akhopkar01/lyft-l5-trajectory-prediction/blob/master/media/Flow.png)

### Authors
Aditya Khopkar

Kartik Venkat

Kushagra Agrawal

Patan Sanaulla Khan

## Dataset
Lyft Level 5 dataset consists of more than 1000+ hours of data with more than 170000 scenes. The dataset could be found on the [Lyft L5 Website](https://self-driving.lyft.com/level5/data/). It is encouraged to download the dataset.The dataset consists of the following structure:
```
prediction-dataset/
  +- scenes/
        +- sample.zarr
        +- train.zarr
        +- train_full.zarr
  +- aerial_map/
        +- aerial_map.png
  +- semantic_map/
        +- semantic_map.pb
  +- meta.json
```
This repository uses the sample.zarr file in the pipeline.

## Dependencies
This repository needs following dependencies:
1. [L5Kit](https://github.com/lyft/l5kit)
2. [PyTorch](https://pytorch.org/get-started/locally/)

## Instructions
This repo comes along with pre-processed sample dataset in DATASET/ directory. If you choose to work on the same DATASET files, you may simply execute the following commands:
```
cd src/
python dataPostProcess.py
```
This post processes the data (Stage 2 Input Processing).
Else, you may generate your own data with processing using the bash script. In this case, you may execute the command ```sh run.sh```

In the file ```src/data_process.py``` you may want to change the line [37](https://github.com/akhopkar01/lyft-l5-trajectory-prediction/blob/0a13b004ec2f155a4c4d1ec1d8c82814f146d1cc/src/data_process.py#L24-L37) to the local location of your downloaded dataset. For example, ```/home/user/path/to/dataset/train.zarr```

You may then run the bash script to run the entire project.

If you choose to run individual commands, the following commands may be useful -
```
cd src/
python data_process.py # Stage 1 Input pre-processing
python dataPostProcess.py # Stage 2 Input processing
python main.py # Training stage
```


