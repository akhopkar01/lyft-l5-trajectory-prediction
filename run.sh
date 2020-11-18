#!/bin/sh

current_dir = $(pwd)
echo "##### STAGE 1 INPUT PRE PROCESSING #####"
cd src/
python data_process.py
echo "Stage 1- Complete"
echo "Commencing Stage 2.."
echo "##### STAGE 2 INPUT PROCESSING #####"
python dataPostProcess.py
echo "Data processing - DONE!"
python main.py