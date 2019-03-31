#!/bin/bash
work_path=$(dirname $(readlink -f $0))
cd ${work_path}/Forward_Warp/cuda/
conda activate pytorch
python setup.py install | grep "error"
cd ../../
python setup.py install
