#!/bin/bash

#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu32
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user zh265998@ucf.edu

python proto.py -m llava --usegpu
