#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80"
#SBATCH --time=1:00:00
#SBATCH --account=wjacobs
#SBATCH --mail-type=fail
#SBATCH --mail-user=by7175@princeton.edu

i=$1
python generate_fe_landscapes.py $i
