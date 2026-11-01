#!/bin/bash -l

#SBATCH --job-name=cooc1x2-classifier	# Name of your job
#SBATCH --account=xxx	# Your Slurm account
#SBATCH --partition=xxx	# Run on tier3
#SBATCH --mail-user=slack:@rs9466	# Slack username to notify
#SBATCH --mail-type=ALL			# Type of slack notifications to send
#SBATCH --output=/home/all_data/run_log/%x_%j.out    # Output file
#SBATCH --error=/home/all_data/run_log/%x_%j.err # Error file
#SBATCH --time=00-0:30:00	# 30 min
#SBATCH --ntasks=1		# 1 tasks (i.e. processes)
#SBATCH --mem=16g		# 16GB RAM
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1	# 1 a100 GPU

export PYTHONNOUSERSITE=1
cd /home/all_data/cooccurrence_1x2/code
python cell_classifier.py "$@"
