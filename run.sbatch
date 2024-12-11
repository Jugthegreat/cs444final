#!/bin/bash

### Do not change the following 8 lines.
#SBATCH --time=12:00:00                         # Job run time (hh:mm:ss)
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=8                     # Number of tasks (cores/ppn) per node
#SBATCH --mem=16g                               # Amount of CPU memory
#SBATCH --job-name=srgan_training               # Name of batch job
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=cs444-ic                      # Replace with your account name if needed
#SBATCH --gres=gpu:A100                         # Request specific GPU type (adjust as needed)
### End Do not change block

#SBATCH --mail-user=NETID@illinois.edu          # Replace with your email
#SBATCH --mail-type=END                         # Email notifications when job ends

#SBATCH --output=srgan_training.out.%j          # Output file name
#SBATCH --error=srgan_training.err.%j           # Error file name

###############################################################################
# Change to the directory from which the batch job was submitted
cd ${SLURM_SUBMIT_DIR}
###############################################################################

echo "Running on `hostname`"

# Load Python/Anaconda module (Enable Python in batch job environment)
module load git/2.19.0 python/3.9.16 cuda/12.4

# Activate virtual environment
source /path/to/your/venv/bin/activate          # Update with the path to your virtual environment

nvidia-smi
export PYTHONUNBUFFERED=1

# Directory to save logs and outputs for this run
OUTPUT_DIR=${OUTPUT_DIR:="runs/srgan_run1"}     # Change for each experiment
mkdir -p ${OUTPUT_DIR}

LOG_FILE="${OUTPUT_DIR}/log.txt"

# Run your Python training script
python /scratch/users/ju12/USR-248-2/train_srgan.py 2>&1 | tee ${LOG_FILE}
