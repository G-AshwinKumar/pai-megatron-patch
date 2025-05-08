#!/bin/bash
#SBATCH -J pai_megatron
#SBATCH --exclusive
#SBATCH --account=bsc70
#SBATCH --qos=gp_debug
#SBATCH --output=slurm_output/out.txt
#SBATCH --error=slurm_output/err.txt
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --constraint=highmem



export SLURM_CPU_BIND=none
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export USE_SYSTEM_NCCL=1

export PATH=/apps/ACC/CUDNN/9.1.0/cuda12/lib:$PATH

echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "START TIME: $(date)"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export GPUS_PER_NODE=4
export NNODES=$SLURM_NNODES
export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# so processes know who to talk to
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# export SINGULARITY_TMPDIR=/dev/shm/
module purge
module load singularity

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

# export TORCH_USE_CUDA_DSA=1

# py-spy top -s -i -n -- $LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD
clear; srun $SRUN_ARGS --jobid $SLURM_JOBID singularity exec -B /gpfs/projects/bsc70 \
    -B /gpfs/scratch/bsc70 \
    -B /gpfs/home/bsc/bsc070997/opt/nvidia/nsight-systems/2024.7.1:/nsys-home \
    -B $PWD:/workspace/Pai-Megatron-Patch/ \
    --nv /gpfs/projects/bsc70/hpai/storage/data/heka/singularity/pai_megatron/pai-megatron-patch_25.02.sif bash -c "cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/mistral; \
    sh hf2mcore_convertor.sh \
    8x7B \
    /gpfs/projects/bsc70/hpai/storage/data/heka/pai_megatron/mistral-ckpts/Mixtral-8x7B-v0.1 \
    ../../../     \
    /gpfs/projects/bsc70/hpai/storage/data/heka/pai_megatron/mistral-ckpts/Mixtral-8x7B-v0.1 \
    /gpfs/projects/bsc70/hpai/storage/data/heka/pai_megatron/mistral-ckpts/Mixtral-8x7B-v0.1-to-mcore-tp4-pp1-ep4-exp8-ws16 \
    4  \
    1  \
    0  \
    8  \
    2  \
    4 \
    false \
    16"

echo "END TIME: $(date)"  
# relaunch

