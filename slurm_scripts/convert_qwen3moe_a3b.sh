#!/bin/bash
#SBATCH -J pai_megatron
#SBATCH --exclusive
#SBATCH --account=bsc70
#SBATCH --qos=acc_debug
#SBATCH --output=slurm_output/out.txt
#SBATCH --error=slurm_output/err.txt
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80


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
    --nv /gpfs/projects/bsc70/hpai/storage/data/heka/singularity/pai_megatron/pai-megatron-patch_25.02.sif bash -c "cd /workspace/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor; \
    sh scripts/qwen3/run_4xH100.sh \
    A3B \
    /gpfs/projects/bsc70/hpai/storage/data/heka/pai_megatron/qwen-ckpts/Qwen3-30B-A3B-Base \
    /gpfs/projects/bsc70/hpai/storage/data/heka/pai_megatron/qwen-ckpts/Qwen3-30B-A3B-Base-to-mcore \
    false \
    true \
    bf16"

echo "END TIME: $(date)"  
#rerelaunch