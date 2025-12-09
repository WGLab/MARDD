#!/bin/bash

main_dir=/home/wangz12/Qwen2-VL-Finetune
device=h100

CUDA_VISIBLE_DEVICES="0,1,2,3"
module add CUDA
echo "Starting batch job submission..."
echo "Starting job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running as user: $(whoami)"
echo "Start time: $(date)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

batch_sizes=(2)
ratio_values=(0.1) #logs_external_detailed_analysis_RAG/logs_"${disease_name}"_"${peft_model_id}"
seeds=(42)
disease_names=('x') #'bws' 'cdls' 'sotos' 'kbgs'
peft_model_ids=('x') #logs_external_detailed_analysis_RAG/${disease_name}/${peft_model_id})  # ${llama3b_orpo} ${llama3b_dpo} ${llama3b_sft} #logs_external_detailed_analysis_RAG/${disease_name}/${peft_model_id}"
for peft_model_id in "${peft_model_ids[@]}"; do
    for disease_name in "${disease_names[@]}"; do
        for ratio in "${ratio_values[@]}"; do
            log_dir="${main_dir}/Qwen_inference"
            runs_dir="${main_dir}/runs_ratio${ratio}"

            mkdir -p "${log_dir}"
            mkdir -p "${runs_dir}"
            for seed in "${seeds[@]}"; do
                for batch_size in "${batch_sizes[@]}"; do
                    echo -e "\nSubmitting job for batch_size: ${batch_size}"
                    echo -e "Running commands on: $(hostname)"
                    echo -e "Start time: $(date '+%F %H:%M:%S')"
                    echo $CUDA_VISIBLE_DEVICES

                    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=inference_${batch_size}
#SBATCH --output=${log_dir}/${seed}_%j.stdout
#SBATCH --error=${log_dir}/${seed}_%j.stderr
#SBATCH -N 1
#SBATCH --gres=gpu:${device}:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=25:20:00
#SBATCH --cpus-per-task=48
#SBATCH --partition=gpu-xe9680q
#SBATCH --account=hpcusers
#SBATCH --mail-user=Zhanliang.Wang666@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mem=500G           
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export NCCL_DEBUG=INFO 
export MAIN_DIR="${main_dir}"
export PORT=$((1024 + RANDOM % 64512))
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE

${main_dir}/scripts/inference.sh
EOF
                echo "Submitted job for batch_size: ${batch_size}"
                done
            done
        done
    done
done
echo "Batch job submission completed."
echo "Allocated GPU IDs: $CUDA_VISIBLE_DEVICES"
