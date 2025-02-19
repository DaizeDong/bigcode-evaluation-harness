model_path=/mnt/petrelfs/share_data/quxiaoye/models/tzhu_merge_candidates/phi-2-coder
save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/bigcode-evaluation-harness/results/ds1000/phi-2-coder"

#model_path="/mnt/petrelfs/share_data/quxiaoye/models/tzhu_merge_candidates/phi-2-sft-dpo-gpt4_en-ep1"
#save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/bigcode-evaluation-harness/results/ds1000/phi-2-sft-dpo-gpt4_en-ep1"

#model_path="/mnt/petrelfs/share_data/quxiaoye/models/tzhu_merge_candidates/phi-2-dpo-new"
#save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/bigcode-evaluation-harness/results/ds1000/phi-2-dpo-new"

#model_path="/mnt/petrelfs/share_data/quxiaoye/models/tzhu_merge_candidates/dolphin-2_6-phi-2"
#save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/bigcode-evaluation-harness/results/ds1000/dolphin-2_6-phi-2"

#model_path="/mnt/petrelfs/share_data/quxiaoye/models/tzhu_merge_candidates/phixtral-2x2_8"
#save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/bigcode-evaluation-harness/results/ds1000/phixtral-2x2_8"

#model_path="/mnt/petrelfs/share_data/quxiaoye/models/tzhu_merge_candidates/phixtral-4x2_8"
#save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/bigcode-evaluation-harness/results/ds1000/phixtral-4x2_8"

#model_path="/mnt/petrelfs/share_data/quxiaoye/models/microsoft--phi-2"
#save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/bigcode-evaluation-harness/results/ds1000/microsoft--phi-2"

mkdir -p ${save_path}

export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3

gpus=8
cpus=$(($gpus * 16))
quotatype=auto # auto spot reserved
OMP_NUM_THREADS=4 srun --partition=MoE --job-name="eval" --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
  accelerate launch main.py \
  --model ${model_path} \
  --batch_size 16 \
  --tasks ds1000-all-insertion \
  --n_samples 40 \
  --max_length_generation 1024 \
  --temperature 0.2 \
  --top_p 0.95 \
  --precision bf16 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path ${save_path}/generations.json \
  --metric_output_path ${save_path}/evaluation_results.json
