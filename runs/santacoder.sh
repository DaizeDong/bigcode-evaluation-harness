model_path=/mnt/petrelfs/share_data/quxiaoye/models/tzhu_merge_candidates/phi-2-coder
save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/bigcode-evaluation-harness/results/santacoder/phi-2-coder"

#model_path="/mnt/petrelfs/share_data/quxiaoye/models/tzhu_merge_candidates/phixtral-4x2_8"
#save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/bigcode-evaluation-harness/results/santacoder/hixtral-4x2_8"

mkdir -p ${save_path}

gpus=8
cpus=$(($gpus * 16))
quotatype=auto # auto spot reserved
OMP_NUM_THREADS=4 srun --partition=MoE --job-name="eval" --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
  accelerate launch main.py \
  --model ${model_path} \
  --max_length_generation 512 \
  --tasks santacoder_fim \
  --n_samples 1 \
  --temperature 0.2 \
  --batch_size 1 \
  --precision bf16 \
  --save_generations \
  --save_generations_path ${save_path}/generations.json \
  --metric_output_path ${save_path}/evaluation_results.json
