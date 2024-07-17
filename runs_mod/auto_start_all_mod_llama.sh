############################################################################
root_dir="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/bigcode-evaluation-harness-github"

folder_name0="llama_mod"

####################################################
folder_name1="paper"

####################################################
#folder_name2="main_results/baseline-llama2"
#folder_name2="main_results/llama-mod-interleave-zeros32-freq1-NoExtended-Scale2.0-Gap0.05-freq1-cap0.7-cos-global1.0-Anneal1000-DyLr"
#folder_name2="main_results/llama-mod-interleave-zeros32-freq1-NoExtended-Scale2.0-Gap0.05-freq1-cap0.75-cos-global1.0-Anneal1000-DyLr"
#folder_name2="main_results/llama-mod-interleave-zeros32-freq1-NoExtended-Scale2.0-Gap0.05-freq1-cap0.8-cos-global1.0-Anneal1000-DyLr"
#folder_name2="main_results/llama-mod-interleave-zeros32-freq1-NoExtended-Scale2.0-Gap0.05-freq1-cap0.85-cos-global1.0-Anneal1000-DyLr"
#folder_name2="main_results/llama-mod-interleave-zeros32-freq1-NoExtended-Scale2.0-Gap0.05-freq1-cap0.9-cos-global1.0"
#folder_name2="main_results/llama-mod-interleave-zeros32-freq1-NoExtended-Scale2.0-Gap0.05-freq1-cap0.95-cos-global1.0-Anneal1000-DyLr"

#folder_name2="main_results/llama-mod-interleave-zeros40-freq1-Scale2.0-Gap0.05-freq1-cap0.7-cos-global1.0-Anneal1000-DyLr"
#folder_name2="main_results/llama-mod-interleave-zeros40-freq1-Scale2.0-Gap0.05-freq1-cap0.75-cos-global1.0-Anneal1000-DyLr"
#folder_name2="main_results/llama-mod-interleave-zeros40-freq1-Scale2.0-Gap0.05-freq1-cap0.8-cos-global1.0-Anneal1000-DyLr"
#folder_name2="main_results/llama-mod-interleave-zeros40-freq1-Scale2.0-Gap0.05-freq1-cap0.85-cos-global1.0-Anneal1000-DyLr"
#folder_name2="main_results/llama-mod-interleave-zeros40-freq1-Scale2.0-Gap0.05-freq1-cap0.9-cos-global1.0-Anneal1000-DyLr"
#folder_name2="main_results/llama-mod-interleave-zeros40-freq1-Scale2.0-Gap0.05-freq1-cap0.95-cos-global1.0-Anneal1000-DyLr"
#folder_name2="main_results/llama-mod-interleave-zeros40-freq1-Scale2.0-Gap0.05-freq1-cap0.9-cos1.0"

#folder_name2="main_results/llama-mod-interleave-random40-freq1-Scale2.0-Gap0.05-freq1-cap0.8-cos-global1.0-Anneal1000-DyLr"
#folder_name2="main_results/llama-mod-interleave-normal40-freq1-Scale2.0-Gap0.05-freq1-cap0.9-cos-global1.0-Anneal1000-DyLr"
#folder_name2="main_results/llama-mod-interleave-merge-linear40-freq1-Scale2.0-Gap0.05-freq1-cap0.9-cos-global1.0-Anneal1000-DyLr"
#folder_name2="main_results/llama-mod-interleave-merge-slerp40-freq1-Scale2.0-Gap0.05-freq1-cap0.9-cos-global1.0-Anneal1000-DyLr"

#folder_name2="main_results/llama-mod-interleave-zeros40-freq1-Scale1.0-Gap1.0-freq1-cap0.9-cos-global1.0-Anneal1000-DyLr"
#folder_name2="main_results/llama-mod-interleave-zeros40-randomG-freq1-Scale2.0-Gap0.05-freq1-cap0.9-cos-global1.0-Anneal1000-DyLr"

#folder_name2="main_results_rerun/llama-mod-interleave-zeros40-zeroG-freq1-Scale2.0-Gap0.05-freq1-cap0.95-cos-global1.0-Anneal1000-DyLr"
folder_name2="main_results_rerun/llama-mod-interleave-zeros40-zeroG-freq1-Scale2.0-Gap0.05-freq1-cap0.9-cos-global1.0-Anneal1000-DyLr"
####################################################

max_length_list=(512 2048)
batch_size_list=(4 8)
task_name_list=("human_eval" "mbpp")

for ((i = 0; i < ${#max_length_list[@]}; i++)); do
  batch_size=${batch_size_list[i]}
  max_length=${max_length_list[i]}
  task_name=${task_name_list[i]}
  echo "${task_name}"

  #################### BASELINE ########################
  #  model_path="/mnt/petrelfs/dongdaize.d/quxioaye/models/llama2_7B"
  #  save_path="${root_dir}/results_mod/${task_name}/llama2_7B"

  #  model_path="/mnt/petrelfs/dongdaize.d/quxioaye/models/LLaMA-Pro-8B"
  #  save_path="${root_dir}/results_mod/${task_name}/LLaMA-Pro-8B"

  #  model_path="/mnt/petrelfs/dongdaize.d/quxioaye/models/Mistral-7B-v0.1"
  #  save_path="${root_dir}/results_mod/${task_name}/Mistral-7B-v0.1"

  #  model_path="/mnt/petrelfs/dongdaize.d/quxioaye/models/Mistral_Pro_8B_v0.1"
  #  save_path="${root_dir}/results_mod/${task_name}/Mistral_Pro_8B_v0.1"

  ######################################################
  model_path="/mnt/petrelfs/dongdaize.d/workspace/depth-llama/results/finetune/${folder_name0}/${folder_name1}/${folder_name2}"
  save_path="${root_dir}/results_mod/${task_name}/${folder_name0}-${folder_name1}/${folder_name2}"

  result_file="${save_path}/evaluation_results.json"
  #  rm ${result_file}
  if ls ${result_file} >/dev/null 2>&1; then
    echo "Result file \"${result_file}\" already exists. Do not apply the task."
  else
    if [ ! -d ${model_path} ]; then
      echo "Model path \"${model_path}\" not exists. Do not apply the task."
    else
      sbatch ${root_dir}/runs_mod/sub_tasks/${task_name}.sh ${model_path} ${save_path} ${batch_size} ${max_length} &
      sleep 1
    fi
  fi
done
wait
