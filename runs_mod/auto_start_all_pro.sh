############################################################################
root_dir="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/bigcode-evaluation-harness-github"

folder_name0="llama_pro"
parallelize=True

##########################
folder_name1="converted"
#folder_name1="converted-cpt"

##########################
#folder_name2="mixed"
folder_name2="mixed-FULL"
#folder_name2="vicuna_sharegpt"
#folder_name2="vicuna_sharegpt-FULL"
#folder_name2="evol_instruct"
#folder_name2="evol_instruct-FULL"
#folder_name2="slim_orca"
#folder_name2="slim_orca-FULL"
#folder_name2="meta_math_qa"
#folder_name2="meta_math_qa-FULL"
#folder_name2="evol_code_alpaca"
#folder_name2="evol_code_alpaca-FULL"

##########################

max_length_list=(512 2048)
batch_size_list=(8 8)
task_name_list=("human_eval" "mbpp")

for ((i = 0; i < ${#max_length_list[@]}; i++)); do
  batch_size=${batch_size_list[i]}
  max_length=${max_length_list[i]}
  task_name=${task_name_list[i]}
  echo "${task_name}"

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
