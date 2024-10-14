# Configs
seed=0
job_name=${1}_humanoid_goal
if [ $1 == "cic" ]; then
    cp_path="/path/to/cic/option_policy*.pt" # NEEDS TO BE REPLACED WITH REAL PATH
    cp_unit_length=0
    policy_type=gaussian
    use_discrete_sac=0
elif [ $1 == "csf" ]; then
    cp_path="/path/to/csf/option_policy*.pt" # NEEDS TO BE REPLACED WITH REAL PATH
    cp_unit_length=1
    policy_type=gaussian
    use_discrete_sac=0
elif [ $1 == "diayn" ]; then
    cp_path="/path/to/diayn/option_policy*.pt" # NEEDS TO BE REPLACED WITH REAL PATH
    cp_unit_length=0
    policy_type=categorical
    use_discrete_sac=1
elif [ $1 == "metra" ]; then
    cp_path="/path/to/metra/option_policy*.pt" # NEEDS TO BE REPLACED WITH REAL PATH
    cp_unit_length=1
    policy_type=gaussian
    use_discrete_sac=0
elif [ $1 == "visr" ]; then
    cp_path="/path/to/visr/option_policy*.pt" # NEEDS TO BE REPLACED WITH REAL PATH
    cp_unit_length=1
    policy_type=gaussian
    use_discrete_sac=0
else
    echo "Unknown option: $1"
    exit 1
fi

# Run command
python3 -u -m run.train --run_group $job_name \
                        --env dmc_humanoid_goal \
                        --max_path_length 8 \
                        --dim_option 4 \
                        --num_random_trajectories 48 \
                        --seed $seed \
                        --normalizer_type off \
                        --use_gpu 1 \
                        --traj_batch_size 8 \
                        --n_parallel 8 \
                        --algo sac \
                        --n_epochs_per_eval 25 \
                        --n_thread 1 \
                        --model_master_dim 1024 \
                        --n_epochs_per_log 10 \
                        --eval_record_video 1 \
                        --n_epochs 200001 \
                        --discrete 0 \
                        --sac_discount 0.99 \
                        --video_skip_frames 2 \
                        --frame_stack 3 \
                        --trans_optimization_epochs 50 \
                        --sac_max_buffer_size 300000 \
                        --eval_plot_axis -15 15 -15 15 \
                        --common_lr 0.0001 \
                        --trans_minibatch_size 256 \
                        --encoder 1 \
                        --sample_cpu 0 \
                        --goal_range 5.0 \
                        --cp_multi_step 25 \
                        --downstream_reward_type esparse \
                        --downstream_num_goal_steps 200 \
                        --cp_path $cp_path \
                        --cp_path_idx 0 \
                        --cp_unit_length $cp_unit_length \
                        --alpha 0.1 \
                        --policy_type $policy_type \
                        --use_discrete_sac $use_discrete_sac