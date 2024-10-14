# Configs
seed=0
if [ $1 == "cic" ]; then
    job_name=cic_ant_multi_goal
    cp_path=/path/to/cic/option_policy*.pt # NEEDS TO BE REPLACED WITH REAL PATH
    cp_unit_length=0
    policy_type=gaussian
    use_discrete_sac=0
elif [ $1 == "csf" ]; then
    job_name=csf_ant_multi_goal
    cp_path="/path/to/csf/option_policy*.pt" # NEEDS TO BE REPLACED WITH REAL PATH
    cp_unit_length=1
    policy_type=gaussian
    use_discrete_sac=0
elif [ $1 == "dads" ]; then
    job_name=dads_ant_multi_goal
    cp_path="/path/to/dads/option_policy*.pt" # NEEDS TO BE REPLACED WITH REAL PATH
    cp_unit_length=0
    policy_type=gaussian
    use_discrete_sac=0
elif [ $1 == "diayn" ]; then
    job_name=diayn_ant_multi_goal
    cp_path="/path/to/diayn/option_policy*.pt" # NEEDS TO BE REPLACED WITH REAL PATH
    cp_unit_length=0
    policy_type=categorical
    use_discrete_sac=1
elif [ $1 == "metra" ]; then
    job_name=metra_ant_multi_goal
    cp_path="/path/to/metra/option_policy*.pt" # NEEDS TO BE REPLACED WITH REAL PATH
    cp_unit_length=1
    policy_type=gaussian
    use_discrete_sac=0
elif [ $1 == "visr" ]; then
    job_name=visr_ant_multi_goal
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
                        --env ant_nav_prime \
                        --max_path_length 8 \
                        --dim_option 2 \
                        --num_random_trajectories 48 \
                        --seed $seed \
                        --normalizer_type preset \
                        --use_gpu 1 \
                        --traj_batch_size 8 \
                        --n_parallel 8 \
                        --algo sac \
                        --n_epochs_per_eval 100 \
                        --n_thread 1 \
                        --model_master_dim 1024 \
                        --n_epochs_per_log 25 \
                        --eval_record_video 1 \
                        --n_epochs 200001 \
                        --discrete 0 \
                        --sac_discount 0.99 \
                        --eval_plot_axis -50 50 -50 50 \
                        --trans_optimization_epochs 50 \
                        --sac_max_buffer_size 1000000 \
                        --common_lr 0.0001 \
                        --trans_minibatch_size 256 \
                        --goal_range 7.5 \
                        --alpha 0.1 \
                        --cp_multi_step 25 \
                        --downstream_reward_type esparse \
                        --downstream_num_goal_steps 50 \
                        --cp_path $cp_path \
                        --cp_path_idx 0 \
                        --cp_unit_length $cp_unit_length \
                        --policy_type $policy_type \
                        --use_discrete_sac $use_discrete_sac