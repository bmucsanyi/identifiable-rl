# Configs
job_name=metra_ant
seed=0

# Run command
python3 -u -m run.train --run_group $job_name \
                        --env ant \
                        --max_path_length 200 \
                        --seed $seed \
                        --traj_batch_size 8 \
                        --n_parallel 4 \
                        --normalizer_type preset \
                        --eval_plot_axis -50 50 -50 50 \
                        --trans_optimization_epochs 50 \
                        --n_epochs_per_log 500 \
                        --n_epochs_per_eval 500 \
                        --n_epochs_per_save 10000 \
                        --sac_max_buffer_size 1000000 \
                        --algo metra \
                        --discrete 0 \
                        --dim_option 2 \
                        --eval_goal_metrics 1 \
                        --goal_range 50 \
                        --turn_off_dones 1