# Configs
job_name=csf_humanoid
seed=0
option_dim=8

# Run command
python3 -u -m run.train --run_group $job_name \
                        --env dmc_humanoid \
                        --max_path_length 200 \
                        --seed $seed \
                        --traj_batch_size 8 \
                        --n_parallel 8 \
                        --normalizer_type off \
                        --video_skip_frames 2 \
                        --frame_stack 3 \
                        --sac_max_buffer_size 300000 \
                        --eval_plot_axis -15 15 -15 15 \
                        --algo metra_sf \
                        --trans_optimization_epochs 200 \
                        --n_epochs_per_log 25 \
                        --n_epochs_per_eval 125 \
                        --n_epochs_per_save 1000 \
                        --n_epochs_per_pt_save 1000 \
                        --discrete 0 \
                        --encoder 1 \
                        --sample_cpu 0 \
                        --eval_goal_metrics 1 \
                        --goal_range 10 \
                        --turn_off_dones 1 \
                        --sample_new_z 1 \
                        --num_negative_z 256 \
                        --log_sum_exp 1 \
                        --infonce_lam 5.0 \
                        --dim_option $option_dim