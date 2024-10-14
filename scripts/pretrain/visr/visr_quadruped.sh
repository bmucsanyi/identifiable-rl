# Configs
job_name=visr_quadruped
seed=0

# Run command
python3 -u -m run.train --run_group $job_name \
                        --env dmc_quadruped \
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
                        --n_epochs_per_log 50 \
                        --n_epochs_per_eval 250 \
                        --n_epochs_per_save 1000 \
                        --n_epochs_per_pt_save 1000 \
                        --discrete 0 \
                        --encoder 1 \
                        --sample_cpu 0 \
                        --trans_minibatch_size 256 \
                        --goal_range 15 \
                        --eval_goal_metrics 1 \
                        --turn_off_dones 1 \
                        --dim_option 5 \
                        --eval_record_video 0 \
                        --no_diff_in_rep 1 \
                        --self_normalizing 1 \
                        --dual_reg 0