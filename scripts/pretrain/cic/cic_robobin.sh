# Configs
job_name=cic_robobin
seed=0

# Run command
python3 -u -m run.train --run_group $job_name \
                        --env robobin_image \
                        --max_path_length 200 \
                        --seed $seed \
                        --traj_batch_size 10 \
                        --n_parallel 10 \
                        --normalizer_type off \
                        --video_skip_frames 2 \
                        --frame_stack 3 \
                        --sac_max_buffer_size 300_000 \
                        --eval_plot_axis -50 50 -50 50 \
                        --algo cic \
                        --trans_optimization_epochs 100 \
                        --n_epochs_per_log 100 \
                        --n_epochs_per_eval 500 \
                        --n_epochs_per_save 1_000 \
                        --n_epochs_per_pt_save 1000 \
                        --discrete 0 \
                        --dim_option 64 \
                        --encoder 1 \
                        --sample_cpu 0 \
                        --trans_minibatch_size 256 \
                        --eval_goal_metrics 0 \
                        --turn_off_dones 1 \
                        --dual_reg 0 \
                        --eval_record_video 0 \
                        --unit_length 0 \
                        --joint_train 1