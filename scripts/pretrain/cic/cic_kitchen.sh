# Configs
job_name=cic_kitchen
seed=0

# Run command
python3 -u -m run.train --run_group $job_name \
                        --env kitchen \
                        --max_path_length 50 \
                        --seed $seed \
                        --traj_batch_size 8 \
                        --n_parallel 8 \
                        --normalizer_type off \
                        --num_video_repeats 1 \
                        --frame_stack 3 \
                        --sac_max_buffer_size 100000 \
                        --algo cic \
                        --trans_optimization_epochs 100 \
                        --n_epochs_per_log 25 \
                        --n_epochs_per_eval 250 \
                        --n_epochs_per_save 1000 \
                        --n_epochs_per_pt_save 1000 \
                        --discrete 0 \
                        --dim_option 64 \
                        --encoder 1 \
                        --sample_cpu 0 \
                        --eval_goal_metrics 0 \
                        --turn_off_dones 1 \
                        --eval_record_video 0 \
                        --dual_reg 0 \
                        --unit_length 0 \
                        --joint_train 1