# Configs
job_name=cic_half_cheetah
seed=0

# Run command
python3 -u -m run.train --run_group $job_name \
                        --env half_cheetah \
                        --max_path_length 200 \
                        --seed $seed \
                        --traj_batch_size 8 \
                        --n_parallel 8 \
                        --normalizer_type preset \
                        --trans_optimization_epochs 50 \
                        --n_epochs_per_log 500 \
                        --n_epochs_per_eval 500 \
                        --n_epochs_per_save 10000 \
                        --sac_max_buffer_size 1000000 \
                        --algo cic \
                        --discrete 0 \
                        --dim_option 64 \
                        --goal_range 100 \
                        --eval_goal_metrics 0 \
                        --turn_off_dones 1 \
                        --eval_record_video 0 \
                        --unit_length 0 \
                        --dual_reg 0 \
                        --joint_train 1