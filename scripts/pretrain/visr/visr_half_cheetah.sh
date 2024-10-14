# Configs
job_name=visr_half_cheetah
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
                        --n_epochs_per_eval 1000 \
                        --n_epochs_per_save 10000 \
                        --sac_max_buffer_size 1000000 \
                        --algo metra_sf \
                        --discrete 0 \
                        --eval_goal_metrics 1 \
                        --goal_range 100 \
                        --turn_off_dones 1 \
                        --dim_option 5 \
                        --eval_record_video 0 \
                        --no_diff_in_rep 1 \
                        --self_normalizing 1 \
                        --dual_reg 0