# Configs
job_name=dads_half_cheetah
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
                        --n_epochs_per_log 100 \
                        --n_epochs_per_eval 1000 \
                        --n_epochs_per_save 10000 \
                        --sac_max_buffer_size 1000000 \
                        --algo dads \
                        --discrete 0 \
                        --dim_option 3 \
                        --goal_range 100 \
                        --eval_goal_metrics 0 \
                        --turn_off_dones 1 \
                        --unit_length 0 \
                        --sd_const_std 1 \
                        --sd_batch_norm 1 \
                        --uniform_z 1 \
                        --eval_record_video 0 \
                        --common_lr 0.0003 \
                        --sac_lr_a -1 \
                        --alpha 0.1