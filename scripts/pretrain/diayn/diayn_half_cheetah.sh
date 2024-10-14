# Configs
job_name=diayn_half_cheetah
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
                        --algo metra \
                        --discrete 1 \
                        --dim_option 50 \
                        --goal_range 100 \
                        --eval_goal_metrics 1 \
                        --turn_off_dones 1 \
                        --inner 0 \
                        --unit_length 0 \
                        --dual_reg 0 \
                        --diayn_include_baseline 1 \
                        --sac_lr_a -1 \
                        --alpha 0.1