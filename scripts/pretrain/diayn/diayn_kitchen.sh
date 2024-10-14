# Configs
job_name=diayn_kitchen
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
                        --algo metra \
                        --trans_optimization_epochs 100 \
                        --n_epochs_per_log 25 \
                        --n_epochs_per_eval 250 \
                        --n_epochs_per_save 1000 \
                        --n_epochs_per_pt_save 1000 \
                        --discrete 1 \
                        --dim_option 50 \
                        --encoder 1 \
                        --sample_cpu 0 \
                        --eval_goal_metrics 1 \
                        --turn_off_dones 1 \
                        --inner 0 \
                        --unit_length 0 \
                        --dual_reg 0 \
                        --diayn_include_baseline 1 \
                        --sac_lr_a -1 \
                        --alpha 0.1