# Can a MISL Fly? Analysis and Ingredients for Mutual Information Skill Learning
Official code repo for the paper "Can a MISL Fly? Analysis and Ingredients for Mutual Information Skill Learning" by [Chongyi Zheng](https://chongyi-zheng.github.io), [Jens Tuyls](https://jens321.github.io), [Joanne Peng](https://www.joannepeng.com), and [Benjamin Eysenbach](https://ben-eysenbach.github.io). This paper introduces a new method which we call **Contrastive Successor Features (CSF)**, which achieves compareable performance to current SOTA unsupervised skill discovery methods while at its core relying on mutual information maximization.

## Installation 🔌

After cloning this repo, please run the following commands at the root of the project:
```
# Setting up the conda environment
conda create --name csf python=3.9
conda activate csf

# Installing dependencies
pip install -r requirements.txt --no-deps
pip install -e .
pip install -e garaged
pip install --upgrade joblib
pip install patchelf
```

> [!WARNING] 
> Pip might complain about incompatible versions -- this is expected and can be safely ignored.

Next, we need to do some Mujoco setup.
```
conda activate csf
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c anaconda mesa-libgl-cos6-x86_64
conda install -c menpo glfw3
```

We also need to tell Mujoco which backend to use. This can be done by setting the appropriate environment variables.
```
conda env config vars set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
conda deactivate && conda activate csf
```

If you don't already have Mujoco, you will need it. Install Mujoco in a folder called `.mujoco`. More instructions on how to do so are linked [here](https://pytorch.org/rl/main/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html).

Finally, you may want to add the following environment variables to your `.bashrc` file:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/your/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CPATH=$CONDA_PREFIX/include
```

Remember to source your `.bashrc` file after changing it: `source ~/.bashrc`.

## Running Experiments 🏃‍♂️

(1) For **unsupervised pretraining** (state coverage), you can use the following general command. Make sure to run this from the root of the project.
```
sh scripts/pretrain/[method_name]/[method_name]_[env_name].sh
```
For example, in order to run our CSF method on the Ant environment, you would run:
```
sh scripts/pretrain/csf/csf_ant.sh
```

> [!NOTE] 
> The zero-shot goal reaching performance gets logged during the pretraining phase, and hence we don't have separate scripts for them.

(2) For **hierarchical control**, you can use the following general command. Again, make sure to run this from the root of the project.
```
sh scripts/hierarchical_control/[task].sh [method_name]
```
For example, in order to run a pretrained CSF policy on the AntMultiGoal environment, you would run:
```
sh scripts/hierarchical_control/ant_multi_goal.sh csf
```

> [!WARNING]
> All hierarchical control experiments require a pretrained policy path referred to using the `cp_path` argument. 
> Make sure to update this in the corresponding scripts.

Once experiments are running, they will be logged under the `exp` folder.

> [!NOTE] 
> All experiments were run on a single GPU, usually with between 8 - 10 workers (see the `--n_parallel` flag).
> In addition, we found we needed 32GB of CPU memory (RAM) for all state-based experiments (Ant and HalfCheetah), while
> we needed 40GB of CPU memory for all image-based experiments (Humanoid, Quadruped, Kitchen, Robobin).

## Acknowledgements
This code repo was built on the original [METRA repo](https://github.com/seohongpark/METRA).