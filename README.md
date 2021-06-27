# Deep Reinforcement Learning for Adapting Dynamical Systems

## Description
In this work we study ways for augmenting dynamical systems by combining them with deep reinforcement learning in order to improve their performance in the context of contact rich insertion. We propose two approaches to adapt the dynamical systems, SAC-GMM-Residual and SAC-GMM. <br/>
SAC-GMM-Residual aims to learn a residual action on top of the dynamical system's action by exploring the environment safely throughout the learning process. <br/>
In contrast, SAC-GMM adapts the dynamical system's own parameter space to allow using its slightly modified version in the face of noisy observations.

![SAC GMM Example](./example_video.gif)


## Installation
The package was tested using python 3.7 and Ubuntu 20.04 LTS. <br/>
To install use the package manager [pip](https://pip.pypa.io/en/stable/).
```bash
# At root of project
$ pip install -e .
```

## Usage

### Peg insertion environment
It consists in a PyBullet environment with a Sawyer robot. The goal of the task is to insert a peg into a box, the environment works similar to Gym from OpenAI environments.
```python
from env.sawyer_peg_env import custom_sawyer_peg_env, register_sawyer_env

@hydra.main(config_path="../config", config_name="sac_gmm_config")
def main(cfg):
    env = custom_sawyer_peg_env(cfg.env)
    for episode in range(100):
        observation = env.reset()
        episode_length, episode_reward = 0,0
        for step in range(100):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break

if __name__ == "__main__":
    register_sawyer_env()
    main()
```
The environment observation consists in a dictionary with different values that could be accessed when requested in the [hydra env configuration file](./config/env/sawyer_env.yaml). <br/>
All the possible parameters are listed below
```python
        observation_space["position"]
        observation_space["gripper_width"]
        observation_space["tactile_sensor"]
        observation_space["force"]
```
- "position" contains the end effector cartesian pose
- "gripper_width" defines the end effector distance between each finger
- "tactile_sensor" contains the image measurement of each finger
- "force" contains the force readings in each finger 

### Training process
To reproduce the results we provide the following guidelines to understand the training roadmap of each proposed model.
- SAC
    - Train SAC
- GMM
    - Saving demonstrations > Train GMM
- SAC GMM Residual
    - Saving demonstrations > Train GMM > Train SAC GMM Residual
- SAC GMM
    - Saving demonstrations > Train GMM > Train SAC GMM

A detailed explanation of each substep is provided below

### Saving demonstrations
The demonstrations to learn the GMM are recorded by following a [PD Controller](./drlfads/PD/sawyer_peg_pd.py), this program is defined to follow waypoints until inserting the peg.
To save the demonstrations you need to execute the python file [save_demonstrations.py](./scripts/save_demonstrations.py) you can adjust the configuration in this [hydra file](./config/save_demonstrations.yaml).</br>
As output a folder with several demonstrations will be produced, each demonstration consist in a json file with the requested observation space.
You can also use the json files to extract txt files that contain the pose and force in each timestep, a helper script is available on [json_dem_to_txt.py](./scripts/json_dem_to_txt.py).

```bash 
$ # Saving json file demonstrations
$ python scripts/save_demonstrations.py number_demonstrations=20 output_dir="demonstrations/"
$ # Generate txt demonstrations
$ python scripts/json_dem_to_txt.py --input_dir="demonstrations/" --output_dir="demonstrations_txt/"
```

### GMM
Gaussian Mixture Models are used to learn the dynamical system from demonstrations. <br/>
All the files related to this task are inside the GMM directory.
The training of the GMM is based on [ds-opt](https://github.com/nbfigueroa/ds-opt), the model is trained on MATLAB. <br/>
There are useful scripts to train and test the GMM inside this directory, further more the script [gmm.py](./drlfads/GMM/gmm.py) contain several utilities to use the GMM as a dynamical system, i.e. you can predict velocity by providing an observation, or modify the model by making a change in the parameter. Additionally, it supports loading GMM trained with either matlab or python.

```bash 
$ # Train
$ python scripts/gmm_train.py model_name="models/GMM/gmm_peg"
$ # Test
$ python scripts/gmm_test.py model_names=["models/GMM/gmm_peg_pose_3.npy"] show_gui=True

```
As further reference, please see [gmm_config.yaml](./config/gmm_config.yaml) which contain the default configuration parameters for both training and testing.


### SAC
Soft Actor Critic is a state of the art reinforcement learning algorithm inside the [Soft_Actor_Critic](./drlfads/Soft_Actor_Critic) directory you could find the agent implementation, there are also Hyperparameter optimization routines and it is adapted to interact directly with the provided environment.

```bash 
$ # Test
$ python scripts/sac_test.py
$ # Train
$ python scripts/sac_train.py
```

### SAC GMM Residual
In this implementation SAC is capable of predicting a residual action on top of the GMM model prediction, such that it can uses high dimensional observations to adjust the final velocity predicted by the model. The main implementation code can be found in [sac_gmm_residual_agent.py](./drlfads/SAC_GMM_Residual/sac_gmm_residual_agent.py).

```bash 
$ # Test
$ python scripts/sac_gmm_residual_test.py
$ # Train
$ python scripts/sac_gmm_residual_train.py
```

### SAC GMM 
In this code, SAC predicts a change in the parameters for the initial GMM configuration. Consequently, the modified GMM is used to interact with the environment throughout a window size. This allows the SAC agent to predict from a trajectory level perspective, the implementation can be found in [sac_gmm_agent.py](./drlfads/SAC_GMM/sac_gmm_agent.py).

```bash 
$ # Test
$ python scripts/sac_gmm_test.py
$ # Train
$ python scripts/sac_gmm_train.py
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. <br/>
If you have any question please contact me through my email erickrosetebeas@hotmail.com.

## License
[MIT](https://choosealicense.com/licenses/mit/)
