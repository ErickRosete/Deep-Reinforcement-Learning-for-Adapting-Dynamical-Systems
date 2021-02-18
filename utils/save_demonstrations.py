import sys
import json
import hydra
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from env.sawyer_peg_env import custom_sawyer_peg_env, register_sawyer_env
from PD.sawyer_peg_pd import PegPD

def dict_arrays_to_list(dict):
    for key, value in dict.items():
        if type(value) is list: 
            dict[key] = [x.tolist() for x in value]
        else:
            dict[key] = value.tolist()
    return dict

def save_file(obj, directory, filename):
    Path(directory).mkdir(parents=True, exist_ok=True)
    filename = Path(directory) / filename
    with open(filename, 'w') as fout:
        json.dump(obj, fout)
    print("Saved file %s" % filename)

@hydra.main(config_path="../config", config_name="save_demonstrations")
def main(cfg):
    logger = logging.getLogger('tacto.renderer')
    logger.propagate = False
    
    env = custom_sawyer_peg_env(cfg.env)
    pd = PegPD(env)

    succesful_episodes = 0
    while succesful_episodes < cfg.number_demonstrations:
        episode_return = 0
        observation = env.reset()
        observations = [dict_arrays_to_list(observation)]
        pd.reset()
        for t in range(env.max_episode_steps):
            action = pd.get_action()
            observation, reward, done, info = env.step(action)
            observations.append(dict_arrays_to_list(observation))
            episode_return += reward
            if done:
                if info["success"]:
                    directory = 'demonstrations'
                    filename = "demonstration_%d.json" % (succesful_episodes + 1)
                    save_file(observations, directory, filename)
                    succesful_episodes += 1
                break
        print("Episode_return", episode_return)
    print("Total succesful episodes : %d/%d" % (succesful_episodes, cfg.number_demonstrations))
    env.close()

if __name__ == '__main__':
    register_sawyer_env()
    main()

