import sys
import hydra
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from env.sawyer_peg_env import custom_sawyer_peg_env
from PD.sawyer_peg_pd import PegPD


@hydra.main(config_path="./config", config_name="save_demonstrations")
def main(cfg):
    logger = logging.getLogger('tacto.renderer')
    logger.propagate = False
    
    env = custom_sawyer_peg_env(cfg.env)
    pd = PegPD(env)

    succesful_episodes = 0 
    for i_episode in range(cfg.number_demonstrations):
        episode_return = 0
        observation = env.reset()
        pd.reset()
        for t in range(env.max_episode_steps):
            action = pd.get_action()
            observation, reward, done, info = env.step(action)
            episode_return += reward
            if done:
                if info["success"]:
                    succesful_episodes += 1
                print("Episode finished after {} timesteps".format(t+1))
                break
        print("Episode_return", episode_return)
    print("Total succesful episodes : %d/%d" % (succesful_episodes, cfg.number_demonstrations))
    env.close()

if __name__ == '__main__':
    main()

