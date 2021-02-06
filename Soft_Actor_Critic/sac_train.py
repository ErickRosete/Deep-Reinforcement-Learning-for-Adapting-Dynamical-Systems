import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import hydra
import logging
from Soft_Actor_Critic.sac_agent import SAC_Agent
from env.sawyer_peg_env import custom_sawyer_peg_env

def get_save_filename(cfg, it=0):
    noise = 'noise_' if cfg.env.observation.with_noise else ''
    force = "force" if cfg.env.observation.with_force else "pose"
    rs = "_rs_" + str(it) if hasattr(cfg.train, 'num_random_seeds') and cfg.train.num_random_seeds > 1 else ''
    save_filename = "sac_peg_v2_" + noise + force  + rs
    return save_filename

@hydra.main(config_path="../config", config_name="sac_config")
def main(cfg):
    # Do not show tacto renderer output
    logger = logging.getLogger('tacto.renderer')
    logger.propagate = False
    
    for i in range(cfg.train.num_random_seeds):
        # Training 
        env = custom_sawyer_peg_env(cfg.env)
        agent = SAC_Agent(env, **cfg.agent)
        save_filename = get_save_filename(cfg, i)
        agent.train(**cfg.train.run, save_filename=save_filename)
        agent.env.close()

        # Testing
        # agent.env = custom_sawyer_peg_env(**cfg.test.env)
        # agent.evaluate(**cfg.test.run)
        # agent.env.close()

if __name__ == "__main__":
    main()
