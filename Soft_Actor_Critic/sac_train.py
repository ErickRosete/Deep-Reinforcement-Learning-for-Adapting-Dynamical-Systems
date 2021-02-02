import sys
import hydra
import logging
from pathlib import Path
from sac_agent import SAC_Agent
sys.path.insert(0, str(Path(__file__).parents[1]))
from env.sawyer_peg_env import custom_sawyer_peg_env

def get_save_filename(cfg, it=0):
    noise = 'noise_' if cfg.env.observation.with_noise else ''
    force = "force" if cfg.env.observation.with_force else "pose"
    rs = "_rs_" + str(it) if hasattr(cfg, 'num_random_seeds') and cfg.num_random_seeds > 1 else ''
    save_filename = "sac_peg_v2_" + noise + force  + rs
    return save_filename


@hydra.main(config_path="../config", config_name="sac_config")
def main(cfg):
    # Do not show tacto renderer output
    logger = logging.getLogger('tacto.renderer')
    logger.propagate = False
    
    for i in range(cfg.train.num_random_seeds):
        # Training 
        env = custom_sawyer_peg_env(cfg.train.env)
        agent = SAC_Agent(env, **cfg.agent)
        save_filename = get_save_filename(cfg.train, i)
        agent.train(**cfg.train.run, save_filename=save_filename)
        agent.env.close()

        # Testing
        # agent.env = panda_peg_v2(**cfg.test.env)
        # agent.evaluate(**cfg.test.run)
        # agent.env.close()

if __name__ == "__main__":
    main()
