import hydra
import logging
from drlfads.Soft_Actor_Critic.sac_agent import SAC_Agent
from drlfads.env.sawyer_peg_env          import custom_sawyer_peg_env, register_sawyer_env
from drlfads.utils.misc                  import get_save_filename

@hydra.main(config_path="../config", config_name="sac_config")
def main(cfg):
    # Do not show tacto renderer output
    logger = logging.getLogger('tacto.renderer')
    logger.propagate = False
    
    for i in range(cfg.train.num_random_seeds):
        # Training 
        env = custom_sawyer_peg_env(cfg.env)
        agent = SAC_Agent(env, **cfg.agent)
        save_filename = get_save_filename("sac_peg", cfg, i)
        agent.train(**cfg.train.run, save_filename=save_filename)
        agent.env.close()

        # Testing
        # agent.env = custom_sawyer_peg_env(**cfg.test.env)
        # agent.evaluate(**cfg.test.run)
        # agent.env.close()

if __name__ == "__main__":
    register_sawyer_env()
    main()
