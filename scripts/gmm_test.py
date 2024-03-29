import hydra
import logging
from drlfads.utils.path         import add_cwd
from drlfads.env.sawyer_peg_env import  custom_sawyer_peg_env, register_sawyer_env
from drlfads.GMM.gmm            import GMM

@hydra.main(config_path="../config", config_name="gmm_config")
def main(cfg):
    logger = logging.getLogger('tacto.renderer')
    logger.propagate = False
    
    env = custom_sawyer_peg_env(cfg.env)
    for model_name in cfg.model_names:
        print(model_name)
        model = GMM(add_cwd(model_name))
        accuracy, mean_return, mean_length = model.evaluate(env=env, **cfg.test)
        logger = logging.getLogger(__name__)
        logger.info("Accuracy: %.2f, Mean return: %.2f, mean length: %.2f" % (accuracy, mean_return, mean_length))

if __name__ == '__main__':
    register_sawyer_env()
    main()

