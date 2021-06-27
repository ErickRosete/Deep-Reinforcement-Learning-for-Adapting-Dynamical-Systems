import hydra
import logging
import drlfads
if drlfads.USE_MATLAB:
    import matlab.engine

from drlfads.utils.path         import add_cwd
from drlfads.env.sawyer_peg_env import  custom_sawyer_peg_env, register_sawyer_env
from drlfads.GMM.gmm            import GMM

@hydra.main(config_path="../config", config_name="gmm_config")
def main(cfg):
    logger = logging.getLogger('tacto.renderer')
    logger.propagate = False
    logger = logging.getLogger('env.sawyer_peg_env')
    logger.propagate = False
    logger = logging.getLogger('pybulletX._wrapper')
    logger.propagate = False
    #Hyperparams
    type = "pose" # "pose" or "force"
    demonstration_dir = add_cwd("demonstrations_txt")
    K = 3
    budget = 20

    #Start matlab
    log_likelihood = []
    best_ret = 0
    if not drlfads.USE_MATLAB:
        raise NotImplementedError(f'This function requires matlab')

    eng = matlab.engine.start_matlab()
    eng.addpath(add_cwd(str(Path(__file__).parents[0])))
    env = custom_sawyer_peg_env(cfg.env)
    for _ in range(budget):
        name = "gmm_peg_%s_%d" % (type, K)
        bll = eng.train_model(demonstration_dir, name, type, K, 1)
        print("model trained, final log likelihood:", bll)
        
        # Test new configurations
        if not bll in log_likelihood:
            # Evaluate model in actual environment
            log_likelihood.append(bll)
            model = GMM(name+".mat")
            accuracy, mean_return, mean_length = model.evaluate(env=env, **cfg.test)
            print("Accuracy:", accuracy, "Mean return:", mean_return, "Mean length:", mean_length)
            if mean_return > best_ret:
                print("Best model so far!")
                best_ret = mean_return
                model.save_model(name+".npy")

    eng.quit()

if __name__ == '__main__':
    register_sawyer_env()
    main()

