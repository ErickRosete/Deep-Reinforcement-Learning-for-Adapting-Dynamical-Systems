import hydra
import drlfads
if not drlfads.USE_MATLAB:
    raise NotImplementedError('This script requires matlab')
import matlab.engine
from pathlib import Path

#Hyperparam
@hydra.main(config_path="../config", config_name="gmm_config")
def main(cfg):
    #Start matlab
    eng = matlab.engine.start_matlab()
    eng.addpath(str(Path(__file__).parents[0]))
    for K in range(cfg.K_range[0], cfg.K_range[1]):
        name = "%s_%s_%d" % (cfg.model_name, cfg.type, K)
        bll = eng.train_model(cfg.demonstration_dir, name, cfg.type, K, cfg.num_models)
        print(name, bll)
    eng.quit()

if __name__ == '__main__':
    main()
