import hydra
import logging

from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

from drlfads.Soft_Actor_Critic.sac_agent import SAC_Agent
from drlfads.env.sawyer_peg_env          import custom_sawyer_peg_env, \
                                                register_sawyer_env

# Simple program for optimizing parameters using Ray
def train(config, params=None):
    #TODO: Save models
    logger = logging.getLogger(__name__)
    budget = 300
    env = custom_sawyer_peg_env(params.env)
    agent = SAC_Agent(env, **config)
    agent.train(**params.train, num_episodes = int(budget))
    accuracy, val_return, val_length = agent.evaluate(**params.validation)
    agent.env.close()
    logger.info("Final return reported to the optimizer: %2f" % val_return)
    tune.report(val_return=val_return, val_accuracy=accuracy)

def optimize(cfg):
    search_space = {
        "ae_lr":  tune.loguniform(1e-6, 1e-2),
    }

    # Points to evaluate
    best_params = [{
        "ae_lr": 3e-4
    }]

    search_alg = HyperOptSearch(metric="val_return", mode="max", points_to_evaluate=best_params)
    analysis = tune.run(
        tune.with_parameters(train, params=cfg.worker),
        num_samples=1,
        config=search_space, resources_per_trial={'cpu': 4, 'gpu': 1})
    search_alg.save("./opt_checkpoint.pkl")
    print("best config: ", analysis.get_best_config(metric="return", mode="max"))

@hydra.main(config_path="../config", config_name="sac_ray_hpo_config")
def main(cfg):
    optimize(cfg)

if __name__ == "__main__":
    register_sawyer_env()
    logger = logging.getLogger('tacto.renderer')
    logger.propagate = False
    main()