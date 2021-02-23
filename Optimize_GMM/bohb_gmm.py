import os
import sys
import json 
import hydra
import logging
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import ConfigSpace as CS
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import ConfigSpace.hyperparameters as CSH
from hpbandster.optimizers import BOHB as BOHB
sys.path.insert(0, str(Path(__file__).parents[1]))
from env.sawyer_peg_env import custom_sawyer_peg_env, register_sawyer_env
from GMM.gmm import GMM
from utils.path import add_cwd

def config_to_change_dict(config):
    dprior, dmu = [], []
    for key, value in sorted(config.items()):
        if key.startswith("prior"):
            dprior.append(value)
        if key.startswith("mu"):
            dmu.append(value)
    change_dict = {"prior": np.array(dprior), "mu": np.array(dmu)}
    return change_dict

class GMM_Worker(Worker):
    def __init__(self, cfg, run_id, nameserver):
        super(GMM_Worker,self).__init__(run_id, nameserver = nameserver)
        self.cfg = cfg
        model_name =  add_cwd(cfg.gmm_name)
        self.initial_model = GMM(model_name)
        self.total_episodes = 0
        self.logger = logging.getLogger(__name__)

    def compute(self, config, budget, working_directory, *args, **kwargs):
        int_budget = int(budget)
        model = GMM()
        model.copy_model(self.initial_model)
        x = config_to_change_dict(config)
        model.update_gaussians(x)
        env = custom_sawyer_peg_env(self.cfg.env)
        accuracy, mean_return, mean_length = model.evaluate(env, **self.cfg.validation, num_episodes=int_budget)
        env.close()
        print("Accuracy:", accuracy, "mean_return:", mean_return, "budget:", int_budget)
        self.total_episodes += int_budget
        return ({'loss': - mean_return, # remember: HpBandSter always minimizes!
                 'info': {  'num_episodes': int_budget,
                            'mean_episode_length': mean_length ,
                            'accuracy': accuracy}})

    def get_configspace(self):
        cs = CS.ConfigurationSpace()
        dprior = [CSH.UniformFloatHyperparameter('prior_%d'%i, lower=-0.15, upper=0.15) 
                   for i in range(self.initial_model.priors.size)]
        dmu = [CSH.UniformFloatHyperparameter('mu_%d'%i, lower=-0.005, upper=0.005) 
                for i in range(self.initial_model.mu.size)]
        cs.add_hyperparameters([*dprior, *dmu])
        return cs

def optimize(cfg):
    logger = logging.getLogger(__name__)
    
    NS = hpns.NameServer(run_id=cfg.bohb.run_id, host=cfg.bohb.nameserver, port=None)
    NS.start()

    w = GMM_Worker(cfg.worker, nameserver=cfg.bohb.nameserver, run_id=cfg.bohb.run_id)
    w.run(background=True)


    bohb = BOHB(  configspace = w.get_configspace(), eta=4,
                run_id = cfg.bohb.run_id, nameserver=cfg.bohb.nameserver,
                min_budget=cfg.bohb.min_budget, max_budget=cfg.bohb.max_budget )
            
    res = bohb.run(n_iterations=cfg.bohb.n_iterations)

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    inc_runs = res.get_runs_by_id(incumbent)
    inc_run = inc_runs[-1]

    # Log information
    logger.info('Best found configuration: %s' % id2config[incumbent]['config'])
    logger.info("Best run result - Average reward: %.2f" %(-inc_run.loss))
    logger.info("info: %s" % json.dumps(inc_run.info))
    logger.info('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    logger.info('A total of %i runs where executed.' % len(res.get_all_runs()))
    logger.info('Total budget corresponds to %.1f full function evaluations.' %(sum([r.budget for r in res.get_all_runs()])/cfg.bohb.max_budget))
    logger.info('Total number of episodes ran: %d' % w.total_episodes)


    # Store optimization results
    if not os.path.exists("optimization_results/"): 
        os.makedirs("optimization_results/")
    with open(os.path.join("optimization_results/", "%s.pkl" % cfg.bohb.run_id), 'wb') as fh:
        pickle.dump(res, fh)

    # Save model
    opt_model_dir = Path(cfg.optimized_gmm_name).parents[0]
    if not os.path.exists(opt_model_dir): 
        os.makedirs(opt_model_dir)
    model = GMM()
    model.copy_model(w.initial_model)
    dparam = config_to_change_dict(id2config[incumbent]['config'])
    model.update_gaussians(dparam)
    model.save_model(cfg.optimized_gmm_name)

    # Shutdown
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()


@hydra.main(config_path="../config", config_name="bohb_gmm_config")
def single_gmm(cfg):
    optimize(cfg)

if __name__ == "__main__":
    register_sawyer_env()
    logger = logging.getLogger('tacto.renderer')
    logger.propagate = False
    single_gmm()