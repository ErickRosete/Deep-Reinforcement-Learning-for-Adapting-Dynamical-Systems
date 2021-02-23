from skopt import gp_minimize
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from GMM.gmm import GMM
from env.sawyer_peg_env import  custom_sawyer_peg_env, register_sawyer_env

class GMMOptimizer:
    def __init__(self, env, model):
        self.initial_model = model
        self.env = env

    def optimize(self):
        bounds = [[-0.15, 0.15]] * self.initial_model.priors.size + [[-0.005, 0.005]] * self.initial_model.mu.size
        res = gp_minimize(self.objective,        # the function to minimize
                  bounds,                        # the bounds on each dimension of x
                  acq_func="EI",                 # the acquisition function
                  n_calls=200,                    # the number of evaluations of f
                  n_random_starts=20,             # the number of random initialization points
                  noise="gaussian" ,             # the objective returns noisy gaussian observations
                  random_state=1234)             # the random seed
        return res

    def objective(self, x):
        model = GMM() 
        model.copy_model(self.initial_model) 
        model.update_gaussians(np.asarray(x))
        accuracy, mean_return, mean_length = model.evaluate(self.env, max_steps=600, num_episodes=1)
        print("Accuracy:", accuracy, "mean_return:", mean_return)
        return - mean_return


def main():
    # Environment hyperparameters
    env_params = {"show_gui": False, "with_force": False, "with_joint": False,
                    "relative": True, "with_noise": False, "dt": 0.05}
    env = custom_sawyer_peg_env(**env_params)

    # Evaluation parameters
    model_name = "models/GMM_models/gmm_peg_v2_pose_9.npy"
    model = GMM(model_name)

    optimizer = GMMOptimizer(env, model)
    res = optimizer.optimize()
    print(res.x)
    model.update_gaussians(np.asarray(res.x))
    new_model_name = "models/optimizer/test.npy"
    model.save_model(new_model_name)
    print("Best model - Average reward:", -res.fun)
    print("Model saved as:", new_model_name)


if __name__ == '__main__':
    register_sawyer_env()
    main()
