import matlab.engine
from pathlib import Path

#Hyperparams
K_range = [3, 8]
type = "pose" # "pose" or "force"
demonstration_dir = "demonstrations_txt"
num_models = 100

#Start matlab
eng = matlab.engine.start_matlab()
eng.addpath(str(Path(__file__).parents[0]))
for K in range(K_range[0], K_range[1]):
    name = "gmm_peg_%s_%d" % (type, K)
    bll = eng.train_model(demonstration_dir, name, type, K, num_models)
    print(name, bll)
eng.quit()
