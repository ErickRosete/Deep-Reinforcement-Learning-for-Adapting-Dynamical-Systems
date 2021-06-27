import os
import glob
import json
import numpy as np
from pathlib import Path
import argparse


def save_demo_txt(filename, demonstration, dt):
    observations = []
    for t, observation in enumerate(demonstration):
        cur_obs = [t*dt]
        cur_obs.extend(observation['position'])
        cur_obs.extend(observation['force'])
        observations.append(cur_obs)
    np.savetxt(filename, observations)

def main(cfg):
    files = glob.glob( "%s*.json" % cfg.input_dir)
    dt = 0.004 # Time between each timestep

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    for f in files:
            new_filename = os.path.basename(f).replace(".json", ".txt")
            new_filename = os.path.join(cfg.output_dir, new_filename)
            with open(f, "r") as json_file:
                demonstration = json.load(json_file)
            save_demo_txt(new_filename, demonstration, dt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract pose and force to txt files from json demonstrations files.')
    parser.add_argument("--input_dir", default="demonstrations/")
    parser.add_argument("--output_dir", default="demonstrations_txt/")
    cfg = parser.parse_args()
    main(cfg)