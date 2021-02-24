import numpy as np
from pathlib import Path
import glob
import json
import os

def save_demo_txt(filename, demonstration, dt):
    observations = []
    for t, observation in enumerate(demonstration):
        cur_obs = [t*dt]
        cur_obs.extend(observation['position'])
        cur_obs.extend(observation['force'])
        observations.append(cur_obs)
    np.savetxt(filename, observations)

def main():
    files = glob.glob( "demonstrations/*.json")
    save_dir = "demonstrations_txt/" 
    dt = 0.004 # Time between each timestep

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for f in files:
            new_filename = os.path.basename(f).replace(".json", ".txt")
            new_filename = os.path.join(save_dir, new_filename)
            with open(f, "r") as json_file:
                demonstration = json.load(json_file)
            save_demo_txt(new_filename, demonstration, dt)

if __name__ == "__main__":
    main()