import glob, os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from functools import reduce

csv_dir = str((Path(__file__).parents[0] / "csv/").resolve())
search_string = csv_dir + "/*.csv"

csv_files = []
for i, filename in enumerate(glob.glob(search_string)):
    run = "Run_"+str(i)
    csv_file = pd.read_csv(filename)
    csv_file = csv_file.drop(columns=['Wall time'])
    csv_file = csv_file.rename(columns={"Value": run})
    csv_files.append(csv_file)


all_data = reduce(lambda df1, df2: df1.join(df2.set_index('Step'), on="Step"), csv_files)
run_keys = [col for col in all_data if col.startswith('Run')]
all_data['Mean'] = all_data[run_keys].mean(numeric_only=True, axis=1)
all_data["Std"] = all_data[run_keys].std(numeric_only=True, axis=1)
ax = all_data.plot("Step", "Mean")
all_data.plot("Step", run_keys, alpha=0.40, ax=ax)
plt.show()

ax = all_data.plot("Step", "Mean")
lb = all_data["Mean"] - all_data["Std"]
ub = all_data["Mean"] + all_data["Std"]
ax.fill_between(all_data["Step"], lb, ub, facecolor='purple', alpha=0.5)
plt.show()
