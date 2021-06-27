import glob, os
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cbook  as cbook
from pathlib   import Path
from functools import reduce
# from tabulate import tabulate
import seaborn as sns

def plot_dirs(dirs):
    csv_dir = str((Path(__file__).parents[0] / "csv/").resolve())
    cat_dirs = ['validation','train']
    for cat_dir in cat_dirs:
        sns.set(style="darkgrid", font_scale=2.5)
        with sns.axes_style("darkgrid"):
            f, ax = plt.subplots()
        f.tight_layout(pad=0.5)
        ax.set_title(cat_dir)
        ax.set_ylabel('return')
        for plot_dir in dirs:
            search_string = '%s/%s/%s/*.csv' % (csv_dir, cat_dir, plot_dir)
            csv_files = []
            for i, filename in enumerate(glob.glob(search_string)):
                run = "Run_"+str(i)
                csv_file = pd.read_csv(filename)
                csv_file = csv_file.drop(columns=['Wall time'])
                csv_file = csv_file.rename(columns={"Step": "episode", "Value": run})
                csv_files.append(csv_file)

            # Join csv files in a single df
            all_data = reduce(lambda df1, df2: df1.join(df2.set_index('episode'), on="episode"), csv_files)
            run_keys = [col for col in all_data if col.startswith('Run')]

            # Smoothing with a moving window average
            y = np.ones(20) 
            for key in run_keys:
                x = np.asarray(all_data[key])
                z = np.ones(len(x))
                smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
                all_data[key] = smoothed_x

            # Calculate mean and std
            all_data[plot_dir] = all_data[run_keys].mean(numeric_only=True, axis=1)
            all_data["std"] = all_data[run_keys].std(numeric_only=True, axis=1)

            all_data.plot("episode", plot_dir, ax=ax)
            lb = all_data[plot_dir] - all_data["std"]
            ub = all_data[plot_dir] + all_data["std"]
            ax.fill_between(all_data["episode"], lb, ub, alpha=0.5)

        plt.legend(loc='best').set_draggable(True)
        plt.show()

if __name__ == '__main__':
    dirs = ['SAC GMM Residual - hard without noise', 'SAC GMM Residual tactile - hard without noise']
    dirs = ['SAC GMM - hard without noise', 'SAC GMM tactile - hard without noise']
    dirs = ['SAC pose - hard with noise', 'SAC tactile - hard with noise',]
    #dirs = ['SAC pose - hard without noise', 'SAC tactile - hard without noise',]
    #dirs = ['SAC GMM - easy with noise', 'SAC GMM tactile - easy with noise']
    plot_dirs(dirs)