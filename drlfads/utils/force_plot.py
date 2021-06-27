import matplotlib.pyplot as plt

class ForcePlot():
    def __init__(self):
        self.fig, ax = plt.subplots(figsize=(3,4), constrained_layout=True)
        ax.set_title('Fingers force')
        ax.set_ylabel('Force')
        ax.set_ylim([0, 1])
        self.rects = ax.bar(range(2), [0, 0])

    def update(self, force):
        for i, rect in enumerate(self.rects):
            rect.set_height(force[i])
        self.fig.canvas.draw()
        plt.pause(0.001)

