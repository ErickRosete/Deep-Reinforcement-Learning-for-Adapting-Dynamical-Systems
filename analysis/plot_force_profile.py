import hydra
from Soft_Actor_Critic.sac_agent import SAC_Agent
import matplotlib.pyplot as plt
from env.sawyer_peg_env import custom_sawyer_peg_env
from utils.utils import add_cwd

def plot_force_sequences(force_sequences):
    fig = plt.figure()
    for side in ["left", "right"]:
        ax = fig.add_subplot(2, 1, 1 if side=="left" else 2)
        kwargs = {'color': 'b', 'linewidth': 2, 'linestyle': 'solid'}
        for seq in force_sequences["successes"][side]:
            ax.plot(seq, **kwargs)
        ax.plot([], [], label='Successes', **kwargs)
        kwargs = {'color': 'r', 'linewidth': 2, 'linestyle': 'dashed'}
        for seq in force_sequences["failures"][side]:
            ax.plot(seq, **kwargs)
        ax.plot([], [], label='Failures', **kwargs)
        ax.legend()
    plt.show()

@hydra.main(config_path="./config", config_name="sac_config")
def main(cfg):
    env = custom_sawyer_peg_env(cfg.env)
    agent = SAC_Agent(env, cfg.agent)
    agent.load(add_cwd(cfg.test.model_name))

    min_seq = 5
    force_sequences = {"successes":{"left":[], "right":[]}, "failures":{"left":[], "right":[]}}
    successes, failures = 0, 0
    while successes < min_seq or failures < min_seq:
        force_readings = {"left":[], "right":[]}
        state = env.reset()
        for step in range(env.max_episode_steps):
            force_readings["left"].append(state[-2])
            force_readings["right"].append(state[-1])
            action = agent.getAction(state, deterministic = True) 
            next_state, reward, done, info = env.step(action)
            state = next_state
            if done:
                break
        if "success" in info and info['success']:
            if successes < min_seq:
                force_sequences["successes"]["left"].append(force_readings["left"])
                force_sequences["successes"]["right"].append(force_readings["right"])
            successes += 1
        else:
            if failures < min_seq:
                force_sequences["failures"]["left"].append(force_readings["left"])
                force_sequences["failures"]["right"].append(force_readings["right"])
            failures += 1

    plot_force_sequences(force_sequences)

if __name__ == "__main__":
    main()