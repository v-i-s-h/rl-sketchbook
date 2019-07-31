# Main interface to run classic control experiments

import os, argparse
from tqdm import tqdm

import gym

from stable_baselines.bench import Monitor

# To handle command line options
parser = argparse.ArgumentParser()

parser.add_argument("--env",        help="OpenAI Gym environment name", 
                                        type=str, default="CartPole-v1")
parser.add_argument("--episodes",   help="Maximum episodes to run", 
                                        type=int, default=200)
parser.add_argument("--rounds",     help="Number of rounds", 
                                        type=int, default=10)
parser.add_argument("--logdir",     help="Directory to save results",
                                        type=str, default="./logs/")

args = parser.parse_args()

# --------------- Gym related functions ----------------------------------------

# --------------- RL related functions -----------------------------------------
from stable_baselines.deepq.policies import FeedForwardPolicy, MlpPolicy, LnMlpPolicy
from stable_baselines import DQN

# Max episode limiter
class MaxEpisodeCb(object):
    def __init__(self, max_episodes):
        self.max_episodes   = max_episodes
        self.n_episodes     = 0

    def callback(self, _locals, _globals):
        if _locals["_"] > 0:    # Are we past first step?
            if _locals["done"]:
                self.n_episodes += 1
        # Continue of if we did not complete required number of episodes
        return self.n_episodes < self.max_episodes

    def reset(self):
        self.n_episodes = 0


# --------------- Main function ------------------------------------------------

if __name__ == "__main__":

    # Print experiment configuration
    print("Environment    :", args.env)
    print("No.of episodes :", args.episodes)
    print("No.of rounds   :", args.rounds)
    print("Log directory  :", args.logdir)

    # Make sure the directory exists
    os.makedirs(args.logdir, exist_ok=True)

    for i in tqdm(range(args.rounds)):
        # Log files prefix
        log_file_prefix = "{}/{:03d}".format(args.logdir,i)
        
        # Make environment
        env = gym.make(args.env)
        env = Monitor(env, log_file_prefix, allow_early_resets=True)

        maxep_cb = MaxEpisodeCb(max_episodes=args.episodes)

        # Create model
        model = DQN(MlpPolicy, env, verbose=0)

        # Train model
        model.learn(total_timesteps=100000, callback=maxep_cb.callback)

        model.save("{}.model".format(log_file_prefix))