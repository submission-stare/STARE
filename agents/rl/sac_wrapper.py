from stable_baselines3 import SAC

class SACOptimizedTrainer:
    """
    Trainer wrapper implementing Soft Actor-Critic (SAC) logic using Stable-Baselines3.
    SAC maximizes trade rewards while also maximizing agent entropy (exploration).
    """
    def __init__(self, env, learning_rate=3e-4, batch_size=256, buffer_size=100000):
        self.env = env
        self.model = SAC("MlpPolicy", env,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         buffer_size=buffer_size,
                         ent_coef='auto',
                         verbose=1)

    def train(self, total_timesteps=50000):
        print(f"Training SAC Agent for {total_timesteps} steps...")
        self.model.learn(total_timesteps=total_timesteps)
        return self.model
