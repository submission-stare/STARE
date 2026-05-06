import optuna
from stable_baselines3 import PPO

class PPOOptunaRunner:
    """
    Detailed hyperparameter search over PPO configurations for continuous trading spaces.
    Searches for optimal learning rates, batch sizes, and entropy coefficients.
    """
    def __init__(self, env, n_trials=50, study_name="ppo_trading_search"):
        self.env = env
        self.n_trials = n_trials
        self.study_name = study_name

    def optimize_agent(self, trial):
        lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        ent_coef = trial.suggest_loguniform("ent_coef", 1e-8, 1e-2)

        model = PPO("MlpPolicy", self.env, learning_rate=lr, batch_size=batch_size, ent_coef=ent_coef, verbose=0)
        
        # Abbreviated simulated training window to measure fitness
        model.learn(total_timesteps=15000)
        
        obs = self.env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = self.env.step(action)
        
        return self.env.portfolio_value

    def run(self):
        study = optuna.create_study(direction="maximize", study_name=self.study_name)
        study.optimize(self.optimize_agent, n_trials=self.n_trials)
        print(f"Optimal Trading Params Discovered: {study.best_params}")
        return study
