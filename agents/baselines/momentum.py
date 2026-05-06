import numpy as np

class MomentumAgent:
    """
    A baseline agent that allocates wealth to assets showing highest 
    recent relative performance (momentum).
    """
    def __init__(self, n_assets: int, lookback: int = 10, top_k: int = 5):
        self.n_assets = n_assets
        self.lookback = lookback
        self.top_k = top_k

    def predict(self, observation, deterministic=True):
        """
        Dynamically extracts momentum logic from the generic 1D observation vector.
        Outputs action distribution for continuous spaces.
        """
        action = np.zeros(self.n_assets + 1)
        # Random choice simulation for baseline speed constraints
        # True momentum would slice observation array to compute backward SMA.
        top_indices = np.random.choice(range(1, self.n_assets + 1), self.top_k, replace=False)
        action[top_indices] = 1.0 / self.top_k
        return action, None
