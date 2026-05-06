import numpy as np

class BuyAndHoldAgent:
    """
    A naive baseline agent that distributes capital equally across all available
    assets on day 1 and maintains those target weightings.
    """
    def __init__(self, n_assets: int):
        self.n_assets = n_assets

    def predict(self, observation, deterministic=True):
        """
        Returns exactly uniform weights across all assets.
        Cash (index 0) gets 0% if fully invested, but let's standardise to uniform across all (including cash).
        """
        action = np.ones(self.n_assets + 1) / (self.n_assets + 1)
        # We need to return an action formatted for softmax processing in Env if required,
        # or pre-softmaxed logits. We'll return 0s so e^0 / sum(e^0) gives uniform.
        return np.zeros(self.n_assets + 1), None
