import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_wealth_trajectory(portfolio_values: list, title="Agent Wealth Evolution", save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values, label='Agent Strategy', linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Total Value Portfolio', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.5)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_return_density(returns: list, title="Density of Geometric Returns", save_path=None):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(returns, fill=True, color="indigo", alpha=0.4, label='Agent Return Freq')
    plt.axvline(np.mean(returns), color='red', linestyle='--', label=f"Mean Return: {np.mean(returns):.4f}")
    plt.title(title, fontsize=14)
    plt.xlabel('Daily Return (%)', fontsize=12)
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
