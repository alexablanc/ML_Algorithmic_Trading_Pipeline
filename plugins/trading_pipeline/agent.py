import os
import pickle
import numpy as np
import sys

# Add plugins path to sys.path
sys.path.insert(0, '/usr/local/airflow/plugins')
from trading_pipeline.QLearner import QLearner

class TradingAgent:
    """
    Wrapper for the QLearner, managing state discretization
    and storage across daily DAG runs.
    """

    def __init__(self, model_path='/tmp/qlearner_model.pkl'):
        self.model_path = model_path
        self.learner = self._load_or_create_learner()
        self.pending_reward = None

    def _load_or_create_learner(self):
        """Load the existing Q-Learner from disk or create a new one."""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                return pickle.load(f)

        # Create a new learner if none exists
        # State space: 10 bins for BB, 10 for RSI, 10 for Mom, 3 for LLM Signal = 3000 states
        # Actions: 3 (0=Short, 1=Cash, 2=Long)
        return QLearner(
            num_states=3000,
            num_actions=3,
            alpha=0.2,
            gamma=0.9,
            rar=0.5,
            radr=0.99,
            dyna=200,
            verbose=False
        )

    def save_learner(self):
        """Save the updated Q-Learner to disk."""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.learner, f)

    def discretize_state(self, bb, rsi, mom, llm_signal):
        """
        Convert continuous indicators and the LLM signal into a single discrete state integer.
        - BB%B: 10 bins (0 to 9)
        - RSI: 10 bins (0 to 9)
        - Momentum: 10 bins (0 to 9)
        - LLM Signal: 3 bins (0, 1, 2)
        Total states = 10 * 10 * 10 * 3 = 3000
        """
        bb_bin = np.digitize(bb, np.linspace(-0.2, 1.2, 9))
        rsi_bin = np.digitize(rsi, np.linspace(10, 90, 9))
        mom_bin = np.digitize(mom, np.linspace(-0.1, 0.1, 9))
        llm_bin = llm_signal

        state = (bb_bin * 300) + (rsi_bin * 30) + (mom_bin * 3) + llm_bin
        return int(state)

    def set_reward(self, reward):
        """
        Store the reward from yesterday's action.
        It will be applied when get_action() is called with today's state.
        """
        self.pending_reward = reward

    def get_action(self, state):
        """
        Query the Q-Learner for the next action based on the current state.
        If a pending reward exists, it updates the Q-table first.
        """
        if self.pending_reward is not None:
            # We have a reward from yesterday, so update the Q-table and get today's action
            action = self.learner.query(state, self.pending_reward)
            self.pending_reward = None
        else:
            # First run or no reward available, get an action w/o updating
            action = self.learner.querysetstate(state)

        self.save_learner()
        return action


# --- Backward compatibility functions for the existing DAG ---
# The DAG calls these procedural functions, so we wrap them around a singleton instance.

_global_agent = None


def _get_agent():
    global _global_agent
    if _global_agent is None:
        _global_agent = TradingAgent()
    return _global_agent


def discretize_state(bb, rsi, mom, llm_signal):
    return _get_agent().discretize_state(bb, rsi, mom, llm_signal)


def update_q_learner(reward):
    _get_agent().set_reward(reward)


def get_q_action(state):
    return _get_agent().get_action(state)
