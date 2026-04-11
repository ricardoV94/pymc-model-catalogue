"""
Model: Reinforcement Learning (Bernoulli likelihood)
Source: pymc-examples/examples/case_studies/reinforcement_learning.ipynb, Section: "Alternative model using Bernoulli for the likelihood"
Authors: Ricardo Vieira
Description: Two-armed bandit reinforcement learning model with Q-value updating via pytensor.scan.
    Uses pm.Bernoulli likelihood for the observed right-action choices, with alpha (learning rate)
    and beta (inverse temperature) as free parameters. This is the alternative formulation that
    enables posterior predictive sampling and model comparison.

Changes from original:
- Inlined simulated data (actions and rewards arrays, 150 trials, seed derived from "RL_PyMC")
- Removed MLE estimation, plotting, and sampling code

Benchmark results:
- Original:  logp = -56.2869, grad norm = 26.8556, 82.9 us/call (100000 evals)
- Frozen:    logp = -56.2869, grad norm = 26.8556, 79.0 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

def build_model():
    def _generate_data(rng, alpha, beta, n=100, p_r=None):
        """Generate simulated RL data (reproduced from notebook)."""
        if p_r is None:
            p_r = [0.4, 0.6]
        actions = np.zeros(n, dtype="int")
        rewards = np.zeros(n, dtype="int")
        Q = np.array([0.5, 0.5])
        for i in range(n):
            exp_Q = np.exp(beta * Q)
            prob_a = exp_Q / np.sum(exp_Q)
            a = rng.choice([0, 1], p=prob_a)
            r = rng.random() < p_r[a]
            Q[a] = Q[a] + alpha * (r - Q[a])
            actions[i] = a
            rewards[i] = r
        return actions, rewards
    def update_Q(action, reward, Qs, alpha):
        """
        This function updates the Q table according to the RL update rule.
        It will be called by pytensor.scan to do so recursevely, given the observed data and the alpha parameter
        This could have been replaced be the following lamba expression in the pytensor.scan fn argument:
            fn=lamba action, reward, Qs, alpha: pt.set_subtensor(Qs[action], Qs[action] + alpha * (reward - Qs[action]))
        """
        Qs = pt.set_subtensor(Qs[action], Qs[action] + alpha * (reward - Qs[action]))
        return Qs
    def right_action_probs(alpha, beta, actions, rewards):
        rewards = pt.as_tensor_variable(rewards, dtype="int32")
        actions = pt.as_tensor_variable(actions, dtype="int32")
        # Compute the Qs values
        Qs = 0.5 * pt.ones((2,), dtype="float64")
        Qs, updates = pytensor.scan(
            fn=update_Q, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha]
        )
        # Apply the sotfmax transformation
        Qs = Qs[:-1] * beta
        logp_actions = Qs - pt.logsumexp(Qs, axis=1, keepdims=True)
        # Return the probabilities for the right action, in the original scale
        return pt.exp(logp_actions[:, 1])
    seed = sum(map(ord, "RL_PyMC"))
    rng = np.random.default_rng(seed)
    true_alpha = 0.5
    true_beta = 5
    n = 150
    actions, rewards = _generate_data(rng, true_alpha, true_beta, n)

    with pm.Model() as m_alt:
        alpha = pm.Beta(name="alpha", alpha=1, beta=1)
        beta = pm.HalfNormal(name="beta", sigma=10)

        action_probs = right_action_probs(alpha, beta, actions, rewards)
        like = pm.Bernoulli(name="like", p=action_probs, observed=actions[1:])

    ip = m_alt.initial_point()
    m_alt.rvs_to_initial_values = {rv: None for rv in m_alt.free_RVs}
    return m_alt, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
