"""hmm_regime_decode.py — Viterbi-decode market regimes from a return series.

Demonstrates:
  - `kuant.qm.ghmm.viterbi` for the maximum-likelihood state sequence
  - `kuant.qm.ghmm.forward` for the log-marginal likelihood
  - `kuant.qm.ghmm.posterior` for smoothed per-bar state probabilities

Uses the Gaussian-emission HMM (GHMM) since returns are continuous. Two
states: 'quiet' (low vol) and 'stress' (high vol). We simulate a hidden
regime sequence, generate returns from it, then recover the sequence.

Run:
    python docs/examples/hmm_regime_decode.py
"""

from __future__ import annotations

import numpy as np

from kuant.qm.ghmm import forward, posterior, viterbi


def main() -> None:
    rng = np.random.default_rng(0)

    # 1) Model (known to us — the whole point is to see if we recover it).
    pi = np.array([0.9, 0.1])  # start in quiet 90%
    A = np.array(
        [
            [0.98, 0.02],  # sticky quiet
            [0.10, 0.90],
        ]
    )  # stress persists too
    mu = np.array([0.0005, -0.001])  # small pos drift / neg drift
    sigma = np.array([0.008, 0.025])  # quiet vol / stress vol

    # 2) Simulate hidden state sequence + observed returns.
    n = 800
    states_true = np.zeros(n, dtype=int)
    obs = np.zeros(n)
    states_true[0] = rng.choice(2, p=pi)
    obs[0] = rng.normal(mu[states_true[0]], sigma[states_true[0]])
    for t in range(1, n):
        states_true[t] = rng.choice(2, p=A[states_true[t - 1]])
        obs[t] = rng.normal(mu[states_true[t]], sigma[states_true[t]])

    # 3) Recover the regime via Viterbi (best single-path decoding).
    states_viterbi, log_prob_path = viterbi(obs, pi, A, mu, sigma)

    # 4) Forward pass for log-likelihood of the whole sequence.
    log_alpha, log_lik = forward(obs, pi, A, mu, sigma)

    # 5) Smoothed per-bar state posteriors (soft alternative to Viterbi).
    gamma, xi, _ = posterior(obs, pi, A, mu, sigma)
    # gamma[t, s] = P(state=s at time t | full sequence)

    # 6) Compare recovered vs. true regimes.
    accuracy = float(np.mean(states_viterbi == states_true))
    print(f"n bars:              {n}")
    print(f"Viterbi path log-prob: {log_prob_path:.3f}")
    print(f"Total sequence log-lik: {log_lik:.3f}")
    print(f"Viterbi state accuracy: {accuracy * 100:.2f}%")
    print()

    # 7) Regime-share breakdown.
    frac_true_stress = float(np.mean(states_true == 1))
    frac_recovered_stress = float(np.mean(states_viterbi == 1))
    frac_gamma_stress = float(np.mean(gamma[:, 1]))
    print(f"Stress-state share — true:      {frac_true_stress:.3f}")
    print(f"Stress-state share — Viterbi:   {frac_recovered_stress:.3f}")
    print(f"Stress-state share — posterior: {frac_gamma_stress:.3f}")
    print()

    # 8) Show a slice where the model transitions.
    #    Find the first transition into stress state.
    transitions = np.where(np.diff(states_true) == 1)[0]
    if len(transitions):
        first_stress = int(transitions[0])
        lo = max(0, first_stress - 3)
        hi = min(n, first_stress + 5)
        print(f"Transition into stress at t={first_stress}:")
        print(f"  {'t':>4s}  {'obs':>8s}  {'true':>5s}  {'viterbi':>7s}  {'P(stress)':>9s}")
        for t in range(lo, hi):
            marker = " ← switch" if t == first_stress + 1 else ""
            print(
                f"  {t:>4d}  {obs[t]:>+8.4f}  {states_true[t]:>5d}  "
                f"{states_viterbi[t]:>7d}  {gamma[t, 1]:>9.4f}{marker}"
            )


if __name__ == "__main__":
    main()
