# Training objective

[ALoDLM](../README.md)

The transformer runs a prelude once, repeats a shared middle block, and
branches through a coda at every depth. The recurrent carry uses the model's
final RMS normalization between passes. A shared linear gate observes detached
readout features and has a learned bias for each depth.

Training uses a clean and a masked stream. Each independently attended segment
has its own positions and attention boundary. Response tokens are masked within
blocks; observed tokens precede masked tokens while retaining their original
rotary positions.

The gate samples an exit for each masked token. Once a token exits, its ground
truth embedding replaces its latent state on subsequent passes. The final depth
is absorbing. Supervision excludes readouts after the sampled exit.

For token `i`, let `z_i` be its sampled exit, `l_i(d)` its cross-entropy, and
`logq_i` the log probability of its sampled decisions. The detached token cost is

```text
cost_i = l_i(z_i) - l_i(1)
       + beta_mi   * (logq_i - log(qbar[z_i]))
       + beta_marg * (log(qbar[z_i]) - log(prior[z_i]))
prior[d] = softmax(-c * d), d = 1,...,K
```

Costs are summed within each independently attended segment. Each segment's
cost multiplies the sum of its decision log probabilities, with one final
division by the total number of masked tokens. The first-pass loss is an
action-independent baseline. Costs are not centered across tokens.

`qbar` is a detached mean of the stick distributions observed along the sampled
rollout. This is a sampled-history regularization estimate. The two KL
coefficients separately control token-depth variation and the average depth
profile. The geometric parameter `c` is an exponential slope, not a halting
probability.

The denoiser receives detached, truncated-stick-weighted supervision at all
depths up to the sampled exit, with normalized inverse-mask-probability weights.
A next-token auxiliary loss is averaged across depths on the clean stream.
The final loss is `(denoising + outcome_actor + autoregressive) / 2`.

The supplied configuration uses four depths, `c=0.4`, `beta_mi=0.1`,
`beta_marg=1.0`, and uniform initial exit masses `[0.25, 0.25, 0.25, 0.25]`.
These masses correspond to conditional halting probabilities `1/4`, `1/3`,
and `1/2`, followed by a forced exit at the final depth. Gate initialization
and the geometric regularization prior are separate: the prior retains `c=0.4`.
These initialization settings apply to newly created gates. Loading a saved
model restores the learned gate parameters from `exit_gate.pt`.
For the 36-layer 8B backbone, layers `[0, 10)` form the prelude, `[10, 26)`
the recurrent block, and `[26, 36)` the coda; interval ends are exclusive.
There is one training objective; no token-local advantage or ablation switch
is exposed.
