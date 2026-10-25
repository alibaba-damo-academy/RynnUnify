# Branch-Conditional Policy Optimization (BCPO)

BCPO learns **how to reason within a path** and **when generation is useful** with separate comparisons of the same question. The reasoner is the trainable actor; the video generator, initial reference policy, and answer judge remain frozen.

## Interaction and exploration

For each routable question, sample four direct (D) trajectories and four generation-assisted (G) trajectories. D answers from the original observations. G writes a generation instruction and a duration in (0, 10] seconds, receives the generated video, and answers from the augmented history. The training sampler forces the branch to explore both; the policy learns its inference-time decision with routing supervision. Native multi-image choice questions use eight D trajectories and receive no routing supervision.

The generator receives observed images (one for I2V, two for first/last-frame conditioning). For an observed video, its **last decoded frame** is the generation condition. The reasoner retains the original video. The returned complete MP4 is checksum-verified, decoded, and passed through the checkpoint's native video processor. The processor's own temporal sampling is model dependent; full-video transport does not imply every frame becomes a model token.

The policy sees only question, observed media, its own actions, and returned media. References are accessible only to scoring. Infrastructure failure raises an error and cancels the batch; it is never a low task reward.

## Within-branch improvement

For each question and branch, standardize its rewards:

\[
A_{z,i}=(r_{z,i}-\mu_z)/(\sigma_z+10^{-6}),\qquad
\rho_i=\exp\left[\frac{1}{|T_i|}\sum_{t\in T_i}(\log\pi_\theta-\log\pi_{\rm old})\right].
\]

Group normalization uses population standard deviation. Every trajectory gets equal weight. For G, **concatenate request and answer action log probabilities before taking the sequence mean**; averaging two independently computed losses would implement a different objective. Forced route tokens, prompts, tool observations and padding are not actions.

\[
L_{\rm body}=-\mathbb E_i\min(\rho_i A_i,
\operatorname{clip}(\rho_i,1-0.0003,1+0.0004)A_i).
\]

Uniform rewards within a branch yield zero body advantage. No trajectories are replaced or filtered to obtain a more diverse reward group.

## Learning the invocation policy

\[
q_G=\operatorname{sg}\sigma((\mu_G-\mu_D)/0.1),\qquad
L_{\rm route}=\frac1Q\sum_j m_j\operatorname{BCEWithLogits}(\ell_{G,j}-\ell_{D,j},q_{G,j}).
\]

Logits are evaluated on the original question at the route position. D and G must each be a single distinct token. A question's routing mask is one only if it is routable, all eight responses obey their protocols, and at least one branch has positive mean reward. The denominator is **the full Q**, including masked questions. Masking routing does not remove its body trajectories.

## Combined loss

\[
L=L_{\rm body}+0.01L_{\rm route}+0.001L_{\rm KL}-0.001H.
\]

The portable backend uses the sampled nonnegative k3 KL estimator, `exp(logp_ref-logp)-1-(logp_ref-logp)`, and full-distribution token entropy on action positions. KL and entropy are not added to the task reward. There is no explicit generation cost in the default routing target.

## Source navigation

- `selective_rollout.py`: balanced exploration, protocol validation and tool interaction.
- `selective_objective.py`: branch advantages, soft target and routing mask.
- `selective_loss.py`: sequence ratio, asymmetric clipping, KL, entropy and routing BCE.
- `trainer.py`: one optimizer update per question batch, trajectory gradient accumulation.
- `hf_policy.py`: native processor, generated-token replay, frozen reference and route position.
- `rewards.py`, `legacy/`: public reward adapter and task scoring functions.
