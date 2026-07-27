# Concepts, in plain language

A teaching reference for explaining HiBayES analyses to someone who is a competent researcher but not a practising Bayesian. Load it when the user needs the *why* behind a step (see "Teach as you work" in `SKILL.md`), and lift explanations from here rather than improvising theory.

**How to use an entry.** Each has four parts: the one-liner, the intuition, where it shows up in hibayes, and the trap. In conversation, lead with the one-liner and the intuition, tie it to the user's actual data (their group names, their numbers), and offer the rest only if they want it. Two or three sentences usually lands; a lecture rarely does.

---

## Posterior, and why it isn't a p-value

**One-liner.** The posterior is a probability distribution over the parameter — every value the parameter could take, weighted by how well it's supported by your data and priors.

**Intuition.** Frequentist inference asks "how surprising would my data be if the effect were exactly zero?" and reports a p-value. Bayesian inference asks the question people actually want answered: "given what I saw, what values is the effect plausibly taking?" The answer isn't one number, it's a whole distribution, and you can read anything you want off it — a mean, a 90% interval, or `P(effect > 0)` directly.

**In hibayes.** Every fit returns MCMC *draws* from the posterior: thousands of parameter vectors whose density approximates the distribution. `summary_table` reports mean/median plus interval per parameter; `forest_plot` draws the intervals; `inference_data.nc` holds the raw draws, so any question expressible as "what fraction of draws satisfy X" is one line of code away.

**Trap.** Do not translate an interval back into significance testing. "The 94% interval excludes zero" is not "p < 0.05", and an interval that *includes* zero is weak evidence, not evidence of no effect — often it means "we don't have enough data to say", which is a different finding and should be reported as one.

## Priors, and the fact that there are no neutral ones

**One-liner.** A prior states what parameter values are plausible before seeing this dataset; picking one is unavoidable, so pick it deliberately.

**Intuition.** People hope for a prior that assumes nothing. There isn't one. A prior that looks agnostic about a parameter usually implies something absurd about the observable outcome — and the outcome is what you have intuitions about. So choose priors by simulating the data they imply and checking whether that simulated world is one you'd believe. Weakly regularising priors — mildly sceptical of huge effects — also improve out-of-sample prediction, because they resist chasing noise.

**In hibayes.** Priors are model-config kwargs (`prior_mu_overall_scale`, `prior_sigma_group_scale`, ...), so you can fit the same model under several prior settings by giving each a `tag`. The `prior_predictive_plot` checker does the simulating for you, before any fitting.

**Trap.** "Wide equals uninformative" is false on the logit scale — see the next entry.

## The logit scale (and why "wide" priors go wrong there)

**One-liner.** Binomial and ordinal models estimate effects on a log-odds scale that runs from −∞ to +∞, then squash it into a 0–1 probability; distances on the two scales are not comparable.

**Intuition.** Logit 0 is a 50% chance. Logit ±2 is about 12% / 88%. Logit ±5 is 0.7% / 99.3% — already essentially certain. So a `Normal(0, 10)` prior on an intercept, which *sounds* humble, actually says "this task is almost surely either never solved or always solved, and 50-50 would be a shock". `Normal(0, 1.5)` is roughly flat across probabilities, which is what people mean when they say uninformative. For effects, 0.3–1 is generous: an effect of 2 moves 50% to 88%.

**In hibayes.** `link_function` (default `logit`) sets the squashing; `forest_plot: {transform: true}` maps effects back to the probability scale for reporting.

**Trap.** Reporting raw coefficients to a non-statistical audience. "+0.8 logits" is meaningless to most readers; "raises solve rate from 50% to about 69%" is the same fact, understood. Note that the probability change depends on the baseline — the same logit effect is large in the middle of the scale and small near the ends.

## Partial pooling (hierarchy, shrinkage)

**One-liner.** Estimate each group separately *and* estimate how much groups vary, letting each group's estimate borrow strength from the others in proportion to how little data it has.

**Intuition.** Three ways to handle 20 task groups. Pool completely (one overall accuracy) and you erase real differences. Pool not at all (20 independent estimates) and a group with 4 samples that happened to go 4/4 gets reported at 100%. Partial pooling learns the spread of group performance from the data and pulls each group toward the overall mean by an amount set by that spread and that group's sample size — small, noisy groups move a lot, large groups barely budge. It isn't a fudge: with many groups of uneven size it's the estimator that predicts new data best.

**When flat is the better answer.** Hierarchy earns its place when there are many groups, some of them small, and you care about the population they're drawn from. It's the wrong tool when the levels *are* the question — comparing three named models, two graders, treatment vs control. Those aren't a sample from a population of models, and shrinking them toward each other pulls on precisely the contrast being asked about; with only a few levels the between-group variance is barely identifiable anyway, which shows up as divergences or a `sigma_group` posterior that won't settle. A flat fixed-effects fit is then simpler, better identified, and more honest. Worth saying out loud to users, who often assume "hierarchical" means "more rigorous".

**In hibayes.** Hierarchical: `two_level_group_binomial` and friends — `mu_overall` is the population mean, `sigma_group` the learned spread, `group_effects` the per-group deviations, `prior_sigma_group_scale` your scepticism about large spread. Flat: `linear_group_binomial` (or `ordered_logistic_model`) with `main_effects`, where each level's coefficient stands on its own data.

**Trap.** Whichever you pick, the grouping in the model has to match the grouping in the data-generating story. If your rows are per-sample scores with 5 epochs per task and you don't aggregate (`groupby`) or model the repetition, the model treats 5 correlated observations as 5 independent ones and your intervals come out too narrow (see pseudo-replication, below). Dropping the hierarchy does not excuse you from that.

## Ordinal outcomes and cutpoints

**One-liner.** Judge ratings on a 0–10 scale are ordered categories, not measurements, and need a model that only assumes the order.

**Intuition.** Treating a rating as a number asserts that the step 3→4 is the same size as 7→8, and that 10.4 is conceivable. Neither is true of a rating scale. An ordered-logistic model instead posits a continuous latent quality and a set of *cutpoints* along it: whichever band the latent value falls into is the rating you observe. The cutpoints are estimated, so unevenly-used scale points (graders love 7, never say 2) are absorbed rather than distorting the effects.

**In hibayes.** `ordered_logistic_model` with `num_classes` covering the full range (0–10 → 11), scores as integers starting at 0. Effects are shifts of the latent scale, so report them as changes in category probabilities.

**Trap.** Averaging ratings across graders before modelling. That both assumes metric spacing and throws away the grader-level structure you probably wanted to measure.

## MCMC, in one paragraph

**One-liner.** The posterior can't be computed in closed form, so the sampler walks the parameter space and its footprints become the draws you analyse.

**Intuition.** NUTS (the default) uses the gradient of the posterior to take long, well-chosen steps rather than blundering about randomly. Several independent chains start from different points; if they all end up wandering the same region in the same way, that's evidence the walk actually explored the posterior. All the diagnostics below are answers to one question: *did the walk work?* Nothing about them concerns whether the model is scientifically right — a perfectly-sampled fit of a silly model is still silly.

**In hibayes.** `fit: {samples, warmup, chains, target_accept, ...}`. Warmup draws are discarded tuning; `samples` are what you keep.

**Trap.** Reading numbers off a fit whose diagnostics failed. Convergence comes first, always.

## The diagnostics, and what each one actually detects

- **R-hat** (want < 1.01) — compares variance *between* chains to variance *within* chains. Above threshold means the chains disagree about where the posterior is, so they can't all be right. Usual causes: too little warmup, a stuck chain, or a model that isn't identified (two parameters that can trade off freely with no constraint).
- **ESS, bulk and tail** (want > 1000) — how many *effectively independent* draws you have, once autocorrelation is accounted for. 8,000 sticky draws can be worth 200 independent ones. Bulk ESS governs the reliability of means and medians; tail ESS governs interval edges, which is why a fit can look fine but have untrustworthy 90% intervals.
- **Divergences** (want exactly 0) — the sampler hit a region where its numerical integration broke down and had to reject the trajectory. Crucially, this biases the result rather than just adding noise: the divergent regions are systematically under-visited, so the posterior you keep is the wrong shape. The classic cause is a *funnel* — a variance parameter that can approach zero, making the space narrow to a spike the sampler can't get into. Fix by non-centering the parameterisation (built-ins already are) or by tightening the variance prior, which is usually the real problem: the data cannot rule out an implausibly huge group spread.
- **BFMI** (want ≥ 0.2) — whether the sampler's energy moves adequately between iterations. Low BFMI means heavy tails or a funnel that more samples won't fix; it's a signal to reparameterise or rethink priors.
- **Trace plots** — the visual version of all of the above. A healthy trace looks like a fuzzy caterpillar: chains overlapping, no trend, no long flat stretches. Traces tell you *which* parameter is sick, which the scalar diagnostics don't.

**Trap.** Raising a threshold or deleting a checker to make a fit "pass". That doesn't remove the problem, it removes your ability to see it.

## Prior and posterior predictive checks

**One-liner.** Both simulate fake datasets from the model and compare them to reality — the prior version before fitting (are these priors sane?), the posterior version after (can the fitted model reproduce what it saw?).

**Intuition.** A model is a machine for generating data. Run it forward from the priors alone and you see the range of worlds the model considers possible before evidence; if that includes only 0% and 100% accuracies, your priors are broken and you've learned it in seconds. Run it forward from the fitted posterior and overlay the real data: systematic mismatch is the model telling you which structure it's missing — bimodal observations against unimodal predictions usually means a grouping factor you left out.

**In hibayes.** `prior_predictive_plot` and `posterior_predictive_plot` checkers (`plot_proportion: true` for binomial). Both save figures to the model's diagnostics dir whether or not anyone approves them interactively.

**Trap.** Reading a passed posterior predictive check as "the model is right". It can only reveal *lack* of fit, and a flexible model can retrodict its own training data while predicting new data badly — which is what LOO is for.

## LOO, WAIC, elpd, Pareto k

**One-liner.** These estimate how well a model would predict *new* data, so you can compare candidate models without refitting on held-out sets.

**Intuition.** In-sample fit always improves as you add parameters, so it can't tell you which model to prefer. Leave-one-out cross-validation would answer the question but means refitting once per observation; PSIS-LOO approximates it from a single fit by reweighting draws. The score is elpd (expected log pointwise predictive density) — higher is better — and it comes with a standard error, which is the part people drop. A difference of 3 elpd with an SE of 5 is not a difference.

**In hibayes.** `loo` / `waic` checkers store `elpd_loo`/`elpd_waic`; `model_comparison_plot: {ic: loo}` runs the comparison with error bars. Prefer LOO: same target, better behaviour on small samples, and it self-diagnoses via **Pareto k** — any observation with k > 0.7 is one the approximation can't handle, which means it's unusually influential. Those flagged rows are information: a group that keeps showing up is often a group needing its own model term.

**Trap.** Treating the winner as the true model. Comparison ranks predictive accuracy; it doesn't establish structure, and it will happily prefer a confounded model that predicts well. The estimand is still read off the posterior of the model whose parameters answer the question.

## Effect coding versus dummy coding

**One-liner.** Two ways to turn a categorical variable into numbers; they make the same predictions but the coefficients mean different things.

**Intuition.** Under **effect coding** (the hibayes default), each level's effect is its deviation from the grand mean and the effects sum to zero: "grader H is 0.4 above average". Under **dummy coding**, one level is the reference and every other coefficient is a contrast against it: "grader H is 0.4 above grader A". Neither is more correct — but reading effect-coded numbers as if they were contrasts (or vice versa) silently misstates every result.

**In hibayes.** `effect_coding_for_main_effects` in `extract_features` and in model configs; `reference_categories` picks the dummy-coding baseline (default: alphabetically first).

**Trap.** Not saying which one you used in the write-up. Always state the coding scheme alongside the numbers.

## Pseudo-replication and exchangeability

**One-liner.** Rows that share a cause aren't independent evidence, and a model that assumes they are will be overconfident.

**Intuition.** "Exchangeable" means: before seeing the data, you have no reason to expect one observation to differ systematically from another within the same group. Five epochs of the same task are not exchangeable with five different tasks — they share the task's difficulty. Model that shared cause (aggregate to counts, or add a task level) and the uncertainty comes out honest. Ignore it and you get intervals that are too narrow, which is the failure mode that makes analyses claim differences that later evaporate.

**In hibayes.** The `groupby` processor aggregates per-sample scores into `n_correct`/`n_total` at whatever level you specify; hierarchy levels in the model absorb the rest.

**Trap.** Choosing the aggregation level for convenience rather than from the generative story. The level should follow from your answer to "what varies together, and why".

## Reporting language that survives scrutiny

- Give a posterior mean or median **with an interval**, on a scale the reader understands, and say which scale it is.
- Where the question is directional, give the direct probability: `(draws > 0).mean()` → "P(grader A is stricter than average) ≈ 0.97". This is the sentence Bayesian analysis licenses and frequentist analysis doesn't; use it.
- Say "the interval spans zero, so this data can't resolve the direction" rather than "no significant effect".
- State the priors and, if conclusions are close, whether they change under a different prior setting.
- If any diagnostic failed, that *is* the headline finding until it's fixed.
