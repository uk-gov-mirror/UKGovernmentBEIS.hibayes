---
name: hibayes-statistical-workflow
description: >-
  Use whenever making STATISTICAL decisions in a HiBayES analysis: choosing a
  model/likelihood, setting or justifying priors, interpreting prior/posterior
  predictive checks, diagnosing bad MCMC fits (divergences, R-hat, low ESS,
  BFMI), comparing models with WAIC/LOO, or writing up results. Encodes the
  principled Bayesian workflow applied to AI-eval data, and carries a
  plain-language concepts reference for explaining each step to the user.
  Triggers on "which model should I use", "are these priors OK", "the fit has
  divergences", "is the difference between models real", "how do I report
  this", "what does R-hat/elpd/partial pooling mean", "is this significant",
  or reviewing any hibayes config's model/check sections.
---

# The Bayesian workflow in HiBayES

Mechanics of configs, CLIs, and registries live in the **`running-hibayes`** skill. The methods themselves come from two papers worth citing in write-ups: [HiBayES](https://arxiv.org/abs/2505.05602) for the hierarchical eval-modelling framework, and [*Quantifying Biases in LLM-as-Judge Evals*](https://openreview.net/pdf?id=q33S4QMuay) (Dubois et al., ICML 2026) for judge/grader analyses. This skill is the judgment layer: a model does exactly what it is told, so the work is deciding what to tell it and verifying you believe the result. Never skip from "data loaded" to "fit the default model and report the forest plot" — walk the loop below, and push back (constructively) if the user asks you to skip steps like prior checks or fit diagnostics.

**Explain as you go.** Every statistical decision you make should reach the user with a one-or-two-sentence reason in plain language, not just as a config diff — they are the ones who will defend this analysis. `references/concepts.md` in this skill directory holds ready explanations of every concept below (posterior vs p-value, priors, the logit scale, partial pooling, ordinal cutpoints, each diagnostic, elpd/Pareto k, coding schemes, pseudo-replication); read it when the user needs the *why* and lift from it rather than improvising theory. See "Teach as you work" below for how much to say and when to say more.

## The loop

1. **State the estimand.** Write one sentence before touching config: *"the difference in P(solve) between model A and B, averaged over task groups"*, *"the grader × generator interaction on the latent rating scale"*. Every later choice (grouping columns, effects, plots) is derived from this sentence. **If the question is too vague to yield that sentence, stop and ask** — see below.
2. **Think generatively.** How did each row come to be? Which sources of variation intervene between "model capability" and "score in the log" — task difficulty, grader identity, epoch repeats, prompt template? Each named source becomes a candidate effect or hierarchy level; anything that repeats across rows (same task scored 5 epochs) is non-independence the model must absorb, not ignore.
3. **Choose the likelihood from the data type** (table below) — never from convenience.
4. **Prior predictive check** — simulate from the priors alone and look at the implied data. Mandatory, cheap, catches most disasters. Details below.
5. **Fit the simplest defensible model first**, then add structure one step at a time (an effect, an interaction, a hierarchy level), tagging each variant so all fits survive side by side.
6. **Diagnose every fit** (divergences, R-hat, ESS, BFMI, traces). An unconverged fit is not evidence about anything.
7. **Posterior predictive check** — can the fitted model retrodict the data it saw?
8. **Compare variants** with LOO/WAIC — as predictive-accuracy estimates, not truth detectors.
9. **Report full posteriors on an interpretable scale**, with uncertainty attached.

Steps 4–8 are literally hibayes checkers and communicators; the loop is one config with several tagged models plus an honest `check` section.

## When the question is too vague to model: ask

"Analyse these logs", "tell me what's interesting in this data", "run hibayes on this" do not determine an estimand, and a model fitted to a guessed estimand answers a question nobody asked — expensively, and with a false air of authority. **Don't silently pick one. Look at the data, then come back with specifics and ask.**

A good probe is short, concrete, and grounded in the columns actually present:

> The extracted data has 4 models × 37 tasks in 5 capability domains, 5 epochs each, plus a `refusal` flag from your custom extractor. A few different questions I could fit here:
> 1. **Model ranking** — differences in P(solve) between the four models, pooled across tasks.
> 2. **Where the differences live** — model × domain, i.e. does the ranking flip by capability area.
> 3. **Refusal behaviour** — whether refusal rate differs by model, separately from correctness.
>
> Which of those is the actual question? (Or is it something else — I'd rather fit the right thing once.)

Rules for this:

- **Load and process first, ask second.** Inspecting the columns, group sizes, and score distribution costs one cheap stage and makes the question specific instead of abstract. Ask with numbers in hand.
- **Two or three candidate estimands, not a menu of ten** — each phrased as the comparison it makes, not as a model name. Say which one you'd pick if forced, and why.
- **Flag what the data can't support** while you're there: a factor with two levels and four samples per cell won't sustain an interaction; if that's what they were hoping for, better to know before fitting.
- **Explicit license to choose is fine.** If the user says "you decide", "just explore it", "give me the standard analysis", or has already told you to work autonomously — proceed. Then state the estimand you chose and the assumptions behind it prominently in the write-up, not in a footnote, and note which other questions the same data could answer.
- **Don't hide behind the question either.** One round of clarification, then work with what you get. If they answer partially, fit the part that's determined and name what's still open.

## Teach as you work

The typical user here is a capable researcher who is not a practising Bayesian. They can evaluate your reasoning if you show it, and cannot if you only show YAML. So: **state the reason with the decision, every time, in one or two sentences of plain language, and offer to go deeper.** "I'm aggregating to counts per task before fitting, because five epochs of the same task aren't five independent observations — treating them as independent would make the intervals look tighter than the data supports. Happy to expand on that if useful." That costs a sentence and buys a user who can defend the analysis.

**Signals the user needs more scaffolding** — watch for these and step up the teaching, without being asked and without commenting on their level of knowledge:

| Signal | What it usually means | Where to go |
|---|---|---|
| "Is the difference significant?", asks for a p-value | Reaching for a frequentist frame the output doesn't have | Give the posterior probability statement instead, and explain in two sentences why it answers their question better — `concepts.md` → *Posterior* |
| "Just use the default model / whatever's standard" | No mental model of what the choice changes | Name the choice, give the one-line consequence of getting it wrong, propose a default *with* its reason |
| "What's a prior? / can we make the priors uninformative?" | Assumes a neutral prior exists | `concepts.md` → *Priors*, then the logit-scale entry; show them the prior predictive plot as the concrete artefact |
| "The interval includes zero, so there's no effect" | Reading intervals as accept/reject | Absence of resolution ≠ absence of effect — `concepts.md` → *Reporting language* |
| "Can we just drop that checker / raise the threshold?" | Sees diagnostics as bureaucracy, not bias detection | Explain what that specific diagnostic detects (divergences bias, they don't just add noise), then offer the remediation ladder |
| Averages ordinal ratings, or wants a t-test on 0–10 scores | Metric assumptions on a rating scale | `concepts.md` → *Ordinal outcomes* |
| Reports raw logit coefficients as the headline number | Scale confusion | `concepts.md` → *The logit scale*; convert to probabilities for them |
| "Why are there so many models in this config?" | Doesn't see the ladder as a comparison | Explain the baseline/variant logic in one line — the simple fits are the yardstick |
| Confusion about what a plot is showing (forest, trace, pair) | Reasonable — these are unfamiliar plots | Say what a healthy version looks like *before* interpreting theirs |

**Signals to stay terse instead**: they set priors themselves, mention elpd/PSIS/non-centering/shrinkage unprompted, or ask for specific parameterisations. Then drop to decisions and numbers — over-explaining to an expert wastes their time and reads as condescension. If you genuinely can't tell, one clarifying question up front ("how much statistical detail do you want in the write-up?") is cheaper than guessing wrong for a whole analysis.

**Three ways to teach that work better than prose:** show the artefact (a prior predictive plot makes the flat-prior trap self-evident in a way a paragraph doesn't); use their numbers ("group `crypto` has 4 samples, so partial pooling pulls its 100% estimate down to about 62%" beats a definition of shrinkage); and name the counterfactual ("without the interaction term, the model literally cannot express the claim you're testing").

Finally: teaching is not hedging. Once the question is settled, give a clear recommendation and act on it. "I'd fit the binomial with fixed model effects — here's why in one line" is teaching; a menu of five options with no recommendation is abdication. (Clarifying the *question* is different from dithering over the *method* — do the first, not the second.)

## Choosing the likelihood

| Data | Likelihood | HiBayES model | Notes |
|---|---|---|---|
| Per-sample 0/1 scores, grouped (tasks, capability areas) | Binomial on aggregated counts | `two_level_group_binomial` (partial pooling) | `groupby` processor makes `n_correct`/`n_total` first. Pooling choice: see below |
| As above, deeper nesting (task ⊂ suite ⊂ domain) | Binomial, 3 levels | `three_level_group_binomial`(`_exponential`) | |
| 0/1 scores, comparing a few named levels or several factors + interactions | Binomial regression, flat | `linear_group_binomial` | Fixed effects, no pooling — the right choice when the levels are the objects of interest (see below) |
| **Ordinal ratings (judge scores 0–10, Likert)** | **Ordered logistic** | `ordered_logistic_model` with `num_classes` | **Never model ordinal as metric.** A Gaussian/binomial on a 0–10 rating assumes 3→4 equals 7→8 and invents mass outside the scale. Estimated cutpoints are the honest treatment — explain them as bands on a latent quality scale (`references/concepts.md` → *Ordinal outcomes*). For judge/grader data specifically, start from the paper's configs (below) |
| Pairwise preference ("which response is better?") | Bernoulli logistic on the choice | custom `pairwise_logistic_model` (ships in `examples/llm-as-a-judge-usecases/custom_model.py`) | Model the *choice*, and include the confounds that drive it — response length, presentation order — as covariates rather than hoping they cancel |
| Anything else (counts, censored times, mixtures, zero-inflation) | Custom | `@model` custom NumPyro model | Follow the custom-model discipline below |

### Judge and grader questions have a worked reference

For anything of the form "do these graders agree / is this autograder biased", the methods paper is *Quantifying Biases in LLM-as-Judge Evals* ([Dubois et al., ICML 2026](https://openreview.net/pdf?id=q33S4QMuay)), and its analyses ship as runnable configs in `examples/llm-as-a-judge-usecases/`. Use them as the estimand menu when the user's judge question is vague — each is a distinct question, and naming them concretely is exactly the probe the section above asks for:

| Question | Example | Structure |
|---|---|---|
| Do graders differ in strictness? | `q1` | grader effects on the latent rating scale |
| Does the generating model matter, and do graders treat generators differently (self-preference)? | `q2` | grader + LLM main effects, grader × LLM interaction |
| Do *types* of grader (e.g. human vs model) behave differently? | `q3` | adds grader type |
| How much of the variation is the item being rated? | `q4` | adds item effects |
| Which response do judges pick, and is that driven by length? | `q5` | pairwise choice, standardised `length_diff` covariate |
| How much do raters agree at all? | `krippendorff_alpha` communicator | inter-rater agreement, with a bias-corrected α |

The ladder discipline below is already baked into those configs (each fits interaction and no-interaction variants), so they double as a template for build-up-don't-build-in.

### How much pooling? (hierarchical is a tool, not a badge)

Three options, none of them automatically right: *complete pooling* (one grand mean — pretends groups are identical), *no pooling* / flat fixed effects (an independent estimate per level), *partial pooling* (hierarchical — each group shrinks toward the population mean in proportion to its evidence). Choose from the estimand and the group structure, and say which you chose and why:

| Situation | Reach for | Why |
|---|---|---|
| Many groups (roughly ≥ 8–10), several of them small, and you care about the *population* of groups or about per-group estimates that don't overreact to noise | **Hierarchical** (`two_level_group_binomial`, `three_level_*`) | Shrinkage is what stops a 4/4 group being reported at 100%, and the learned `sigma_group` is itself an answer ("how much do tasks vary?") |
| A handful of named levels you want to compare *as themselves* — model A vs B vs C, three graders, treatment vs control | **Flat fixed effects** (`linear_group_binomial`, or `ordered_logistic_model` with `main_effects`) | These are the comparison, not a sample from a population. Shrinking them toward each other biases exactly the contrast you were asked about, and with < ~5 levels the variance parameter is barely identified anyway |
| Plenty of data in every group | **Either** — often flat, for simplicity | Shrinkage becomes negligible; the hierarchy costs you fitting complexity and a prior to defend for no change in conclusions |
| Nested structure that matters (task ⊂ suite ⊂ domain) and enough groups per level | **Hierarchical, multi-level** | Non-independence at each level has to go somewhere |
| Hierarchy fits badly (divergences that survive the remediation ladder, `sigma_group` posterior pinned at ~0 or wandering) with few groups | **Flat**, and say so | A variance parameter the data can't identify is a fitting problem, not extra rigour |

So: **a flat model is a perfectly respectable answer** — often the *right* one for the common eval question "which of these N models is better", where N is small and the models are the objects of interest rather than a random draw. Don't add a hierarchy to look sophisticated, and don't apologise for a fixed-effects fit that matches the estimand. Where the choice is genuinely close, fit both (they're two entries in the same `model.models` list), compare with LOO, and check whether the conclusion even depends on it — if it doesn't, say that and pick the simpler one.

What is *not* optional is dealing with non-independence somehow: repeated measurements of the same unit have to be aggregated or modelled (next paragraph) whichever pooling regime you pick. Worth explaining shrinkage to the user with their own smallest group when you do use it, since it sounds like a fudge until you see the alternative (`references/concepts.md` → *Partial pooling*).

**Pseudo-replication check.** Inspect logs often have multiple epochs per task and one row per sample. If you `groupby` over `[model, task]`, epochs are correctly pooled into `n_total`. If you skip grouping and model rows as independent Bernoulli trials while tasks repeat, you'll understate uncertainty. Make the aggregation level match the exchangeability story from step 2.

## Priors: never accept them silently

The core discipline: priors are chosen by **simulating what they imply about observable data**, not by tradition and not by making them "uninformative". There is no neutral prior — a prior that looks agnostic about a *parameter* usually implies something absurd about the *outcome*, and the outcome is the thing anyone has intuitions about. Say this to users who ask for uninformative priors, then show them the prior predictive plot.

**The flat-prior trap on the logit scale.** All the binomial/ordinal models here operate through a logit link, and "wide" is not "neutral" there: a `Normal(0, 10)` prior on an intercept implies P(success) is almost certainly ~0 or ~1 — an absurd, strongly bimodal belief. Rules of thumb: `Normal(0, 1.5)` on a logit intercept is roughly uniform on probability; effect scales of 0.3–1 are generous for realistic differences (an effect of 2 is an ~0.5→0.88 probability swing); hibayes defaults (e.g. `prior_main_effects_scale: 0.5` in `linear_group_binomial`, `prior_sigma_group_scale: 0.1` in `two_level_group_binomial`) are deliberately regularizing — question them upward only with a reason, and prefer *slightly-too-tight* to *absurdly-loose*: weakly regularizing priors reduce overfitting the same way information criteria say they should.

**Prior predictive check, the hibayes way** — keep these first in `check.checkers`:

```yaml
check:
  checkers:
    - prior_predictive_plot: {plot_proportion: true, interactive: false}   # binomial models
    # ordinal models: omit plot_proportion; you're checking the implied rating distribution
```

Read the saved plot against three questions: (a) does the implied outcome distribution cover the plausible range *without* piling mass on impossible or absurd values (all-0%/all-100% spikes = logit trap); (b) are extreme-but-possible outcomes rare rather than dominant; (c) would a colleague laugh at any simulated dataset? If yes, tighten `prior_*` kwargs in the model config and re-check — this iteration costs seconds since nothing is fitted yet. **Prior sensitivity doubles as a robustness check**: fitting the same model under 2–3 tagged prior settings (see the example in `running-hibayes`) shows whether conclusions are prior-driven; if they are, say so in the write-up.

## Build up, don't build in

Fit a ladder of tagged variants in one config rather than the full model first:

```yaml
model:
  models:
    - name: ordered_logistic_model
      config: {tag: graders_only, main_effects: [grader], num_classes: 11}
    - name: ordered_logistic_model
      config: {tag: main_effects, main_effects: [grader, LLM], num_classes: 11}
    - name: ordered_logistic_model
      config: {tag: interaction, main_effects: [grader, LLM], interactions: [[grader, LLM]], num_classes: 11}
```

The simple models are not throwaways: they are the baselines that tell you what the added structure buys (via LOO/WAIC below), and when a complex model misbehaves, the point in the ladder where fitting broke localizes the problem. If the estimand is an interaction ("do autograders favour their own generations?"), the no-interaction variant is the null story — fit both, always.

**Validate on simulated data when the model is custom or the stakes are high.** Generate a synthetic dataset from known parameter values (a short pandas/NumPy script → CSV → `extracted_data:` loader), fit, and confirm the posteriors cover the truth. A model that can't recover parameters it generated has no business seeing real data.

## Diagnostics: what failure means and what to do

Run the full battery on every fit (`r_hat`, `ess_bulk`, `ess_tail`, `divergences`, `bfmi`, plus `trace_plot: {best_model: false}` in communicators). Remediation, in the order to try it:

| Symptom | Meaning | Fix, in order |
|---|---|---|
| `divergences` fail (any > 0) | The sampler cannot follow the posterior geometry; estimates are **biased, not just noisy**. Classic cause: centered hierarchical parameterizations with weak data ("funnels") | (1) raise `fit.target_accept` to 0.99; (2) non-center the parameterization — built-ins already are (`z_group` + `sigma * z`), so for them go to (3) tighten the variance prior (`prior_sigma_group_scale`), which is usually the real problem: the data can't rule out huge group spread |
| `r_hat` fail (> 1.01) | Chains disagree — not sampling the same distribution | more `warmup`; check trace plot for multimodality or a stuck chain; consider whether the model is unidentified (e.g. dummy-coded effects plus a free intercept with no constraint) |
| `ess_bulk`/`ess_tail` fail (< 1000) | Autocorrelated chains; posterior summaries (esp. intervals, from tail ESS) unreliable | more `samples`; if ESS stays disproportionately low vs sample count, reparameterize — it's geometry, not length |
| `bfmi` fail (< 0.2) | Energy transitions too small; heavy-tailed or funnel-shaped posterior | reparameterize / rethink priors; more samples won't fix it |
| Trace plots not "hairy caterpillars" (trending, sticky, chains at different levels) | Visual confirmation of any of the above | as above — traces tell you *which* parameter is sick |

Never report estimates from a fit with failing convergence checks, and never "fix" a diagnostic by deleting the checker or raising its threshold without a stated justification. If the user asks to do that, flag the bias risk and offer the remediation ladder instead.

## Posterior predictive: retrodiction, honestly

`posterior_predictive_plot: {plot_proportion: true, interactive: false}` (proportion for binomial; raw scale for ordinal). You're asking: *does data simulated from the fitted model look like the data it was fitted to?* Systematic misses are the model telling you what structure it's missing — e.g. observed proportions bimodal while predictions are unimodal (a missing grouping factor), or ordinal categories at the scale ends over/under-predicted (cutpoint priors too tight, or graders using the scale in a way main effects can't express). A posterior predictive check can only reveal *lack* of fit — passing it doesn't make the model right, and a flexible-enough model retrodicts anything; that's what out-of-sample comparison is for.

## Model comparison: prediction, not truth

`loo` and `waic` checkers store `elpd_loo`/`elpd_waic` per model; `model_comparison_plot: {ic: loo}` runs `az.compare` across every fitted variant. Discipline:

- **elpd is an estimate of out-of-sample predictive accuracy, with a standard error.** Higher is better; treat differences smaller than ~2× their SE as "no meaningful difference" — the comparison plot draws the error bars, use them. Never rank on point elpd alone.
- **Prefer LOO (PSIS) over WAIC**; same target, better small-sample behaviour and it comes with its own alarm: **Pareto k > 0.7** flags observations where the approximation fails — which are precisely your influential outliers. The `loo` checker fails on these; look at *which* rows they are (`diagnostics["pareto_k"]` is pointwise, aligned with the grouped data) before trusting the comparison. Recurring high-k groups often deserve their own model term, not deletion.
- **Comparison is not selection.** If `interaction` beats `no_interaction` by a solid elpd margin, that supports the interaction *predictively*; the estimand is still read off the interaction model's posterior (is the `grader_LLM_effects` interaction's interval away from 0?), not off the ranking. And a causal claim ("the autograder favours its own outputs *because*...") needs the generative story from step 2, not an information criterion — IC will happily prefer a confounded model that predicts well.
- **`get_best_model` semantics**: communicators with `best_model: true` pick max `elpd_waic` ignoring SE. Fine as a plotting default; not a substitute for reading the comparison plot. (And it requires the `waic` checker to have run — see `running-hibayes`.)

## Interpreting and reporting

- **Know your scale.** All binomial-model parameters live on the logit scale. Report effects there ("+0.8 logits") only alongside the outcome scale: `forest_plot: {transform: true}` applies the model's link function, or transform draws yourself from `inference_data` (`az.summary` on `jax.nn.sigmoid`-mapped quantities). For ordered logistic, effects are shifts of a latent scale — report implied category-probability changes, not raw coefficients.
- **Effect coding** (`effect_coding_for_main_effects: true`, the model default) makes each level's `*_effects` a deviation from the grand mean, summing to zero — "grader H scores 0.4 latent units above average". Dummy coding makes them contrasts against the reference level (alphabetically first unless `reference_categories` says otherwise in `extract_features`). State which one you're reading; they answer different questions.
- **Report distributions, not verdicts.** Posterior mean/median with a compatibility interval (`summary_table`, forest plots with `vertical_line: 0`), plus a direct probability statement for the estimand: `(draws > 0).mean()` → "P(grader A is stricter than average) ≈ 0.97". Avoid "significant"/"not significant" framing; an interval touching zero is weak evidence, not proof of absence.
- **Standardise continuous predictors** (`standardise: true` in `extract_features`) so slopes read "per SD" and default priors stay sensible.
- **Reproducibility**: `fit.seed` defaults to 0 — set it explicitly per analysis, and let DVC + the saved `AnalysisState` (config JSONs live next to each model) carry the provenance.

## Custom models: the same discipline applies

When the built-ins don't fit the generative story, write a NumPyro model (mechanics in `running-hibayes`), and inherit the house style: **non-centered hierarchies** (`z ~ Normal(0,1)`, `effect = mu + sigma * z`), **explicit weakly-regularizing priors as builder kwargs** (so the config can sweep them), `numpyro.deterministic` for every quantity you'll plot, `check_features` up front. Then run it through the *entire* loop — prior predictive first, parameter recovery on simulated data, full diagnostics — before believing anything it says about real data.

## Pre-report checklist

Before presenting results, confirm — and if any box fails, that's the finding to report first:

- [ ] Estimand stated — confirmed with the user, or chosen under explicit license and flagged as chosen; model structure follows from a generative story (not from "it's the default")
- [ ] Ordinal outcomes got cutpoints; the pooling choice (flat or hierarchical) is stated with its reason, and repeated measurements are aggregated or modelled either way
- [ ] Prior predictive plot inspected and priors adjusted/justified; sensitivity checked if conclusions are close
- [ ] Zero divergences; R-hat, ESS, BFMI pass; traces inspected
- [ ] Posterior predictive plot inspected against the observed data
- [ ] Variants compared with LOO/WAIC *with SEs*; Pareto k warnings investigated
- [ ] Effects reported on an interpretable scale with intervals, coding scheme stated, no bare "significant"
- [ ] The user has been told *why* each substantive choice was made, in language they can repeat to someone else — and knows which of the conclusions are prior- or assumption-dependent

---

Last reviewed: 2026-07-27 against hibayes 1.0.0
