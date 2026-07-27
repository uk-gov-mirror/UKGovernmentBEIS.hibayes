---
name: running-hibayes
description: >-
  Use whenever the user wants to run, configure, extend, or debug a HiBayES
  analysis — hierarchical Bayesian statistical modelling of Inspect AI eval
  results (or any tabular eval data). Covers the five-stage pipeline (load →
  process → model → check → communicate), the YAML config surface, the CLI
  (`hibayes-full`, `hibayes-load`, `hibayes-process`, `hibayes-model`,
  `hibayes-comm`), the component registries (extractors, processors, models,
  checkers, communicators), and the AnalysisState output layout. Triggers on
  "analyse these eval scores with hibayes", "is model X actually better than
  Y", "fit a hierarchical model to my eval results", "add a custom
  extractor/processor/model", installing or setting up hibayes, watching a
  run's TUI, or any config.yaml editing for hibayes.
---

# Running HiBayES analyses

HiBayES ([paper](https://arxiv.org/abs/2505.05602)) is AISI's Python package for statistical modelling of AI-evaluation data. Its judge-specific modelling comes from a second paper, *Quantifying Biases in LLM-as-Judge Evals* ([Dubois et al., ICML 2026](https://openreview.net/pdf?id=q33S4QMuay)), whose analyses ship as `examples/llm-as-a-judge-usecases/` — the reference to reach for on any grader/judge question. It answers questions raw accuracy tables can't: *is model A actually better than model B given the noise*, *how much do graders disagree*, *does an autograder favour its own generations*. It reads Inspect AI eval logs (or pre-extracted CSV/parquet) and fits hierarchical Bayesian GLMs with NumPyro/JAX, with ArviZ for diagnostics.

This skill covers the **mechanics**: installation, pipeline, config, CLI, extension points. For the **statistical judgment** — which model to reach for, how to set priors, what to do when diagnostics fail, how to compare and report models — use the companion **`hibayes-statistical-workflow`** skill. Read both before building an analysis; mechanics without the workflow produces fits nobody should trust.

**Say what you're doing and why.** Config edits are opaque to most users — a processor list and a set of `prior_*` kwargs don't explain themselves. When you add or change something, give the one-line reason alongside it ("`groupby` first, so five epochs of one task count as one binomial trial rather than five independent ones"). The `hibayes-statistical-workflow` skill carries a `references/concepts.md` with plain-language explanations for the statistical side, and a "Teach as you work" section on how much to explain to whom.

**Don't guess the question.** A config encodes a specific comparison, so a vague request ("analyse these logs") doesn't determine one. Run load/process to see what's actually in the data, then ask which comparison matters, offering two or three concrete candidates — unless the user has explicitly handed you the choice ("you decide", "just explore it"), in which case proceed and say prominently what you picked. Details in `hibayes-statistical-workflow` → "When the question is too vague to model".

## Installing HiBayES

Python **≥ 3.12** required. `pip show hibayes` (or `python -c "from importlib.metadata import version; print(version('hibayes'))"` — the package has no `__version__` attribute) tells you whether it's already installed and where; check before installing anything.

**Not on PyPI** — every install goes through the GitHub repo, so `pip install hibayes` fails and `uv add hibayes` won't resolve.

```bash
# In an existing uv project — add it as a dependency
uv add "git+https://github.com/UKGovernmentBEIS/hibayes.git"

# Standalone, without cloning
uv pip install "git+https://github.com/UKGovernmentBEIS/hibayes.git"

# From a clone (what you want if you'll read the source, run the bundled examples, or patch hibayes)
git clone https://github.com/UKGovernmentBEIS/hibayes.git && cd hibayes
uv venv .venv && uv sync            # uv sync pins the locked dependency set
uv pip install -e .                 # editable
```

- **GPU:** `uv pip install "git+https://github.com/UKGovernmentBEIS/hibayes.git[gpu]"` (or `.[gpu]` from a clone) — pulls `numpyro[cuda]`. Verify with `python -c "import jax; print(jax.devices())"` — if that prints `CpuDevice`, JAX isn't seeing the GPU and `platform.device_type: gpu` will not help. MCMC on eval-sized data is usually fine on CPU; don't chase GPU unless fits are genuinely slow.
- **Dev extras:** `uv pip install -e ".[dev]"` then `uv run pre-commit install` (contributing to hibayes itself).
- **Verify the install:** `hibayes-full --help` should list `--config`, `--out`, `--no-tui`. If the console scripts aren't found but the import works, the venv isn't active — prefix commands with `uv run`.
- **In a uv-managed project, always `uv run hibayes-*`** rather than the bare command; the bundled examples and `dvc.yaml` stages assume it.

The examples in the hibayes repo (`examples/hibayes-usecases/`, `examples/llm-as-a-judge-usecases/`) are the fastest way to confirm an install end to end — each is a config plus data, runnable with `uv run hibayes-full --config files/config.yaml --out .output`.

**Source of truth.** The docs site (https://ukgovernmentbeis.github.io/hibayes/) is useful but has drift in places (this skill flags known instances). When exactness matters — registered component names, config keys, parameter defaults — read the installed source: `pip show hibayes` for the location, then `src/hibayes/{load,process,model,check,communicate}/`. Component registration is decorator-based, so every registered name is a `def` directly under `@extractor`, `@process`, `@model`, `@checker`, or `@communicate` in those modules. Check the user's installed version against the repo before relying on newer features.

## The pipeline

Five stages, one flowing state object. Each stage reads its section of a single `config.yaml` and adds to an `AnalysisState` that is saved to the output dir after each stage (checkpointed, resumable):

| Stage | CLI | Config key | What it does |
|---|---|---|---|
| Load | `hibayes-load` | `data_loader` | Extract one pandas row per Inspect sample from `.eval`/`.json` logs (local or S3), or read pre-extracted CSV/parquet/JSONL |
| Process | `hibayes-process` | `data_process` | Pandas transforms → JAX feature arrays, coords, dims. Raw data stays immutable; a new `processed_data` is created |
| Model | `hibayes-model` | `model` + `check` + `platform` | Fit each configured NumPyro model by MCMC (NUTS default); run before/after checkers on each fit |
| Check | (runs inside model stage) | `check` | Prior/posterior predictive plots, R-hat, ESS, divergences, BFMI, LOO, WAIC per model |
| Communicate | `hibayes-comm` | `communicate` | Forest/trace/pair/comparison plots and summary tables from the fitted models |

The split matters for explaining runs to users, not just for resuming them: **load** answers "what counts as one observation", **process** answers "what structure does the model get to see" (this is where a modelling mistake usually originates — a grouping column not aggregated, a factor never extracted), **model** answers "what does the data imply about the parameters", **check** answers "should we believe this fit at all", and **communicate** answers "what do we tell people". When something looks wrong in the results, walk back up that list — the culprit is upstream of where it showed up more often than not.

```bash
# Full pipeline in one shot
hibayes-full --config config.yaml --out .output

# Or stage by stage (later stages take the prior stage's output dir)
hibayes-load    --config config.yaml --out .output/load
hibayes-process --config config.yaml --analysis-state .output/load --out .output/process
hibayes-model   --config config.yaml --analysis-state .output/process --out .output/model
hibayes-comm    --config config.yaml --analysis-state .output/model --out .output/comm
```

Two CLI traps:

- **The communicate command is `hibayes-comm`**, not `hibayes-communicate` (`pyproject.toml` `[project.scripts]` is authoritative).
- **Running unattended (as an agent, in CI, via DVC): pass `--no-tui` and disable interactive checkers.** By default the CLI launches a Textual TUI, and the `prior_predictive_plot` / `posterior_predictive_plot` checkers default to `interactive: true`, which waits for a human to approve each plot. For any non-interactive run, add `--no-tui` and set `interactive: false` on both plot checkers — the plots are still saved to the model's output dir for review afterwards. (Since hibayes 1.0.0 the checkers detect a non-promptable display, log a warning, and return `NA` rather than hanging forever — but you still get a warning-shaped hole where a check should be, so set the flag.) `--no-frequent-save` exists too but the default (save after every model fit and communicator) is usually what you want: crash recovery is nearly free thanks to incremental saves.

## Letting the human watch the TUI

The TUI is genuinely useful — live logs, per-model progress, an in-terminal ASCII predictive-check plot, and inline Yes/No approval prompts — and it's exactly what an agent-driven `--no-tui` run hides from the user. Two ways to give it back:

**1. Hand the command over (highest fidelity).** Write the config, then let the user run it in their own terminal:

```bash
uv run hibayes-full --config config.yaml --out .output
```

In Claude Code they can prefix it with `!` to run it in-session, but full-screen Textual apps want a real TTY — output comes back captured and redrawn, so a separate terminal is the better experience.

**2. Run it in tmux and let them attach (best of both).** This works, including the interactive approval prompts — the pipeline runs under your control while the user watches or drives:

```bash
tmux new-session -d -s hibayes -x 200 -y 50 "uv run hibayes-full --config config.yaml --out .output"
# tell the user: tmux attach -t hibayes     (detach with Ctrl-b then d)
tmux capture-pane -p -t hibayes             # you can read the current screen too
```

Set the width/height explicitly (`-x`/`-y`): a detached session defaults to 80×24 and the TUI's panels get cramped. Keep `interactive: true` on the predictive-check plot checkers *only* if a human will actually be attached to answer — the fit blocks at the prompt until they do (Ctrl-C in the TUI aborts pending prompts and exits). Note the TUI clears the screen on exit, so `capture-pane` after completion shows an empty pane; read `.output/logs/` for the record.

**3. `--no-tui` for anything else.** The classic Rich display prints linearly, so it survives log capture, CI, and DVC. Every plot still lands as a PNG under `.output/models/<name>/diagnostics/` — for a user who wants to *inspect* rather than *watch*, pointing them at those files after the run beats the live view anyway.

How "can I be prompted?" is decided differs between the two displays, which is why interactive checks behave differently depending on how you launched: the TUI can prompt whenever the app is running, while the Rich display prompts on **stdin and therefore requires a TTY** (`sys.stdin.isatty()`). So `--no-tui` from a tool call or CI is non-promptable — checks warn and return `NA` — but `--no-tui` in a human's own terminal *will* stop and ask on stdin. Set `interactive: false` explicitly rather than relying on either behaviour.

## Config anatomy

One YAML file drives everything. **Top-level keys are exactly**: `data_loader`, `data_process`, `model`, `check`, `communicate`, `platform` (all optional — sensible defaults fill in). Two documented spellings that are wrong: the processors doc page uses `process:` and some snippets use `extractors.enabled: [base]`; the real key is `data_process:` and the real extractor name is `base_extractor`. Section keys *are* validated with did-you-mean suggestions for `model`/`check`/`communicate`, but a mistyped top-level key is silently ignored and you get defaults.

A complete, real-shaped config (hierarchical binomial accuracy across task groups, fitted under two prior settings):

```yaml
data_loader:
  paths:
    files_to_process:        # .eval/.json files, dirs of logs, or a .txt listing them; local or s3://
      - path/to/logs/
  extractors:
    path: files/custom_extractor.py   # optional: file(s) with your @extractor defs
    enabled:
      - base_extractor
      - my_custom_extractor: {keyword: refuse}   # kwargs form

data_process:
  processors:
    - map_columns: {column_mapping: {task: group}}          # the *_group_* models want a literal "group" column
    - groupby: {groupby_columns: [group]}                   # aggregates score rows -> n_correct, n_total
    - extract_features: {categorical_features: [group], continuous_features: [n_total]}
    - extract_observed_feature: {feature_name: n_correct}   # -> features["obs"]

model:
  models:
    - name: two_level_group_binomial            # built-in; tag distinguishes reruns
      config:
        tag: base
        prior_sigma_group_scale: 0.1            # unknown config keys pass through to the model builder
        fit: {samples: 2000, warmup: 1000, chains: 4, seed: 0, target_accept: 0.95}
    - name: two_level_group_binomial
      config: {tag: looser_prior, prior_sigma_group_scale: 0.4}

check:
  checkers:
    - prior_predictive_plot: {plot_proportion: true, interactive: false}
    - r_hat
    - divergences
    - ess_bulk
    - ess_tail
    - bfmi
    - loo
    - waic
    - posterior_predictive_plot: {plot_proportion: true, interactive: false}

communicate:
  communicators:
    - forest_plot: {vars: [group_effects], combined: true, vertical_line: 0}
    - trace_plot: {best_model: false}
    - summary_table
    - model_comparison_plot: {ic: waic}

platform:
  device_type: cpu          # or gpu; chain_method: parallel|sequential|vectorized
```

Everywhere a component is listed you can write either the bare name (defaults) or `name: {kwarg: value}`. Every section takes an optional `path:` (string or list) pointing at Python file(s) whose decorated functions get registered before names are resolved — that's the whole custom-component story.

**Data sources are mutually exclusive**: `paths.files_to_process` (Inspect logs, run through extractors) XOR `paths.extracted_data` (pre-extracted `.csv`/`.parquet`/`.jsonl`/`.json` read straight into the dataframe, extractors skipped). Use `extracted_data` when the data didn't come from Inspect or was already flattened.

### Defaults when a section is omitted

- `data_process`: `extract_observed_feature` (expects a `score` column) + `extract_features` with default args — almost never what you want; write this section.
- `model`: fits `two_level_group_binomial` with default priors.
- `check`: nine default checkers (every built-in except `prior_predictive_check`) **with `interactive: true`** — will block an unattended run (see above).
- `communicate`: `trace_plot`, `forest_plot`, `pair_plot`, `model_comparison_plot`, `summary_table`.
- `data_loader.extractors`: `base_extractor` only.

## Built-in components

Registered names, verified against source. Configure any of them with kwargs using the `name: {kwarg: value}` form.

**Extractors** (`data_loader.extractors.enabled`; each returns dict-of-columns per Inspect sample):

| Name | Extracts |
|---|---|
| `base_extractor` | score (C/I → 1/0), target, model, model_raw, dataset, task (sample id), epoch, num_messages, log_path |
| `token_extractor` | total/input/output/cache-read/cache-write token counts |
| `message_count_extractor` | per-role message counts (`include_system/user/assistant` flags) |
| `tools_extractor` | tools used, from the eval plan |
| `metadata_field_extractor` | any nested metadata field via dot-path (`field_path`, `default_value`, `rename_to`) |
| `score_normalizer` | score normalisation with `custom_mapping` |
| `cyber_extractor` | cyber-eval challenge metadata |

**Processors** (`data_process.processors`; run in order, each mutates `processed_data`/`features`):

| Name | Purpose | Key params |
|---|---|---|
| `map_columns` | rename columns | `column_mapping` |
| `drop_rows_with_missing_features` | drop rows with NAs in given columns | `feature_names` (default `[model, task]`) — **not** `features`, despite the getting-started doc's example |
| `groupby` | aggregate row-per-sample scores into `n_correct`/`n_total` for binomial models | `groupby_columns` |
| `extract_observed_feature` | one column → `features["obs"]` | `feature_name`, `test` |
| `extract_features` | categorical → `{f}_index`+`num_{f}`+coords/dims; continuous → arrays | `categorical_features`, `continuous_features`, `interactions` (**bool** here), `effect_coding_for_main_effects`, `standardise`, `test`, `reference_categories`, `category_order` |
| `merge_scout_results` | join Inspect Scout scan verdicts onto the data | `scout_scan_path`, `scanner_name`, `join_on_left/right`, `prefix` |

**Models** (`model.models[].name`) — statistical guidance on choosing between these lives in `hibayes-statistical-workflow`. `two_level_group_binomial` is the *config default*, which is not the same as the right choice: a flat `linear_group_binomial` is often the better fit for "compare these few named models/graders", where the levels are the objects of interest rather than a sample from a population.

| Name | Likelihood / structure | Model-level params |
|---|---|---|
| `two_level_group_binomial` | hierarchical binomial, HalfNormal variance priors, non-centered | `prior_mu_overall_{loc,scale}`, `prior_sigma_{overall,group}_scale` |
| `simplified_group_binomial_exponential` | hierarchical binomial, Exponential variance prior | `prior_mu_overall_{loc,scale}`, `prior_sigma_group_rate` |
| `three_level_group_binomial`(`_exponential`) | 3-level nesting (group → subgroup → subsubgroup) | analogous per-level priors |
| `linear_group_binomial` | non-hierarchical binomial regression, fixed effects | `main_effects`, `interactions` (**list of pairs**), `effect_coding_for_main_effects`, `prior_*` |
| `ordered_logistic_model` | ordinal outcomes (e.g. 0–10 judge ratings) via ordered cutpoints | `main_effects`, `continuous_effects`, `interactions`, `num_classes`, cutpoint priors, `effect_coding_for_main_effects` |

Per-model `config` also takes: `tag` (unique label when fitting the same model twice), `fit` (see below), `link_function` (`logit` default; `identity`/`probit`/`cloglog`), `main_effect_params` (which params default plots show). **Any other key is forwarded as a kwarg to the model-builder function** — that's how `main_effects` and `prior_*` reach the model, and also why a typo'd prior name errors as an unexpected-keyword TypeError at build time rather than a config validation message.

`fit` keys (NumPyro MCMC): `method` (NUTS|HMC), `samples` (2000), `warmup` (1000), `chains` (4), `seed` (0), `target_accept` (0.95), `max_tree_depth` (10), `init_strategy` (median|mean|uniform), `progress_bar`.

**Checkers** (`check.checkers`; each returns pass/fail/error/NA per model):

| Name | When | Pass condition (default) |
|---|---|---|
| `prior_predictive_check` | before fit | NA — just stores prior predictive samples |
| `prior_predictive_plot` | before fit | user approval (`interactive: true`) or NA; `plot_proportion: true` for binomial |
| `r_hat` | after | all R-hat < 1.01 (`threshold`) |
| `ess_bulk` / `ess_tail` | after | all ESS > 1000 (`threshold`) |
| `divergences` | after | count ≤ 0 (`threshold`) |
| `bfmi` | after | all chains ≥ 0.20 (`threshold`) |
| `loo` | after | no Pareto k > 0.7 (`reff_threshold`); stores `elpd_loo` |
| `waic` | after | NA — stores `elpd_waic` for comparison |
| `posterior_predictive_plot` | after | user approval; `num_samples`, `plot_proportion`, `plot_kwargs` |

**Communicators** (`communicate.communicators`):

| Name | Output | Key params |
|---|---|---|
| `forest_plot` | credible-interval forest plot | `vars`, `vertical_line`, `combined`, `transform` (apply link function → probability scale), `best_model` |
| `trace_plot` | MCMC traces | `vars`, `best_model` |
| `pair_plot` | pairwise parameter KDEs | `vars`, `best_model` |
| `model_comparison_plot` | `az.compare` across fitted models | `ic: waic|loo`; needs ≥2 fitted models |
| `summary_table` | posterior summary table (CSV) | `vars`, `round_to`, `best_model` |

**`best_model: true` is the default** for forest/trace/pair/summary. It calls `state.get_best_model(with_respect_to="elpd_waic")` (highest elpd wins), which **raises if no model has that diagnostic** — so keep `waic` in your checkers, or set `best_model: false` to plot every model. With a single model, still keep `waic` (or set `best_model: false`).

## Naming conventions (how features, params, and plots connect)

`extract_features: {categorical_features: [grader]}` on a column `grader` produces features `grader_index` + `num_grader`, coord `grader`, and the built-in models then name their parameters `grader_effects` (with `grader_effects_constrained`/`_raw` as sampling-space nuisance versions under effect/dummy coding). Interaction parameters are named `{a}_{b}_effects` (e.g. `grader_LLM_effects` — the getting-started doc's naming-conventions section says `{a}_{b}_interaction`, which is drift; `create_interaction_effects` in `model/utils.py` is authoritative). These are the strings you pass to `forest_plot: {vars: [...]}` and `summary_table`. The hierarchical `*_group_binomial` models hardcode the feature names `obs`, `n_total`, `group_index`, `num_group` (plus `subgroup`/`subsubgroup` for three-level) — hence the `map_columns` rename of your grouping column to literally `group` in the recipe above.

**Trap — two different `interactions`:** in `extract_features` it's a **boolean** (extract all pairwise interaction dims); in a model config it's a **list of pairs**, YAML `interactions: [["grader", "LLM"]]`. Enable it in both places or the model will fail its `check_features` with a missing-feature error (those errors list what's missing and what exists — read them, they're good).

## Canonical recipes

**1. "Which model is better?" from Inspect logs.** The AI model has to enter the statistical model as a *feature* — this is the mistake to avoid: the `*_group_binomial` models read only `group_index` (plus `subgroup`/`subsubgroup`), so grouping by `[model, group]` while extracting only `group` as a feature leaves both AI models sharing one group index and their difference unestimated. Two correct patterns:

```yaml
# (a) Flat — model effects estimated directly. Usually what "compare these N models" means.
data_process:
  processors:
    - map_columns: {column_mapping: {task: group}}
    - groupby: {groupby_columns: [model, group]}
    - extract_features: {categorical_features: [model, group], continuous_features: [n_total]}
    - extract_observed_feature: {feature_name: n_correct}
model:
  models:
    - name: linear_group_binomial
      config: {tag: _flat, main_effects: [model, group], fit: {seed: 0}}
```

Effect-coded `model_effects[<name>]` are then deviations from the grand mean; the contrast between two models is the difference of their draws (and `group_effects` absorbs task difficulty). Add `interactions: [[model, group]]` — plus `interactions: true` in `extract_features` — only if the question is whether the ranking flips by task group.

```yaml
# (b) Hierarchical, with the model as the top level (the pattern in examples/hibayes-usecases/usecase2)
- map_columns: {column_mapping: {model: group, domain: subgroup, sub_domain: subsubgroup}}
- groupby: {groupby_columns: [group, subgroup, subsubgroup]}
- extract_features: {categorical_features: [group, subgroup, subsubgroup], continuous_features: [n_total]}
# then: three_level_group_binomial
```

Pick (a) for a handful of named models you're comparing as themselves, (b) when there's real nesting below and you want per-level variance estimates — `hibayes-statistical-workflow` → "How much pooling?" has the decision criteria.

**2. Ordinal judge scores from a CSV** (0–10 ratings; never model these as metric — see `hibayes-statistical-workflow`):

```yaml
data_loader:
  paths:
    extracted_data: [data/ratings.csv]
data_process:
  processors:
    - extract_observed_feature: {feature_name: score}
    - extract_features: {categorical_features: [grader, LLM], effect_coding_for_main_effects: true, interactions: true}
model:
  models:
    - name: ordered_logistic_model
      config: {tag: with_interaction, main_effects: [grader, LLM], interactions: [[grader, LLM]], num_classes: 11}
    - name: ordered_logistic_model
      config: {tag: no_interaction, main_effects: [grader, LLM], num_classes: 11}
```

`num_classes` must cover the score range (0–10 → 11) and scores must be integers starting at 0.

**Don't start a judge analysis from scratch.** `examples/llm-as-a-judge-usecases/` is the config set from *Quantifying Biases in LLM-as-Judge Evals* ([paper](https://openreview.net/pdf?id=q33S4QMuay)), and it's a ready-made ladder — `q1` grader effects alone, `q2` adds the generating LLM plus the grader × LLM interaction, `q3` adds grader type, `q4` adds item effects, `q5` switches to pairwise preference (`first_is_chosen` binary outcome with a standardised `length_diff` covariate for length confounds). Read the config whose question is closest to the user's and adapt it, rather than deriving the feature/effect wiring again.

That directory also ships the paper's custom components, which are the best worked examples in the repo (register them with `path:` — they are **not** built-ins): `hierarchical_ordered_logistic_model` and `pairwise_logistic_model` in `custom_model.py`, `add_categorical_column` in `custom_processors.py`, and `cutpoint_plot` / `ordered_residuals_plot` / `category_frequency_plot` / `krippendorff_alpha` in `custom_communicators.py` — the last of these computes inter-rater agreement, including a bias-corrected α, which no built-in communicator does.

## Custom components

All five stages extend the same way: decorated builder function → outer kwargs come from YAML → inner function does the work → reference the file via `path:`. The decorators are `@extractor` (`hibayes.load`), `@process` (`hibayes.process`), `@model` (`hibayes.model`), `@checker` (`hibayes.check`, takes `when="before"|"after"`), `@communicate` (`hibayes.communicate`). Minimal custom processor:

```python
from hibayes.process import DataProcessor, process

@process
def add_categorical_column(new_column: str, source_column: str, mapping: dict, default: str = "other") -> DataProcessor:
    def _process(state, display=None):
        state.processed_data[new_column] = (
            state.processed_data[source_column].map(mapping).fillna(default)
        )
        return state          # always return the state; it flows onward
    return _process
```

Custom models must sample the likelihood site named `"obs"` with `obs=features["obs"]` (enforced by the `@model` decorator) and should call `check_features(features, [...])` (`hibayes.model.models`) up front for readable errors. Use `numpyro.deterministic("<name>_effects", ...)` for any derived quantity you'll want to plot. For the prior discipline a custom model should follow, see `hibayes-statistical-workflow`.

## Outputs: reading the AnalysisState

Everything lands in the `--out` dir:

```
.output/
├── data.parquet              # raw extracted rows
├── processed_data.parquet
├── features.pkl / coords.json / dims.json
├── logs/                     # per-stage logs
├── communicate/              # model_<model_name>_<vars>_forest.png, model_<model_name>_summary.csv, model_comparison.png, ...
└── models/<name><tag>/       # e.g. two_level_group_binomialbase — name+tag, no separator; start tags with _ if that bothers you
    ├── inference_data.nc     # ArviZ InferenceData (posterior, predictives)
    ├── model_config.json
    └── diagnostics/
        ├── diagnostics.json  # r_hat, ess, elpd_waic/loo, pareto_k, ...
        └── *.png             # prior/posterior predictive & diagnostic figures
```

For follow-up questions, don't re-run the pipeline — load the state:

```python
from pathlib import Path
from hibayes.analysis_state import AnalysisState

state = AnalysisState.load(Path(".output"))
m = state.get_model("ordered_logistic_modelwith_interaction")   # model_name = name + tag, concatenated with NO separator
m.diagnostics["elpd_waic"], m.diagnostic("r_hat")
m.inference_data                     # ArviZ InferenceData -> az.summary(...), az.plot_*
best = state.get_best_model(with_respect_to="elpd_waic")
```

`inference_data.nc` is standard ArviZ — ad-hoc analysis beyond the built-in communicators is just `arviz` on that file.

## DVC integration

Examples ship with `dvc.yaml` pipelines (stages = the four CLIs; `params` sections track the matching config keys so `dvc repro` reruns only changed stages). If the user has a `dvc.yaml`, prefer `uv run dvc repro` over invoking stages by hand; edit the config and let DVC decide what reruns.

---

Last reviewed: 2026-07-27 against hibayes 1.0.0 (source, docs, and bundled examples)
