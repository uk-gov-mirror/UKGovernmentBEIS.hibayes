# Claude Code skills for HiBayES

Skills for working with HiBayES in Claude Code. **This directory is their only home** — real files,
no symlinks, no second copy to keep in sync. They are picked up automatically when Claude Code runs
in a checkout of this repository, with no install step.

They cover the methods from both papers behind the package: [HiBayES](https://arxiv.org/abs/2505.05602)
for hierarchical modelling of eval data, and [*Quantifying Biases in LLM-as-Judge Evals*](https://openreview.net/pdf?id=q33S4QMuay)
(Dubois et al., ICML 2026) for grader/judge analyses, whose configs ship in
`examples/llm-as-a-judge-usecases/`.

| Skill | When it fires |
|---|---|
| [`running-hibayes`](running-hibayes/SKILL.md) | Installing, running, configuring, extending, or debugging an analysis: install and verification (HiBayES isn't on PyPI), the five-stage pipeline (load → process → model → check → communicate), the YAML config surface, the CLI (`hibayes-full` / `-load` / `-process` / `-model` / `-comm`), the component registries with verified built-in reference tables, letting a human watch the live TUI, custom components, the `AnalysisState` output layout, and known doc-drift traps. |
| [`hibayes-statistical-workflow`](hibayes-statistical-workflow/SKILL.md) | Statistical decisions inside that pipeline: stating an estimand (and asking when the question is too vague), likelihood choice, how much pooling — flat or hierarchical, priors and prior predictive checks, MCMC diagnostics with a remediation ladder, LOO/WAIC comparison, and reporting on interpretable scales. Ships [`references/concepts.md`](hibayes-statistical-workflow/references/concepts.md), a plain-language explanation of every concept for teaching users as the analysis proceeds. |

The two cross-reference each other; an agent doing a real analysis will typically load both.

## Using them outside a hibayes checkout

If you analyse eval data in your own repository, either copy these two directories into that
project's `.claude/skills/`, or install them as a plugin — `.claude-plugin/marketplace.json` at the
repo root lists this directory as a plugin, fetched as a sparse subdirectory so installing doesn't
pull the repo's example data:

```
/plugin marketplace add UKGovernmentBEIS/hibayes
/plugin install hibayes-skills@hibayes
```

To have changes reach you automatically, add this to `~/.claude/settings.json` (user-wide) or your
own repo's `.claude/settings.json` (shared with everyone who opens it):

```json
{
  "extraKnownMarketplaces": {
    "hibayes": {
      "source": { "source": "github", "repo": "UKGovernmentBEIS/hibayes" },
      "autoUpdate": true
    }
  },
  "enabledPlugins": { "hibayes-skills@hibayes": true }
}
```

`enabledPlugins` is what installs and enables the plugin at session start; `extraKnownMarketplaces`
registers the marketplace and turns on auto-update. Registering the marketplace alone installs
nothing. To update manually: `/plugin marketplace update hibayes` then `/reload-plugins`.

## Design notes

- **Verified against source, not just docs.** Every registered component name, config key, CLI flag,
  and default was checked against `src/hibayes/` and the configs under `examples/`. Known drift in
  the docs site (`hibayes-comm` vs `hibayes-communicate`, `data_process:` vs `process:`, interaction
  parameter naming) is called out explicitly so agents don't reproduce it.
- **Agent-first, without hiding the tool from the human.** The skills front-load what breaks
  unattended runs (`--no-tui`, `interactive: false` on the predictive-check plots) and then show how
  to give the TUI back via tmux when a human wants to watch a run or approve plots.
- **Explaining is part of the job.** Statistical decisions are supposed to reach the user with a
  plain-language reason; `references/concepts.md` supplies the explanations, and the workflow skill's
  "Teach as you work" section says how much to say to whom.
- **No thumb on the scale for hierarchical models.** Partial pooling is a tool with conditions, not a
  badge of rigour; the "How much pooling?" table covers when a flat fixed-effects fit is the better
  answer.
- **Ask when the question is vague.** A model fitted to a guessed estimand answers a question nobody
  asked, so the workflow skill requires probing for specifics unless the user has explicitly
  delegated the choice.

## Maintenance

When hibayes changes, re-verify each `SKILL.md` against `src/hibayes/` and bump its "Last reviewed"
footer. Registered component names are the `def`s directly under the `@extractor` / `@process` /
`@model` / `@checker` / `@communicate` decorators; config keys live in each stage's `*_config.py`.
Component tables, defaults, and CLI flags go stale first.
