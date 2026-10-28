# Event-aware RL for patch foraging — GitHub Pages report

This repository is for a synthetic two-action patch-foraging simulation. It compares five policies under a common `exploit` / `leave` action set: event-aware average-reward policy iteration, event-blind policy iteration, a Zeraati-style event-free adapted baseline, an MVT-inspired threshold, and a tuned fixed threshold.



## Start here

- [`index.html`](index.html): web report with theoretical background, model definitions, equations, figures, tables, results, limitations and references.
- [`REPORT.md`](REPORT.md): the same report in Markdown for code review, citation and diffing.
- [`references.bib`](references.bib): BibTeX entries for the cited literature.
- [`code/event_aware_two_action_foraging.py`](code/event_aware_two_action_foraging.py): standalone simulation and plotting code.
- [`notebooks/Event_Aware_Two_Action_Foraging_RL.ipynb`](notebooks/Event_Aware_Two_Action_Foraging_RL.ipynb): Google Colab notebook.
- [`data/`](data/): seed-level results, summaries, policy iteration, threshold tuning and action diagnostics.
- [`assets/figures/`](assets/figures/): PNG/PDF plots and SVG model-state diagram.
- [`downloads/event_aware_two_action_results.zip`](downloads/event_aware_two_action_results.zip): results bundle.
- Source paper: [Zeraati et al. (2025), CCN abstract](https://2025.ccneuro.org/abstract_pdf/Zeraati_2025_Optimal_Foraging_Learning_World_Model.pdf).

## Run locally

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
python code/event_aware_two_action_foraging.py --output-dir results --seeds 20 --horizon 3000 --warmup 200 --quiet
```

The script writes reports, CSVs, convergence histories, example traces, plots and a results archive.

## Important scientific scope

This is a synthetic proof-of-concept, not an empirical biological result. The event-aware model receives an oracle persistent phase label; it does not learn event perception. The event-blind/event-free MDPs use a fixed-prior aggregation of hidden phases, which may not be Markov. The Zeraati-style event-free model adapts the broad state/action and average-reward PI framework to this environment; it is **not an exact replication** of Zeraati et al. (2025). The MVT-inspired rule is also a task-specific heuristic, not the classical theorem under its canonical assumptions. Read the Limitations section before interpreting the reward-rate differences.

## Literature

Full linked references are listed in [`index.html`](index.html) and [`REPORT.md`](REPORT.md).
