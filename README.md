# FineTuningDataSelection

This is the code for **RETHINKING DATA SELECTION: THE IMPORTANCE OF COVERAGE OVER DIFFICULTY IN GENERATIVE FINE-TUNING**. It has been adapted from the [open-instruct](https://github.com/allenai/open-instruct) repository of Allen AI (Ai2): https://github.com/allenai/open-instruct

Datasets, per-example score dumps, SLURM job scripts and raw evaluation outputs are not included in this repository. Paths such as `/path/to/scratch` in the scripts are placeholders for your own directories.

- `open-instruct/open_instruct/` – selection metrics (perplexity, PVI, VoG, data-cartography, prototypicality, PCA/clustering) and fine-tuning code
- `open-instruct/eval/` – evaluation code
- `results/` – summary CSVs (accuracy, FLOPs, training trajectories, per-method scores)
- `*.py` at top level – plotting and result-aggregation scripts

## Licensing

This codebase is licensed under Apache 2.0 as given in [LICENSE](LICENSE). It is adapted from [allenai/open-instruct](https://github.com/allenai/open-instruct) by the Allen Institute for AI, which is also released under Apache 2.0.
