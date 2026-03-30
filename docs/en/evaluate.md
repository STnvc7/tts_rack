# How to Evaluate Models
---

- In `src/train.py`, an evaluation loop is automatically executed at the end of training, calculating objective evaluation metrics.
- Results are saved in the `results` folder within the experiment directory.
- If you want to re-evaluate or use a different evaluation set, you can run the evaluation script.
```bash
uv run python src/evaluate.py --exp=[path_to_experiment_directory] --metrics=[evaluation_set]
```
- For `--metrics`, specify the name of the configuration file located under `conf/metrics`.
- If `none` is specified for `--metrics`, only the Real-Time Factor (RTF), memory consumption, and the number of parameters will be calculated.
- The `--device` flag can be set to either `cpu` or `gpu`.