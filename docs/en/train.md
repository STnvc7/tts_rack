# Model Training Procedure
---

## Dataset Preparation

### Dataset Placement
- Create a `data` directory and place your dataset under it.
- Example: `data/hifi_captain/[raw data]`
- Since the `data` directory will also store utterance lists and other files, make sure not to mix them with the raw corpus data.

---

### Creating Utterance Lists
- For the Hi-Fi-CAPTAIN and JSUT corpora, modules for generating utterance lists are provided.
- Specify a configuration file under `conf/corpus` to create the utterance list.
- Utterance lists (`train.yaml`, `valid.yaml`, `test.yaml`) will be generated under:
  `data/hifi_captain/male_jp/utterance_list`
```bash
uv run python src/make_utterance_list.py corpus=hifi_captain_male_jp
```

#### Other Corpora
- Create a corpus configuration file under `conf/corpus`.
- In the `Corpus` class, describe items such as the number of speakers, number of phonemes, and the storage path for utterance lists.
- Refer to `src/utterance_list_maker` to implement a module for generating utterance lists.

**Basic workflow:**
1. Load the corpus utterance list
2. Create an instance of the `Utterance` class for each utterance
3. Create lists of `Utterance` objects for `train/valid/test` and save them as YAML

---

### Calculating Statistics of Acoustic Features
- Since acoustic models may normalize acoustic features, compute their statistics in advance.
- Specify the model and DSP configuration file used for training.
```bash
uv run python src/calculate_stats.py engine=acoustic/fastspeech2 dsp=24k corpus=hifi_captain_male_jp
```
- The statistics will be saved to the path specified by `feature_stats_path` in the corpus configuration file.
- The `feature_stats` field in the corpus config will be automatically populated when running training scripts.
- You can assign values in the model configuration file as follows:
  `f0_mean: corpus.config.features_stats.pitch.mean`
- Refer to files such as `conf/engine/acoustic/model/fastspeech2.yaml`.

---

## Downloading Files for Forced Alignment
- Perform this step if you use duration information obtained via forced alignment (e.g., in FastSpeech2) as training targets.
- Forced alignment is implemented using the pydomino library.
```bash
curl -L https://github.com/DwangoMediaVillage/pydomino/raw/refs/heads/main/onnx_model/phoneme_transition_model.onnx --create-dirs -o ./bin/phoneme_transition_model.onnx
```

---

## Training
- You can train models while switching between datasets and configurations.
- By specifying `general.exp_name`, an experiment directory will be created and results will be saved.
  The save location is determined by the `general` settings:
  `exp/[project_name]/[group_name]/[exp_name]`
- If `general.analysis=True`, logs will be sent to Weights & Biases.
```bash
uv run python src/train.py engine=acoustic/fastspeech2 text=ja_duration general.exp_name=fs2
uv run python src/train.py engine=e2e/vits corpus=hifi_captain_female_jp general.exp_name=vits
uv run python src/train.py engine=vocoder/hifi_gan dsp=48k general.exp_name=hifi_gan general.analysis=True
```
- By default, `conf/config.yaml` is loaded.
- Using Hydra overrides, you can change settings at runtime without modifying configuration files.
```bash
uv run python src/train.py engine=e2e/vits general.batch_size=16 general.segment_size.train=8192 engine.lightning_module.model.kernel_size=7
```