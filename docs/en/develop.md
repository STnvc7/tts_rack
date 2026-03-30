# Model Research and Development
- This framework is designed with specific interfaces for each module.
- You can add your own custom modules by following these interfaces.
- For more details, please refer to the `src/interface` directory.

---

## DataModule
- Most standard operations can be handled by modifying the arguments of the `DataModule`.
- You can change the acoustic features to be extracted by modifying `features_to_extract` in the configuration file.
- The available acoustic features are limited to the methods provided in the `Processor` class of [dsp_board](git@github.com:STnvc7/dsp_board.git) (most common features are already implemented).
- Specify the method names exactly as they appear. Example: `features_to_extract: [mel_spectrogram, pitch]`
- If you want to use features not available in the `Processor`, please extend the class (see the DSP section below).
- Set `require_duration` to `true` if your model requires duration information.

## Engine
- Please review the `lightning_module.py` for each model type (Acoustic Model, E2E Model, Neural Vocoder) to understand the data flow.

### AcousticModel / E2EModel / Generator
- **Input**: `DataLoaderOutput`
- **Output**: `[AcousticModelOutput / E2EModelOutput / GeneratorOutput]`
- `DataLoaderOutput` contains phoneme sequences, acoustic features, audio data, etc.
- While the output interface name varies by model type, the basic structure remains the same.
- Store intermediate features used for loss calculations in the `outputs` property of the output interface.
- Additionally, store data intended for logging in the `loggable_outputs` property using the `Loggable` class. These will be automatically logged to **wandb** during the validation loop.

### Discriminator
- **Input**: `target`, `[E2EModelOutput / GeneratorOutput]`, `mode`
- **Output**: `DiscriminatorOutput`
- This is the interface for discriminators used in E2E models and neural vocoders.
- The output includes both the validity scores (true/false) and feature maps, which are required for calculating feature matching loss.
- Since you must detach gradients from the generated data when training the discriminator, specify either `generator` or `discriminator` for the `mode`.

### AcousticModelLoss / E2EModelLoss / GeneratorLoss
- **Input**: `DataLoaderOutput`, `[AcousticModelOutput / E2EModelOutput / GeneratorOutput]`
- **Output**: `LossOutput`
- Calculate the loss using the outputs from each model.
- Gradient calculation is performed based on the `total_loss` passed to `LossOutput`.
- If you are summing multiple loss functions, you can log them by storing them as a dictionary in `loss_components`.

---

## DSP
- If the methods in [dsp_board](git@github.com:STnvc7/dsp_board.git) are insufficient, please extend them in `src/dsp/processor.py`.
- Currently, methods for continuous F0 and rescaled mel-spectrograms have been implemented.
- If you list these method names in `features_to_extract`, they will be calculated within the `Dataset`.

---

## Text
- Currently, only Japanese is supported.
- To use other languages, inherit from the `TextProcessor` class and refer to the Japanese implementation as a guide.
- The default `TextProcessor` is designed for use with MAS (Monotonic Alignment Search) or neural vocoders that do not pre-calculate durations.
- If you are using models like FastSpeech that rely on forced alignment durations as ground truth, set the `TextProcessor` to `ja_