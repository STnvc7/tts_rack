# モデルの研究・開発
- 本フレームワークは各モジュールに対してインターフェースを設計しています。
- このインターフェースに則って、自作のモジュールを追加可能です。
- 詳しくはsrc/interface以下を確認してください。

---

## DataModule
- DataModuleの引数を変更することで、基本的な動作には対応可能です。
- 設定ファイルにおいてfeatures_to_extractを変更することで、抽出する音響特徴量を変更することができます。
- 抽出可能な音響特徴量は[dsp_board](git@github.com:STnvc7/dsp_board.git)のProcessorクラスのメソッドにあるものに限られます（基本的なものは揃っています）。
- メソッド名をそのまま記述してください。　例）features_to_extract: [mel_spectrogram, pitch]
- Processorにない特徴量を使用したい場合は、クラスを拡張してください。(下記DSPを参照)
- 継続長が必要な場合は、require_durationをtrueにしてください。

## Engine
- 各モデル（音響モデル、E2Eモデル、ニューラルボコーダ）のlightning_module.pyを見て、データの流れを確認してください。
### AcousticModel / E2EModel / Generator
- 入力: DataLoaderOutput
- 出力: [AcousticModelOutput / E2EModelOutput / GeneratorOutput]
- DataLoaderOutputには、音素列、音響特徴量、音声データなどが含まれます。
- 各モデルごとに出力のインターフェース名が変わりますが、基本的には同じです。
- 出力インターフェースのoutputsプロパティには、誤差計算などで使用する中間特徴量などを格納します。
- また、出力インターフェースのloggable_outputsプロパティには、Loggableクラスのデータを格納してください。検証ループにおいて、wandbにロギングされます。

### Discriminator
- 入力: target, [E2EModelOutput / GeneratorOutput], mode
- 出力: DiscriminatorOutput
- E2Eモデルとニューラルボコーダで使用する識別器のインターフェースです。
- 出力には、真偽判定の結果だけでなく、特徴マップも返すようにしています。特徴マッチング誤差のためです。
- 識別器の学習時には、生成データの勾配を切り離す必要があるため、modeにはgeneratorとdiscriminatorのどちらかを指定してください。

### AcousticModelLoss / E2EModelLoss / GeneratorLoss
- 入力: DataLoaderOutput, [AcousticModelOutput / E2EModelOutput / GeneratorOutput]
- 出力: LossOutput
- 各モデルの出力を使用して誤差計算をしてください。
- LossOutputのtotal_lossに渡された誤差に基づいて勾配計算が行われます。
- 複数の誤差関数を合計している場合は、loss_componentsに辞書型で格納することで、ロギングされます。

---

## DSP
- [dsp_board](git@github.com:STnvc7/dsp_board.git)内のメソッドでまかなえない場合は、src/dsp/processor.pyにて拡張をおこなってください。
- 現時点で、continuous f0とスケールを変更したメルスペクトログラムのメソッドを作成しています。
- このメソッド名をそのままfeatures_to_extractの記述すると、Dataset内で計算されます。

---

## Text
- 現在、日本語のみの対応となっています。
- その他の言語を使用する場合は、TextProcessorクラスを継承し、日本語のTextProcessorを参考にして作成してください。
- デフォルトで設定されているTextProcessorは、継続長の計算を行わないMASやニューラルボコーダの使用を前提にしたものです。
- FastSpeechなど、強制アライメントによる継続長を教師データとして使用する場合は、ja_durationをTextProcessorに設定してください。
- また、継続長はDataset内で毎回計算を行っています。学習時間を短縮したい際は、事前計算済みの継続長を使用することを検討してください。
---

## Metircs
- [metric_board](git@github.com:STnvc7/metric_board.git)を使用して、客観評価をおこないます。
- metric_boardはtorchmetricsを基にした評価指標ライブラリです。
- conf/metricsには、metrics_boardのEvaluatorクラスを作成する設定を記述してください。
- 設定ファイル内のmetrics以下には、metrics_board.metrics以下の指標を指定してください。
- computeメソッドにてmetrics_board.MetricOutputを返すクラスであれば、自作したものも使用可能です。
