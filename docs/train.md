# モデルの学習方法
---

## データセットの準備
### データセットの配置
- dataディレクトリを作成し、その下にデータを配置します。
- 例) data/hifi_captain/[生データ]
- dataディレクトリには発話リストなども保存されるため、コーパスの生データと混ざらないようにしてください。

---

### 発話リストの作成
- [Hi-Fi-CAPTAIN](https://ast-astrec.nict.go.jp/release/hi-fi-captain/)及び[JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)コーパスにおいては、発話リストの作成用モジュールを用意しています。
- conf/corpus以下にある設定ファイルを指定して、発話リストを作成します。
```bash
uv run python src/make_utterance_list.py corpus=hifi_captain_male_jp
```
- data/hifi_captain/male_jp/utterance_list　以下に発話リスト(train.yaml, valid.yaml, test.yaml)が作成されます。

#### その他のコーパス
- conf/corpus　以下にコーパスの設定ファイルを作成してください。
- Corpusクラスには、話者数・音素の数・発話リストの保存場所などを記述します。
- src/utterance_list_maker　を参考にして発話リスト作成モジュールを作成してください。
- 基本的な流れ
  1. コーパスの発話リストを読み込む
  2. 一発話ごとにUtteranceクラスのインスタンスを作成
  3. train/valid/testごとにUtteranceのリストを作成し、yamlとして保存

---

### 音響特徴量の統計量の計算
- 音響モデルでは、音響特徴量の正規化をおこなう場合があるため、事前に統計量を計算します。
- 学習に使用するモデルと、信号処理の設定ファイルを指定して計算します。
```bash
uv run python src/calculate_stats.py engine=acoustic/fastspeech2 dsp=24k corpus=hifi_captain_male_jp
```
- コーパスの設定ファイルに指定したfeature_stats_pathの場所に統計量が保存されます。
- コーパスの設定ファイルのfeature_statsには、学習スクリプトの実行時などに自動で統計量が格納されます。
- f0_mean: corpus.config.features_stats.pitch.mean のようにしてモデルの設定ファイルに値を代入できます。
- conf/engine/acoustic/model/fastspeech2.yamlなどを参考にしてください。

---

## 学習
- 用意したデータセットやモデルを切り替えながら学習が可能です。
- general.exp_nameを指定することで、実験ディレクトリが作成され、実験データが保存されます。保存場所は設定ファイルのgeneralの値によって決定されます。　exp/[project_name]/[group_name]/[exp_name]
- general.analysis=Trueとすると、wandbにロギングされます。
```bash
uv run python src/train.py engine=acoustic/fastspeech2 text=ja_duration general.exp_name=fs2
uv run python src/train.py engine=e2e/vits corpus=hifi_captain_female_jp general.exp_name=vits
uv run python src/train.py engine=vocoder/hifi_gan dsp=48k general.exp_name=hifi_gan general.analysis=True
```
- デフォルトではconf/config.yamlが読み込まれます。
- hydraのオーバーライドを使えば、設定ファイルを書き換えずに実行時の設定を変更することができます。
```bash
uv run python src/train.py engine=e2e/vits general.batch_size=16 general.segment_size.train=8192 engine.lightning_module.model.kernel_size=7 
```