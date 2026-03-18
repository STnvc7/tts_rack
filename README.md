# tts_rack

## 概要
音響モデル、E2Eモデル、ニューラルボコーダの研究のためのフレームワークです。
各モデルにおいて共通の学習ループ(LightningModule)を使用するため、モデル構造以外の条件を揃えやすくなっています。

## 使用方法
- docs/train.md: モデルの学習方法
- docs/evaluate.md: モデルの評価方法
- docs/develop.md: モデルの設計・追加方法

## 参考文献
以下のリポジトリおよび論文を参考にしています。

||モデル|リポジトリ|論文|
|---|---|---|---|
|Acoustic Models|FastSpeech2|https://github.com/ming024/FastSpeech2.git|https://arxiv.org/abs/2006.04558v1|
|E2E Models|VITS|https://github.com/jaywalnut310/vits.git|https://arxiv.org/abs/2106.06103|
|Neural Vocoders|APNet|https://github.com/YangAi520/APNet.git|https://arxiv.org/abs/2305.07952|
||APNet2|https://github.com/redmist328/APNet2.git|https://arxiv.org/abs/2311.11545|
||BigVGAN|https://github.com/NVIDIA/BigVGAN.git|https://arxiv.org/abs/2206.04658|
||DiffWave|https://github.com/lmnt-com/diffwave.git|https://arxiv.org/abs/2009.09761|
||FreeV|https://github.com/BakerBunker/FreeV.git|https://arxiv.org/abs/2406.08196|
||HiFi-GAN|https://github.com/jik876/hifi-gan.git|https://arxiv.org/abs/2010.05646|
||iSTFTNet|https://github.com/rishikksh20/iSTFTNet-pytorch.git|https://arxiv.org/abs/2203.02395|
||SiFiGAN|https://github.com/chomeyama/SiFiGAN.git|https://arxiv.org/abs/2210.15533|
||UnivNet|https://github.com/maum-ai/univnet.git|https://arxiv.org/abs/2106.07889|
||Vocos|https://github.com/gemelo-ai/vocos.git|https://arxiv.org/abs/2306.00814|
||WaveGrad|https://github.com/lmnt-com/wavegrad.git|https://arxiv.org/abs/2009.00713|
||WaveHax|https://github.com/chomeyama/wavehax.git|https://arxiv.org/abs/2411.06807|
||WaveNeXt|https://github.com/wetdog/wavenext_pytorch.git|https://ast-astrec.nict.go.jp/release/preprints/preprint_asru_2023_okamoto.pdf|