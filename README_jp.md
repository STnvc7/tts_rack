# tts_rack

## 概要
音響モデル、E2Eモデル、ニューラルボコーダの研究のためのフレームワークです。
各モデルにおいて共通の学習ループ(LightningModule)を使用するため、モデル構造以外の条件を揃えやすくなっています。

## 使用方法
- docs/ja/train.md: モデルの学習方法
- docs/ja/evaluate.md: モデルの評価方法
- docs/ja/develop.md: モデルの設計・追加方法

## 参考文献
以下のリポジトリおよび論文を参考にしています。

| | モデル | リポジトリ | 論文 |
|---|---|---|---|
| **Acoustic Models** | FastSpeech2 | [GitHub](https://github.com/ming024/FastSpeech2.git) | [arXiv](https://arxiv.org/abs/2006.04558v1) |
| **E2E Models** | VITS | [GitHub](https://github.com/jaywalnut310/vits.git) | [arXiv](https://arxiv.org/abs/2106.06103) |
| **Neural Vocoders** | APNet | [GitHub](https://github.com/YangAi520/APNet.git) | [arXiv](https://arxiv.org/abs/2305.07952) |
| | APNet2 | [GitHub](https://github.com/redmist328/APNet2.git) | [arXiv](https://arxiv.org/abs/2311.11545) |
| | BigVGAN | [GitHub](https://github.com/NVIDIA/BigVGAN.git) | [arXiv](https://arxiv.org/abs/2206.04658) |
| | DiffWave | [GitHub](https://github.com/lmnt-com/diffwave.git) | [arXiv](https://arxiv.org/abs/2009.09761) |
| | FreeV | [GitHub](https://github.com/BakerBunker/FreeV.git) | [arXiv](https://arxiv.org/abs/2406.08196) |
| | HiFi-GAN | [GitHub](https://github.com/jik876/hifi-gan.git) | [arXiv](https://arxiv.org/abs/2010.05646) |
| | iSTFTNet | [GitHub](https://github.com/rishikksh20/iSTFTNet-pytorch.git) | [arXiv](https://arxiv.org/abs/2203.02395) |
| | SiFiGAN | [GitHub](https://github.com/chomeyama/SiFiGAN.git) | [arXiv](https://arxiv.org/abs/2210.15533) |
| | UnivNet | [GitHub](https://github.com/maum-ai/univnet.git) | [arXiv](https://arxiv.org/abs/2106.07889) |
| | Vocos | [GitHub](https://github.com/gemelo-ai/vocos.git) | [arXiv](https://arxiv.org/abs/2306.00814) |
| | WaveGrad | [GitHub](https://github.com/lmnt-com/wavegrad.git) | [arXiv](https://arxiv.org/abs/2009.00713) |
| | WaveHax | [GitHub](https://github.com/chomeyama/wavehax.git) | [arXiv](https://arxiv.org/abs/2411.06807) |
| | WaveNeXt | [GitHub](https://github.com/wetdog/wavenext_pytorch.git) | [Paper](https://ast-astrec.nict.go.jp/release/preprints/preprint_asru_2023_okamoto.pdf) |