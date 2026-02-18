from typing import Literal

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
AcousticFeature = Literal[
    "linear_spectrogram",
    "log_spectrogram",
    "phase_spectrogram",
    "linear_energy",
    "mel_spectrogram",
    "log_energy",
    "mel_energy",
    "mel_cepstrum",
    "mel_generalized_cepstrum",
    "mfcc",
    "pitch",
    "spectral_envelope",
    "aperiodicity",
    "band_aperiodicity",
    "vuv",
    "continuous_pitch"
]