"""Segmentação por energia num áudio sintético. Uso: uv run python test_poc.py"""
import numpy as np

import poc


def test_utterance_onsets():
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 1e-3, poc.SR * 10).astype(np.float32)  # 10 s de ruído de fundo
    for start, dur in [(1.0, 1.5), (4.0, 1.2), (7.0, 1.8)]:  # 3 "frases"
        i = int(start * poc.SR)
        audio[i:i + int(dur * poc.SR)] += 0.3 * np.sin(2 * np.pi * 150 * np.arange(int(dur * poc.SR)) / poc.SR)
    # Uma pausa interna de 0,3 s (menor que min_gap) não pode partir a frase em duas.
    audio[int(4.5 * poc.SR):int(4.8 * poc.SR)] = rng.normal(0, 1e-3, int(0.3 * poc.SR))
    # Um estalo curto (menor que min_len) não pode virar frase.
    audio[int(9.0 * poc.SR):int(9.1 * poc.SR)] += 0.3
    onsets = poc.utterance_onsets(audio)
    assert len(onsets) == 3, onsets
    assert all(abs(o - s) < 0.05 for o, s in zip(onsets, [1.0, 4.0, 7.0])), onsets


if __name__ == "__main__":
    test_utterance_onsets()
    print("ok")
