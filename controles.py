"""Controles para os resultados do poc.py, sem dado novo.

1. Controle negativo: mesmas gravações, mas janelas das PAUSAS (boca parada, sem consoante).
   Se o classificador também separa pato.mov de bato.mov aí, ele está reconhecendo a sessão.
2. Teste de permutação: acaso empírico, embaralhando os rótulos.

Uso: uv run python controles.py data [--modelo todos] [--perm 1000]
"""
import argparse

import numpy as np
import torch
from sklearn.model_selection import LeaveOneOut, cross_val_predict

import poc
from modelos import MODELOS

MARGIN_S = 0.15  # silêncio exigido antes e depois da janela de pausa


def pause_starts(audio, n_frames, k):
    """Até k inícios (em quadros) de janelas inteiras dentro do silêncio, espalhados pela gravação."""
    speech = poc.speech_mask(audio)  # 1 valor a cada 10 ms
    win_s, starts, t = poc.WIN / poc.FPS, [], 0.0
    while t + win_s <= n_frames / poc.FPS:
        a, b = int((t - MARGIN_S) * 100), int((t + win_s + MARGIN_S) * 100)
        if a >= 0 and b <= len(speech) and not speech[a:b].any():
            starts.append(round(t * poc.FPS))
            t += win_s  # janelas sem sobreposição
        else:
            t += 1 / poc.FPS
    return [starts[i] for i in np.linspace(0, len(starts) - 1, min(k, len(starts))).round().astype(int)]


def collect(data_dir, landmarker):
    """Clipes de fala (como no poc.py) e o mesmo número de clipes de pausa, por vídeo."""
    fala, pausa = ([], []), ([], [])
    for path, label, frames, audio in poc.recordings(data_dir):
        onsets = poc.utterance_onsets(audio)
        pauses = pause_starts(audio, len(frames), len(onsets))
        for (clips, labels), starts in [(fala, [round((t + poc.WIN_OFFSET_S) * poc.FPS) for t in onsets]),
                                        (pausa, pauses)]:
            found = poc.clips_at(frames, starts, landmarker)
            clips += found
            labels += [label] * len(found)
        print(f"{path.name}: {len(onsets)} janelas de fala, {len(pauses)} de pausa")
    return fala, pausa


def loo_hits(X, y):
    return int((cross_val_predict(poc.make_clf(), X, y, cv=LeaveOneOut()) == y).sum())


def perm_p(X, y, hits, n, rng):
    """Fração de embaralhamentos que acertam pelo menos tanto quanto os rótulos reais."""
    null = [loo_hits(X, rng.permutation(y)) for _ in range(n)]
    return (1 + sum(h >= hits for h in null)) / (n + 1)


def main(data_dir, modelos, n_perm):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    (fc, fy), (pc, py) = collect(data_dir, poc.make_landmarker())
    fy, py = np.array(fy), np.array(py)
    rng = np.random.default_rng(0)

    print(f"\nleave-one-out; p de permutação com {n_perm} embaralhamentos")
    print(f"{'modelo':10s} {'fala':>16s} {'pausa (controle)':>20s}")
    for nome in modelos:
        f = MODELOS[nome](device)
        with torch.no_grad():
            Xf, Xp = f(fc), f(pc)
        hf, hp = loo_hits(Xf, fy), loo_hits(Xp, py)
        pf, pp = perm_p(Xf, fy, hf, n_perm, rng), perm_p(Xp, py, hp, n_perm, rng)
        print(f"{nome:10s} {hf:>3d}/{len(fy)} p={pf:.3f}   {hp:>3d}/{len(py)} p={pp:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", nargs="?", default="data")
    ap.add_argument("--modelo", default="todos", choices=[*MODELOS, "todos"])
    ap.add_argument("--perm", type=int, default=1000)
    args = ap.parse_args()
    main(args.data, list(MODELOS) if args.modelo == "todos" else [args.modelo], args.perm)
