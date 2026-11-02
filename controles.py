"""Controles para os resultados do poc.py, sem dado novo.

1. Controle negativo: mesmas gravações, mas janelas das PAUSAS (boca parada, sem consoante).
   Se o classificador também separa pato.mov de bato.mov aí, ele está reconhecendo a sessão.
2. Teste de permutação: acaso empírico, embaralhando os rótulos.

Uso: uv run python controles.py data [--modelo todos] [--perm 1000]
"""
import argparse
from pathlib import Path

import mediapipe as mp
import numpy as np
import torch
from sklearn.model_selection import LeaveOneOut, cross_val_predict

import poc

MARGIN = 0.15  # s de silêncio exigidos antes e depois da janela de pausa


def pause_starts(audio, n_frames, k):
    """Até k inícios (em quadros) de janelas inteiras dentro do silêncio, sem sobreposição."""
    speech = poc.speech_mask(audio)  # 1 valor a cada 10 ms
    win_s, starts, t = poc.WIN / poc.FPS, [], 0.0
    while t + win_s <= n_frames / poc.FPS:
        a, b = int((t - MARGIN) * 100), int((t + win_s + MARGIN) * 100)
        if a >= 0 and b <= len(speech) and not speech[a:b].any():
            starts.append(round(t * poc.FPS))
            t += win_s
        else:
            t += 0.04
    # Espalha as escolhidas pela gravação inteira, em vez de pegar só o começo.
    return [starts[i] for i in np.linspace(0, len(starts) - 1, min(k, len(starts))).round().astype(int)]


def collect(data_dir, landmarker):
    fala, pausa = ([], []), ([], [])
    for path in sorted(p for p in Path(data_dir).iterdir() if p.suffix.lower() in {".mov", ".mp4"}):
        label, frames, audio = poc.label_from_name(path), poc.read_frames(path), poc.read_audio(path)
        onsets = poc.utterance_onsets(audio)
        speech_starts = [round((t + poc.WIN_OFFSET_S) * poc.FPS) for t in onsets]
        for out, starts in ((fala, speech_starts), (pausa, pause_starts(audio, len(frames), len(onsets)))):
            for i in starts:
                if i + poc.WIN <= len(frames) and (c := poc.crop_mouths(frames[i:i + poc.WIN], landmarker)) is not None:
                    out[0].append(c)
                    out[1].append(label)
        print(f"{path.name}: {len(onsets)} janelas de fala, {sum(1 for l in pausa[1] if l == label)} de pausa")
    return fala, pausa


def loo_hits(X, y):
    return int((cross_val_predict(poc.make_pipeline(poc.StandardScaler(), poc.LogisticRegression(max_iter=5000)),
                                  X, y, cv=LeaveOneOut()) == y).sum())


def perm_p(X, y, hits, n, rng):
    """Fração de embaralhamentos que acertam pelo menos tanto quanto os rótulos reais."""
    null = [loo_hits(X, rng.permutation(y)) for _ in range(n)]
    return (1 + sum(h >= hits for h in null)) / (n + 1)


def main(data_dir, modelos, n_perm):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(poc.ROOT / "checkpoints/face_landmarker.task"))))
    (fc, fy), (pc, py) = collect(data_dir, landmarker)
    fy, py = np.array(fy), np.array(py)
    rng = np.random.default_rng(0)

    print(f"\nleave-one-out; p de permutação com {n_perm} embaralhamentos")
    print(f"{'modelo':10s} {'fala':>16s} {'pausa (controle)':>20s}")
    for nome in modelos:
        f = poc.MODELOS[nome](device)
        with torch.no_grad():
            Xf, Xp = f(fc), f(pc)
        hf, hp = loo_hits(Xf, fy), loo_hits(Xp, py)
        pf, pp = perm_p(Xf, fy, hf, n_perm, rng), perm_p(Xp, py, hp, n_perm, rng)
        print(f"{nome:10s} {hf:>3d}/{len(fy)} p={pf:.3f}   {hp:>3d}/{len(py)} p={pp:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", nargs="?", default="data")
    ap.add_argument("--modelo", default="todos", choices=[*poc.MODELOS, "todos"])
    ap.add_argument("--perm", type=int, default=1000)
    args = ap.parse_args()
    main(args.data, list(poc.MODELOS) if args.modelo == "todos" else [args.modelo], args.perm)
