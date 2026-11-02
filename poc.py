"""POC: vídeo bruto → recorte da boca → features de um modelo pré-treinado congelado → classificador das 6 oclusivas.

Uso: uv run python poc.py data/ [--modelo lrw|pt|braven|avhubert|todos]
Cada arquivo em data/ tem o nome da palavra (pato.mov, bato_2.mp4 ...) e contém várias
repetições de "Digo ___ baixinho" separadas por pausas.
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modelos import MODELOS

ROOT = Path(__file__).parent

FPS = 25  # taxa em que todos os modelos foram treinados
SR = 16000
WIN = 29  # quadros por clipe no LRW (1,16 s)
# Início da janela relativo ao começo da frase: pula o "Di" de "Digo" e cobre "go ___ bai".
# Calibrar olhando alguns recortes; nível 2 de rotulagem (MFA) substitui isto.
WIN_OFFSET_S = 0.2
# Rosto médio do LRW: os 4 modelos foram treinados com recortes alinhados a ele.
MEAN_FACE = np.load(ROOT / "third_party/vsr_ml/pipelines/detectors/mediapipe/20words_mean_face.npy")
# Pontos estáveis (olho dir., olho esq., base do nariz, centro da boca) no rosto médio e no MediaPipe.
REF_POINTS = np.stack([MEAN_FACE[36:42].mean(0), MEAN_FACE[42:48].mean(0), MEAN_FACE[31:36].mean(0),
                       MEAN_FACE[48:68].mean(0)]).astype(np.float32)
MP_POINTS = [(33, 133), (362, 263), (1,), (13, 14)]

CLASSES = list("pbtdkg")
PLACE = {"p": "bilabial", "b": "bilabial", "t": "alveolar", "d": "alveolar", "k": "velar", "g": "velar"}
PAIRS = [("p", "b"), ("t", "d"), ("k", "g")]


# ---- Leitura dos vídeos

def label_from_name(path):
    letter = path.stem.split("_")[0][0].lower()
    return {"c": "k"}.get(letter, letter)  # "cato" é /k/


def read_audio(path):
    raw = subprocess.run(
        # async=1 preenche buracos pelos timestamps; sem isso o áudio da webcam deriva até 2,5 s do vídeo
        ["ffmpeg", "-v", "error", "-i", str(path), "-af", "aresample=async=1", "-ac", "1", "-ar", str(SR),
         "-f", "s16le", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def read_frames(path, width=960):
    """Decodifica já reamostrado para 25 fps; o quadro i fica no instante i/25 s."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    w, h = map(int, probe.split(",")[:2])
    width = min(width, w)
    height = round(h * width / w / 2) * 2
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={FPS},scale={width}:{height}",
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


def recordings(data_dir):
    """(caminho, rótulo, quadros, áudio) de cada vídeo da pasta."""
    for path in sorted(p for p in Path(data_dir).iterdir() if p.suffix.lower() in {".mov", ".mp4"}):
        yield path, label_from_name(path), read_frames(path), read_audio(path)


# ---- Segmentação pelo áudio

def speech_mask(audio):
    """Fala ou silêncio a cada 10 ms, pela energia do som."""
    hop = SR // 100
    rms = np.sqrt(np.mean(audio[: len(audio) // hop * hop].reshape(-1, hop) ** 2, axis=1)) + 1e-8
    db = np.convolve(20 * np.log10(rms), np.ones(5) / 5, "same")
    if db.max() < -60:
        raise ValueError("áudio mudo: confira o microfone usado na gravação")
    # Limiar no meio entre ruído (mediana: a maior parte da gravação é pausa) e fala (p99).
    return db > (np.median(db) + np.percentile(db, 99)) / 2


def utterance_onsets(audio, min_gap_s=0.6, min_len_s=0.6):
    """Começo (s) de cada frase: trechos de fala separados por pausas de pelo menos min_gap_s."""
    idx = np.flatnonzero(speech_mask(audio))  # instantes com fala, em centésimos de segundo
    if len(idx) == 0:
        return []
    cut = np.flatnonzero(np.diff(idx) > min_gap_s * 100)
    starts, ends = idx[np.r_[0, cut + 1]], idx[np.r_[cut, len(idx) - 1]]
    return [s / 100 for s, e in zip(starts, ends) if e - s >= min_len_s * 100]


# ---- Recorte da boca

def make_landmarker():
    return mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT / "checkpoints/face_landmarker.task"))))


def crop_mouths(frames, landmarker):
    """Alinha o rosto ao rosto médio do LRW (256×256), recorta 96×96 na boca e fica com o centro 88×88.

    Mesma receita do pré-processamento do Ma et al. (VSR_for_Multiple_Languages). None se perder o rosto.
    """
    pts = []
    for f in frames:
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f)))
        if not res.face_landmarks:
            return None
        lm = res.face_landmarks[0]
        pts.append([np.mean([(lm[i].x * f.shape[1], lm[i].y * f.shape[0]) for i in idx], 0) for idx in MP_POINTS])
    pts = np.array(pts, np.float32)
    out = []
    for t, f in enumerate(frames):
        # Média móvel de ±6 quadros tira o tremor dos landmarks, mantendo a posição do quadro atual.
        w = pts[max(0, t - 6):t + 7].mean(0)
        w += pts[t].mean(0) - w.mean(0)
        M = cv2.estimateAffinePartial2D(w, REF_POINTS, method=cv2.LMEDS)[0]
        face = cv2.warpAffine(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), M, (256, 256))
        cx, cy = (M[:, :2] @ w[3] + M[:, 2]).round().astype(int)
        # Borda preta de 48 px: o recorte 96×96 centrado na boca nunca sai da imagem.
        face = cv2.copyMakeBorder(face, 48, 48, 48, 48, cv2.BORDER_CONSTANT)
        out.append(face[cy + 4:cy + 92, cx + 4:cx + 92])
    return (np.stack(out).astype(np.float32) / 255 - 0.421) / 0.165


def clips_at(frames, starts, landmarker):
    """Clipes de WIN quadros a partir de cada início; pula os que saem do vídeo ou perdem o rosto."""
    clips = (crop_mouths(frames[i:i + WIN], landmarker) for i in starts if i + WIN <= len(frames))
    return [c for c in clips if c is not None]


def load_clips(data_dir, landmarker):
    """Um clipe por frase (janela em volta da consoante) de cada vídeo, com o rótulo do arquivo."""
    clips, labels = [], []
    for path, label, frames, audio in recordings(data_dir):
        onsets = utterance_onsets(audio)
        found = clips_at(frames, [round((t + WIN_OFFSET_S) * FPS) for t in onsets], landmarker)
        clips += found
        labels += [label] * len(found)
        print(f"{path.name}: /{label}/, {len(onsets)} frases, {len(found)} recortes")
    return clips, np.array(labels)


# ---- Avaliação

def make_clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def cv_eval(X, y):
    k = min(5, min(np.unique(y, return_counts=True)[1]))
    return cross_val_predict(make_clf(), X, y, cv=StratifiedKFold(k, shuffle=True, random_state=0))


def evaluate(X, y):
    """Imprime as métricas e devolve um resumo de uma linha por medida."""
    present = [c for c in CLASSES if c in y]
    pred = cv_eval(X, y)
    resumo = {"fonema": f"{accuracy_score(y, pred):.1%}"}
    print(f"\n== Fonema ({len(present)} classes, acaso {1 / len(present):.0%}) ==")
    print(f"acurácia {resumo['fonema']}")
    print(classification_report(y, pred, labels=present, zero_division=0))
    print("matriz de confusão (linha = verdade, coluna = previsto):", present)
    print(confusion_matrix(y, pred, labels=present))

    place = np.array([PLACE[c] for c in y])
    if len(set(place)) > 1:
        resumo["ponto"] = f"{accuracy_score(place, cv_eval(X, place)):.1%}"
        print(f"\n== Ponto ({len(set(place))} classes) == acurácia {resumo['ponto']}")

    print("\n== Vozeamento dentro do par (acaso 50%) ==")
    for pair in PAIRS:
        m = np.isin(y, pair)
        if len(set(y[m])) < 2:
            continue
        hits = int((cv_eval(X[m], y[m]) == y[m]).sum())
        p = binomtest(hits, m.sum(), 0.5, alternative="greater").pvalue
        resumo["/".join(pair)] = f"{hits}/{m.sum()} (p={p:.3f})"
        print(f"/{pair[0]}/×/{pair[1]}/: {hits}/{m.sum()} = {hits / m.sum():.1%}  (p = {p:.3f})")
    return resumo


def main(data_dir, modelos):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    clips, y = load_clips(data_dir, make_landmarker())

    resumos = {}
    for nome in modelos:
        print(f"\n######## modelo: {nome} ########")
        with torch.no_grad():
            X = MODELOS[nome](device)(clips)
        print(f"features: {X.shape[1]} dimensões")
        resumos[nome] = evaluate(X, y)

    if len(resumos) > 1:
        print("\n######## comparação ########")
        for nome, r in resumos.items():
            print(f"{nome:10s} " + "  ".join(f"{k}: {v}" for k, v in r.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", nargs="?", default="data", help="pasta com os vídeos")
    ap.add_argument("--modelo", default="lrw", choices=[*MODELOS, "todos"])
    args = ap.parse_args()
    main(args.data, list(MODELOS) if args.modelo == "todos" else [args.modelo])
