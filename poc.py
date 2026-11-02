"""POC: vídeo bruto → recorte da boca → features do LRW-TCN congelado → classificador das 6 oclusivas.

Uso: uv run python poc.py data/
Cada arquivo em data/ tem o nome da palavra (pato.mov, bato_2.mp4 ...) e contém várias
repetições de "Digo ___ baixinho" separadas por pausas.
"""
import json
import subprocess
import sys
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

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "third_party/lrw_tcn"))
from lipreading.model import Lipreading  # noqa: E402

FPS = 25  # taxa em que o LRW foi treinado
SR = 16000
WIN = 29  # quadros por clipe no LRW (1,16 s)
# Início da janela relativo ao começo da frase: pula o "Di" de "Digo" e cobre "go ___ bai".
# Calibrar olhando alguns recortes; nível 2 de rotulagem (MFA) substitui isto.
WIN_OFFSET_S = 0.2
MOUTH_SCALE = 2.0  # lado do recorte = MOUTH_SCALE × largura da boca (cantos 61-291)

CLASSES = list("pbtdkg")
PLACE = {"p": "bilabial", "b": "bilabial", "t": "alveolar", "d": "alveolar", "k": "velar", "g": "velar"}
PAIRS = [("p", "b"), ("t", "d"), ("k", "g")]
LIPS = [61, 291, 0, 17, 13, 14, 78, 308]  # landmarks do MediaPipe em volta da boca


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


def utterance_onsets(audio, min_gap_s=0.6, min_len_s=0.6):
    """Começo de cada frase, pela energia: trechos acima do limiar separados por pausas."""
    hop = SR // 100
    rms = np.sqrt(np.mean(audio[: len(audio) // hop * hop].reshape(-1, hop) ** 2, axis=1)) + 1e-8
    db = np.convolve(20 * np.log10(rms), np.ones(5) / 5, "same")
    if db.max() < -60:
        raise ValueError("áudio mudo: confira o microfone usado na gravação")
    # Limiar no meio entre ruído (mediana: a maior parte da gravação é pausa) e fala (p99).
    active = db > (np.median(db) + np.percentile(db, 99)) / 2
    onsets, start, last = [], None, -10**9
    for i, a in enumerate(active):
        if not a:
            continue
        if start is None or i - last > min_gap_s * 100:
            if start is not None and (last - start) >= min_len_s * 100:
                onsets.append(start / 100)
            start = i
        last = i
    if start is not None and (last - start) >= min_len_s * 100:
        onsets.append(start / 100)
    return onsets


def crop_mouths(frames, landmarker):
    """Recorte 88×88 em cinza, normalizado como no treino do LRW. None se perder o rosto."""
    centers, widths = [], []
    for f in frames:
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f)))
        if not res.face_landmarks:
            return None
        pts = np.array([(res.face_landmarks[0][i].x * f.shape[1], res.face_landmarks[0][i].y * f.shape[0])
                        for i in LIPS])
        centers.append(pts.mean(0))
        widths.append(np.linalg.norm(pts[0] - pts[1]))
    # ponytail: sem alinhamento por rotação (assume rosto de frente); adicionar se a cabeça inclinar.
    kernel = np.ones(5) / 5
    centers = np.stack([np.convolve(np.pad(c, 2, mode="edge"), kernel, "valid") for c in np.array(centers).T], 1)
    half = MOUTH_SCALE * np.median(widths) / 2
    out = []
    for f, (cx, cy) in zip(frames, centers):
        x0, y0 = int(cx - half), int(cy - half)
        pad = int(2 * half)
        crop = cv2.copyMakeBorder(f, pad, pad, pad, pad, cv2.BORDER_REPLICATE)[y0 + pad:y0 + 2 * pad, x0 + pad:x0 + 2 * pad]
        gray = cv2.cvtColor(cv2.resize(crop, (96, 96)), cv2.COLOR_RGB2GRAY)[4:92, 4:92]
        out.append(gray)
    return (np.stack(out).astype(np.float32) / 255 - 0.421) / 0.165


def load_lrw(device):
    cfg = json.loads((ROOT / "third_party/lrw_tcn/configs/lrw_resnet18_dctcn.json").read_text())
    model = Lipreading(
        modality="video", num_classes=500, backbone_type=cfg["backbone_type"], relu_type=cfg["relu_type"],
        width_mult=cfg["width_mult"], use_boundary=False,
        densetcn_options={
            "block_config": cfg["densetcn_block_config"], "growth_rate_set": cfg["densetcn_growth_rate_set"],
            "reduced_size": cfg["densetcn_reduced_size"], "kernel_size_set": cfg["densetcn_kernel_size_set"],
            "dilation_size_set": cfg["densetcn_dilation_size_set"], "squeeze_excitation": cfg["densetcn_se"],
            "dropout": cfg["densetcn_dropout"]})
    ckpt = torch.load(ROOT / "checkpoints/lrw_resnet18_dctcn_video.pth", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    # Tira a camada das 500 palavras: sai a média temporal da saída do TCN (1664-d).
    model.tcn.tcn_output = torch.nn.Identity()
    return model.eval().to(device)


@torch.no_grad()
def extract(model, clips, device):
    """Ponto de troca do benchmark: recebe clipes (N, 29, 88, 88), devolve features (N, D)."""
    x = torch.from_numpy(np.stack(clips))[:, None].to(device)
    # ponytail: média no tempo perde a duração da oclusão (pista de vozeamento); testar features por quadro depois.
    return model(x, lengths=[WIN] * len(clips)).cpu().numpy()


def cv_eval(X, y):
    k = min(5, min(np.unique(y, return_counts=True)[1]))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    return cross_val_predict(clf, X, y, cv=StratifiedKFold(k, shuffle=True, random_state=0))


def main(data_dir):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT / "checkpoints/face_landmarker.task"))))
    clips, labels = [], []
    for path in sorted(p for p in Path(data_dir).iterdir() if p.suffix.lower() in {".mov", ".mp4"}):
        label, frames = label_from_name(path), read_frames(path)
        onsets = utterance_onsets(read_audio(path))
        kept = 0
        for t in onsets:
            i = round((t + WIN_OFFSET_S) * FPS)
            if i + WIN > len(frames):
                continue
            mouths = crop_mouths(frames[i:i + WIN], landmarker)
            if mouths is not None:
                clips.append(mouths)
                labels.append(label)
                kept += 1
        print(f"{path.name}: /{label}/, {len(onsets)} frases, {kept} recortes")

    X, y = extract(load_lrw(device), clips, device), np.array(labels)
    present = [c for c in CLASSES if c in y]

    pred = cv_eval(X, y)
    print(f"\n== Fonema ({len(present)} classes, acaso {1 / len(present):.0%}) ==")
    print(f"acurácia {accuracy_score(y, pred):.1%}")
    print(classification_report(y, pred, labels=present, zero_division=0))
    print("matriz de confusão (linha = verdade, coluna = previsto):", present)
    print(confusion_matrix(y, pred, labels=present))

    place = np.array([PLACE[c] for c in y])
    if len(set(place)) > 1:
        print(f"\n== Ponto ({len(set(place))} classes) == acurácia {accuracy_score(place, cv_eval(X, place)):.1%}")

    print("\n== Vozeamento dentro do par (acaso 50%) ==")
    for pair in PAIRS:
        m = np.isin(y, pair)
        if len(set(y[m])) < 2:
            continue
        hits = int((cv_eval(X[m], y[m]) == y[m]).sum())
        p = binomtest(hits, m.sum(), 0.5, alternative="greater").pvalue
        print(f"/{pair[0]}/×/{pair[1]}/: {hits}/{m.sum()} = {hits / m.sum():.1%}  (p = {p:.3f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
