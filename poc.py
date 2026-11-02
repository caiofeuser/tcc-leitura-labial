"""POC: vídeo bruto → recorte da boca → features de um modelo pré-treinado congelado → classificador das 6 oclusivas.

Uso: uv run python poc.py data/ [--modelo lrw|pt|todos]
Cada arquivo em data/ tem o nome da palavra (pato.mov, bato_2.mp4 ...) e contém várias
repetições de "Digo ___ baixinho" separadas por pausas.
"""
import argparse
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

FPS = 25  # taxa em que todos os modelos foram treinados
SR = 16000
WIN = 29  # quadros por clipe no LRW (1,16 s)
# Início da janela relativo ao começo da frase: pula o "Di" de "Digo" e cobre "go ___ bai".
# Calibrar olhando alguns recortes; nível 2 de rotulagem (MFA) substitui isto.
WIN_OFFSET_S = 0.2
# Rosto médio do LRW: os 4 modelos foram treinados com recortes alinhados a ele.
MEAN_FACE = np.load(ROOT / "third_party/vsr_ml/pipelines/detectors/mediapipe/20words_mean_face.npy")
# Pontos estáveis (olho dir., olho esq., ponta do nariz, centro da boca) no rosto médio e no MediaPipe.
REF_POINTS = np.stack([MEAN_FACE[36:42].mean(0), MEAN_FACE[42:48].mean(0), MEAN_FACE[31:36].mean(0),
                       MEAN_FACE[48:68].mean(0)]).astype(np.float32)
MP_POINTS = [(33, 133), (362, 263), (1,), (13, 14)]

CLASSES = list("pbtdkg")
PLACE = {"p": "bilabial", "b": "bilabial", "t": "alveolar", "d": "alveolar", "k": "velar", "g": "velar"}
PAIRS = [("p", "b"), ("t", "d"), ("k", "g")]


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
    """Começo de cada frase: trechos de fala separados por pausas."""
    active = speech_mask(audio)
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
        crop = face[max(cy - 48, 0):cy + 48, max(cx - 48, 0):cx + 48]
        out.append(cv2.resize(crop, (96, 96))[4:92, 4:92] if crop.shape != (96, 96) else crop[4:92, 4:92])
    return (np.stack(out).astype(np.float32) / 255 - 0.421) / 0.165


# ---- Modelos: cada loader recebe o device e devolve f(clipes (N, 29, 88, 88)) -> features (N, D).
# ponytail: média no tempo perde a duração da oclusão (pista de vozeamento); testar features por quadro depois.

def load_lrw(device):
    sys.path.insert(0, str(ROOT / "third_party/lrw_tcn"))
    from lipreading.model import Lipreading

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
    model = model.eval().to(device)
    return lambda clips: model(_batch(clips, device), lengths=[WIN] * len(clips)).cpu().numpy()


def load_pt(device):
    """Ma et al. (2022), treinado em português (CMU-MOSEAS-pt): ResNet-18 + Conformer de 12 blocos."""
    _use_repo("vsr_ml")
    from espnet.nets.pytorch_backend.transformer.encoder import Encoder

    a = json.loads((ROOT / "checkpoints/vsr_pt/model.json").read_text())[2]
    encoder = Encoder(
        idim=80, attention_dim=a["adim"], attention_heads=a["aheads"], linear_units=a["eunits"],
        num_blocks=a["elayers"], input_layer=a["transformer_input_layer"], dropout_rate=a["dropout_rate"],
        positional_dropout_rate=a["dropout_rate"], attention_dropout_rate=a["transformer_attn_dropout_rate"],
        encoder_attn_layer_type=a["transformer_encoder_attn_layer_type"], macaron_style=a["macaron_style"],
        use_cnn_module=a["use_cnn_module"], cnn_module_kernel=a["cnn_module_kernel"],
        zero_triu=a.get("zero_triu", False), a_upsample_ratio=a["a_upsample_ratio"], relu_type=a["relu_type"])
    # Só o encoder: o decodificador de texto e a CTC ficam de fora.
    sd = torch.load(ROOT / "checkpoints/vsr_pt/model.pth", map_location="cpu")
    encoder.load_state_dict({k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")})
    encoder = encoder.eval().to(device)
    return lambda clips: encoder(_batch(clips, device), None)[0].mean(1).cpu().numpy()


def load_braven(device):
    """BRAVEn Large (Haliassos et al., 2024), auto-supervisionado em inglês e ajustado para VSR no LRS3."""
    _use_repo("raven")
    from espnet.nets.pytorch_backend.transformer.encoder import Encoder

    # Hiperparâmetros de conf/model/visual_backbone/resnet_transformer_large.yaml (dropout zerado: só inferência).
    encoder = Encoder(
        idim=512, attention_dim=1024, attention_heads=16, linear_units=4096, num_blocks=24,
        frontend="conv3d", input_layer="vanilla_linear", dropout_rate=0.0, positional_dropout_rate=0.0,
        attention_dropout_rate=0.0, encoder_attn_layer_type="rel_mha", macaron_style=False,
        use_cnn_module=False, cnn_module_kernel=31, zero_triu=False, a_upsample_ratio=1, relu_type="swish",
        layerscale=True, init_values=0.1, ff_bn_pre=True, post_norm=False, gamma_zero=False, gamma_init=0.1,
        mask_init_type=None, drop_path=0.0)
    sd = torch.load(ROOT / "checkpoints/braven_large_lrs3vox2avs_vsr.pth", map_location="cpu")
    encoder.load_state_dict({k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")})
    encoder = encoder.eval().to(device)
    # Este encoder espera (N, T, 88, 88): ele mesmo adiciona o canal.
    return lambda clips: encoder(_batch(clips, device)[:, 0], None)[0].mean(1).cpu().numpy()


def load_avhubert(device):
    """AV-HuBERT Large ajustado em português (MuAViC-pt), port para Hugging Face sem fairseq. Só vídeo."""
    sys.path.insert(0, str(ROOT / "third_party/avhubert"))
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from src.model.av2text_config import AV2TextConfig
    from src.model.avhubert import AVHubertModel

    repo = "nguyenvulebinh/AV-HuBERT-MuAViC-pt"
    model = AVHubertModel(AV2TextConfig.from_pretrained(repo, token=False))
    # Só o encoder (prefixo model.encoder.); o decodificador de texto fica de fora.
    sd = load_file(hf_hub_download(repo, "model.safetensors", token=False))
    pre = "model.encoder."
    missing, _ = model.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, strict=False)
    assert not missing, missing
    model = model.eval().to(device)
    # audio=None: o próprio código zera o áudio antes da fusão (modo só-vídeo, visto no treino).
    return lambda clips: model.extract_finetune({"audio": None, "video": _batch(clips, device)})[0].mean(1).cpu().numpy()


MODELOS = {"lrw": load_lrw, "pt": load_pt, "braven": load_braven, "avhubert": load_avhubert}


def _use_repo(name):
    """vsr_ml e raven têm cada um seu pacote `espnet`; descarta o já importado antes de trocar de repo."""
    for m in [m for m in sys.modules if m == "espnet" or m.startswith("espnet.")]:
        del sys.modules[m]
    sys.path.insert(0, str(ROOT / "third_party" / name))


def _batch(clips, device):
    return torch.from_numpy(np.stack(clips))[:, None].to(device)


def cv_eval(X, y):
    k = min(5, min(np.unique(y, return_counts=True)[1]))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    return cross_val_predict(clf, X, y, cv=StratifiedKFold(k, shuffle=True, random_state=0))


def evaluate(X, y):
    """Imprime as métricas e devolve um resumo (acurácia de fonema e de vozeamento por par)."""
    present = [c for c in CLASSES if c in y]
    pred = cv_eval(X, y)
    resumo = {"fonema": accuracy_score(y, pred)}
    print(f"\n== Fonema ({len(present)} classes, acaso {1 / len(present):.0%}) ==")
    print(f"acurácia {resumo['fonema']:.1%}")
    print(classification_report(y, pred, labels=present, zero_division=0))
    print("matriz de confusão (linha = verdade, coluna = previsto):", present)
    print(confusion_matrix(y, pred, labels=present))

    place = np.array([PLACE[c] for c in y])
    if len(set(place)) > 1:
        resumo["ponto"] = accuracy_score(place, cv_eval(X, place))
        print(f"\n== Ponto ({len(set(place))} classes) == acurácia {resumo['ponto']:.1%}")

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
    y = np.array(labels)

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
            print(f"{nome:10s} " + "  ".join(f"{k}: {v:.1%}" if isinstance(v, float) else f"{k}: {v}"
                                             for k, v in r.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", nargs="?", default="data", help="pasta com os vídeos")
    ap.add_argument("--modelo", default="lrw", choices=[*MODELOS, "todos"])
    args = ap.parse_args()
    main(args.data, list(MODELOS) if args.modelo == "todos" else [args.modelo])
