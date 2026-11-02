"""Extratores pré-treinados congelados.

Cada loader recebe o device e devolve f(clipes) -> features: clipes é uma lista de arrays
(T, 88, 88) já normalizados, e a saída é (N, D) com a média no tempo da saída do encoder.
O decodificador de texto de cada modelo é sempre descartado.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent

# ponytail: média no tempo perde a duração da oclusão (pista de vozeamento); testar features por quadro depois.


def load_lrw(device):
    """ResNet-18 + DC-TCN (Ma et al., 2022), palavras isoladas em inglês (LRW)."""
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
    model.load_state_dict(torch.load(ROOT / "checkpoints/lrw_resnet18_dctcn_video.pth", map_location="cpu")
                          ["model_state_dict"])
    # Tira a camada das 500 palavras: sai a média temporal da saída do TCN (1664-d).
    model.tcn.tcn_output = torch.nn.Identity()
    model = model.eval().to(device)
    return lambda clips: model(_batch(clips, device), lengths=[len(c) for c in clips]).cpu().numpy()


def load_pt(device):
    """ResNet-18 + Conformer de 12 blocos (Ma et al., 2022), treinado em português (CMU-MOSEAS-pt)."""
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
    encoder.load_state_dict(_only(torch.load(ROOT / "checkpoints/vsr_pt/model.pth", map_location="cpu"), "encoder."))
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
    encoder.load_state_dict(_only(sd, "encoder."))
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
    missing, _ = model.load_state_dict(
        _only(load_file(hf_hub_download(repo, "model.safetensors", token=False)), "model.encoder."), strict=False)
    assert not missing, missing
    model = model.eval().to(device)
    # audio=None: o próprio código zera o áudio antes da fusão (modo só-vídeo, visto no treino).
    return lambda clips: model.extract_finetune({"audio": None, "video": _batch(clips, device)})[0].mean(1).cpu().numpy()


MODELOS = {"lrw": load_lrw, "pt": load_pt, "braven": load_braven, "avhubert": load_avhubert}


def _only(state_dict, prefix):
    """Fica só com os pesos de um submódulo, sem o prefixo."""
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _use_repo(name):
    """vsr_ml e raven têm cada um seu pacote `espnet`; descarta o já importado antes de trocar de repo."""
    for m in [m for m in sys.modules if m == "espnet" or m.startswith("espnet.")]:
        del sys.modules[m]
    sys.path.insert(0, str(ROOT / "third_party" / name))


def _batch(clips, device):
    return torch.from_numpy(np.stack(clips))[:, None].to(device)
