# TCC: leitura labial das oclusivas no português brasileiro

Reconhecer pela imagem da boca qual das consoantes /p b t d k g/ foi dita, usando o áudio só para recortar o vídeo.

Contexto completo em `CLAUDE.md` e na pasta `docs/`.

## POC: como rodar

```
uv run python poc.py data                  # modelo padrão (lrw)
uv run python poc.py data --modelo pt      # lrw | pt | braven | avhubert
uv run python poc.py data --modelo todos   # roda todos e imprime uma tabela comparando
uv run python controles.py data            # controle negativo (pausas) + teste de permutação
```

`controles.py` repete a classificação em janelas das **pausas**, onde não há consoante. Se ali também der acima do acaso, o modelo está separando as gravações (sessão), não as consoantes.

`data/` tem um vídeo por palavra, com o nome da palavra (`pato.mov`, `bato.mov`...). Em cada vídeo, a frase "Digo ___ baixinho" é repetida, com ~2 s de pausa entre as repetições. O rótulo sai do nome do arquivo (`cato` → /k/).

Código e pesos dos modelos ficam fora do git (licenças não comerciais): `third_party/` e `checkpoints/`.

| `--modelo` | Modelo | Treino | Código | Pesos em `checkpoints/` |
|---|---|---|---|---|
| `lrw` | ResNet-18 + DC-TCN (Ma et al., 2022) | LRW, inglês, palavras isoladas | `mpc001/Lipreading_using_Temporal_Convolutional_Networks` → `third_party/lrw_tcn` | `lrw_resnet18_dctcn_video.pth` |
| `pt` | ResNet-18 + Conformer (Ma et al., 2022) | CMU-MOSEAS, **português** | `mpc001/Visual_Speech_Recognition_for_Multiple_Languages` → `third_party/vsr_ml` | `vsr_pt/` |
| `braven` | BRAVEn Large (Haliassos et al., 2024) | auto-supervisionado + LRS3, inglês | `ahaliassos/raven` → `third_party/raven` | `braven_large_lrs3vox2avs_vsr.pth` |
| `avhubert` | AV-HuBERT Large (Shi et al., 2022) | MuAViC, **português** | `nguyenvulebinh/AV-HuBERT-S2S` → `third_party/avhubert` | cache do Hugging Face (`nguyenvulebinh/AV-HuBERT-MuAViC-pt`, baixa sozinho) |

`third_party/vsr_ml` é usado por todos: ele traz o rosto médio do LRW, que serve de referência para alinhar o recorte. O MediaPipe precisa de `checkpoints/face_landmarker.task`.

## Arquivos

| Arquivo | O que tem |
|---|---|
| `poc.py` | O pipeline: leitura do vídeo, segmentação pelo áudio, recorte da boca e avaliação |
| `modelos.py` | Os 4 extratores congelados e o `MODELOS`, que liga o nome do `--modelo` ao loader |
| `controles.py` | Controle negativo (pausas) e teste de permutação |
| `test_poc.py` | Teste da segmentação num áudio sintético (`uv run python test_poc.py`) |

## O que o script faz

1. **Segmenta pelo áudio:** acha o começo de cada frase pelas pausas (energia do som).
2. **Recorta no tempo:** 29 quadros a 25 fps (1,16 s), começando 0,2 s depois do início da frase (`WIN_OFFSET_S`).
3. **Recorta a boca:** alinha o rosto ao rosto médio do LRW (olhos, nariz e boca, pelo MediaPipe), recorta 96×96 na boca e fica com o centro 88×88, em cinza. É a mesma receita com que os modelos foram treinados.
4. **Extrai features:** o modelo escolhido roda **congelado**, e a média no tempo da saída do encoder vira um vetor por clipe.
5. **Classifica:** uma regressão logística é treinada nesses vetores (*linear probe*) e avaliada por validação cruzada de 5 partes.

## Como ler a saída

- `pato.mov: /p/, 5 frases, 5 recortes`: controle de qualidade. "frases" tem que bater com o que foi dito.
- **acurácia**: fração de clipes certos. Compare sempre com o acaso (50% com 2 classes, 17% com 6).
- **precision / recall / f1**: por classe. Precision responde "quando disse p, era p?"; recall responde "dos p, quantos achou?".
- **matriz de confusão**: as linhas são a verdade e as colunas a previsão. Com as 6 classes, mostra se o erro é de ponto (p↔t) ou de vozeamento (p↔b).
- **vozeamento `/p/×/b/: 9/10 (p = 0.011)`**: teste binomial contra o acaso (50%). Só p < 0,05 permite dizer que o vídeo separa surda de sonora.

## Limites conhecidos

- Com um falante e ~10 clipes, os números servem para provar o pipeline, não para tirar conclusões.
- Um vídeo por palavra mistura classe com sessão (luz, posição): o modelo pode aprender a sessão. O piloto deve misturar as palavras na mesma gravação.
- A média no tempo perde durações (oclusão, vogal), que são pistas de vozeamento.
