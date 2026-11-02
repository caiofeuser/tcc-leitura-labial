# TCC: leitura labial das oclusivas /p b t d k g/ no português brasileiro

Projeto de TCC do Caio (Eng. Mecânica, UFSC). Orientador: prof. Andrey Ricardo da Silva (LVA, Laboratório de Vibrações e Acústica). Apoio em fonética: profa. Izabel Christine Seara (FONAPLI/UFSC).

Escreva comentários, docs e mensagens para o Caio em **português (pt-BR)**. Ele vem da engenharia e é iniciante em fonética: explique termos de fonética antes de usá-los (glossário em `docs/guia-fonetica.md`, seção 0.8).

## Objetivo

Reconhecer, **só pelo vídeo da boca**, qual das seis consoantes oclusivas /p b t d k g/ foi dita. O áudio não entra no modelo; serve só para localizar a consoante e recortar o vídeo.

A parte difícil (e a pergunta científica): /p/-/b/, /t/-/d/ e /k/-/g/ têm a mesma forma de boca (mesmo *visema*). A diferença é o **vozeamento** (vibração das pregas vocais), que acontece na laringe e não aparece direto na imagem. No PB, /p t k/ não são aspiradas e /b d g/ têm pré-vozeamento, então resultados de leitura labial em inglês não transferem direto.

Classes: 6 (`p b t d k g`). Avaliar em três níveis: ponto de articulação (3 classes: bilabial/alveolar/velar), vozeamento dentro do par (2 classes, acaso = 50%) e fonema (6 classes).

## Modelo

**Um modelo só, de vídeo.** Recorte da boca → modelo de leitura labial pré-treinado (inglês, LRS3/LRW) → camada final nova com 6 classes, ajustada às nossas gravações.

O áudio **não** é entrada do modelo: serve só para localizar a consoante no tempo e recortar o vídeo. O rótulo vem do roteiro (sabemos qual palavra foi pedida). O modelo de áudio pré-treinado (alinhador automático) é a **ferramenta de anotação**: marca o instante da consoante. Nunca é entrada do modelo de vídeo.

## Primeira POC (o que fazer agora)

Objetivo da POC: provar o pipeline de ponta a ponta com **um modelo pré-treinado**, antes do piloto. Prazo: 04/11 a 03/12/2026 (1 mês). A primeira semana é para escolher um modelo que rode no Mac. A revisão bibliográfica aprofundada só vem depois do piloto; aqui basta a busca pelo modelo.

1. Escolher o modelo pré-treinado de leitura labial (verificar disponibilidade de pesos, licença e se roda em Mac Apple Silicon/MPS). Candidatos a avaliar, **não verificados**:
   - Auto-AVSR (Ma et al., 2023), repositório `mpc001/auto_avsr`, pesos treinados em LRS3;
   - AV-HuBERT (Meta), `facebookresearch/av_hubert`, auto-supervisionado em áudio + vídeo;
   - modelos de reconhecimento de palavras isoladas do LRW (ResNet-18 + TCN, Ma/Martinez/Petridis/Pantic).
2. Usar o front-end visual pré-treinado **congelado** como extrator de features e treinar só um classificador pequeno (linear ou MLP) para as 6 classes.
3. Dados da POC: alguns vídeos curtos gravados no celular pelo próprio Caio (2 ou 3 pares de palavras, ex.: pato/bato, cato/gato, tava/dava). O piloto completo (30/60/240 fps) vem depois da POC.
4. Saída esperada: script que vai de vídeo bruto → recorte da boca → features → acurácia + matriz de confusão.

Pergunte ao Caio antes de decisões que mudem o objetivo (classes, modelo, formato dos dados).

## Dados e rotulagem

- Frase de gravação: "Digo ___ baixinho". Lista de palavras **ainda pendente** (pares mínimos, oclusiva inicial + vogal tônica /a/ /o/ /u/; evitar /t d/ + [i] por causa da palatalização "tch/dj").
- Vídeo: ≥ 60 fps (ideal 120–240), rosto inteiro e pescoço no quadro. Sincronizar áudio e vídeo com um sinal no início (palma/claquete): 1 quadro de erro já é do tamanho do VOT.
- Anotação do instante da consoante, do menos ao mais trabalhoso (só avança se compensar): (1) janela fixa em volta da palavra (localizada pela energia do áudio; antes de /p b/ etc. há um quase-silêncio seguido do estouro); (2) anotação automática pelo áudio com alinhador (Montreal Forced Aligner, modelo de PB); (3) anotação manual no Praat.
- Tamanho-alvo do corpus final: 300–500 exemplos por classe, 10–15 falantes. Decidir se precisa de mais com curva de aprendizado (25/50/75/100% dos dados).
- Sem Comitê de Ética neste projeto.

## Avaliação

- Sempre separar por falante (leave-one-speaker-out); nunca o mesmo falante em treino e teste.
- Métricas: acurácia, F1 por classe, matriz de confusão 6×6, acurácia de vozeamento por par + teste binomial contra 50%.
- Comparações: 30 × 60 × 120+ fps (reduzindo o mesmo vídeo), vogal seguinte, nível de rotulagem.

## Ambiente

- MacBook Pro Apple Silicon (arm64). Usar PyTorch com MPS quando possível; treinos pesados podem ir para GPU do LVA ou Google Colab.
- Python com `uv` (já instalado).
- Dados, vídeos e pesos ficam fora do git (`data/`, `checkpoints/`, ver `.gitignore`).

## Documentos

- `docs/planejamento.md`: planejamento completo (hipóteses, riscos, evolução).
- `docs/cronograma-tcc.xlsx`: etapas e datas (01/11/2026 a 27/06/2027).
- `docs/guia-fonetica.md`: fonética do zero (VOT, oclusão, soltura, visema etc.).
