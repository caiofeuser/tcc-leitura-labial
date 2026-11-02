# Planejamento do TCC (draft v1)

**Status:** rascunho para iterar. A única pendência marcada é o conjunto de palavras (**[PENDENTE]**); as demais escolhas estão fechadas como premissas (seção 10) e podem ser revistas.
**Data:** 02/11/2026 (v3: um modelo só, de vídeo; sem Comitê de Ética)
**Cronograma detalhado:** [cronograma-tcc.xlsx](cronograma-tcc.xlsx)
**Base:** [guia da reunião com a profa. Izabel](guia-fonetica.md) e os quadros do Excalidraw. Os termos de fonética usados aqui estão explicados na seção 0 do guia (glossário em 0.8).

---

## 1. Objetivo

### 1.1 Objetivo geral (sua formulação)

Reconhecer **fonemas** a partir do vídeo da boca, começando pelo menor problema que ainda é interessante, e depois ir ampliando.

### 1.2 Objetivo específico desta primeira etapa

> Classificar, a partir do **vídeo**, qual das seis consoantes oclusivas **/p b t d k g/** foi dita, usando **um modelo de vídeo pré-treinado** ajustado às nossas gravações. O áudio não entra no modelo: serve só para localizar a consoante no tempo e recortar o vídeo.

Por que esse recorte e não "todos os fonemas":

- É exatamente o foco que o Andrey pediu (pares surda/sonora).
- Reconhecer **fonema** (e não visema) é o que obriga o modelo a separar /p/ de /b/. Se o alvo fosse visema, /p/ e /b/ seriam a mesma classe e o problema do vozeamento sumiria.
- Seis classes, em palavras controladas, cabem num corpus pequeno gravado por você.

Leitura alternativa de "reconhecer os fonemas primeiro": reconhecer **todos** os fonemas do PB numa fala contínua. Deixei isso como evolução (seção 9), porque precisa de muito mais dados e esconderia a pergunta principal.

### 1.3 Pergunta de pesquisa

> Existem pistas **visuais** que distinguem as oclusivas surdas das sonoras no português brasileiro, e um modelo treinado com áudio + vídeo consegue aprendê-las a ponto de acertar acima do acaso **só com o vídeo**?

### 1.4 Hipóteses (para testar, não para assumir)

- **H1 (ponto de articulação):** o vídeo separa bem os três pontos (bilabial, alveolar, velar), com /p b/ mais fácil e /k g/ mais difícil.
- **H2 (vozeamento):** dentro de cada par, o vídeo acerta acima de 50% (acaso), principalmente em /p b/, usando pistas como duração da oclusão e duração da vogal anterior (guia, seção 3.3).
- **H3 (taxa de quadros):** vídeo a 60 fps ou mais acerta mais o vozeamento do que o mesmo vídeo reduzido para 30 fps.

Se H2 der negativo (o vídeo não separa surda/sonora), isso **também é um resultado válido** para o TCC: mostra, com dados medidos, o limite da leitura labial no PB.

---

## 2. A tarefa, em termos concretos

**Entrada:** um trecho curto de vídeo (e, no treino, o áudio sincronizado) recortado em volta de uma oclusiva, por exemplo a palavra "pato" dentro de "Digo pato baixinho".
**Saída:** uma das seis classes /p b t d k g/.

Dá para olhar o mesmo resultado em três níveis, do mais fácil ao mais difícil:

| Nível | Pergunta | Classes | Acaso |
|---|---|---|---|
| A | Qual o **ponto** de articulação? (= qual visema) | 3 | 33% |
| B | Dentro do par, **surda ou sonora**? | 2 | 50% |
| C | Qual **fonema** exatamente? (A e B juntos) | 6 | 17% |

O nível C é o objetivo; A e B ajudam a entender onde o modelo erra.

---

## 3. Escopo

**Dentro:**
- Seis oclusivas em palavras controladas (pares mínimos em frase-veículo).
- Corpus próprio pequeno, gravado em condição controlada.
- Avaliação com falantes que o modelo **não viu** no treino.
- Comparação áudio só × vídeo só × áudio no treino + vídeo no teste.

**Fora (por enquanto):**
- Fala contínua, frases livres, outros fonemas.
- Vídeo "do mundo real" (YouTube, TV, iluminação ruim, rosto de lado).
- Sistema em tempo real.

---

## 4. Dados

### 4.1 Corpus (proposta do guia, a ajustar com o que a Izabel disse)

- **Palavras: [PENDENTE].** Critério: pares mínimos com a oclusiva no início, seguida de vogal tônica /a/, /o/, /u/ (e /i/ só onde não houver palatalização de /t d/). Tabela inicial na seção 4 do guia.
- **Frase-veículo:** "Digo ___ baixinho" (troca-se se a Izabel indicar outra).
- **Repetições:** 5 a 10 por palavra, em ordem aleatória.
- **Falantes:** começar com 5, meta de 10 a 15, recrutados na UFSC. A região de origem de cada falante é registrada, e a palatalização é evitada pela escolha das palavras.
- **Balanceado ou frequência natural?** Para esta etapa, proponho **balanceado** (mesmo número de exemplos por classe), porque facilita a avaliação. A frequência natural do PB passa a importar quando formos para fala contínua.

Tamanho estimado: ~24 palavras × 8 repetições × 10 falantes ≈ 1.900 exemplos (~320 por classe). A seção 4.2 explica de onde vem esse número.

### 4.2 Quanto dado é preciso?

**Resposta curta:** algumas horas de gravação no máximo, não milhares. O que importa aqui não são horas, e sim **exemplos por classe** e **número de falantes**.

**Por que os números da literatura são enganosos.** As "milhares de horas" aparecem em leitura labial de **frases livres** (o modelo precisa aprender vocabulário inteiro, gramática, milhares de palavras). Este TCC é um problema de **classificação** de trechos curtos em 6 classes, em fala controlada. É outro tamanho de problema.

**Referências para comparar** (valores aproximados, conferir nos artigos antes de citar no texto):

| Base | Tarefa | Tamanho | Observação |
|---|---|---|---|
| LRS3 / LRS2 | Frases livres (inglês, TV/TED) | centenas de horas cada | É daí que vêm os "milhares de horas" somando bases. Não é o nosso caso. |
| LRW ([Chung & Zisserman, 2016](https://twine.net/dataset/lip-reading-in-the-wild-lrw)) | Classificar 1 de 500 palavras | ~1.000 exemplos por palavra, centenas de falantes | Problema de classificação, como o nosso, mas com 500 classes "no mundo real". Modelos com ResNet + TCN passam de 85% ([Stafylakis et al.](https://arxiv.org/pdf/1811.01194)). |
| GRID ([site](https://spandh.dcs.shef.ac.uk/gridcorpus/)) | Frases de vocabulário fixo (51 palavras) | 34 falantes × 1.000 frases (~28 h) | Gravado em estúdio, condição controlada como a nossa. |
| Estudos de fonética (VOT) | Medir surdas × sonoras | 5 a 20 falantes × poucas repetições | Padrão da fonética experimental para mostrar diferença entre /p/ e /b/. |

O nosso caso fica entre a fonética experimental e o LRW: 6 classes (não 500), fala controlada (não TV) e um modelo que parte de pesos já treinados em centenas de horas de inglês.

**Três parâmetros práticos para decidir o tamanho:**

1. **Exemplos por classe para o modelo aprender.** Com pré-treino, ajustar só a parte final de uma rede costuma funcionar com **algumas centenas de exemplos por classe**. Meta proposta: **300 a 500 por classe** (≈ 1.800 a 3.000 exemplos).
2. **Exemplos de teste para provar o resultado.** Para afirmar que o vídeo separa surda de sonora acima do acaso (50%), com 95% de confiança e 80% de chance de detectar o efeito, um teste binomial precisa aproximadamente de:

   | Acurácia real de vozeamento | Exemplos de teste por par |
   |---|---|
   | 70% | ~40 |
   | 60% | ~150 |
   | 55% | ~600 |

   Ou seja: se o efeito for pequeno (55%), precisa de bem mais dados para conseguir mostrar que ele existe.
3. **Número de falantes.** Para generalizar para pessoas que o modelo não viu, **mais falantes vale mais do que mais repetições**. Mínimo razoável: **10 falantes**; 15 a 20 é confortável.

**Em tempo de gravação:** cada frase-veículo dura ~2 s. 3.000 exemplos ≈ 1h40 de fala útil, ou cerca de **10 a 15 minutos por falante** (com pausas, uns 20 a 30 min de sessão). É viável para um TCC.

**Como saber se precisa de mais:** fazer uma **curva de aprendizado**: treinar com 25%, 50%, 75% e 100% dos dados e ver se a acurácia ainda está subindo. Se ainda sobe no 100%, vale gravar mais falantes; se estabilizou, o tamanho basta. Isso pode ser feito já com os primeiros 5 falantes, e decide se a coleta continua.

### 4.3 Gravação

- **Câmera:** 60 fps no mínimo, idealmente 120 ou 240 fps (celulares atuais gravam câmera lenta em 240 fps). Rosto inteiro e pescoço no quadro, de frente, iluminação uniforme.
- **Áudio:** microfone de lapela ou condensador, sala silenciosa (perguntar ao FONAPLI/LVA por cabine).
- **Sincronia áudio-vídeo:** gravar um sinal de referência no início (bater palmas ou claquete), porque um erro de sincronia de 1 quadro já é do tamanho do VOT.

### 4.4 Rotulagem (do menos ao mais trabalhoso)

Testam-se os níveis na ordem; só se passa ao seguinte se o anterior não for preciso o bastante.

1. **Roteiro + janela fixa:** a classe vem do roteiro (sabemos qual palavra foi pedida) e o recorte é uma janela de tamanho fixo em volta da palavra, localizada pela energia do áudio. Quase nenhum trabalho manual.
2. **Automático pelo áudio:** um modelo de áudio pré-treinado ou o alinhador automático (MFA). Detalhe do alinhador: o Montreal Forced Aligner, com modelo de PB, marca no áudio onde começa e termina cada som, permitindo recortar a consoante com mais precisão.
3. **MFA + correção manual no Praat:** ajustar à mão as fronteiras de oclusão e soltura e medir o VOT, numa amostra ou no corpus todo.

Critério para avançar: comparar o resultado do modelo com cada nível numa amostra pequena corrigida no Praat. Se o ganho for pequeno, fica-se no nível mais barato.

O piloto é feito com gravações do próprio autor.

---

## 5. Modelo

Um modelo só, de vídeo: recorte da boca → modelo de leitura labial **pré-treinado** em inglês (bases LRS3/LRW) → camada final nova com 6 classes. Ajustamos o modelo às nossas gravações em vez de treinar do zero.

O áudio não é entrada do modelo. Ele só ajuda a achar o momento da consoante para recortar o vídeo; qual consoante foi dita já se sabe pelo roteiro.

Extra, se sobrar tempo: usar um modelo de áudio como "professor" do modelo de vídeo e ver se melhora o resultado.

## 6. Avaliação

- **Métricas:** acurácia, F1 por classe, **matriz de confusão** 6×6 (mostra se o erro é de ponto ou de vozeamento), acurácia de vozeamento por par.
- **Separação por falante:** treinar com uns falantes e testar com outros (*leave-one-speaker-out*). Sem isso o modelo pode decorar o rosto de cada um.
- **Comparações:** 30 fps × 60 fps × 120+ fps (reduzindo a taxa do mesmo vídeo); palavras com /a/ × /o/ × /u/; nível de rotulagem 1 × 2 × 3.
- **Teste estatístico:** verificar se a acurácia de vozeamento é significativamente acima de 50% (ex.: teste binomial), não só "um pouco acima".

---

## 7. Etapas e cronograma

A descrição de cada etapa, com datas e entregas, está em [cronograma-tcc.xlsx](cronograma-tcc.xlsx). Premissa: início em 02/11/2026 e cerca de 8 meses (32 semanas) até a defesa. Basta trocar a data de início na planilha para que todas as datas se ajustem.

O piloto é o passo mais importante do começo: em poucas semanas ele mostra se a ideia tem sinal, antes de investir em gravar muita gente.

## 8. Riscos

| Risco | Efeito | Plano B |
|---|---|---|
| Efeito de vozeamento pequeno | Precisa de muitos dados para provar | Curva de aprendizado com os primeiros falantes (seção 4.2); ampliar falantes se necessário |
| Vídeo não separa surda/sonora (H2 falha) | Resultado "negativo" | Tratar como resultado: medir e explicar o limite; reforçar a análise do nível A e das medidas acústicas (VOT, duração da oclusão) |
| Poucos dados para deep learning | O modelo não aprende | Pré-treino em inglês, aumento de dados (espelhar, variar brilho), modelo menor |
| Rotulagem barata imprecisa | Recortes ruins | Subir de nível (seção 4.4) só onde for preciso |
| Sem GPU suficiente | Treino lento | Modelos pequenos, Google Colab, perguntar ao LVA |

---

## 9. Como pode evoluir depois

Na ordem em que eu proporia, cada passo reaproveitando o anterior:

1. **Outras consoantes com o mesmo problema visual:** /f/×/v/, /s/×/z/, /m/ (nasal no mesmo visema de /p b/).
2. **Todas as consoantes e vogais em palavras controladas** → mapa de visemas do PB medido (comparando com os mapas da literatura).
3. **Fala contínua** (frases lidas) → reconhecimento de sequência de fonemas; aqui a frequência natural dos fonemas no PB passa a importar.
4. **Dataset público** de leitura labial em PB (não existe nenhum hoje), se os falantes autorizarem a divulgação das gravações.

---

## 10. Premissas adotadas

Escolhas fechadas para seguir em frente; qualquer uma pode ser revista.

1. Objetivo: as 6 oclusivas, só o vídeo entra no modelo; o áudio serve só para recortar (seção 1.2).
2. Modelo: um só, de vídeo, pré-treinado e ajustado aos nossos dados (seção 5).
3. Rotulagem: do menos ao mais trabalhoso (seção 4.4).
4. Corpus balanceado, frase-veículo "Digo ___ baixinho", falantes da UFSC.
5. Um resultado negativo em H2 é aceito como resultado.
6. Piloto com gravações do próprio autor; sem submissão ao Comitê de Ética.
7. Prazo: ~32 semanas a partir de 02/11/2026.
8. Treino em GPU do LVA ou Google Colab.
9. Tamanho do corpus: 300 a 500 exemplos por classe, 10 a 15 falantes, ajustado pela curva de aprendizado (seção 4.2).

**Pendente:** conjunto de palavras (seção 4.1).
