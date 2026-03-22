# Event-based Temporal Action Detection (TFG)

Este repositorio contén unha reprodución e extensión dun pipeline de detección temporal de accións baseado en cámaras de eventos, tomando como base o repositorio orixinal *event_penguins* e o método publicado por Hamann et al. en CVPR 2024. O repositorio orixinal estrutura o proxecto en `config/`, `docs/`, `scripts/`, `src/`, `requirements.txt` e un fluxo de uso baseado en preprocesado e inferencia. :contentReference[oaicite:0]{index=0} O paper define un pipeline en dúas etapas: xeración de propostas temporais e clasificación posterior mediante CNN. :contentReference[oaicite:1]{index=1}

---

## Descrición

O problema abordado é a **Temporal Action Detection (TAD)** sobre datos de cámaras de eventos. A entrada do sistema é unha secuencia de eventos da forma:

E = {(x, y, t, p)}

onde cada evento representa un cambio de intensidade nun píxel concreto, nun instante temporal concreto, cunha polaridade asociada.

A diferenza dos sistemas baseados en vídeo convencional, as cámaras de eventos ofrecen propiedades especialmente útiles para observación continua:

- alta resolución temporal
- baixo consumo enerxético
- robustez fronte a condicións de iluminación difíciles
- representación naturalmente centrada no movemento

Este proxecto céntrase na análise e mellora da **etapa de xeración de propostas temporais**.

---

## Pipeline base

O método orixinal segue, a nivel conceptual, o seguinte fluxo:

1. Cálculo da taxa de eventos r(t)
2. Normalización robusta do sinal
3. Definición de actionness a partir da magnitude da actividade
4. Xeración de propostas temporais mediante **reTAG**
5. Clasificación das propostas mediante **ATSN**

Segundo o repositorio orixinal, o fluxo práctico de uso baséase nun paso de preprocesado que xera `preprocessed.h5` e nun paso de inferencia que executa o pipeline completo. :contentReference[oaicite:2]{index=2}

---

## Limitación do método base

A principal limitación detectada é que o actionness do método base depende case exclusivamente da magnitude da actividade, é dicir, da intensidade do sinal de eventos.

Isto implica que o sistema responde ben á pregunta:

“hai movemento?”

pero non necesariamente ás preguntas:

- “ese movemento é relevante?”
- “ese movemento corresponde á acción obxectivo?”
- “ese movemento é ruído ambiental?”

Na práctica, isto pode producir:

- falsos positivos debidos a ruído
- propostas temporais pouco discriminativas
- dificultades para separar accións con patróns de actividade similares

---

## Obxectivo do TFG

O obxectivo deste TFG é mellorar a calidade das propostas temporais redefinindo o concepto de **actionness**.

En lugar de empregar un score baseado só na magnitude de actividade, proponse un score máis completo para cada proposta temporal I = (t_a, t_b), combinando:

- magnitude da actividade
- consistencia temporal
- estrutura espacial dos eventos
- indicadores ou penalizacións de ruído

A CNN final non se modifica. O foco do traballo está exclusivamente na fase previa de xeración e selección de propostas.

---

## Liña de traballo

A estratexia xeral consiste en:

1. reproducir o pipeline base
2. analizar o comportamento de reTAG
3. incorporar descritores adicionais ao cálculo de actionness
4. reordenar ou filtrar propostas segundo o novo score
5. avaliar o impacto na calidade das propostas e no rendemento final

O enfoque é deliberadamente clásico e interpretable, evitando introducir deep learning adicional na fase de propostas.

---

## Estrutura do repositorio

A organización do repositorio mantense próxima á do proxecto orixinal, con especial atención ás partes relevantes para o TFG:

- `config/`
  - ficheiros de configuración de experimentos e anotacións

- `docs/`
  - documentación auxiliar do proxecto

- `scripts/`
  - `preprocess.py`: preparación e reestruturación dos datos
  - `inference.py`: execución do pipeline completo

- `src/`
  - implementación da lóxica principal do sistema
  - módulos relacionados coa xeración de propostas, descritores e clasificación

- `requirements.txt`
  - dependencias do proxecto

No repositorio orixinal, os datos procesados almacénanse nun único ficheiro `preprocessed.h5`, organizado por gravación e por ROI ou niño. :contentReference[oaicite:3]{index=3}

---

## Instalación

Crear contorno:

```bash
conda create --name eventpenguins python=3.8
conda activate eventpenguins
```

Instalar PyTorch segundo a versión de CUDA correspondente.

Instalar o resto de dependencias:

pip install -r requirements.txt

No repositorio orixinal, PyTorch foi probado con versión 2.2.2. :contentReference[oaicite:4]{index=4}

---

## Preparación dos datos

Crear directorio de datos:

```bash
mkdir data
```

Descargar os datos do proxecto base e gardalos dentro de `data/`.

Executar o preprocesado:

```bash
python scripts/preprocess.py --data_root data/EventPenguins --output_dir data --recording_info_path config/annotations/recording_info.csv
```

Isto xera o ficheiro:

```bash
`data/preprocessed.h5`
```

Segundo o repositorio orixinal, o preprocesado recorta os eventos segundo os niños previamente anotados e organiza as gravacións segundo o split definido polo método. :contentReference[oaicite:5]{index=5}

---

## Modelos

Crear directorio de modelos:

```bash
mkdir models
```

Descargar os pesos preadestrados do proxecto base e gardalos en `models/` se se desexa executar a inferencia orixinal completa. :contentReference[oaicite:6]{index=6}

---

## Execución

Para executar a inferencia do pipeline:

```bash
python scripts/inference.py --config config/exp/inference.yaml --verbose
```

---

## Foco deste repositorio

Este traballo céntrase especificamente en:

- análise da taxa de eventos
- reformulación do actionness
- mellora da xeración de propostas temporais
- redución de falsos positivos
- maior robustez fronte a ruído e patróns ambiguos

Non forma parte do alcance deste TFG:

- redeseñar a arquitectura ATSN
- introducir modelos adicionais de deep learning
- converter o sistema nun problema multi-clase xeral

---

## Dataset e anotacións

O proxecto base emprega un conxunto de 24 gravacións de dez minutos con 16 niños anotados, e as anotacións seguen unha estrutura similar a ActivityNet, incorporando unha capa adicional por niño. O método traballa sobre eventos recortados por ROI, non sobre a escena completa. :contentReference[oaicite:7]{index=7}

---

## Referencias

Se empregas este repositorio ou o método base no teu traballo, considera citar o artigo orixinal:

Hamann, F., Ghosh, S., Juarez Martinez, I., Hart, T., Kacelnik, A., Gallego, G.
Low-power Continuous Remote Behavioral Localization with Event Cameras.
CVPR 2024.

Repositorio base:
tub-rip/event_penguins
