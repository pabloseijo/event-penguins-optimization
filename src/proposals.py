import os
from multiprocessing import Pool
import pickle

import numpy as np
from absl import logging
import pandas as pd
import h5py

from src.utils import temporal_nms

def log_to_txt(file_path, text):
    with open(file_path, "a") as f:
        f.write(text + "\n")

def get_event_rate(events, bin_width):
    '''
    events: é un array de dimensión (n_events, 4) onde cada fila é un evento e as columnas son [x, y, t, p], sendo x e y as coordenadas do evento e t o tempo do evento en microsegundos.
    bin_width: ancho do bin para calcular a taxa de eventos, en microsegundos. Por exemplo, se queremos calcular a taxa de eventos cada 10 ms, o ancho do bin sería 10 * 1e3 = 1e4 microsegundos.
    '''
 
    # o menos un en python é pra coller o último elemento, neste caso a última linha da matriz. Cóllese entón fila 0, columna 2 (t) da primeira fila e fila -1, columna 2 (t) da última fila.
    # Implícase que os eventos están ordeados por tempo, polo que asumimos que o dataset funciona así, pero en calquera caso pode ser unha fonte de erro de cara a xeralización.
    t_min, t_max = events[0, 2], events[-1, 2] 
    
    # calcúlase o numero de bins necesarios para cubrir o rango de tempo dos eventos. O resultado é un número enteiro que indica cantos bins (defínese nos apuntes de obsidian) se necesitan para cubrir todo o rango de tempo.
    bin_num = int((t_max - t_min) / bin_width)
    
    # calculamos entón cantos eventos (elementos de counts) hai por cada intervalo temporal (elementos de bins). Por exemplo 
    # [0.00–0.03) → 10 eventos
    # [0.03–0.06) → 30 eventos
    # (0.06–0.09) → 5 eventos
    # Isto é a imaxe mental que obtemos de facer o event rate, que calculamos pasandolle de parámetros a columna de tempo dos eventos como un array, e o número de bins, para que calcule internamente as divisiós do tempo 
    counts, bins = np.histogram(events[:, 2], bins=bin_num)
    return counts, bins


def apply_robust_min_max(rate, percentile):
    #Isto é o que se fai antes de pasarlle o event rate ao actioness para facelo lixeiramente máis robusto. Veremos o que se fai. 
    
    # Entón pasaselle rate, que como viumos na función anterior é o número de eventos por bin temporal. Pero pode ter unha serie de problemas, de ruido mesmamente. Vemos o que fan pra solucionalo.
    
    # Este paso aplica un recorte robusto á taxa de eventos (rate) antes de calcular
    # o actionness. O obxectivo é reducir o impacto de valores extremos (outliers),
    # que poden aparecer debido a ruído puntual ou artefactos do sensor.
    #
    # O método consiste en calcular percentís inferior e superior da distribución
    # de rate e saturar os valores fóra dese rango.
    #
    # Limitacións:
    # - Os percentís son fixos a nivel global, polo que non se adaptan ás características
    #   específicas de cada secuencia.
    # - Non distingue entre ruído sostido e actividade relevante, xa que ambos poden
    #   producir valores elevados de forma consistente.
    # - A saturación pode reducir a variabilidade interna do sinal en rexións de alta
    #   actividade, diminuíndo a capacidade discriminativa posterior.
    #
    # Isto motiva a necesidade de mellorar a modelización do sinal de entrada,
    # incorporando criterios adicionais (por exemplo, información espacial ou
    # estrutura temporal) en lugar de depender unicamente da magnitude da actividade.
    
    rmin = np.percentile(rate.flat, 0.5 * percentile)
    rmax = np.percentile(rate.flat, 100 - 0.5 * percentile)
    
    rate[rate < rmin] = rmin
    rate[rate > rmax] = rmax
    
    return rate


def get_index_proposals_from_1d_score(
    score1d: np.ndarray, threshold: float
) -> np.ndarray:
    """
    Xera propostas temporais a partir dun sinal 1D mediante segmentación por limiar.

    ENTRADA:
    - score1d: array 1D que representa un sinal ao longo do tempo.
      No contexto deste proxecto, corresponde ao actionness, é dicir,
      unha medida da actividade baseada na taxa de eventos.
    - threshold: limiar escalar (o lambda para seren máis exactos). Define a partir de que valor do sinal
      se considera que hai actividade relevante.

    PROCESO:
    - Constrúese unha máscara booleana (score1d > threshold), que indica
      en que instantes o sinal supera o limiar. É dicir, un array de 0 e 1 onde 
      1 indica que o sinal supera o limiar e 0 indica que non.
    - Calcúlase a diferenza discreta (np.diff) sobre esa máscara, o que
      permite detectar transicións:
          0 → 1 : inicio dun segmento activo
          1 → 0 : fin dun segmento activo
    - Engádense valores ao principio e ao final (prepend, append) para
      garantir que se detectan correctamente os segmentos nos bordes.
    - Extráense os índices destas transicións e reagrúpanse en pares
      (inicio, fin), representando intervalos continuos de actividade.

    SAÍDA:
    - Array de tamaño (n, 2), onde cada fila define unha proposta temporal
      en índices do sinal: [start_index, end_index].

    LIMITACIÓNS (IMPORTANTE PARA O TFG):
    - A xeración de propostas depende exclusivamente dun limiar fixo, polo
      que é altamente sensible á súa elección.
    - Non existe adaptación ao contexto da secuencia (nivel base de actividade).
    - Non distingue entre actividade relevante e ruído: ambos poden superar
      o limiar se teñen magnitude suficiente.
    - Ignora completamente a estrutura interna do sinal (forma, patrón temporal)
      e toda a información espacial dos eventos.
    - Pode producir fragmentación excesiva (múltiples segmentos curtos) que
      posteriormente deben ser corrixidos mediante merging.

    IMPLICACIÓN:
    Este método implementa unha segmentación baseada en magnitude, o que
    motiva a necesidade de mellorar a definición de actionness ou introducir
    limiares adaptativos no contexto do TFG.
    """
    return np.where(np.diff(score1d > threshold, prepend=0, append=0))[0].reshape(-1, 2)


def check_merge_possible(proposal_1, proposal_2, basin_durations, threshold):
    """
    Determina se dúas propostas temporais deben fusionarse nun único intervalo.

    ENTRADA:
    - proposal_1: proposta actual [start_index, end_index]
    - proposal_2: seguinte proposta [start_index, end_index]
    - basin_durations: duración acumulada de rexións activas (sinal por riba do limiar)
      dentro do segmento que se está a construír.
    - threshold: limiar de agrupamento (entre 0 e 1)

    PROCESO:
    - Engádese a duración de proposal_2 á duración acumulada de actividade.
    - Calcúlase a duración total do intervalo combinado, desde o inicio de
      proposal_1 ata o final de proposal_2.
    - Avalíase a proporción de tempo activo dentro do intervalo total:

          activity_ratio = basin_durations / merged_duration

    - Se esta proporción supera o limiar, considérase que as propostas forman
      parte dun mesmo evento continuo e deben fusionarse.

    SAÍDA:
    - True  → fusionar propostas
    - False → manter propostas separadas

    LIMITACIÓNS:
    - O criterio baséase exclusivamente na ocupación temporal, sen ter en conta
      a intensidade real do sinal (score) nin a súa estrutura interna.
    - Non distingue entre actividade relevante e ruído, xa que ambos poden
      producir rexións densas en termos de tempo.
    - Non incorpora información espacial nin patróns de movemento.
    - A decisión depende dun limiar fixo, que pode non ser óptimo para todas
      as secuencias.

    IMPLICACIÓN:
    Este mecanismo tenta corrixir a fragmentación xerada pola segmentación por
    limiar, pero segue sendo un criterio heurístico limitado baseado unicamente
    en proporción temporal.
    """
    
    basin_durations += proposal_2[1] - proposal_2[0]
    merged_duration = proposal_2[1] - proposal_1[0]

    if basin_durations / merged_duration > threshold:
        return True
    else:
        return False


def merge_proposals(unmerged, score, grouping_thres, times):
    """
    Fusiona propostas temporais iniciais en propostas máis longas e consistentes.

    ENTRADA:
    - unmerged: array de propostas en índices, de tamaño (n, 2), onde cada fila
      representa un intervalo [start_index, end_index] no sinal temporal.
    - score: sinal 1D asociado ao tempo (neste proxecto, o actioness), usado para
      calcular a puntuación media da proposta resultante.
    - grouping_thres: limiar de agrupamento que controla se dúas propostas
      consecutivas deben fusionarse.
    - times: vector cos tempos reais asociados aos índices do sinal, utilizado
      para converter as propostas de índices a tempo físico.

    PROCESO:
    - Percórrese a lista de propostas iniciais en orde temporal.
    - Mantense unha proposta actual ('current') que se intenta ampliar coa
      seguinte proposta ('next_proposal').
    - A decisión de fusionar ou non tómase mediante check_merge_possible(...),
      que avalía a proporción de actividade acumulada no intervalo combinado.
    - Se se decide fusionar, exténdese o final da proposta actual.
    - Se non se fusiona, péchase a proposta actual, convértese de índices a tempo
      real e asígnaselle unha puntuación igual á media do score dentro do intervalo.
    - O resultado é unha lista de propostas fusionadas con formato:
          [t_start, t_end, mean_score]

    SAÍDA:
    - Lista de propostas fusionadas, expresadas en tempo real e con score asociado.

    LIMITACIÓNS:
    - A fusión baséase nun criterio heurístico puramente temporal.
    - Non usa información espacial nin estrutura interna do movemento.
    - O score final da proposta depende directamente do sinal de entrada; se o
      actioness non é discriminativo, a proposta fusionada tampouco o será.
    - O comportamento depende dun limiar fixo (grouping_thres), sen adaptación
      específica á secuencia.
    - A implementación modifica a proposta actual en sitio (current[1] = ...),
      o que pode facer o código menos claro e máis fráxil de manter.

    IMPLICACIÓN:
    Esta función compensa a fragmentación producida pola segmentación por limiar,
    pero non resolve as limitacións estruturais do modelo. Por iso, no contexto
    do TFG, resulta máis prometedor mellorar o sinal de actioness de entrada ca
    modificar esta heurística de agrupamento.
    """
    
    merged = []
    current = None
    accumulated_basin_durations = 0

    for next_proposal in unmerged:
        if current is None:
            current = next_proposal
            accumulated_basin_durations += next_proposal[1] - next_proposal[0]
            
        else:
            
            do_merge = check_merge_possible(
                current, next_proposal, accumulated_basin_durations, grouping_thres
            )
            
            if do_merge:
                current[1] = next_proposal[1] # é dicir, que o final do intervalo actual se estenda ata o final da seguinte proposta, fusionando así os dous intervalos nun só.
                accumulated_basin_durations += next_proposal[1] - next_proposal[0] # actualizamos a duración acumulada de actividade, sumando a duración da seguinte proposta, xa que agora forman parte do mesmo intervalo fusionado.

            elif not do_merge or (next_proposal == unmerged[-1]).all(): # se non se fusionan ou se é a última proposta, entón engadimos a proposta actual ao resultado final.
                t_start = times[current[0]]
                t_end = times[current[1]]

                merged.append([
                        t_start,
                        t_end,
                        np.mean(score[current[0] : current[1]]), # media do score dentro do intervalo fusionado, que se usará como puntuación da proposta final.
                    ])

                current = next_proposal
                accumulated_basin_durations = 0

    return merged

def get_adaptive_actioness_thresholds(actioness, central_percentile=80, delta=0.10, step=0.05):
    """
    Calcula un conxunto pequeno de thresholds adaptados á distribución do actioness.

    Parámetros
    ----------
    actioness : np.ndarray
        Sinal normalizado en [0, 1].
    central_percentile : float
        Percentil central usado para estimar lambda.
    delta : float
        Marxe arredor do lambda central.
    step : float
        Paso entre thresholds.

    Retorna
    -------
    np.ndarray
        Array cos thresholds adaptativos.
    """
    
    # Calculamos cal é o valor do actioness correspondente ao percentil. 
    lambda_center = np.percentile(actioness, central_percentile)

    # Como queremos xerar un conxunto pequeno, xeneramos thresholds arredor dese valor central, 
    # nun rango definido por delta, e cun paso definido por step. Por exemplo, se lambda_center 
    # é 0.6, delta é 0.10 e step é 0.05, xeneraríamos thresholds en [0.5, 0.55, 0.6, 0.65, 0.7], 
    # pero se lambda_center é 0.2, xeneraríamos thresholds en [0.1, 0.15, 0.2, 0.25, 0.3].
    thresholds = np.arange(
        lambda_center - delta,
        lambda_center + delta + 1e-9, # 1e-9 para asegurar que se inclúe o límite superior cando se xenera o array con np.arange, por fallos cos flotantes
        step
    )

    # Aseguramos que os thresholds están dentro do rango [0.05, 0.95] 
    # para evitar valores extremos que poidan xerar demasiados ou poucos segmentos.
    # E por último eliminamos duplicados. 
    thresholds = np.clip(thresholds, 0.05, 0.95)
    thresholds = np.unique(np.round(thresholds, 4))

    return thresholds

class ProposalGenerator:
    """
    Generates action proposals based on temporal actioness scores across
    multiple regions of interest (ROI) within recordings.

    Attributes:
        data_dir (str): Directory containing the preprocessed data.
        bin_width (float): Width of the bins used for event rate calculation [us].
        percentile (float): Percentile used for robust scaling of event rates.
        nms_threshold (float): Threshold for non-maximal suppression.
    """

    def __init__(self, data_path, bin_width, percentile, nms_threshold, use_adaptive_lambda=False) -> None:
        if use_adaptive_lambda:
            self.log_file = os.path.join("output", "retag_adaptive_full.txt")
        else:
            self.log_file = os.path.join("output", "retag_baseline_full.txt")

        with open(self.log_file, "w") as f:
            if use_adaptive_lambda:
                f.write("=== RUN RETAG ADAPTIVE ===\n")
            else:
                f.write("=== RUN RETAG BASELINE ===\n")
            
        self.data_path = data_path
        self.bin_width = bin_width * 1e6  # us
        
        self.use_adaptive_lambda = use_adaptive_lambda
        
        self.percentile = percentile
        self.nms_threshold = nms_threshold
        
        self.actioness_thresholds = np.arange(0.05, 1, 0.05)
        
        self.lambda_percentile = 80
        self.lambda_delta = 0.10
        self.lambda_step = 0.05

        self.grouping_thresholds = np.arange(0.05, 1, 0.05)

    def process_recording(self, rec):
        """
        Processes a single recording to generate action proposals for each ROI.

        Args:
            rec (str): The name of the recording to process.

        Returns:
            dict: Dictionary with ROI ids as keys and proposal arrays as values.
        """
        rec_proposal_data = {}

        log_to_txt(self.log_file, f"\n[RECORDING] {rec}")

        with h5py.File(self.data_path, "r") as file:
            data = file[rec]
            roi_ids = data.keys()

            for roi_id in roi_ids:
                events = np.array(data[roi_id]["events"])

                log_to_txt(self.log_file, f"\n[ROI] {roi_id}")
                log_to_txt(self.log_file, f"n_events: {len(events)}")

                rate, bins = get_event_rate(events, self.bin_width)
                rate = apply_robust_min_max(rate, self.percentile)

                log_to_txt(self.log_file, f"rate min/max: {rate.min()} / {rate.max()}")
                log_to_txt(self.log_file, f"rate mean: {rate.mean():.4f}")

                min_rate, max_rate = np.min(rate), np.max(rate)

                # Protección contra división por cero
                if max_rate == min_rate:
                    actioness = np.zeros_like(rate, dtype=np.float64)
                else:
                    actioness = (rate - min_rate) / (max_rate - min_rate)

                log_to_txt(
                    self.log_file,
                    f"actioness min/max: {actioness.min():.4f} / {actioness.max():.4f}"
                )
                log_to_txt(self.log_file, f"actioness mean: {actioness.mean():.4f}")
                
                if self.use_adaptive_lambda:
                    actioness_thresholds = get_adaptive_actioness_thresholds(
                        actioness,
                        central_percentile=self.lambda_percentile,
                        delta=self.lambda_delta,
                        step=self.lambda_step,
                    )
                    log_to_txt(self.log_file, f"adaptive thresholds: {actioness_thresholds.tolist()}")
                else:
                    actioness_thresholds = self.actioness_thresholds
                    log_to_txt(self.log_file, f"fixed thresholds: {actioness_thresholds.tolist()}")

                proposals = []

                for at in actioness_thresholds:
                    for gt in self.grouping_thresholds:
                        unmerged = get_index_proposals_from_1d_score(actioness, at)
                        proposals += merge_proposals(unmerged, actioness, gt, bins)

                log_to_txt(self.log_file, f"proposals raw: {len(proposals)}")

                proposals = np.array(proposals)

                if len(proposals) > 0:
                    proposals = proposals[proposals[:, 1] - proposals[:, 0] > 2 * 1e6]
                    proposals = temporal_nms(proposals, self.nms_threshold)

                    log_to_txt(self.log_file, f"proposals after NMS: {len(proposals)}")

                    if len(proposals) > 0:
                        durations = proposals[:, 1] - proposals[:, 0]
                        log_to_txt(self.log_file, f"mean duration: {durations.mean():.2f}")
                        log_to_txt(self.log_file, f"min duration: {durations.min():.2f}")
                        log_to_txt(self.log_file, f"max duration: {durations.max():.2f}")
                    else:
                        log_to_txt(self.log_file, "proposals after NMS: 0")
                else:
                    proposals = np.empty((0, 3))
                    log_to_txt(self.log_file, "proposals after NMS: 0")

                rec_proposal_data[roi_id] = proposals

        return rec_proposal_data
    
    
    def run(self):
        """
        Processes all recordings in the data directory to generate action proposals.

        Returns:
            pd.DataFrame: DataFrame with all generated proposals.
        """
        logging.info("Running Proposal Generator.")

        with h5py.File(self.data_path, "r") as f:
            recordings = [rec for rec in f.keys() if f[rec].attrs["split"] == "test"]

        log_to_txt(self.log_file, f"\n[INFO] recordings found: {recordings}")

        proposal_data = {}

        with Pool(processes=16) as pool:
          results = pool.map(self.process_recording, recordings)

        for rec, rec_proposals in zip(recordings, results):
            rec_name = os.path.splitext(rec)[0]
            proposal_data[rec_name] = rec_proposals

        proposal_df = {
            "rec_name": [],
            "roi_id": [],
            "t_start": [],
            "t_end": [],
            "score": [],
        }

        for rec_name, rec_proposals in proposal_data.items():
            for roi_id, roi_proposals in rec_proposals.items():
                for proposal in roi_proposals:
                    proposal_df["rec_name"].append(rec_name)
                    proposal_df["roi_id"].append(roi_id)
                    proposal_df["t_start"].append(proposal[0])
                    proposal_df["t_end"].append(proposal[1])
                    proposal_df["score"].append(proposal[2])

        return pd.DataFrame(proposal_df)
