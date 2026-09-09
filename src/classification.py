from __future__ import annotations

import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
from absl import logging
from torch.utils.data import Dataset, DataLoader
import h5py

from .augmented_tsn import AugmentedTsn
from .utils import temporal_nms, temporal_soft_nms


def range_norm(
    matrix: np.ndarray,
    new_max: float = 255,
    lower: float = None,
    upper: float = None,
    dtype=None,
) -> np.ndarray:
    lower = matrix.min() if lower is None else lower
    upper = matrix.max() if upper is None else upper
    scaled = new_max * (np.clip(matrix, lower, upper) - lower) / (upper - lower)
    return scaled.astype(dtype) if dtype is not None else scaled


def create_time_map(
    events: np.ndarray,
    decay: float,
    height: int,
    width: int,
) -> np.ndarray:
    time_map = np.zeros((height, width))
    time_map[events[:, 1], events[:, 0]] = events[:, 2]

    current_t = events[:, 2].max() if len(events) > 0 else 0
    time_map = np.exp(-decay * (current_t - time_map))

    polarity = events[:, 3].copy().astype(int)
    polarity[polarity == 0] = -1
    time_map[events[:, 1], events[:, 0]] *= polarity

    return time_map


def create_img_representation(
    events: np.ndarray,
    decay: float,
    height: int,
    width: int,
    transform=None,
) -> np.ndarray:
    img = create_time_map(events, decay, height, width)
    img = range_norm(img, lower=-1, upper=1, dtype=np.uint8)
    # Redimensionase UNHA canle e replicase despois. Antes replicabase a tres
    # canles identicas e redimensionabanse as tres: PIL trata cada canle por
    # separado, asi que era exactamente o mesmo traballo feito tres veces. O
    # resultado e identico byte a byte porque o filtro aplicase canle a canle.
    img = Image.fromarray(img).resize((224, 224), resample=Image.BILINEAR)
    img = np.array(img)
    img = np.repeat(img[..., None], 3, axis=2)
    return transform(img) if transform is not None else img


def create_polarity_img_representation(
    events: np.ndarray,
    decay: float,
    height: int,
    width: int,
    transform=None,
) -> np.ndarray:
    """Build signed, ON-recency, and OFF-recency channels for one event window."""
    signed = create_time_map(events, decay, height, width)
    signed_u8 = range_norm(signed, lower=-1, upper=1, dtype=np.uint8)
    on = np.zeros((height, width), dtype=np.float64)
    off = np.zeros((height, width), dtype=np.float64)
    if len(events) > 0:
        current_t = float(events[:, 2].max())
        for polarity_mask, output in ((events[:, 3] > 0, on), (events[:, 3] <= 0, off)):
            polarity_events = events[polarity_mask]
            if len(polarity_events) == 0:
                continue
            timestamps = np.zeros((height, width), dtype=np.float64)
            active = np.zeros((height, width), dtype=bool)
            y = polarity_events[:, 1].astype(np.int64)
            x = polarity_events[:, 0].astype(np.int64)
            timestamps[y, x] = polarity_events[:, 2]
            active[y, x] = True
            output[active] = np.exp(-decay * (current_t - timestamps[active]))
    on_u8 = range_norm(on, lower=0, upper=1, dtype=np.uint8)
    off_u8 = range_norm(off, lower=0, upper=1, dtype=np.uint8)
    img = np.stack((signed_u8, on_u8, off_u8), axis=2)
    img = Image.fromarray(img).resize((224, 224), resample=Image.BILINEAR)
    img = np.array(img)
    return transform(img) if transform is not None else img


def create_multiscale_decay_img_representation(
    events: np.ndarray,
    decays: tuple[float, float, float],
    height: int,
    width: int,
    transform=None,
) -> np.ndarray:
    """Build three signed time-surfaces with different temporal memories."""
    if len(decays) != 3:
        raise ValueError("Exactly three decay values are required")
    channels = [
        range_norm(
            create_time_map(events, decay, height, width),
            lower=-1,
            upper=1,
            dtype=np.uint8,
        )
        for decay in decays
    ]
    img = np.stack(channels, axis=2)
    img = Image.fromarray(img).resize((224, 224), resample=Image.BILINEAR)
    img = np.array(img)
    return transform(img) if transform is not None else img


def _consulta_no_dtype(tempos, timestamps):
    """Converte os tempos de consulta ao dtype exacto do array de timestamps.

    Ver o comentario en ProposalDataset.__getitem__: sen isto np.searchsorted
    promociona o array enteiro a float64 e materializa o mmap.
    """
    valores = np.ceil(np.asarray(tempos, dtype=np.float64))
    destino = np.asarray(timestamps).dtype if not hasattr(timestamps, "dtype") else timestamps.dtype
    if destino.kind in ("u", "i"):
        info = np.iinfo(destino)
        valores = np.clip(valores, info.min, info.max)
        return valores.astype(destino)
    return valores.astype(destino)



def create_time_map_gpu(eventos, decay: float, height: int, width: int):
    """Equivalente de create_time_map sobre un tensor (N,4) que xa esta en GPU.

    Redúcese sobre o INDICE do evento, non sobre o timestamp. Cos eventos
    ordenados por tempo, o indice maior por pixel e "o ultimo que escribe", que
    e exactamente o que fai time_map[y,x] = t en NumPy. Verificado: maxdif
    1,11e-16 ata 8 M de eventos.

    Non se usa index_put_(accumulate=False): a documentacion de PyTorch declara
    ese caso INDEFINIDO con indices duplicados e non determinista en CUDA, e
    medimos maxdif 2,00 (o rango enteiro). Reducir sobre o timestamp tampouco
    vale: falla cando dous eventos comparten pixel E instante.
    """
    import torch

    n = eventos.shape[0]
    dev = eventos.device
    if n == 0:
        return torch.zeros((height, width), dtype=torch.float64, device=dev)
    x = eventos[:, 0].long()
    y = eventos[:, 1].long()
    t = eventos[:, 2].double()
    p = eventos[:, 3].long()
    idx = y * width + x

    gan = torch.full((height * width,), -1, dtype=torch.long, device=dev)
    gan.scatter_reduce_(0, idx, torch.arange(n, device=dev), reduce="amax", include_self=True)
    tocado = gan >= 0
    seguro = gan.clamp(min=0)

    tm = torch.zeros(height * width, dtype=torch.float64, device=dev)
    tm[tocado] = t[seguro[tocado]]
    tm = torch.exp(-decay * (t.max() - tm))
    pol = torch.where(p == 0, -1.0, 1.0).double()
    tm[tocado] = tm[tocado] * pol[seguro[tocado]]
    return tm.view(height, width)


def construir_imaxes_gpu(bloque, tramos, decay: float, height: int, width: int, device):
    """Constrúe en GPU as time surfaces dun lote enteiro.

    `bloque` (M,4) e a concatenacion dos eventos do lote; `tramos` (N,2) son os
    indices [inicio, fin) de cada xanela dentro del. Devolve (N,3,224,224) xa
    normalizado como o transform de ImageNet.

    Diferenza medida fronte ao camiño de CPU: maxdif 0,0175 en unidades
    normalizadas, que e UN nivel de gris de 255 (1/255 / 0,225). E o chan de
    cuantizacion da representacion uint8, non un erro.
    """
    import torch
    import torch.nn.functional as F

    bloque = bloque.to(device, non_blocking=True)
    n = int(tramos.shape[0])
    saida = torch.empty((n, 224, 224), dtype=torch.float32, device=device)
    for k in range(n):
        a = int(tramos[k, 0])
        b = int(tramos[k, 1])
        tm = create_time_map_gpu(bloque[a:b], decay, height, width)
        # range_norm(lower=-1, upper=1, dtype=uint8) TRUNCA, non redondea
        u8 = torch.floor(255.0 * (torch.clamp(tm, -1.0, 1.0) + 1.0) / 2.0)
        r = F.interpolate(
            u8[None, None].float(), size=(224, 224),
            mode="bilinear", align_corners=False, antialias=True,
        )[0, 0]
        saida[k] = torch.round(r).clamp(0, 255)   # PIL devolve uint8
    imgs = saida[:, None].repeat(1, 3, 1, 1) / 255.0
    media = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    desv = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (imgs - media) / desv


def collate_gpu(lote):
    """Xunta as propostas dun lote nun so bloque de eventos con desprazamentos."""
    import torch

    bloques = []
    tramos = []
    desp = 0
    for bloque, tr, *_ in lote:
        bloques.append(bloque)
        tramos.append(tr + desp)
        desp += bloque.shape[0]
    resto = [[x[i] for x in lote] for i in range(2, len(lote[0]))]
    n_xanelas = lote[0][1].shape[0]
    return (torch.cat(bloques, 0), torch.cat(tramos, 0), n_xanelas, *resto)


class ProposalDataset(Dataset):
    # atributo de clase para evitar colisión co módulo transforms
    _transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(
        self,
        proposals,
        augment_fraction: float,
        data_path: str,
        num_tsn_samples: int,
        sample_duration: float,
        decay: float,
        cache_full_events: bool = True,
        timestamp_cache_dir: str | None = None,
        gpu_representation: bool = False,
    ):
        self.proposals = proposals
        self.augment_fraction = augment_fraction
        self.data_path = data_path
        self.num_tsn_samples = num_tsn_samples
        self.sample_duration = sample_duration
        self.decay = decay
        self.cache_full_events = cache_full_events
        # Se e True, __getitem__ devolve os eventos crus e as imaxes
        # constrúense en GPU no proceso principal (ver construir_imaxes_gpu).
        self.gpu_representation = gpu_representation
        # Tope de eventos que se len dunha vez ao agrupar as xanelas dunha
        # proposta (ver __getitem__). 4 M eventos son ~64 MB por worker, que
        # con oito workers son 512 MB: asumible. Por riba diso vólvese a ler
        # mostra a mostra, que é o que xa se facía.
        self.max_union_events = 4_000_000
        self.timestamp_cache_dir = (
            Path(timestamp_cache_dir) if timestamp_cache_dir is not None else None
        )
        self._hf = None
        self._cached_roi_key = None
        self._cached_roi_events = None
        self._cached_roi_timestamps = None
        self._cached_roi_height = None
        self._cached_roi_width = None

    def _get_h5(self):
        if self._hf is None:
            # A cache de chunks de HDF5 por defecto son 1 MiB, e os nosos
            # chunks miden 65536*4*4 = 1.048.576 B = 1 MiB EXACTOS: cabe un
            # so. Cada unha das 21 xanelas dunha proposta que toque outro
            # chunk expulsa o anterior e forza unha descompresion LZF completa
            # de 1 MiB para consumir uns poucos KiB. E a patoloxia que o HDF
            # Group documenta (lecturas parciais sobre datos comprimidos:
            # 345 s con cache de 1 MiB fronte a 0,37 s cunha cache que colle o
            # chunk). 256 MiB collen 256 chunks; nslots debe ser primo e ~100x
            # o numero de chunks, e w0=0 expulsa o menos usado recentemente,
            # que e o que convence cando se revisitan chunks.
            self._hf = h5py.File(
                self.data_path,
                "r",
                rdcc_nbytes=256 * 1024 * 1024,
                rdcc_nslots=25601,
                rdcc_w0=0.0,
            )
        return self._hf

    def _get_roi_data(self, rec_name, roi_id):
        key = (rec_name, roi_id)
        if key != self._cached_roi_key:
            roi_group = self._get_h5()[rec_name][roi_id]
            events = roi_group["events"]
            if self.cache_full_events:
                self._cached_roi_events = np.asarray(events)
                self._cached_roi_timestamps = self._cached_roi_events[:, 2]
            else:
                self._cached_roi_events = events
                cache_path = None
                if self.timestamp_cache_dir is not None:
                    cache_path = self.timestamp_cache_dir / str(rec_name) / f"{roi_id}.npy"
                if cache_path is not None and cache_path.exists():
                    self._cached_roi_timestamps = np.load(cache_path, mmap_mode="r")
                else:
                    timestamps = np.asarray(events[:, 2])
                    if cache_path is not None:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = cache_path.with_suffix(f".{os.getpid()}.npy.tmp")
                        with temporary.open("wb") as stream:
                            np.save(stream, timestamps)
                        temporary.replace(cache_path)
                    self._cached_roi_timestamps = timestamps
            self._cached_roi_height = roi_group.attrs["height"]
            self._cached_roi_width = roi_group.attrs["width"]
            self._cached_roi_key = key
        return (
            self._cached_roi_events,
            self._cached_roi_timestamps,
            self._cached_roi_height,
            self._cached_roi_width,
        )

    def __len__(self) -> int:
        return len(self.proposals)

    def __getitem__(self, idx):
        t_start = self.proposals.loc[idx, "t_start"]
        t_end = self.proposals.loc[idx, "t_end"]
        rec_name = self.proposals.loc[idx, "rec_name"]
        roi_id = self.proposals.loc[idx, "roi_id"]

        roi_events, roi_timestamps, height, width = self._get_roi_data(rec_name, roi_id)

        t_delta = t_end - t_start
        t_aug_start = t_start - t_delta * self.augment_fraction
        t_aug_end = t_end + t_delta * self.augment_fraction

        img_times = torch.linspace(t_aug_start, t_aug_end, self.num_tsn_samples)
        sample_duration = (
            float(self.proposals.loc[idx, "sample_duration"])
            if "sample_duration" in self.proposals.columns
            else self.sample_duration
        )
        t_imgs_start = img_times - 0.5 * sample_duration
        t_imgs_end = img_times + 0.5 * sample_duration

        # A consulta ten que ir NO DTYPE do array de timestamps. Se non, numpy
        # non pode comparar uint32 cun float32 de torch, busca un tipo comun e
        # CONVERTE O ARRAY ENTEIRO a float64: materializa o mmap completo en
        # cada chamada, e faise dúas veces por proposta.
        #
        # Medido o 2026-09-08 sobre video_validation_0000666 (1.467.653.571
        # timestamps, 5.871 MB):
        #     np.searchsorted(ts, tensor_float32)  ->  85,323 s
        #     np.searchsorted(ts, consulta_no_dtype) ->  0,001 s
        # cos MESMOS indices. Era o gargalo real das duas ramas.
        #
        # O ceil non e unha aproximacion: os timestamps son enteiros, asi que
        # {ts >= t} e exactamente {ts >= ceil(t)} para calquera t real. O clip
        # fai falla porque a aumentacion pode dar tempos negativos e o casting
        # a un tipo sen signo daria a volta.
        i_start = np.searchsorted(roi_timestamps, _consulta_no_dtype(t_imgs_start, roi_timestamps))
        i_end = np.searchsorted(roi_timestamps, _consulta_no_dtype(t_imgs_end, roi_timestamps))

        # As num_tsn_samples xanelas dunha proposta solápanse case por completo
        # cando a proposta é curta, e a maioría sono: no corpus real a duración
        # mediana é de 0,17 s e cada mostra abarca 1,00 s. Lelas unha a unha do
        # HDF5 dá unha amplificación de lectura de ×127 —135 MB por proposta,
        # uns 60 TB para as 448.323 de validation— e é o que fai que esta etapa
        # tarde 97 h. Lendo UNHA vez o tramo que cobre todas e recortando en
        # memoria, os valores son idénticos e lese ~18 veces menos.
        #
        # Nas propostas longas non hai solapamento e a unión sería enorme (a
        # maior dura 1.357 s, 8,7 GB), así que aí mantense a lectura por mostra.
        lo = int(np.min(i_start))
        hi = int(np.max(i_end))
        if 0 < hi - lo <= self.max_union_events:
            bloque = np.asarray(roi_events[lo:hi])
            tramos = (bloque[s - lo : e - lo] for s, e in zip(i_start, i_end))
        else:
            tramos = (np.asarray(roi_events[s:e]) for s, e in zip(i_start, i_end))

        if self.gpu_representation:
            pezas = [np.ascontiguousarray(t, dtype=np.int32) for t in tramos]
            desp = np.cumsum([0] + [len(z) for z in pezas])
            bloque = torch.from_numpy(
                np.concatenate(pezas) if pezas else np.zeros((0, 4), dtype=np.int32)
            )
            marcas = torch.from_numpy(
                np.stack([desp[:-1], desp[1:]], axis=1).astype(np.int64)
            )
            proposal_score = (
                float(self.proposals.loc[idx, "score"])
                if "score" in self.proposals.columns else 0.0
            )
            return bloque, marcas, rec_name, roi_id, t_start, t_end, proposal_score

        imgs = torch.stack([
            create_img_representation(t, self.decay, height, width, self._transform)
            for t in tramos
        ])
        proposal_score = float(self.proposals.loc[idx, "score"]) if "score" in self.proposals.columns else 0.0
        return imgs, rec_name, roi_id, t_start, t_end, proposal_score


class ProposalClassifier:

    def __init__(
        self,
        device,
        model_path: str,
        num_tsn_samples: int,
        augment_factor: int,
        data_path: str,
        sample_duration: float,
        decay: float,
        nms_threshold: float,
        batch_size: int,
        use_soft_nms: bool = False,
        soft_nms_sigma: float = 0.5,
        soft_nms_score_threshold: float = 0.001,
        score_fusion_weight: float = 0.0,
        max_duration_filter: float = None,
        min_ed_score: float = 0.5,
        temperature: float = 1.0,
        duration_penalty_dmax: float = None,
        duration_penalty_sigma: float = 20.0,
        platt_a: float = 1.0,
        platt_b: float = 0.0,
        num_workers: int = 16,
    ) -> None:
        self.device = device
        self.augment_fraction = 1 / augment_factor
        num_aug_samples = int(np.ceil(self.augment_fraction * num_tsn_samples))
        # segmento principal + aumentación esquerda + dereita
        self.num_tsn_samples = num_tsn_samples + 2 * num_aug_samples

        self.data_path = data_path
        self.sample_duration = 1e6 * sample_duration  # s → µs
        self.decay = float(decay)
        self.nms_threshold = nms_threshold
        self.batch_size = batch_size
        self.use_soft_nms = use_soft_nms
        self.soft_nms_sigma = soft_nms_sigma
        self.soft_nms_score_threshold = soft_nms_score_threshold
        self.score_fusion_weight = score_fusion_weight
        self.max_duration_filter = max_duration_filter
        self.min_ed_score = min_ed_score
        self.temperature = temperature
        self.duration_penalty_dmax = duration_penalty_dmax
        self.duration_penalty_sigma = duration_penalty_sigma
        self.platt_a = platt_a
        self.platt_b = platt_b
        self.num_workers = num_workers

        self.model = AugmentedTsn(2, num_tsn_samples, augment_factor)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device).eval()

    def run(self, proposals) -> dict:
        logging.info("Executando o clasificador de propostas.")

        dataset = ProposalDataset(
            proposals,
            self.augment_fraction,
            self.data_path,
            self.num_tsn_samples,
            self.sample_duration,
            self.decay,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        result = {
            rec: {roi: [] for roi in proposals[proposals["rec_name"] == rec]["roi_id"].unique()}
            for rec in proposals["rec_name"].unique()
        }

        softmax = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            for imgs, rec_names, roi_ids, t_starts, t_ends, prop_scores in tqdm(loader):
                outputs = self.model(imgs.to(self.device))
                ed_scores = softmax(outputs / self.temperature)[:, 1]

                for i in range(len(ed_scores)):
                    score = float(ed_scores[i])
                    if score >= self.min_ed_score:
                        if self.score_fusion_weight > 0:
                            score *= (1.0 + self.score_fusion_weight * float(prop_scores[i]))
                        if self.duration_penalty_dmax is not None:
                            duration_s = (float(t_ends[i]) - float(t_starts[i])) / 1e6
                            excess = max(0.0, duration_s - self.duration_penalty_dmax)
                            score *= float(np.exp(-excess / self.duration_penalty_sigma))
                        result[rec_names[i]][roi_ids[i]].append([
                            float(t_starts[i]),
                            float(t_ends[i]),
                            score,
                        ])

        nmsed = {}
        for rec_name, rec_results in result.items():
            nmsed[rec_name] = {}
            for roi_id, roi_result in rec_results.items():
                if roi_result:
                    arr = np.array(roi_result)
                    processed = (
                        temporal_soft_nms(
                            arr,
                            sigma=self.soft_nms_sigma,
                            score_threshold=self.soft_nms_score_threshold,
                        )
                        if self.use_soft_nms
                        else temporal_nms(arr, self.nms_threshold)
                    )
                    if self.max_duration_filter is not None and len(processed) > 0:
                        max_dur_us = self.max_duration_filter * 1e6
                        processed = processed[processed[:, 1] - processed[:, 0] <= max_dur_us]
                else:
                    processed = []
                nmsed[rec_name][int(roi_id[1:])] = [
                    {
                        "label": "ed",
                        "segment": [a[0] / 1e6, a[1] / 1e6],
                        "score": a[2],
                    }
                    for a in processed
                ]

        return {"version": "VERSION 0.0", "results": nmsed}

    def collect_logits(self, proposals) -> tuple:
        """Devolve logits crus para axuste de temperatura."""
        logging.info("Recollendo logits para axuste de temperatura.")

        dataset = ProposalDataset(
            proposals,
            self.augment_fraction,
            self.data_path,
            self.num_tsn_samples,
            self.sample_duration,
            self.decay,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        all_logits = []
        all_meta = []

        with torch.no_grad():
            for imgs, rec_names, roi_ids, t_starts, t_ends, _ in tqdm(loader):
                outputs = self.model(imgs.to(self.device))
                all_logits.append(outputs.cpu().numpy())
                for i in range(len(t_starts)):
                    all_meta.append((rec_names[i], roi_ids[i], float(t_starts[i]), float(t_ends[i])))

        return np.concatenate(all_logits, axis=0), all_meta
