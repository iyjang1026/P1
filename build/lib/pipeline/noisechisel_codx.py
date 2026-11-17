"""
noisechisel_lite.py  (v4 -- 속도 최적화판)
==========================================

NoiseChisel(Akhlaghi & Ichikawa 2015, ApJS 220, 1 [AI15]; Akhlaghi 2019,
arXiv:1909.11230 [A19])의 핵심 검출 로직을 순수 Python(numpy/scipy)으로
옮긴 경량 구현입니다. v3보다 빠르도록 구현을 바꿨지만, 통계 계산과 convolution의
차이 때문에 결과가 완전히 같다고 보장하지 않습니다.

파이프라인
    1. Convolution (Gaussian, AI15 3.1.1)
    2. (옵션) Tessellation: 타일별 qthresh/noerodequant/Sky 지도 (AI15 4, A19 2.1)
    3. Sub-sky quantile threshold (AI15 3.1.2)
    4. Erosion(+noerodequant) -> Opening (AI15 3.1.3-4)
    5. Pseudo-detection S/N 기반 false-detection 제거 (AI15 3.1.5, Eq.3)
    6. Growth (--detgrowquant, A19 2.2) / 구멍 메우기 / --cleangrowndet

v3 대비 달라진 구현 (결과가 완전히 같은 것은 보장되지 않음)
    * 타일 통계: 타일마다 astropy sigma_clipped_stats / np.quantile 를 부르던
      Python 루프 -> "타일 행(band) 단위로 한 번 정렬 + 누적합 + 이분탐색" 으로
      sigma-clipping·quantile·mean-quantile 을 전부 벡터화 (스레드 병렬).
    * Growth: heapq 픽셀 루프 -> 연결성분 라벨링. priority-flood 는
      "grow_threshold 이상인 픽셀로 이어진 경로가 true detection 에서 닿는가"
      만 결정하므로 최종 마스크는 정확히 동일합니다 (테스트로 확인).
    * pseudo-detection S/N: 라벨별 Python 루프 -> numpy 벡터 연산.
    * binary_fill_holes -> 라벨링 기반 구멍 채우기, 보간은 cKDTree.
    * 내부 작업 배열은 float32 (메모리 절약). 필요하면 WORK_DTYPE=np.float64.
    * widekernel: 타일 분류에만 넓은 커널을 쓰고 threshold는 sharp 이미지에서 계산.

아직 없는 것 (v3와 동일): sorted-distance 방식 outlier rejection,
--skyfracnoblank, blank 픽셀의 전 단계 전파, clump segmentation(Segment).
논문에 싣는 최종 수치는 실제 Gnuastro NoiseChisel 로 교차검증하세요.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from astropy.stats import sigma_clipped_stats

WORK_DTYPE = np.float32          # 내부 작업 배열 dtype (정밀도 검증 시 float64)
STRUCT_4 = ndi.generate_binary_structure(2, 1)
STRUCT_8 = ndi.generate_binary_structure(2, 2)
_NGB = {4: STRUCT_4, 8: STRUCT_8}


class NoiseChiselResult:
    """각 단계 결과(Gnuastro --checkdetection 처럼 확인 가능)."""

    def __init__(self):
        self.convolved = None       # convolved image
        self.threshold = None       # qthresh: float(global) 또는 2D map(tessellation)
        self.foreground = None      # qthresh 위 픽셀
        self.eroded = None
        self.opened = None
        self.protect_mask = None    # noerodequant 로 보호된 픽셀
        self.sky_value = None       # 대표 Sky (tessellation 이면 타일 값의 median)
        self.sky_std = None
        self.sky_map = None         # tessellation 일 때 2D Sky / Sky STD map
        self.sky_std_map = None
        self.sn_threshold = None
        self.true_mask = None       # false detection 제거 후
        self.grown = None           # growth 후
        self.detected = None        # 최종 mask
        self.segmap = None          # photutils SegmentationImage
        self.tile_shape = None      # (n_tiles_y, n_tiles_x)
        self.tile_good_mask = None  # "신호 없음"으로 판정된 타일


# ===================================================================== #
# 1. Convolution
# ===================================================================== #
def _convolve(data: np.ndarray, fwhm: float = 2.0) -> np.ndarray:
    """Gaussian smoothing (가분리 필터). astropy convolve(boundary='extend',
    nan_treatment='interpolate') 와 같은 결과(차이 ~1e-7)를 4~5배 빠르게 냄.
    NaN 이 있으면 normalized convolution 으로 interpolate 와 동일하게 처리."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    radius = (int(np.ceil(8.0 * sigma)) | 1) // 2        # astropy 커널 반폭과 일치
    kw = dict(sigma=sigma, mode="nearest", truncate=radius / sigma, output=data.dtype)
    valid = np.isfinite(data)
    if valid.all():
        return ndi.gaussian_filter(data, **kw)
    num = ndi.gaussian_filter(np.where(valid, data, 0).astype(data.dtype), **kw)
    den = ndi.gaussian_filter(valid.astype(data.dtype), **kw)
    return np.where(den > 1e-5, num / np.maximum(den, 1e-5), np.nan).astype(data.dtype)


# ===================================================================== #
# 2. Tessellation  (타일 통계는 "band 단위 정렬"로 한 번에 계산)
# ===================================================================== #
def _tile_geometry(shape, tilesize):
    ty, tx = (tilesize, tilesize) if np.isscalar(tilesize) else tilesize
    ty, tx = int(ty), int(tx)
    ny, nx = shape
    return ty, tx, -(-ny // ty), -(-nx // tx)                # 마지막 타일은 나머지 크기


def _bisect(X, n, v, right):
    """정렬된 각 행 X[r,:n[r]] 에서 v[r] 의 삽입 위치(searchsorted)를 벡터로 계산."""
    rows = np.arange(len(X))
    lo, hi = np.zeros(len(X), np.int64), n.astype(np.int64).copy()
    last = X.shape[1] - 1
    while (lo < hi).any():
        mid = (lo + hi) // 2
        xm = X[rows, np.minimum(mid, last)]
        go = (xm <= v) if right else (xm < v)
        act = lo < hi
        lo = np.where(act & go, mid + 1, lo)
        hi = np.where(act & ~go, mid, hi)
    return lo


def _band_stats(X, n, quantiles, sigma, maxiters):
    """정렬된 행들(X, 끝에 NaN)에 대해 astropy sigma_clipped_stats(median 중심,
    std ddof=0, maxiters) 와 동일한 clipped mean/std, mean 의 quantile 위치,
    그리고 np.quantile(linear) 와 동일한 quantile 값들을 계산."""
    T = len(X)
    rows = np.arange(T)
    nn = np.maximum(n, 1)
    ref = X[rows, (nn - 1) // 2].astype(np.float64)          # 수치 안정용 기준값
    Xs = X.astype(np.float64) - ref[:, None]
    C1 = np.zeros((T, X.shape[1] + 1)); C2 = np.zeros_like(C1)
    np.cumsum(Xs, axis=1, out=C1[:, 1:]); np.cumsum(Xs * Xs, axis=1, out=C2[:, 1:])

    lo, hi = np.zeros(T, np.int64), n.astype(np.int64).copy()
    for _ in range(maxiters):                                # iterative sigma clip
        m = np.maximum(hi - lo, 1)
        med = 0.5 * (Xs[rows, lo + (m - 1) // 2] + Xs[rows, np.minimum(lo + m // 2, X.shape[1] - 1)])
        mean = (C1[rows, hi] - C1[rows, lo]) / m
        std = np.sqrt(np.maximum((C2[rows, hi] - C2[rows, lo]) / m - mean ** 2, 0))
        nlo = np.maximum(lo, _bisect(Xs, n, med - sigma * std, right=False))
        nhi = np.minimum(hi, _bisect(Xs, n, med + sigma * std, right=True))
        changed = (nlo != lo) | (nhi != hi)
        lo, hi = nlo, nhi
        if not changed.any():
            break
    m = np.maximum(hi - lo, 1)
    cmean = (C1[rows, hi] - C1[rows, lo]) / m
    cstd = np.sqrt(np.maximum((C2[rows, hi] - C2[rows, lo]) / m - cmean ** 2, 0))
    meanq = _bisect(Xs, n, cmean, right=True) / nn            # mean 의 quantile 위치

    qv = []
    for q in quantiles:                                       # np.quantile(method='linear')
        pos = q * (nn - 1)
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, nn - 1)
        t = pos - i0
        a, b = Xs[rows, i0], Xs[rows, i1]
        qv.append(np.where(t >= 0.5, b - (b - a) * (1 - t), a + (b - a) * t) + ref)
    return cmean + ref, cstd, meanq, qv


def _tile_pass(arr, ty, tx, mask=None, quantiles=(), sigma=3.0, maxiters=5, n_jobs=None):
    """arr 를 ty x tx 타일로 나눠 타일별 통계를 한 번에 계산 (타일 행 단위 band 로
    정렬 -> 벡터화, band 들은 스레드 병렬). mask(True=사용 가능) 가 있으면 그 픽셀만 사용.
    반환: dict(n, frac, mean, std, meanq, q=[...]) 각 (nty, ntx)."""
    ny, nx = arr.shape
    nty, ntx = -(-ny // ty), -(-nx // tx)
    widths = np.full(ntx, tx); widths[-1] = nx - (ntx - 1) * tx
    out = {k: np.full((nty, ntx), np.nan) for k in ("n", "frac", "mean", "std", "meanq")}
    out["q"] = [np.full((nty, ntx), np.nan) for _ in quantiles]

    def run(i):
        y0, y1 = i * ty, min((i + 1) * ty, ny)
        h = y1 - y0
        band = np.full((h, ntx * tx), np.nan, dtype=arr.dtype)
        band[:, :nx] = arr[y0:y1]
        if mask is not None:
            ok = np.zeros((h, ntx * tx), bool); ok[:, :nx] = mask[y0:y1]
            frac = ok.reshape(h, ntx, tx).sum((0, 2)) / (h * widths)
            band[~ok] = np.nan
        else:
            frac = np.ones(ntx)
        band[~np.isfinite(band)] = np.nan
        X = band.reshape(h, ntx, tx).transpose(1, 0, 2).reshape(ntx, h * tx)
        X = np.ascontiguousarray(X); X.sort(axis=1)           # NaN 은 끝으로
        n = (~np.isnan(X)).sum(1)
        good = n >= 1
        cm, cs, mq, qv = _band_stats(X, n, quantiles, sigma, maxiters)
        for k, v in (("n", n), ("frac", frac), ("mean", cm), ("std", cs), ("meanq", mq)):
            out[k][i] = np.where(good, v, np.nan) if k != "n" else v
        for k, v in enumerate(qv):
            out["q"][k][i] = np.where(good, v, np.nan)

    workers = n_jobs or 1
    if workers > 1 and nty > 1:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(run, range(nty)))
    else:
        for i in range(nty):
            run(i)
    return out


def _reject_outliers(values, good, outliersigma):
    """'good' 타일 중 값이 전체 good 타일 분포(sigma-clipped)에서 outliersigma
    이상 벗어난 타일(밝은 소스의 wing 등)을 제외. (Gnuastro 의 이웃 기반 방식의 단순화)"""
    keep = good.copy()
    if not outliersigma or outliersigma <= 0 or good.sum() < 5:
        return keep
    _, med, std = sigma_clipped_stats(values[good], sigma=5.0, maxiters=10)
    if std > 0 and np.isfinite(std):
        keep &= np.abs(values - med) <= outliersigma * std
    return keep


def _interpolate(values, good, k):
    """good 가 아닌 타일을 가장 가까운 good 타일 k개의 median 으로 채움."""
    out = values.copy()
    gi, bi = np.argwhere(good), np.argwhere(~good)
    if len(gi) == 0 or len(bi) == 0:
        return out
    k = min(k, len(gi))
    kk = min(len(gi), k + 32)                     # 동거리 후보까지 넉넉히 뽑은 뒤
    _, nn = cKDTree(gi).query(bi, k=kk)
    nn = nn.reshape(len(bi), kk)
    d2 = (gi[nn, 0] - bi[:, :1]) ** 2 + (gi[nn, 1] - bi[:, 1:]) ** 2
    order = np.lexsort((gi[nn, 1], gi[nn, 0], d2), axis=1)[:, :k]   # (거리, 행, 열)로 tie 를 결정
    pick = np.take_along_axis(nn, order, axis=1)
    out[bi[:, 0], bi[:, 1]] = np.median(values[gi[pick, 0], gi[pick, 1]], axis=1)
    return out


def _tile_map(values, good, shape, ty, tx, outliersigma, interpnumngb, smoothwidth, fallback):
    """타일 값 -> outlier 제거 -> 보간 -> smoothing -> 전체 해상도 map."""
    keep = _reject_outliers(values, good & np.isfinite(values), outliersigma)
    if not keep.any():
        values, keep = np.full(values.shape, fallback()), np.ones(values.shape, bool)
    grid = _interpolate(values, keep, interpnumngb)
    if smoothwidth and smoothwidth > 1:
        grid = ndi.uniform_filter(grid, size=smoothwidth, mode="nearest")
    full = np.repeat(np.repeat(grid.astype(WORK_DTYPE), ty, 0), tx, 1)
    return full[:shape[0], :shape[1]]


# ===================================================================== #
# 3-4. Erosion + Opening  (슬라이스 기반 형태학 연산: scipy 보다 ~10x 빠름)
# ===================================================================== #
def _ero(a, ngb):
    """3x3 십자(4) / 정사각(8) 구조요소 침식. 경계 밖은 False (scipy 기본과 동일)."""
    if ngb == 4:
        e = a.copy()
        e[1:] &= a[:-1]; e[:-1] &= a[1:]
        e[:, 1:] &= a[:, :-1]; e[:, :-1] &= a[:, 1:]
    else:                                   # 정사각형은 행/열로 분리 가능
        r = a.copy()
        r[1:] &= a[:-1]; r[:-1] &= a[1:]
        e = r.copy()
        e[:, 1:] &= r[:, :-1]; e[:, :-1] &= r[:, 1:]
    e[0] = e[-1] = False
    e[:, 0] = e[:, -1] = False
    return e


def _dil(a, ngb):
    d = a.copy()
    if ngb == 4:
        d[1:] |= a[:-1]; d[:-1] |= a[1:]
        d[:, 1:] |= a[:, :-1]; d[:, :-1] |= a[:, 1:]
    else:
        r = a.copy(); r[1:] |= a[:-1]; r[:-1] |= a[1:]
        d = r.copy(); d[:, 1:] |= r[:, :-1]; d[:, :-1] |= r[:, 1:]
    return d


def _erode(fg, n, ngb, protect=None):
    if n <= 0:
        return fg
    keep = None if protect is None else fg & protect     # --noerodequant: 보호 픽셀은 매 반복 복원
    cur = fg
    for _ in range(n):
        cur = _ero(cur, ngb)
        if keep is not None:
            cur |= keep
    return cur


def _dilate(a, n, ngb):
    for _ in range(n):
        a = _dil(a, ngb)
    return a


def _erode_open(fg, erode, opening, erodengb=4, openingngb=8, protect=None):
    eroded = _erode(fg, erode, erodengb, protect)
    opened = (_dilate(_erode(eroded, opening, openingngb, protect), opening, openingngb)
              if opening > 0 else eroded)
    return eroded, opened


# ===================================================================== #
# 5. False-detection 제거 (pseudo-detection S/N)
# ===================================================================== #
def _fill_holes(mask, structure):
    """binary_fill_holes 와 같은 결과를 라벨링으로 (훨씬 빠름): 경계와 이어지지
    않는 배경 성분을 전경으로."""
    lab, n = ndi.label(~mask, structure=structure)
    if n == 0:
        return mask
    touch = np.zeros(n + 1, bool)
    for edge in (lab[0], lab[-1], lab[:, 0], lab[:, -1]):
        touch[edge] = True
    touch[0] = True
    return mask | ~touch[lab]


def _sn_values(data, labels, nlab, Sa, snminarea):
    """pseudo-detection 별 S/N = sqrt(N)(F-Sa)/sqrt(F+Sa)  (AI15 Eq.3), 벡터화.
    라벨이 있는 픽셀만 모아 bincount 로 합산 (전체 영상 순회 최소화)."""
    if nlab == 0:
        return np.empty(0)
    sel = labels > 0
    lab = labels[sel]
    area = np.bincount(lab, minlength=nlab + 1)[1:].astype(np.float64)
    cnt = np.maximum(area, 1)
    mean = np.bincount(lab, weights=data[sel], minlength=nlab + 1)[1:] / cnt
    if np.isscalar(Sa):
        sa = np.full(nlab, float(Sa))
    else:
        sa = np.bincount(lab, weights=Sa[sel], minlength=nlab + 1)[1:] / cnt
    den = mean + sa
    ok = (area >= snminarea) & (den > 0)
    sn = np.full(nlab, np.nan)
    sn[ok] = np.sqrt(area[ok]) * (mean[ok] - sa[ok]) / np.sqrt(den[ok])
    return sn


def _component_tile_ids(labels, nlab, tilesize):
    """Return the large-tile index containing each component's centroid."""
    if nlab == 0:
        return np.empty(0, dtype=np.int64)
    ty, tx, nty, ntx = _tile_geometry(labels.shape, tilesize)
    ids = np.arange(1, nlab + 1)
    centers = np.asarray(ndi.center_of_mass(labels, labels, ids))
    iy = np.clip((centers[:, 0] // ty).astype(int), 0, nty - 1)
    ix = np.clip((centers[:, 1] // tx).astype(int), 0, ntx - 1)
    return iy * ntx + ix


def _remove_false_detections(data, opened, dthresh, snminarea, snquant, dopening,
                             dopeningngb, holengb, pseudoconcomp, snthresh, tile_sky,
                             largetilesize=None, minnumfalse=100):
    Rd, Rs = opened, ~opened
    if tile_sky is None:
        Sa, _, sig = sigma_clipped_stats(data, mask=opened, sigma=3.0, maxiters=5)
        Sa_map = sig_map = None
        thresh2, Sa_arg = Sa + dthresh * sig, Sa
    else:
        Sa_map, sig_map = tile_sky(Rs)
        Sa, sig = float(np.median(Sa_map[::10, ::10])), float(np.median(sig_map[::10, ::10]))
        thresh2, Sa_arg = Sa_map + dthresh * sig_map, Sa_map

    def pseudo(region):
        above = _fill_holes((data > thresh2) & region, _NGB[holengb])
        if dopening > 0:
            above = _dilate(_erode(above, dopening, dopeningngb), dopening, dopeningngb)
        return ndi.label(above, structure=_NGB[pseudoconcomp])

    lab_s, ns = pseudo(Rs)
    sn_s = _sn_values(data, lab_s, ns, Sa_arg, snminarea)
    if snthresh is not None:
        sn_thr = snthresh
        threshold_grid = None
    else:
        valid_s = np.isfinite(sn_s)
        global_thr = np.quantile(sn_s[valid_s], snquant) if valid_s.any() else -np.inf
        sn_thr = global_thr
        threshold_grid = None
        if largetilesize is not None:
            ty, tx, nty, ntx = _tile_geometry(data.shape, largetilesize)
            tile_ids = _component_tile_ids(lab_s, ns, (ty, tx))
            flat_values = np.full(nty * ntx, np.nan, dtype=np.float64)
            for tid in range(nty * ntx):
                vals = sn_s[valid_s & (tile_ids == tid)]
                if vals.size >= minnumfalse:
                    flat_values[tid] = np.quantile(vals, snquant)
            good_tiles = np.isfinite(flat_values)
            if good_tiles.any():
                threshold_grid = _interpolate(
                    flat_values.reshape(nty, ntx),
                    good_tiles.reshape(nty, ntx),
                    k=1,
                )
            else:
                threshold_grid = np.full((nty, ntx), global_thr, dtype=np.float64)

    lab_d, nd = pseudo(Rd)
    sn_d = _sn_values(data, lab_d, nd, Sa_arg, snminarea)
    accepted = np.zeros(nd + 1, bool)
    if threshold_grid is None:
        accepted[1:] = sn_d >= sn_thr
    else:
        d_tile_ids = _component_tile_ids(lab_d, nd, largetilesize)
        thresholds = threshold_grid.ravel()[d_tile_ids]
        accepted[1:] = sn_d >= thresholds
    acc_mask = accepted[lab_d]
    del lab_d

    cc, ncc = ndi.label(Rd, structure=STRUCT_8)             # 받아들여진 pseudo-det 와 겹치는 성분만
    good = np.zeros(ncc + 1, bool)
    good[cc[acc_mask]] = True
    good[0] = False
    if threshold_grid is not None:
        sn_thr = float(np.nanmedian(threshold_grid))
    return good[cc], Sa, sig, sn_thr, Sa_map, sig_map


# ===================================================================== #
# 6. Growth / 구멍 메우기 / 정리
# ===================================================================== #
def _grow(data, true_mask, detgrowquant):
    """--detgrowquant. sky(비검출) 픽셀 분포의 detgrowquant quantile 이상인 픽셀로
    true detection 에서 8-연결 경로가 닿는 영역까지 성장. (priority-flood 와
    최종 마스크가 동일 -> 라벨링으로 계산.) 1.0 이면 성장 없음."""
    if detgrowquant >= 1.0 or not true_mask.any():
        return true_mask.copy()
    finite = np.isfinite(data)
    sky = data[~true_mask & finite]
    if sky.size == 0:
        return true_mask.copy()
    thr = np.quantile(sky, detgrowquant)
    del sky
    lab, n = ndi.label(((data >= thr) & finite) | true_mask, structure=STRUCT_8)
    keep = np.zeros(n + 1, bool)
    keep[lab[true_mask]] = True
    keep[0] = False
    return keep[lab]


def _fill_small_holes(mask, max_size):
    """--detgrowmaxholesize: 이미지 경계에 닿지 않는, max_size 이하의 구멍만 채움."""
    lab, n = ndi.label(~mask, structure=STRUCT_4)
    if n == 0:
        return mask
    size = np.bincount(lab.ravel(), minlength=n + 1)
    fill = size <= max_size
    for edge in (lab[0], lab[-1], lab[:, 0], lab[:, -1]):
        fill[edge] = False
    fill[0] = False
    return mask | fill[lab]


def _clean_grown(data, mask, sn_thr, Sa):
    """--cleangrowndet: growth 후 S/N 이 pseudo-detection S/N 한계보다 낮은 detection 제거."""
    lab, n = ndi.label(mask, structure=STRUCT_8)
    keep = np.zeros(n + 1, bool)
    keep[1:] = _sn_values(data, lab, n, Sa, 0) >= sn_thr
    return keep[lab]


# ===================================================================== #
# Public API
# ===================================================================== #
def detect(
    data,
    kernel=2.0, qthresh=0.30, erode=2, opening=1, noerodequant=0.99,
    dthresh=-0.0, snminarea=10, snquant=0.99,
    dopening=1, dopeningngb=4, holengb=8, pseudoconcomp=4, snthresh=None,
    detgrowquant=0.9, detgrowmaxholesize=None, cleangrowndet=False,
    n_dilate=0, min_final_area=1,
    tilesize=None, largetilesize=200, meanmedqdiff=0.01,
    outliersigma=3.0, interpnumngb=15, minskyfrac=0.7, smoothwidth=5,
    blankasforeground=False, numthreads=None, widekernel=None,
    minnumfalse=100, erodengb=4, openingngb=8,
    segmentation=False,
) -> NoiseChiselResult:
    """NoiseChisel-lite 검출. 파라미터는 Gnuastro 옵션과 같은 이름/의미:
    kernel (Gaussian FWHM in pixels in this Python implementation), qthresh,
    erode, opening, erodengb, openingngb, noerodequant,
    dthresh, snminarea, snquant, dopening/dopeningngb/holengb/
    pseudoconcomp, snthresh, detgrowquant, detgrowmaxholesize, cleangrowndet,
    tilesize/largetilesize/meanmedqdiff/outliersigma/interpnumngb/minskyfrac/smoothwidth
    (tessellation, tilesize=None 이면 전체 이미지 통계 하나만 사용),
    blankasforeground. n_dilate, min_final_area, segmentation은 Python 확장 옵션.
    segmentation : bool
        True이면 Photutils SegmentationImage도 계산합니다. False이면 최종 mask만 반환하고
        라벨링 및 Photutils import를 건너뜁니다.

    largetilesize : int 또는 (int, int) 또는 None
        미검출 영역 pseudo-detection의 snquant 기준을 계산할 큰 타일 크기.
        타일마다 기준을 추정하며, 타일 안 유효 pseudo-detection 수가 minnumfalse보다
        적으면 가까운 유효 타일 기준을 사용합니다. None이면 전체 영상 기준 하나를 씁니다.
    widekernel : float 또는 None
        타일 신호 유무 판정에 사용할 넓은 Gaussian FWHM (픽셀). 타일 모드에서만
        사용합니다. threshold quantile은 항상 기본 kernel convolution에서 계산합니다.
        GNU Astro의 --widekernel은 커널 파일을 받으므로 여기서는 Gaussian FWHM으로 받습니다.

    numthreads : 타일 통계 계산에 쓸 스레드 수 (기본 1).
    minnumfalse : largetilesize 타일별 snquant 계산에 필요한 최소 유효 pseudo-detection 수.

    tilesize 는 타일 안 픽셀 수가 충분해야 합니다(타일이 작고 meanmedqdiff 가 너무
    엄격하면 순수 노이즈도 '신호 있음'으로 판정됨; 50px 이상 또는 0.02~0.05 권장).
    """
    data = np.asarray(data, dtype=WORK_DTYPE)
    res = NoiseChiselResult()
    blank = ~np.isfinite(data)
    any_blank = blank.any()

    conv = res.convolved = _convolve(data, kernel)
    protect = None
    tile_sky = None

    if tilesize is None:
        valid = conv[np.isfinite(conv)] if any_blank else conv
        qs = [qthresh] + ([noerodequant] if noerodequant is not None else [])
        qv = np.quantile(valid, qs)
        res.threshold = float(qv[0])
        res.foreground = conv > res.threshold
        if noerodequant is not None:
            protect = conv > qv[1]
        del valid
    else:
        ty, tx, nty, ntx = _tile_geometry(data.shape, tilesize)
        qs = [qthresh] + ([noerodequant] if noerodequant is not None else [])
        # A wider convolution is only for classifying signal-free tiles;
        # threshold quantiles are still measured on the sharp convolution.
        if widekernel is None:
            st = _tile_pass(conv, ty, tx, quantiles=qs, n_jobs=numthreads)
            qstats = st
        else:
            class_conv = _convolve(data, widekernel)
            st = _tile_pass(class_conv, ty, tx, n_jobs=numthreads)
            qstats = _tile_pass(conv, ty, tx, quantiles=qs, n_jobs=numthreads)
        good = (st["n"] >= 10) & (np.abs(st["meanq"] - 0.5) <= meanmedqdiff)

        def tmap(vals, q):
            return _tile_map(np.where(good, vals, np.nan), good, data.shape, ty, tx, outliersigma,
                             interpnumngb, smoothwidth, lambda: np.nanquantile(conv, q))

        res.threshold = tmap(qstats["q"][0], qthresh)
        res.foreground = conv > res.threshold
        if noerodequant is not None:
            protect = conv > tmap(qstats["q"][1], noerodequant)
        res.tile_shape, res.tile_good_mask = (nty, ntx), good

        def tile_sky(Rs):                                   # Rs(비검출) 기준 타일 Sky 지도
            s = _tile_pass(data, ty, tx, mask=Rs, n_jobs=numthreads)
            g = (s["frac"] >= minskyfrac) & (s["n"] >= 10)

            def fallback(j):                                # 쓸 수 있는 타일이 없을 때만
                ok = Rs & ~blank
                v = data[ok] if ok.sum() >= 10 else data[~blank]
                return (lambda r: (r[0], r[2])[j])(sigma_clipped_stats(v, sigma=3.0, maxiters=5))

            return tuple(_tile_map(np.where(g, s[k], np.nan), g, data.shape, ty, tx, None,
                                   interpnumngb, smoothwidth, lambda j=j: fallback(j))
                         for j, k in enumerate(("mean", "std")))

    if any_blank:
        res.foreground[blank] = blankasforeground
    res.protect_mask = protect
    res.eroded, res.opened = _erode_open(
        res.foreground, erode, opening, erodengb, openingngb, protect
    )

    (res.true_mask, res.sky_value, res.sky_std, res.sn_threshold,
     res.sky_map, res.sky_std_map) = _remove_false_detections(
        data, res.opened, dthresh, snminarea, snquant, dopening, dopeningngb,
        holengb, pseudoconcomp, snthresh, tile_sky, largetilesize, minnumfalse)

    res.grown = final = _grow(data, res.true_mask, detgrowquant)
    if detgrowmaxholesize is not None:
        final = _fill_small_holes(final, detgrowmaxholesize)
    if cleangrowndet:
        final = _clean_grown(data, final, res.sn_threshold, res.sky_value)
    if n_dilate > 0:
        final = _dilate(final, n_dilate, 8)
    if any_blank:
        final = final & ~blank
    res.detected = final

    if segmentation:
        from photutils.segmentation import SegmentationImage

        labels, n = ndi.label(final, structure=STRUCT_8)
        if min_final_area > 1 and n > 0:
            size = np.bincount(labels.ravel(), minlength=n + 1)
            labels[(size < min_final_area)[labels]] = 0
            labels, _ = relabel_consecutive(labels)
        res.segmap = SegmentationImage(labels.astype(np.int32, copy=False))
    else:
        res.segmap = None
    return res


def relabel_consecutive(labels):
    """라벨을 1..N 으로 재번호 (0=배경)."""
    uniq = np.unique(labels)
    uniq = uniq[uniq != 0]
    remap = np.zeros(labels.max() + 1, dtype=labels.dtype)
    remap[uniq] = np.arange(1, len(uniq) + 1)
    return remap[labels], len(uniq)


def sky_stats(data, segmap, sigma=3.0):
    """최종 segmap 의 비검출 픽셀로 구한 global sigma-clipped (mean, median, std).
    gradient 가 있으면 detect(tilesize=...) 의 res.sky_map 을 쓰세요."""
    return sigma_clipped_stats(data, mask=segmap.data > 0, sigma=sigma, maxiters=5)


def mask_for_photometry(segmap, keep_labels=None, remove_labels=None):
    """특정 label 만 남기거나(keep_labels) 제거(remove_labels)한 뒤 boolean mask
    (True=마스크) 반환. photutils SegmentationImage.keep_labels/remove_labels 사용."""
    segm = segmap.copy()
    if keep_labels is not None:
        segm.keep_labels(keep_labels)
    if remove_labels is not None:
        segm.remove_labels(remove_labels)
    return segm.data > 0
