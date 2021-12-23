import warnings
import math
from scipy.special import betaln, gammaln
from scipy.optimize import fsolve
import numpy as np

CFAR_METHODS = ['ca', 'soca', 'goca', 'os']
GOS_CFAR_METHODS = ['gosca', 'gosgo', 'gosso']
OUTPUT_MODES = ['none', 'detections', 'all']


def get_training_cells(x, idx, num_guard_cells, num_training_cells):
    n = len(x)
    half_guard = num_guard_cells // 2
    half_training = num_training_cells // 2
    
    available = n - 2*half_guard - 1
    assert(available >= num_training_cells)
    
    lguard = idx - half_guard
    rguard = idx + half_guard
    available_left = max(0, lguard)
    available_right = max(0, n - 1 - rguard)
    if available_left < available_right:
        num_lcells = min(half_training, available_left)
        lcells = x[lguard - num_lcells : lguard]
        rcells = x[rguard + 1 : rguard + 1 + min(available_right, 2*half_training - num_lcells)]
    else:
        num_rcells = min(half_training, available_right)
        rcells = x[rguard + 1 : rguard + 1 + num_rcells]
        lcells = x[lguard - min(available_left, 2*half_training - num_rcells) : lguard]
    return lcells, rcells

def get_training_cells_2d(x, idx, guard_region_size, training_region_size):
    x = np.array(x)
    idx = np.array(idx)
    guard_region_size = np.array(guard_region_size)
    training_region_size = np.array(training_region_size)
    
    guard_index_tl = (idx - (guard_region_size - 1) / 2).astype(int)
    guard_index_br1 = guard_index_tl + guard_region_size
    training_index_tl = (idx - (training_region_size - 1) / 2).astype(int)
    training_index_br1 = training_index_tl + training_region_size
    assert(np.all(guard_index_tl >= 0) and np.all(training_index_tl >= 0) and
           np.all(guard_index_br1 <= x.shape) and np.all(training_index_br1 <= x.shape))
    
    mask = np.zeros(x.shape, dtype=int)
    mask[training_index_tl[0]:idx[0],
         training_index_tl[1]:training_index_br1[1]] = 1
    mask[idx[0], training_index_tl[1]:idx[1]] = 1
    mask[idx[0], idx[1]+1 : training_index_br1[1]]= 2
    mask[idx[0]+1 : training_index_br1[0],
         training_index_tl[1]:training_index_br1[1]] = 2
    mask[guard_index_tl[0]:guard_index_br1[0],
         guard_index_tl[1]:guard_index_br1[1]] = 0
    return x[np.where(mask == 1)], x[np.where(mask == 2)]
    

def threshold_soca_goca(x, n):
    return sum(
        math.exp(gammaln(n/2 + k) - gammaln(k + 1) - gammaln(n/2)) *
        (2 + x/(n/2))**(-k) for k in range(n//2)
    ) * (2 + x/(n/2))**(-n/2)
    
def threshold_soca(x, n, pfa):
    return threshold_soca_goca(x, n) - pfa/2

def threshold_goca(x, n, pfa):
    return (1 + x/(n/2))**(-n/2) - threshold_soca_goca(x, n) - pfa/2

def threshold_os(x, n, k, pfa):
    c = x + n - k + 1
    if c >= 0:
        return math.exp(
            gammaln(n+1) - gammaln(k) - gammaln(n-k+1) +
            betaln(c, k)
        ) - pfa
    return float('nan')

def threshold_gosca(x, n, m, k, l, pfa):
    def subexpr(M, K):
        assert M - K + 1 + x > 0, "Must not violate this condition; cannot determine automatic threshold otherwise. This problem will be addressed in the future."
        return gammaln(M + 1) + gammaln(M - K + 1 + x) - gammaln(M + 1 + x) - gammaln(M - K + 1)
    return math.exp(
        subexpr(m, k) + subexpr(n, l)
    ) - pfa

def threshold_gosgo(x, n, m, k, l, pfa):
    def subexpr(M, N, K, L):
        assert M - K + 1 + x > 0, "Must not violate this condition; cannot determine automatic threshold otherwise. This problem will be addressed in the future."
        return math.exp(gammaln(M + 1) + gammaln(N + 1)) * sum(
            math.exp(gammaln(K + j + 1) + gammaln(M - K + 1 + N - j + x)) / (
                j * math.exp(gammaln(j + 1) + gammaln(N - j + 1)) +
                K * math.exp(gammaln(j + 1) + gammaln(N - j + 1))
            ) for j in range(L, N + 1)
        ) / math.exp(gammaln(K) + gammaln(M - K + 1) + gammaln(M + N + 1 + x))
    return subexpr(m, n, k, l) + subexpr(n, m, l, k) - pfa

def threshold_gosso(x, n, m, k, l, pfa):
    def subexpr(M, N, K, L):
        assert M - K + 1 + x > 0, "Must not violate this condition; cannot determine automatic threshold otherwise. This problem will be addressed in the future."
        return -(
            -np.exp(gammaln(K) + gammaln(M - K + 1 + x)) +
            np.exp(gammaln(M + 1 + x)) * sum(
                np.exp(gammaln(N + 1) + gammaln(K + 1 + j) + gammaln(M - K + 1 + N - j + x)) / (
                    j * np.exp(gammaln(j + 1) + gammaln(N - j + 1) + gammaln(M + N + 1 + x)) +
                    K * np.exp(gammaln(j + 1) + gammaln(N - j + 1) + gammaln(M + N + 1 + x))
                ) for j in range(L, N + 1)              
            )
        ) * np.exp(gammaln(M + 1) - gammaln(K) - gammaln(M - K + 1) - gammaln(M + 1 + x))
    return subexpr(m, n, k, l) + subexpr(n, m, l, k) - pfa

def auto_threshold_factor(n, pfa, meth, rank, rank_left=1, rank_right=1):
    assert(meth in CFAR_METHODS or meth in GOS_CFAR_METHODS)
    def fsolve_success(*args, **kwargs):
        kwargs['full_output'] = True
        s, _, ier, _ = fsolve(*args, **kwargs)
        success = ier == 1
        if not success:
            warnings.warn("Could not reliably determine the automatic threshold factor for method %s" % meth, RuntimeWarning)
        return s[0], success
    ca = n * (pfa**(-1/n) - 1)
    if meth == 'ca':
        return ca, True
    elif meth == 'soca':
        return fsolve_success(threshold_soca, ca, args=(n, pfa))
    elif meth == 'goca':
        return fsolve_success(threshold_goca, ca, args=(n, pfa))
    elif meth == 'os':
        # guess = ca
        # for i in range(50):
        #     sol, success = fsolve_success(threshold_os, guess, args=(n,rank, pfa))
        #     if success:
        #         break
        #     guess *= 10
        # return sol, success
        return fsolve_success(threshold_os, ca, args=(n, rank, pfa))
    elif meth == 'gosca':
        return fsolve_success(threshold_gosca, ca, args=(n // 2, n // 2, rank_left, rank_right, pfa))
    elif meth == 'gosgo':
        return fsolve_success(threshold_gosgo, ca, args=(n // 2, n // 2, rank_left, rank_right, pfa))
    elif meth == 'gosso':
        return fsolve_success(threshold_gosso, ca, args=(n // 2, n // 2, rank_left, rank_right, pfa))

def _arr_index_real(arr, l):
    assert (0 <= l <= 1)
    return arr[-1] if l == 1 else arr[int(l * len(arr))]

def _validate_output_mode(output_thresh, output_noise):
    if output_thresh not in OUTPUT_MODES:
        raise ValueError("output_thresholds_mode must be one of: 'none', 'detections', 'all'")
    if output_noise not in OUTPUT_MODES:
        raise ValueError("output_noise_levels_mode must be one of: 'none', 'detections', 'all'")

def _validate_common_params(x, num_guard_cells, num_training_cells, output_thresh, output_noise):
    _validate_output_mode(output_thresh, output_noise)
    if num_guard_cells % 2 != 0 or num_guard_cells < 0:
        raise ValueError("num_guard_cells must be even and nonnegative")
    if num_training_cells % 2 != 0 or num_training_cells <= 0:
        raise ValueError("num_training_cells must be even and positive")
    n = len(x)
    if n < 3:
        raise ValueError("The length of the sequence must be at least 3")
    if num_training_cells + num_guard_cells > n - 1:
        raise ValueError("Too many guard and/or training cells specified")

def _validate_common_params_2d(x, guard_region_size, training_region_size, output_thresh, output_noise):
    _validate_output_mode(output_thresh, output_noise)
    if type(x) != np.ndarray:
        raise ValueError("x must be a numpy array")
    if x.ndim != 2:
        raise ValueError("x must be a matrix")
    if len(guard_region_size) != 2:
        raise ValueError("guard_band_size must be a sequence of size 2")
    if len(training_region_size) != 2:
        raise ValueError("training_band_size must be a sequence of size 2")
    if np.any(guard_region_size > x.shape):
        raise ValueError("guard_band_size is too large")
    if np.any(training_region_size > x.shape):
        raise ValueError("training_band_size is too large")

def _get_indices_2d(x, guard_band_size, training_band_size):
    mat_h, mat_w = x.shape
    gband_h, gband_w = guard_band_size
    tband_h, tband_w = training_band_size
    return (
        (i, j)
        for i in range(tband_h + gband_h, mat_h - tband_h - gband_h)
        for j in range(tband_w + gband_w, mat_w - tband_w - gband_w)
    )

def _perform_cfar(x, indices, guard_cells_size, training_cells_size,
                  get_training_cells_func, meth, thfac, noise_power_func,
                  output_thresh, output_noise, **kwargs):
    properties = {}
    peaks = []
    if output_thresh != 'none':
        thresholds = []
    if output_noise != 'none':
        noise_levels = []
    
    for i in indices:
        ltraining, rtraining = get_training_cells_func(x, i, guard_cells_size, training_cells_size)
        noise_power = noise_power_func(ltraining, rtraining, meth, **kwargs)
        threshold = thfac * noise_power
        
        detection = x[i] > threshold
        if detection:
            peaks.append(i)
        if output_thresh == 'all' or output_thresh == 'detections' and detection:
            thresholds.append(threshold)
        if output_noise == 'all' or output_noise == 'detections' and detection:
            noise_levels.append(noise_power)
    if output_thresh != 'none':
        properties['thresholds'] = np.array(thresholds)
    if output_noise != 'none':
        properties['noise_levels'] = np.array(noise_levels)
    return np.array(peaks), properties

def _noise_power(ltraining, rtraining, meth, **kwargs):
    os_rank = kwargs['os_rank']
    if meth in {'ca', 'os'}:
            training = np.concatenate((ltraining, rtraining))
        
    if meth == 'ca':
        return np.mean(training)
    elif meth == 'soca':
        return min(np.mean(x) for x in (ltraining, rtraining) if len(x) > 0)
    elif meth == 'goca':
        return max(np.mean(x) for x in (ltraining, rtraining) if len(x) > 0)
    elif meth == 'os':
        training.sort()
        return training[os_rank - 1]

def _noise_power_gos(ltraining, rtraining, meth, **kwargs):
    rank_left = kwargs['rank_left']
    rank_right = kwargs['rank_right']
    ltraining.sort()
    rtraining.sort()
    
    if meth == 'gosca':
        gos_func = sum # np.mean
    elif meth == 'gosgo':
        gos_func = max
    elif meth == 'gosso':
        gos_func = min
    return gos_func(_arr_index_real(x, l) for (x, l) in ((ltraining, rank_left), (rtraining, rank_right)) if len(x) > 0)
    
###### TODO: Merge cfar_detector and cfar_detector_gos
def cfar_detector(x, num_guard_cells, num_training_cells,
                  pfa, method='ca', os_rank=1, custom_threshold_factor=None,
                  output_thresholds_mode='none', output_noise_levels_mode='none'):
    output_thresh = output_thresholds_mode.lower()
    output_noise = output_noise_levels_mode.lower()
    _validate_common_params(x, num_guard_cells, num_training_cells, output_thresh, output_noise)
    if custom_threshold_factor is None and not 0 < pfa < 1:
        raise ValueError("pfa must be greater than 0 and less than 1")
    
    meth = method.lower()
    if meth not in CFAR_METHODS:
        raise ValueError("Unknown method %s" % meth)
    if meth == 'os' and not 1 <= os_rank <= num_training_cells:
        raise ValueError("Rank must be between 1 and num_training_cells inclusive")
    
    if custom_threshold_factor is not None:
        thfac = custom_threshold_factor
    else:
        thfac, _ = auto_threshold_factor(num_training_cells, pfa, meth, os_rank)
    
    return _perform_cfar(x, range(len(x)), num_guard_cells, num_training_cells,
                         get_training_cells, meth, thfac, _noise_power,
                         output_thresh, output_noise, os_rank=os_rank)

def cfar_detector_gos(x, num_guard_cells, num_training_cells, pfa,
                      method='gosca', rank_left=1, rank_right=1, custom_threshold_factor=None,
                      output_thresholds_mode='none', output_noise_levels_mode='none'):
    output_thresh = output_thresholds_mode.lower()
    output_noise = output_noise_levels_mode.lower()
    _validate_common_params(x, num_guard_cells, num_training_cells, output_thresh, output_noise)
    if custom_threshold_factor is None and not 0 < pfa < 1:
        raise ValueError("pfa must be greater than 0 and less than 1")
    
    meth = method.lower()
    if meth not in GOS_CFAR_METHODS:
        raise ValueError("Unknown method %s" % meth)
    if not 0 <= rank_left <= 1 or not 0 <= rank_right <= 1:
        raise ValueError("rank_left and rank_right must have values between 0 and 1 inclusive")
    
    if custom_threshold_factor is not None:
        thfac = custom_threshold_factor
    else:
        thfac, _ = auto_threshold_factor(num_training_cells, pfa, meth, None, rank_left, rank_right)
    
    return _perform_cfar(x, range(len(x)), num_guard_cells, num_training_cells,
                         get_training_cells, meth, thfac, _noise_power_gos,
                         output_thresh, output_noise, rank_left=rank_left, rank_right=rank_right)

###### TODO: Merge cfar_detector_2d and cfar_detector_2d_gos
def cfar_detector_2d(x, guard_band_size, training_band_size,
                  pfa, method='ca', os_rank=1, custom_threshold_factor=None,
                  output_thresholds_mode='none', output_noise_levels_mode='none'):
    output_thresh = output_thresholds_mode.lower()
    output_noise = output_noise_levels_mode.lower()
    guard_region_size = 2 * np.array(guard_band_size) + 1
    training_region_size = 2 * np.array(training_band_size) + guard_region_size
    _validate_common_params_2d(x, guard_region_size, training_region_size, output_thresh, output_noise)
    if custom_threshold_factor is None and not 0 < pfa < 1:
        raise ValueError("pfa must be greater than 0 and less than 1")
    
    meth = method.lower()
    num_training_cells = np.prod(training_region_size) - np.prod(guard_region_size)
    if meth not in CFAR_METHODS:
        raise ValueError("Unknown method %s" % meth)
    if meth == 'os' and not 1 <= os_rank <= num_training_cells:
        raise ValueError("Rank must be between 1 and num_training_cells inclusive")
    
    if custom_threshold_factor is not None:
        thfac = custom_threshold_factor
    else:
        thfac, _ = auto_threshold_factor(num_training_cells, pfa, meth, os_rank)
    
    indices = _get_indices_2d(x, guard_band_size, training_band_size)
    return _perform_cfar(x, indices, guard_region_size, training_region_size,
                         get_training_cells_2d, meth, thfac, _noise_power,
                         output_thresh, output_noise, os_rank=os_rank)
 
def cfar_detector_2d_gos(x, guard_band_size, training_band_size, pfa,
                      method='gosca', rank_left=0, rank_right=0, custom_threshold_factor=None,
                      output_thresholds_mode='none', output_noise_levels_mode='none'):
    output_thresh = output_thresholds_mode.lower()
    output_noise = output_noise_levels_mode.lower()
    guard_region_size = 2 * np.array(guard_band_size) + 1
    training_region_size = 2 * np.array(training_band_size) + guard_region_size
    _validate_common_params_2d(x, guard_region_size, training_region_size, output_thresh, output_noise)
    if custom_threshold_factor is None and not 0 < pfa < 1:
        raise ValueError("pfa must be greater than 0 and less than 1")
    
    meth = method.lower()
    num_training_cells = np.prod(training_region_size) - np.prod(guard_region_size)
    if meth not in GOS_CFAR_METHODS:
        raise ValueError("Unknown method %s" % meth)
    if not 0 <= rank_left <= 1 or not 0 <= rank_right <= 1:
        raise ValueError("rank_left and rank_right must have values between 0 and 1 inclusive")
    
    if custom_threshold_factor is not None:
        thfac = custom_threshold_factor
    else:
        thfac, _ = auto_threshold_factor(num_training_cells, pfa, meth, None, rank_left, rank_right)
    
    indices = _get_indices_2d(x, guard_band_size, training_band_size)
    return _perform_cfar(x, indices, guard_region_size, training_region_size,
                         get_training_cells_2d, meth, thfac, _noise_power_gos,
                         output_thresh, output_noise, rank_left=rank_left, rank_right=rank_right)