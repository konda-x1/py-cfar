# py-cfar

## 1D CFAR Detector
### Description
1D CFAR detection using Cell-Averaging (CA, GOCA/CAGO, SOCA/CASO) and Order Statistics (OS) methods.

### Usage
cfar_detector(**x**, **num_guard_cells**, **num_training_cells**, **pfa**, **method**='ca', **os_rank**=1, *custom_threshold_factor*=None, *output_thresholds_mode*='none', *output_noise_levels_mode*='none')
* **x**: Sequence of numbers, preferably numpy.array. Must be of length 3 or above.
* **num_guard_cells**: Total number of guard cells on both sides of the cell under test (CUT). Must be even and nonnegative
* **num_training_cells**: Total number of training cells on both sides of the cell under test (CUT). Must be even and **positive**
* **pfa**: Probability of false alarm for automatic threshold factor. This parameter is inconsequential if *custom_threshold_factor* is not None
* **method**: CFAR method. Valid methods are: 'ca', 'soca', 'goca', and 'os'
* **os_rank**: Training cell rank for order statistics (OS) CFAR method. 1 <= *os_rank* <= *num_training_cells*. Rank 1 is smallest-valued element/cell, rank num_training_cells is largest-valued element/cell.
* **custom_threshold_factor**: If not None, the value of this parameter will be used as the threshold factor for target detection instead of the automatic threshold factor driven by the *pfa* parameter
* *output_thresholds_mode*: One of 'none', 'detections', and 'all'. Will be explained below.
* *output_noise_levels_mode*: One of 'none', 'detections', and 'all'. Will be explained below.

Returns: (peak_indices, properties)
* **peak_indices**: numpy.array of integers, each being an index in *x* where a peak is detected
* **properties**: Python dict with the following keys
  * *'thresholds'*: If *output_thresholds_mode* is 'none', this key is not present. If *output_thresholds_mode* is 'detections', will contain the calculated thresholds for each peak in *x* corresponding to *peak_indices*. If *output_thresholds_mode* is 'all', will contain the calculated thresholds for every element in *x*
  * *'noise_levels'*: If *output_thresholds_mode* is 'none', this key is not present. If *output_thresholds_mode* is 'detections', will contain the calculated noise levels for each peak in *x* corresponding to *peak_indices*. If *output_thresholds_mode* is 'all', will contain the calculated noise levels for every element in *x*

Notes:
* Training cells are distributed evenly on the left and right sides of the CUT when possible.
In edge cases where there is not enough room on one of the sides of the CUT for the cells to be distributed evently,
the "extra" cells are distributed on the other side of the CUT instead. This way, the total number of training cells
on both sides is always equal to *num_training_cells*. The function will throw a ValueError if there are more training
cells specified than can fit in *x*.

Reference: https://www.mathworks.com/help/phased/ug/constant-false-alarm-rate-cfar-detection.html

## 1D CFAR detector using GOS methods
### Description
1D CFAR detection using Generalised Order Statistics methods (GOSCA, GOSGO, GOSSO)

### Usage
cfar_detector_gos(**x**, **num_guard_cells**, **num_training_cells**, **pfa**, **method='gosca'**, **rank_left=1**, **rank_right=1**, *custom_threshold_factor=None*, *output_thresholds_mode='none', output_noise_levels_mode='none'*)
* **x**: Sequence of numbers, preferably numpy.array. Must be of length 3 or above.
* **num_guard_cells**: Total number of guard cells on both sides of the cell under test (CUT). Must be even and nonnegative
* **num_training_cells**: Total number of training cells on both sides of the cell under test (CUT). Must be even and **positive**
* **pfa**: Probability of false alarm for automatic threshold factor. This parameter is inconsequential if *custom_threshold_factor* is not None
* **method**: CFAR method. Valid methods are: 'gosca', 'gosgo', and 'gosso'
* **rank_left**: Cell rank for leading (left) training cells. Must be a real number where 0 <= *rank_left* <= 1. Rank of 0 means the smallest-valued cell, while the rank of 1 means the largest-valued cell.
* **rank_left**: Cell rank for trailing (right) training cells. Must be a real number where 0 <= *rank_right* <= 1. Rank of 0 means the smallest-valued cell, while the rank of 1 means the largest-valued cell.
* **custom_threshold_factor**: If not None, the value of this parameter will be used as the threshold factor for target detection instead of the automatic threshold factor driven by the *pfa* parameter
* *output_thresholds_mode*: One of 'none', 'detections', and 'all'. Will be explained below.
* *output_noise_levels_mode*: One of 'none', 'detections', and 'all'. Will be explained below.

Returns: (peak_indices, properties)
* **peak_indices**: numpy.array of integers, each being an index in *x* where a peak is detected
* **properties**: Python dict with the following keys
  * *'thresholds'*: If *output_thresholds_mode* is 'none', this key is not present. If *output_thresholds_mode* is 'detections', will contain the calculated thresholds for each peak in *x* corresponding to *peak_indices*. If *output_thresholds_mode* is 'all', will contain the calculated thresholds for every element in *x*
  * *'noise_levels'*: If *output_thresholds_mode* is 'none', this key is not present. If *output_thresholds_mode* is 'detections', will contain the calculated noise levels for each peak in *x* corresponding to *peak_indices*. If *output_thresholds_mode* is 'all', will contain the calculated noise levels for every element in *x*

Notes:
* **There is a possibility of an AssertionError being thrown for violating the condition "M - K + 1 + x > 0".** It will be addressed in the future,
but please do keep track of inputs for which this error appears.
* Training cells are distributed evenly on the left and right sides of the CUT when possible.
In edge cases where there is not enough room on one of the sides of the CUT for the cells to be distributed evently,
the "extra" cells are distributed on the other side of the CUT instead. This way, the total number of training cells
on both sides is always equal to *num_training_cells*. The function will throw a ValueError if there are more training
cells specified than can fit in *x*.

## 2D CFAR Detector
### Description
2D CFAR detection using Cell-Averaging (CA, GOCA/CAGO, SOCA/CASO) and Order Statistics (OS) methods.

### Usage
cfar_detector_2d(x, guard_band_size, training_band_size pfa, method='ca', os_rank=1, custom_threshold_factor=None, output_thresholds_mode='none', output_noise_levels_mode='none')
* **x**: 2D sequence of numbers, preferably numpy.array
* **guard_band_size**: Dimensions of the band of guard cells around the Cell Under Test (CUT), specified as a two-element tuple of ints
* **training_band_size**: Dimensions of the band of training cells around the band of guard cells, specified as a two-element tuple of ints
* **pfa**: Probability of false alarm for automatic threshold factor. This parameter is inconsequential if *custom_threshold_factor* is not None
* **method**: CFAR method. Valid methods are: 'ca', 'soca', 'goca', and 'os'
* **os_rank**: Training cell rank for order statistics (OS) CFAR method. 1 <= *os_rank* <= *num_training_cells*. Rank 1 is smallest-valued element/cell, rank num_training_cells is largest-valued element/cell.
* **custom_threshold_factor**: If not None, the value of this parameter will be used as the threshold factor for target detection instead of the automatic threshold factor driven by the *pfa* parameter
* *output_thresholds_mode*: One of 'none', 'detections', and 'all'. Will be explained below.
* *output_noise_levels_mode*: One of 'none', 'detections', and 'all'. Will be explained below.

Returns: (peak_indices, properties)
* **peak_indices**: numpy.array of two-element tuples of integers, each being an index in *x* where a peak is detected
* **properties**: Python dict with the following keys
  * *'thresholds'*: If *output_thresholds_mode* is 'none', this key is not present. If *output_thresholds_mode* is 'detections', will contain the calculated thresholds for each peak in *x* corresponding to *peak_indices*. If *output_thresholds_mode* is 'all', will contain the calculated thresholds for every element in *x*
  * *'noise_levels'*: If *output_thresholds_mode* is 'none', this key is not present. If *output_thresholds_mode* is 'detections', will contain the calculated noise levels for each peak in *x* corresponding to *peak_indices*. If *output_thresholds_mode* is 'all', will contain the calculated noise levels for every element in *x*

Notes:
* Guard band and training band dimensions are not matrix dimensions.

Reference: https://www.mathworks.com/help/phased/ref/phased.cfardetector2d-system-object.html

## 2D CFAR detector using GOS methods
### Description
2D CFAR detection using Generalised Order Statistics methods (GOSCA, GOSGO, GOSSO)

### Usage
def cfar_detector_2d_gos(x, guard_band_size, training_band_size, pfa, method='gosca', rank_left=0, rank_right=0, custom_threshold_factor=None, output_thresholds_mode='none', output_noise_levels_mode='none')
* **x**: 2D sequence of numbers, preferably numpy.array
* **guard_band_size**: Dimensions of the band of guard cells around the Cell Under Test (CUT), specified as a two-element tuple of ints
* **training_band_size**: Dimensions of the band of training cells around the band of guard cells, specified as a two-element tuple of ints
* **pfa**: Probability of false alarm for automatic threshold factor. This parameter is inconsequential if *custom_threshold_factor* is not None
* **method**: CFAR method. Valid methods are: 'gosca', 'gosgo', and 'gosso'
* **rank_left**: Cell rank for leading training cells. Must be a real number where 0 <= *rank_left* <= 1. Rank of 0 means the smallest-valued cell, while the rank of 1 means the largest-valued cell.
* **rank_left**: Cell rank for trailing training cells. Must be a real number where 0 <= *rank_right* <= 1. Rank of 0 means the smallest-valued cell, while the rank of 1 means the largest-valued cell.
* **custom_threshold_factor**: If not None, the value of this parameter will be used as the threshold factor for target detection instead of the automatic threshold factor driven by the *pfa* parameter
* *output_thresholds_mode*: One of 'none', 'detections', and 'all'. Will be explained below.
* *output_noise_levels_mode*: One of 'none', 'detections', and 'all'. Will be explained below.

Returns: (peak_indices, properties)
* **peak_indices**: numpy.array of two-element tuples of integers, each being an index in *x* where a peak is detected
* **properties**: Python dict with the following keys
  * *'thresholds'*: If *output_thresholds_mode* is 'none', this key is not present. If *output_thresholds_mode* is 'detections', will contain the calculated thresholds for each peak in *x* corresponding to *peak_indices*. If *output_thresholds_mode* is 'all', will contain the calculated thresholds for every element in *x*
  * *'noise_levels'*: If *output_thresholds_mode* is 'none', this key is not present. If *output_thresholds_mode* is 'detections', will contain the calculated noise levels for each peak in *x* corresponding to *peak_indices*. If *output_thresholds_mode* is 'all', will contain the calculated noise levels for every element in *x*

Notes:
* **There is a possibility of an AssertionError being thrown for violating the condition "M - K + 1 + x > 0".** It will be addressed in the future,
but please do keep track of inputs for which this error appears.
* Guard band and training band dimensions are not matrix dimensions.
