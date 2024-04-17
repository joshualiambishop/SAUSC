import numpy.typing as npt
import numpy as np


def pooled_stdev(stdevs: npt.NDArray[np.float_]) -> float:

    # HDX data often contains a 0 timepoint reference
    # At this point, the uptake and stdev is 0, which we don't want to average
    stdevs_without_initial_timepoint = stdevs[stdevs != 0]

    variance = stdevs_without_initial_timepoint**2
    avg_variance = variance.mean()
    return np.sqrt(avg_variance)


def pooled_standard_error_mean(stdevs: npt.NDArray[np.float_], n_repeats: int) -> float:
    pooled_sem = np.sqrt((2 * (pooled_stdev(stdevs) ** 2)) / ((n_repeats * 2) - 2))
    return pooled_sem
