# -*- coding: utf-8 -*-
"""
Correspondence: joshualiambishop@gmail.com
"""

from typing import Mapping, Optional, TypeAlias, Callable, TypeVar, Any, Type
import matplotlib.colors

from tkinter import filedialog
import enum
import numpy as np

import copy
import numpy.typing as npt
import tkinter as tk
import dataclasses

try:
    from scipy import stats
except ImportError as exc:
    raise ImportError("Please install scipy via 'pip install scipy'.") from exc
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib
except ImportError as exc:
    raise ImportError(
        "Please install matplotlib via 'pip install matplotlib'."
    ) from exc

import pathlib


class InvalidFileFormatException(Exception):
    pass


LOCAL_USER_DIR = pathlib.Path.home()


def file_browser() -> Optional[str]:
    try:
        root = tk.Tk()
        results = filedialog.askopenfilenames(
            parent=root,
            initialdir=LOCAL_USER_DIR,
            initialfile="tmp",
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
        )

        results_file = pathlib.Path(results[0])
        LOCAL_USER_DIR = results_file.parent

        return str(results_file)
    finally:
        root.destroy()


class StatisticalTestType(enum.Enum):
    t_test = enum.auto()
    hybrid_test = enum.auto()


# For simplicity, a cumulative operation on all exposures are included as if it were a timepoint.
# I've made this a constant simply for sanity.
CUMULATIVE_EXPOSURE_KEY = "Cumulative"


EXPECTED_STATE_DATA_HEADERS: list[str] = [
    "Protein",
    "Start",
    "End",
    "Sequence",
    "Modification",
    "Fragment",
    "MaxUptake",
    "MHP",
    "State",
    "Exposure",
    "Center",
    "Center SD",
    "Uptake",
    "Uptake SD",
    "RT",
    "RT SD",
]

# Making a new diverging colormap only really works if the pairs are both sequential
ALLOWED_COLORMAPS: list[str] = [
    "Greys",
    "Purples",
    "Blues",
    "Greens",
    "Oranges",
    "Reds",
    "YlOrBr",
    "YlOrRd",
    "OrRd",
    "PuRd",
    "RdPu",
    "BuPu",
    "GnBu",
    "PuBu",
    "YlGnBu",
    "PuBuGn",
    "BuGn",
    "YlGn",
]


# Only makes sense for HDX results to have white in the middle.
# HDX scientists will be most familiar with "bwr".
def make_diverging_colormap(
    protection_colormap: str, deprotection_colormap: str
) -> ListedColormap:
    """
    Combine two sequential colormaps into a diverging one, giving a user flexibility for their (de)protection colors.
    """

    if protection_colormap not in ALLOWED_COLORMAPS:
        raise ValueError(
            "Protection colormap {protection_colormap} must be one of {ALLOWED_COLORMAPS}"
        )

    if deprotection_colormap not in ALLOWED_COLORMAPS:
        raise ValueError(
            "Deprotection colormap {deprotection_colormap} must be one of {ALLOWED_COLORMAPS}"
        )

    sampling = np.linspace(0, 1, 128)

    protection_cmap = matplotlib.colormaps[protection_colormap].resampled(128)
    deprotection_cmap = matplotlib.colormaps[deprotection_colormap].resampled(128)

    new_colours = np.vstack((protection_cmap(sampling), deprotection_cmap(sampling)))

    return ListedColormap(
        colors=new_colours,
        name=f"Combined {protection_colormap}{deprotection_colormap}",
        N=256,
    )


# To my knowledge, pymol requires everything to be loaded in from a single script
# so there are many general utilities defined here.
def enforce_between_0_and_1(value: float) -> None:
    if not (0 <= value <= 1):
        raise ValueError(f"Value {value} must be between 0 and 1")


def convert_percentage_if_necessary(value: float) -> float:
    """
    If a number is above 1, divide it by 100.
    """
    if value <= 1:
        return value
    else:
        return value / 100


def check_valid_file(path: Optional[str]) -> None:
    if path is None:
        raise ValueError("No path supplied.")
    filepath = pathlib.Path(path)
    if not filepath.exists():
        raise IOError(f"File {path} does not exist")
    if not filepath.suffix == ".csv":
        raise IOError(f"File {path} must be a csv file.")


def is_floatable(string: str) -> bool:
    try:
        float(string)
        return True
    except ValueError:
        return False


# Pymol unfortunately passes all arguments as strings
PYMOL_BOOL: TypeAlias = str
PYMOL_TUPLE_FLOAT: TypeAlias = str
PYMOL_INT: TypeAlias = str
PYMOL_FLOAT: TypeAlias = str


def _str_to_bool(string: PYMOL_BOOL) -> bool:
    if string.lower() == "true":
        return True
    if string.lower() == "false":
        return False
    raise TypeError(f"Input {string} must be True or False")


def _str_to_tuple_float(string: PYMOL_TUPLE_FLOAT) -> tuple[float, ...]:
    # Assuming input is of the form "(1.0, 1.0, 1.0)"
    stripped = string.strip("()")
    components = stripped.split(",")
    if not all([is_floatable(possible_float) for possible_float in components]):
        raise ValueError(f"Components {components} are not all floatable.")
    return tuple([float(number) for number in components])


def _str_to_float(string: str) -> float:
    if is_floatable(string):
        return float(string)
    else:
        raise ValueError(f"Argument {string} could not be interpreted as a float.")


def _str_to_int(string: str) -> float:
    return int(_str_to_float(string))


pymol_convertors: dict[type, Callable] = {
    int: _str_to_int,
    float: _str_to_float,
    tuple[float]: _str_to_tuple_float,
}

SameAsInput = TypeVar("SameAsInput")


def convert_from_pymol(argument: Any, requested_type: Type[SameAsInput]) -> SameAsInput:
    assert (
        requested_type in pymol_convertors
    ), f"Haven't implemented a conversion for type {requested_type}."
    convertor = pymol_convertors[requested_type]
    return convertor(argument)


@dataclasses.dataclass(frozen=True)
class ExperimentalParameters:
    """
    Simple container for the parameters of the experiment, how many states (assuming one is the default), and the number of deuterium exposure durations (in minutes)
    """

    states: tuple[str]
    exposures: tuple[str]

    def __post_init__(self):
        assert len(self.states), "SAUSC only supports datafiles with two states."

    @property
    def sequence_parse_length(self) -> int:
        return len(self.states) * len(self.exposures)


@dataclasses.dataclass(frozen=True)
class UserParameters:
    """
    These parameters are only known to the user and cannot be automatically determined by the results.
    """

    n_repeats: int
    confidence_interval: float
    global_normalisation: bool
    statistical_test: StatisticalTestType

    def __post_init__(self):
        enforce_between_0_and_1(self.confidence_interval)


@dataclasses.dataclass(frozen=True)
class Uptake:
    mean: float
    stdev: float
    cumulative: bool = False


@dataclasses.dataclass(frozen=True)
class BaseFragment:
    """
    Represents information that is fundamentally based on a protein fragment
    """

    sequence: str
    start_residue: int
    end_residue: int
    max_deuterium_uptake: float

    def residue_present(self, residue: int) -> bool:
        return self.start_residue <= residue <= self.end_residue

    def __post_init__(self) -> None:
        assert (
            self.start_residue < self.end_residue
        ), "End residue must be after the start residue."

    def is_same_fragment(self, other: "BaseFragment") -> bool:
        return (
            self.sequence == other.sequence
            and self.start_residue == other.start_residue
            and self.end_residue == other.end_residue
            and self.max_deuterium_uptake == other.max_deuterium_uptake
        )


@dataclasses.dataclass(frozen=True)
class StateData(BaseFragment):
    """
    Represents a state for a given fragment, with deuterium uptakes corresponding to each exposure time.
    Naturally, references to the 'cumulative' exposure timestep isn't in this function, as it's generated within this script.
    """

    state: str
    max_deuterium_uptake: float
    exposures: dict[str, Uptake]

    def __post_init__(self) -> None:
        super().__post_init__()
        # For plotting and loop purposes, we will add the cumulative data as a timepoint in the exposures
        self.exposures[CUMULATIVE_EXPOSURE_KEY] = self.cumulative_data

    @property
    def cumulative_data(self) -> Uptake:
        uptakes = self.exposures.values()
        combined_mean = np.sum([uptake.mean for uptake in uptakes])
        combined_stdev = pooled_stdev(np.array([uptake.stdev for uptake in uptakes]))
        return Uptake(combined_mean, combined_stdev, cumulative=True)


def load_state_data(
    filepath: str,
) -> tuple[list[StateData], ExperimentalParameters]:
    """
    This is specifically designed to load a .csv containing state data.
    """
    expected_columns = np.arange(len(EXPECTED_STATE_DATA_HEADERS)).tolist()
    loaded_data = np.loadtxt(
        filepath, delimiter=",", dtype=object, usecols=expected_columns
    )
    headers = loaded_data[0].tolist()

    # Z refers to [something], only present in cluster data, a common mistake.
    if "z" in loaded_data:
        raise InvalidFileFormatException(
            f"{filepath} appears to be cluster data, SAUSC only operates on state data."
        )

    if not np.all(headers == EXPECTED_STATE_DATA_HEADERS):
        raise InvalidFileFormatException(
            f"headers {headers} is not the expected {EXPECTED_STATE_DATA_HEADERS}."
        )

    start_residues = loaded_data[1:, 1].astype(int)
    end_residues = loaded_data[1:, 2].astype(int)
    sequences = loaded_data[1:, 3].astype(str)
    max_deuterium_uptakes = loaded_data[1:, 6].astype(float)
    states = loaded_data[1:, 8].astype(str)
    exposures = loaded_data[1:, 9].astype(str)
    uptake_means = loaded_data[1:, 12].astype(float)
    uptake_stdevs = loaded_data[1:, 13].astype(float)

    # fromkeys acts like a set but preserves order
    unique_states: tuple[str] = tuple(dict.fromkeys(states).keys())
    unique_exposures: tuple[str] = tuple(dict.fromkeys(exposures).keys())

    experimental_parameters = ExperimentalParameters(
        states=unique_states, exposures=unique_exposures
    )

    # This complexity exists only to give the user a helpful targeted error message if there is a missing column
    indexes = np.arange(len(loaded_data) - 1)
    n_chunks = len(indexes) // experimental_parameters.sequence_parse_length

    # States in chunk should follow the expected order, i.e. AAAABBBB
    expected_state_format = np.repeat(unique_states, len(unique_exposures))

    # Exposures in chunk should follow the expected order, i.e. ABCDABCD
    expected_exposure_format = unique_exposures + unique_exposures

    state_data: list[StateData] = []
    for chunk in range(n_chunks):
        chunk_indexes = indexes[
            experimental_parameters.sequence_parse_length
            * chunk : experimental_parameters.sequence_parse_length
            * (chunk + 1)
        ]

        states_in_order: npt.NDArray[np.bool_] = (
            states[chunk_indexes] == expected_state_format
        )

        if not states_in_order.all():
            raise InvalidFileFormatException(
                f"""
                Error parsing state data from {filepath} on lines {chunk_indexes[~states_in_order]+1}:
                Got {states[chunk_indexes]}, expected {expected_state_format[~states_in_order]+1}.
                """
            )

        exposures_in_order: npt.NDArray[np.bool_] = (
            exposures[chunk_indexes] == expected_exposure_format
        )

        if not exposures_in_order.all():
            raise InvalidFileFormatException(
                f"""
                Error parsing exposure data from {filepath} on lines {chunk_indexes[~exposures_in_order]+1}
                Got {exposures[chunk_indexes]}, expected {expected_exposure_format[~exposures_in_order]+1}.
                """
            )

        # There should be only one sequence represented within a single chunk
        if len(set(sequences[chunk_indexes])) != 1:
            raise InvalidFileFormatException(
                f"Error parsing state data from {filepath} between lines {chunk_indexes[0]} - {chunk_indexes[-1]}, expected {expected_state_format[~states_in_order]}."
            )

        uptakes = [
            Uptake(mean=uptake_means[i], stdev=uptake_stdevs[i]) for i in chunk_indexes
        ]

        state_specific_indexes = np.reshape(chunk_indexes, (len(unique_states), -1))

        for specific_indexes, state in zip(state_specific_indexes, unique_states):
            state_data.append(
                StateData(
                    sequence=sequences[chunk_indexes[0]],
                    state=state,
                    start_residue=start_residues[chunk_indexes[0]],
                    end_residue=end_residues[chunk_indexes[0]],
                    max_deuterium_uptake=max_deuterium_uptakes[chunk_indexes[0]],
                    exposures={
                        exposures[i]: uptakes[
                            i % experimental_parameters.sequence_parse_length
                        ]
                        for i in specific_indexes
                    },
                )
            )

    return (state_data, experimental_parameters)


@dataclasses.dataclass(frozen=True)
class Comparison(BaseFragment):
    uptake_difference: float
    p_value: float
    is_significant: bool

    @property
    def neg_log_p(self) -> float:
        return -np.log10(self.p_value)

    @property
    def relative_uptake_difference(self) -> float:
        return self.uptake_difference / self.max_deuterium_uptake

    @classmethod
    def from_reference(
        cls,
        reference: BaseFragment,
        uptake_difference: float,
        p_value: float,
        is_significant: bool,
    ) -> "Comparison":
        return Comparison(
            sequence=reference.sequence,
            start_residue=reference.start_residue,
            end_residue=reference.end_residue,
            max_deuterium_uptake=reference.max_deuterium_uptake,
            uptake_difference=uptake_difference,
            p_value=p_value,
            is_significant=is_significant,
        )


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


def compare_uptakes(
    default: Uptake,
    other: Uptake,
    params: UserParameters,
    SEM_for_hybrid_test: Optional[float] = None,
) -> tuple[float, float, bool]:

    uptake_difference = other.mean - default.mean

    p_value = stats.ttest_ind_from_stats(
        mean1=default.mean,
        std1=default.stdev,
        nobs1=params.n_repeats,
        mean2=other.mean,
        std2=other.stdev,
        nobs2=params.n_repeats,
    ).pvalue

    significant = p_value < (1 - params.confidence_interval)

    if params.statistical_test == StatisticalTestType.hybrid_test:
        assert (
            SEM_for_hybrid_test is not None
        ), "Must provide the standard error of the mean to perform a hybrid test."
        degrees_of_freedom = (params.n_repeats * 2) - 2
        t_critical = stats.t.ppf(params.confidence_interval, degrees_of_freedom)
        globally_significant = abs(uptake_difference) > (
            SEM_for_hybrid_test * t_critical
        )
        significant &= globally_significant

    return uptake_difference, p_value, significant


def compare_states(
    default: StateData,
    other: StateData,
    params: UserParameters,
    SEM_for_hybrid_test: float,
) -> dict[str, Comparison]:

    if not default.is_same_fragment(other):
        raise ValueError(
            f"Cannot compare state data from different sequences ({default} != {other})."
        )

    results_by_exposure: dict[str, Comparison] = {}

    for (default_exposure, default_uptake), (other_exposure, other_uptake) in zip(
        default.exposures.items(), other.exposures.items()
    ):
        if default_exposure != other_exposure:
            raise ValueError(
                f"States for fragment {default.sequence} do not have the same order exposures."
            )

        uptake_difference, p_value, is_significant = compare_uptakes(
            default=default_uptake,
            other=other_uptake,
            params=params,
            SEM_for_hybrid_test=SEM_for_hybrid_test,
        )

        comparison = Comparison.from_reference(
            reference=default,
            uptake_difference=uptake_difference,
            p_value=p_value,
            is_significant=is_significant,
        )

        results_by_exposure[default_exposure] = comparison

    return results_by_exposure


def colour_psb_structure_by_uptake_difference() -> None:
    pass


if __name__ == "__pymol":
    from pymol import cmd

    @cmd.extend
    def SAUSC(
        filepath: Optional[str] = None,
        num_repeats: PYMOL_INT = "3",
        confidence_interval: PYMOL_FLOAT = "0.95",
        hybrid_test: PYMOL_BOOL = "True",
        protection_colour: str = "Blues",
        deprotection_colour: str = "Reds",
        insignificant_colour: PYMOL_TUPLE_FLOAT = "(1.0, 1.0, 1.0)",
        no_coverage_colour: PYMOL_TUPLE_FLOAT = "(0.1, 0.1, 0.1)",
        global_normalisation: PYMOL_BOOL = "True",
        debug_messages: PYMOL_BOOL = "False",
    ):
        """
        DESCRIPTION

            Take a HDX results excel comparing a default and variant state, colour significant differences in a gradient manner.
            Colours are normalised either by per exposure setting, or globally.
            You need to enter the correct number of repeats that the data represents as this information isn't saved.
            The specified confidence interval will determine what sequences are considered significant and are coloured in.


        USAGE

            SAUSC [ results_file ], [ num_repeats ],

        EXAMPLE

            SAUSC your_results_file, normalise_global=True,

        """

        user_params = UserParameters(
            n_repeats=convert_from_pymol(num_repeats, int),
            confidence_interval=convert_percentage_if_necessary(
                convert_from_pymol(confidence_interval, float)
            ),
            global_normalisation=convert_from_pymol(global_normalisation, bool),
            statistical_test=(
                StatisticalTestType.hybrid_test
                if convert_from_pymol(hybrid_test, bool)
                else StatisticalTestType.t_test
            ),
        )

        uptake_colourmap = make_diverging_colormap(
            protection_colormap=protection_colour,
            deprotection_colormap=deprotection_colour,
        )

        if filepath is None:
            filepath = file_browser()

        check_valid_file(filepath)
        assert (
            filepath is not None
        ), "File checker didn't work."  # Essentially just mypy juggling

        loaded_results, experimental_params = load_state_data(filepath)

        global_sem = pooled_standard_error_mean(
            stdevs=np.array(
                [
                    uptake.stdev
                    for data in loaded_results
                    for uptake in data.exposures.values()
                ]
            ),
            n_repeats=user_params.n_repeats,
        )


if __name__ == "__main__":
    data, params = load_state_data(r"C:\Users\joshu\Downloads\Cdstate.csv")
    pass
