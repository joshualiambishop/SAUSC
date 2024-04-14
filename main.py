# -*- coding: utf-8 -*-
"""
Correspondence: joshualiambishop@gmail.com
"""

from typing import Optional, TypeAlias, Callable, TypeVar, Any, Type, cast
from tkinter import filedialog
import tkinter as tk
import enum
import numpy as np
import numpy.typing as npt
import dataclasses

# Non standard libraries will need to be installed within
# PyMol itself, so the script can run in that environment.
try:
    from scipy import stats
except ImportError as exc:
    raise ImportError("Please install scipy via 'pip install scipy'.") from exc
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib
    import matplotlib.collections as mpl_collections
    from matplotlib import patches as mpl_patches

except ImportError as exc:
    raise ImportError(
        "Please install matplotlib via 'pip install matplotlib'."
    ) from exc

import pathlib


### Colouring and mapping ###
Colour = tuple[float, float, float]

# Nice bright purple to represent any errors visually
ERROR_MAGENTA: Colour = (204.0, 0.0, 153.0)

# Making a new diverging colormap only really works if the pairs are both sequential types
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


@dataclasses.dataclass(frozen=True)
class ColourScheme:
    """
    Represents all the colours for structures and visualisations
    """

    uptake_colourmap: ListedColormap
    insignificant: Colour
    no_coverage: Colour
    error: Colour

    def uptake_colourmap_with_symmetrical_normalisation(
        self, value: float
    ) -> ScalarMappable:
        assert value > 0, "Value for normalisation must be positive."
        normalisation = Normalize(vmin=-value, vmax=value)
        return ScalarMappable(norm=normalisation, cmap=self.uptake_colourmap)


def cast_to_colour(values: tuple[float, ...]) -> Colour:
    assert len(values) == 3, "Colour must be formed of 3 values (RGB)."
    return cast(Colour, values)


# Only makes sense for HDX results to have white in the middle.
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

    if deprotection_colormap == protection_colormap:
        raise ValueError(
            "Can't have the same colormap for protection and deprotection."
        )

    sampling = np.linspace(0, 1, 128)

    protection_cmap = matplotlib.colormaps[protection_colormap].resampled(128)
    deprotection_cmap = matplotlib.colormaps[deprotection_colormap].resampled(128)

    new_colours = np.vstack(
        (protection_cmap(sampling)[::-1], deprotection_cmap(sampling))
    )

    return ListedColormap(
        colors=new_colours,
        name=f"Combined {protection_colormap}{deprotection_colormap}",
        N=256,
    )


### Data loading and parsing ###
@dataclasses.dataclass()
class LastVisitedFolder:
    last_location: pathlib.Path

    def update(self, new_location: pathlib.Path) -> None:
        self.last_location = new_location


results_folder_tracker = LastVisitedFolder(pathlib.Path.home())


class InvalidFileFormatException(Exception):
    """Specific exception for if something is wrong with the user's results file."""


def check_valid_hdx_file(path: Optional[str]) -> None:
    if path is None:
        raise ValueError("No path supplied.")
    filepath = pathlib.Path(path)
    if not filepath.exists():
        raise IOError(f"File {path} does not exist")
    if not filepath.suffix == ".csv":
        raise IOError(f"File {path} must be a csv file.")


def file_browser() -> Optional[str]:
    """
    Opens a small window to allow a user to select a results file.
    """
    try:
        root = tk.Tk()
        results_file = filedialog.askopenfilenames(
            parent=root,
            initialdir=results_folder_tracker.last_location,
            initialfile="tmp",
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
        )[0]

        check_valid_hdx_file(results_file)
        results_folder_tracker.update(pathlib.Path(results_file).parent)
        return results_file

    finally:
        root.destroy()


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


class StatisticalTestType(enum.Enum):
    """
    Type of statistical test used to perform comparisons
    """

    T_TEST = enum.auto()
    HYBRID = enum.auto()


# For simplicity, a cumulative operation on all exposures are included as if it were a timepoint.
# I've made this a constant simply for sanity.
CUMULATIVE_EXPOSURE_KEY = "Cumulative"


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


def is_floatable(string: str) -> bool:
    try:
        float(string)
        return True
    except ValueError:
        return False


# Pymol unfortunately passes all arguments as strings
PymolBool: TypeAlias = str
PymolTupleFloat: TypeAlias = str
PymolInt: TypeAlias = str
PymolFloat: TypeAlias = str


def _str_to_bool(string: PymolBool) -> bool:
    if string.lower() == "true":
        return True
    if string.lower() == "false":
        return False
    raise TypeError(f"Input {string} must be True or False")


def _str_to_tuple_float(string: PymolTupleFloat) -> tuple[float, ...]:
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
    bool: _str_to_bool,
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


class DataForVisualisation(enum.Enum):
    UPTAKE_DIFFERENCE = enum.auto()
    RELATIVE_UPTAKE_DIFFERENCE = enum.auto()


class ColourNormalisationMode(enum.Enum):
    INDIVIDUAL = enum.auto()  # Colourmaps are normalised to each individual exposure
    ACROSS_EXPOSURES = (
        enum.auto()
    )  # Colourmaps are normalised across the maximum for each exposure (not including cumulative)
    GLOBAL = (
        enum.auto()
    )  # A single universal colourmap, not recommended with cumulative as that'll naturally skew the colours, but the option is there


pretty_string_for: dict[Any, str] = {
    DataForVisualisation.UPTAKE_DIFFERENCE: "Uptake difference (Da)",
    DataForVisualisation.RELATIVE_UPTAKE_DIFFERENCE: "Relative uptake difference (%)",
    ColourNormalisationMode.INDIVIDUAL: "Normalised by timepoint",
    ColourNormalisationMode.ACROSS_EXPOSURES: "Normalised across timepoints",
    ColourNormalisationMode.GLOBAL: "Normalised globally",
}


@dataclasses.dataclass(frozen=True)
class ExperimentalParameters:
    """
    Simple container for the parameters of the experiment, how many states (assuming one is the default), and the number of deuterium exposure durations (in minutes)
    """

    states: tuple[str, str]
    exposures: tuple[str, ...]
    max_residue: int

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
            self.start_residue <= self.end_residue
        ), "End residue must be after the start residue."
        expected_sequence_length = self.end_residue - self.start_residue + 1
        assert (
            len(self.sequence) == expected_sequence_length
        ), "Sequence must have {expected_sequence_length} residues."

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
        combined_mean = np.sum([uptake.mean for uptake in uptakes if uptake.mean > 0])
        combined_stdev = pooled_stdev(
            np.array([uptake.stdev for uptake in uptakes if uptake.stdev > 0])
        )
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
    # casting is relatively safe here as there are many moments where only two states are enforced
    unique_states: tuple[str, str] = cast(
        tuple[str, str], tuple(dict.fromkeys(states).keys())
    )
    unique_exposures: tuple[str] = cast(
        tuple[str], tuple(dict.fromkeys(exposures).keys())
    )

    experimental_parameters = ExperimentalParameters(
        states=unique_states, exposures=unique_exposures, max_residue=end_residues.max()
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
    """
    Represents the difference between a state and reference of a given fragment, at a given exposure.
    """

    uptake_difference: float
    p_value: float
    is_significant: bool
    exposure: str

    @property
    def neg_log_p(self) -> float:
        return -np.log10(self.p_value)

    @property
    def relative_uptake_difference(self) -> float:
        return self.uptake_difference / self.max_deuterium_uptake

    def request(self, data_type: DataForVisualisation) -> float:
        if data_type == DataForVisualisation.UPTAKE_DIFFERENCE:
            return self.uptake_difference
        elif data_type == DataForVisualisation.RELATIVE_UPTAKE_DIFFERENCE:
            return self.relative_uptake_difference
        else:
            raise ValueError(
                f"Cannot provide information for requested type {data_type}"
            )

    @classmethod
    def from_reference(
        cls,
        reference: BaseFragment,
        uptake_difference: float,
        p_value: float,
        is_significant: bool,
        exposure: str,
    ) -> "Comparison":
        return Comparison(
            sequence=reference.sequence,
            start_residue=reference.start_residue,
            end_residue=reference.end_residue,
            max_deuterium_uptake=reference.max_deuterium_uptake,
            uptake_difference=uptake_difference,
            p_value=p_value,
            is_significant=is_significant,
            exposure=exposure,
        )


class ResidueType(enum.Enum):
    AVERAGED = enum.auto()
    ALL_INSIGNIFICANT = enum.auto()
    NOT_COVERED = enum.auto()


@dataclasses.dataclass(frozen=True)
class SingleResidueComparison:
    amino_acid: str
    residue: int
    uptake_difference: float
    uptake_stdev: float
    exposure: str
    residue_type: ResidueType

    def __post_init__(self) -> None:
        assert len(self.amino_acid) == 1, "Amino acid must be a single character."

    @classmethod
    def as_empty(cls, residue: int, exposure: str) -> "SingleResidueComparison":
        return cls(
            amino_acid="-",
            residue=residue,
            uptake_difference=np.nan,
            uptake_stdev=np.nan,
            exposure=exposure,
            residue_type=ResidueType.NOT_COVERED,
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

    if params.statistical_test == StatisticalTestType.HYBRID:
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

    # This is including the cumulative data as an exposure "timepoint"
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
            exposure=default_exposure,
        )

        results_by_exposure[default_exposure] = comparison

    return results_by_exposure


def split_comparisons_by_residue(
    comparisons: list[Comparison], params: ExperimentalParameters
) -> list[SingleResidueComparison]:
    assert (
        len(set([c.exposure for c in comparisons])) == 1
    ), "Comparisons must be from the same exposure time."
    exposure = comparisons[0].exposure

    all_residues = np.arange(params.max_residue + 1)

    single_residue_comparisons: list[SingleResidueComparison] = []
    for residue in all_residues:

        residue_present = [
            comparison
            for comparison in comparisons
            if comparison.residue_present(residue)
        ]
        # Residue is not covered by any sequences, therefore missed in the digestion process.
        if len(residue_present) == 0:
            single_residue_comparisons.append(
                SingleResidueComparison.as_empty(residue=residue, exposure=exposure)
            )
            continue

        amino_acid: str = residue_present[0].sequence[
            residue - residue_present[0].start_residue
        ]

        is_significant = [
            comparison for comparison in residue_present if comparison.is_significant
        ]

        single_residue_comparisons.append(
            SingleResidueComparison(
                amino_acid=amino_acid,
                residue=residue,
                uptake_difference=(
                    np.nan
                    if len(is_significant) == 0
                    else cast(
                        float, np.mean([i.uptake_difference for i in is_significant])
                    )  # TODO: I'm not particularly sure if this makes sense mathematically...
                ),
                uptake_stdev=(
                    np.nan
                    if len(is_significant) == 0
                    else cast(
                        float, np.std([i.uptake_difference for i in is_significant])
                    )  # TODO: I'm not particularly sure if this makes sense mathematically...
                ),
                exposure=exposure,
                residue_type=(
                    ResidueType.ALL_INSIGNIFICANT
                    if len(is_significant) == 0
                    else ResidueType.AVERAGED
                ),
            )
        )

    return single_residue_comparisons


@dataclasses.dataclass(frozen=True)
class FullSAUSCAnalysis:
    user_params: UserParameters
    experimental_params: ExperimentalParameters
    sequence_comparisons: dict[str, list[Comparison]]
    residue_comparisons: dict[str, list[SingleResidueComparison]]
    colouring: ColourScheme
    full_sequence: str


### Plotting functionality ###


# If for any reason you want to customise the fine details of the plots, please do so in the code here:
@dataclasses.dataclass(frozen=True)
class WoodsPlotOptions:
    y_data = DataForVisualisation.UPTAKE_DIFFERENCE
    colour_data = DataForVisualisation.RELATIVE_UPTAKE_DIFFERENCE
    colour_normalisation = ColourNormalisationMode.ACROSS_EXPOSURES
    box_thickness: float = 0.1
    dpi: int = 150
    scale: float = 12


WOODS_PLOT_PARAMS = WoodsPlotOptions()


def get_strongest_magnitude_of_type(
    data: list[Comparison], data_type: DataForVisualisation
) -> float:
    values = [abs(comparison.request(data_type)) for comparison in data]
    return max(values)


def find_normalisation_value(
    data: dict[str, list[Comparison]],
    data_type: DataForVisualisation,
    normalisation_mode: ColourNormalisationMode,
) -> float:
    if normalisation_mode == ColourNormalisationMode.GLOBAL:
        rel_data = [
            comparison for collection in data.values() for comparison in collection
        ]
    elif normalisation_mode == ColourNormalisationMode.ACROSS_EXPOSURES:
        rel_data = [
            comparison
            for (exposure, collection) in data.items()
            for comparison in collection
            if exposure != CUMULATIVE_EXPOSURE_KEY
        ]
    else:
        raise ValueError(f"Cannot interpret normalisation mode {normalisation_mode}")

    return get_strongest_magnitude_of_type(rel_data, data_type)


def draw_woods_plot(analysis: FullSAUSCAnalysis) -> None:

    fig, axes = plt.subplots(
        nrows=len(analysis.experimental_params.exposures)
        + 1,  # Extra for the cumulative
        ncols=1,
        sharex=True,
        sharey=False,
        dpi=WOODS_PLOT_PARAMS.dpi,
        layout="constrained",
        figsize=(WOODS_PLOT_PARAMS.scale, WOODS_PLOT_PARAMS.scale),
    )
    global_cmap: Optional[ScalarMappable] = None
    if WOODS_PLOT_PARAMS.colour_normalisation != ColourNormalisationMode.INDIVIDUAL:
        global_normalisation_value = find_normalisation_value(
            analysis.sequence_comparisons,
            data_type=WOODS_PLOT_PARAMS.colour_data,
            normalisation_mode=WOODS_PLOT_PARAMS.colour_normalisation,
        )
        global_cmap = (
            analysis.colouring.uptake_colourmap_with_symmetrical_normalisation(
                global_normalisation_value
            )
        )

    for ax, exposure in zip(
        axes, (*analysis.experimental_params.exposures, CUMULATIVE_EXPOSURE_KEY)
    ):

        sequence_comparisons = analysis.sequence_comparisons[exposure]

        ax.set_title(
            (
                CUMULATIVE_EXPOSURE_KEY
                if exposure == CUMULATIVE_EXPOSURE_KEY
                else f"Exposure = {exposure} minutes"
            ),
            loc="left",
        )

        ax.set_xticks(
            np.arange(analysis.experimental_params.max_residue + 1),
            analysis.full_sequence,
        )

        colour_map = (
            analysis.colouring.uptake_colourmap_with_symmetrical_normalisation(
                get_strongest_magnitude_of_type(
                    sequence_comparisons, WOODS_PLOT_PARAMS.colour_data
                )
            )
            if (global_cmap is None) or exposure == CUMULATIVE_EXPOSURE_KEY
            else global_cmap
        )

        fig.colorbar(
            colour_map, ax=ax, label=pretty_string_for[WOODS_PLOT_PARAMS.colour_data]
        )
        ax.set(
            ylabel=pretty_string_for[WOODS_PLOT_PARAMS.y_data],
            xlabel="Residue",
            xlim=(0, analysis.experimental_params.max_residue + 1),
        )

        patches: list[mpl_patches.Rectangle] = []
        for comparison in sequence_comparisons:

            # Middle of the box should be aligned to the correct value
            y_position = comparison.request(WOODS_PLOT_PARAMS.y_data) - (
                WOODS_PLOT_PARAMS.box_thickness / 2
            )

            colour_data = comparison.request(WOODS_PLOT_PARAMS.colour_data)

            colour = (
                colour_map.to_rgba(np.array([colour_data]))
                if comparison.is_significant
                else analysis.colouring.insignificant
            )

            rectangle_patch = mpl_patches.Rectangle(
                xy=(comparison.start_residue, y_position),
                width=len(comparison.sequence),
                height=WOODS_PLOT_PARAMS.box_thickness,
                facecolor=colour,
                edgecolor="black",
                linewidth=0.1,
            )
            patches.append(rectangle_patch)

        ax.add_collection(mpl_collections.PatchCollection(patches, match_original=True))
        ax.autoscale_view(scalex=False, scaley=True)
        ax.axhline(0, linewidth=0.3, color="black", alpha=1)
    plt.show()


def colour_psb_structure_by_uptake_difference() -> None:
    pass


if __name__ == "__main__":
    # from pymol import cmd

    # @cmd.extend
    def SAUSC(
        filepath: Optional[str] = None,
        num_repeats: PymolInt = "3",
        confidence_interval: PymolFloat = "0.95",
        hybrid_test: PymolBool = "True",
        protection_colour: str = "Blues",
        deprotection_colour: str = "Reds",
        insignificant_colour: PymolTupleFloat = "(1.0, 1.0, 1.0)",
        no_coverage_colour: PymolTupleFloat = "(0.1, 0.1, 0.1)",
        global_normalisation: PymolBool = "True",
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
                StatisticalTestType.HYBRID
                if convert_from_pymol(hybrid_test, bool)
                else StatisticalTestType.T_TEST
            ),
        )

        uptake_colourmap = make_diverging_colormap(
            protection_colormap=protection_colour,
            deprotection_colormap=deprotection_colour,
        )

        colour_scheme = ColourScheme(
            uptake_colourmap=uptake_colourmap,
            insignificant=cast_to_colour(
                convert_from_pymol(insignificant_colour, tuple[float])
            ),
            no_coverage=cast_to_colour(
                convert_from_pymol(no_coverage_colour, tuple[float])
            ),
            error=ERROR_MAGENTA,
        )

        if filepath is None:
            filepath = file_browser()
        else:
            check_valid_hdx_file(filepath)

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

        # Hopefully a user will call the default state "default"...
        if "default" in experimental_params.states[0].lower():
            default_is_first = True
        elif "default" in experimental_params.states[1].lower():
            default_is_first = False
        else:
            raise ValueError("Cannot determine which state is the default!")

        # The results should be structured interlaced
        default_states, other_states = np.reshape(loaded_results, (-1, 2)).T

        if not default_is_first:
            default_states, other_states = other_states, default_states

        _comparisons: list[dict[str, Comparison]] = [
            compare_states(default, other, user_params, global_sem)
            for default, other in zip(default_states, other_states)
        ]

        comparisons_by_exposure: dict[str, list[Comparison]] = {
            exposure: [comparison[exposure] for comparison in _comparisons]
            for exposure in _comparisons[0].keys()
        }

        single_residue_comparisons: dict[str, list[SingleResidueComparison]] = {
            exposure: split_comparisons_by_residue(comparisons, experimental_params)
            for exposure, comparisons in comparisons_by_exposure.items()
        }
        protein_sequence = "".join(
            [
                residue.amino_acid
                for residue in list(single_residue_comparisons.values())[0]
            ]
        )

        full_analysis = FullSAUSCAnalysis(
            user_params=user_params,
            experimental_params=experimental_params,
            sequence_comparisons=comparisons_by_exposure,
            residue_comparisons=single_residue_comparisons,
            colouring=colour_scheme,
            full_sequence=protein_sequence,
        )
        return full_analysis


if __name__ == "__main__":
    # data, params = load_state_data()
    full_analysis = SAUSC(r"C:\Users\joshu\Downloads\Cdstate.csv")
    draw_woods_plot(full_analysis)
