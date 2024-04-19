"""
Correspondence: joshualiambishop@gmail.com
"""

import dataclasses
import pathlib
from tkinter import filedialog
from typing import Any, Callable, Sequence, Type, TypeAlias, TypeVar, Optional, cast
from datetime import datetime
import numpy.typing as npt
import numpy as np

import enum

# Non standard libraries will need to be installed within
# PyMol itself, so the script can run in that environment.
try:
    import matplotlib
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpl_patches
    import matplotlib.collections as mpl_collections
except ImportError as exc:
    raise ImportError(
        "Please install matplotlib via 'pip install matplotlib'."
    ) from exc
try:
    from scipy import stats
except ImportError as exc:
    raise ImportError("Please install scipy via 'pip install scipy'.") from exc

if __name__ == "pymol":
    from pymol import cmd

# Unfortunately to run a script in pymol and navigate it's virtual environment
# everything must be defined within a single file. :(

# For simplicity, a cumulative operation on all exposures are included as if it were a timepoint.
# I've made this a constant simply for sanity.
CUMULATIVE_EXPOSURE_KEY = "Cumulative"

SameAsInput = TypeVar("SameAsInput")
SameAsInputEnum = TypeVar("SameAsInputEnum", bound=enum.Enum)


class DataForVisualisation(enum.Enum):
    UPTAKE_DIFFERENCE = enum.auto()
    RELATIVE_UPTAKE_DIFFERENCE = enum.auto()
    P_VALUE = enum.auto()
    NEG_LOG_P = enum.auto()


class NormalisationMode(enum.Enum):
    INDIVIDUAL = enum.auto()  # Colourmaps are normalised to each individual exposure
    ACROSS_EXPOSURES = (
        enum.auto()
    )  # Colourmaps are normalised across the maximum for each exposure (not including cumulative)
    GLOBAL = (
        enum.auto()
    )  # A single universal colourmap, not recommended with cumulative as that'll naturally skew the colours, but the option is there


FIGURE_SAVING_FORMATS: list[str] = [".png", ".svg"]


### General utilities ###
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


def find_matching_enum_from_meta(
    meta: Type[SameAsInputEnum], query: str
) -> SameAsInputEnum:
    for possible_match in meta:
        name: str = possible_match.name
        if name.lower() == query.lower():
            return possible_match
    raise ValueError(f"{query} not found in {enum}.")


def warn_SAUSC_not_run() -> None:
    raise Exception(
        "Cannot perform requested action - SAUSC has not been run on any data"
    )


# Figures and plotting
@dataclasses.dataclass(frozen=True)
class GenericFigureOptions:
    dpi: float
    scale: float


@dataclasses.dataclass(frozen=True)
class BaseVisualisationOptions(GenericFigureOptions):
    y_data: DataForVisualisation
    colour_data: DataForVisualisation


@dataclasses.dataclass(frozen=True)
class WoodsPlotOptions(BaseVisualisationOptions):
    box_thickness: float  # As a percentage of the y axes

    def __post_init__(self) -> None:
        enforce_between_0_and_1(self.box_thickness)


@dataclasses.dataclass(frozen=True)
class VolcanoPlotOptions(BaseVisualisationOptions):
    x_data: DataForVisualisation
    circle_size: float
    circle_transparency: float
    annotation_fontsize: float


#
#
# If for any reason you want to customise the fine details of the plots
# please do so in the code here:
#
#

WOODS_PLOT_PARAMS = WoodsPlotOptions(
    dpi=100.0,
    scale=6.0,
    y_data=DataForVisualisation.UPTAKE_DIFFERENCE,
    colour_data=DataForVisualisation.RELATIVE_UPTAKE_DIFFERENCE,
    box_thickness=0.07,
)

VOLCANO_PLOT_PARAMS = VolcanoPlotOptions(
    dpi=150.0,
    scale=3.0,
    x_data=DataForVisualisation.UPTAKE_DIFFERENCE,
    y_data=DataForVisualisation.NEG_LOG_P,
    colour_data=DataForVisualisation.RELATIVE_UPTAKE_DIFFERENCE,
    circle_size=1.0,
    circle_transparency=0.7,
    annotation_fontsize=5.0,
)


class StatisticalTestType(enum.Enum):
    """
    Type of statistical test used to perform comparisons
    """

    T_TEST = enum.auto()
    HYBRID = enum.auto()


class ResidueType(enum.Enum):
    AVERAGED = enum.auto()
    ALL_INSIGNIFICANT = enum.auto()
    NOT_COVERED = enum.auto()


### File browsing ###


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
    # In PyMol this gets directly injected to QtWidgets.QFileDialog.getOpenFileName
    results_file = filedialog.askopenfilename(
        title="Select state data",
        initialdir=str(pathlib.Path.home()),
        filetypes=[("CSV", "*.csv"), ("All files", "*")],
    )

    check_valid_hdx_file(results_file)
    return results_file


# Pymol passes all arguments to functions as strings
# so here is a simple conversion mechanism
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


def convert_from_pymol(argument: Any, requested_type: Type[SameAsInput]) -> SameAsInput:
    assert (
        requested_type in pymol_convertors
    ), f"Haven't implemented a conversion for type {requested_type}."
    convertor = pymol_convertors[requested_type]
    return convertor(argument)


### Mathematical formulas ###
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


### Data storage and encapsulation ###
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

    def __post_init__(self) -> None:
        assert (
            self.start_residue <= self.end_residue
        ), "End residue must be after the start residue."
        expected_sequence_length = self.end_residue - self.start_residue + 1
        assert (
            len(self.sequence) == expected_sequence_length
        ), "Sequence must have {expected_sequence_length} residues."

    def residue_present(self, residue: int) -> bool:
        return self.start_residue <= residue <= self.end_residue

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
        return (self.uptake_difference / self.max_deuterium_uptake) * 100

    def request(self, data_type: DataForVisualisation) -> float:
        match data_type:
            case DataForVisualisation.UPTAKE_DIFFERENCE:
                return self.uptake_difference
            case DataForVisualisation.RELATIVE_UPTAKE_DIFFERENCE:
                return self.relative_uptake_difference
            case DataForVisualisation.P_VALUE:
                return self.p_value
            case DataForVisualisation.NEG_LOG_P:
                return self.neg_log_p
            case _:
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
    normalisation_type: NormalisationMode
    statistical_test: StatisticalTestType

    def __post_init__(self):
        enforce_between_0_and_1(self.confidence_interval)


### Data loading and parsing
class InvalidFileFormatException(Exception):
    """Specific exception for if something is wrong with the user's results file."""


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


def load_state_data(
    filepath: str,
) -> tuple[list[StateData], ExperimentalParameters]:
    """
    This is specifically designed to load a .csv containing state
    """
    expected_columns = np.arange(len(EXPECTED_STATE_DATA_HEADERS)).tolist()
    loaded_data = np.loadtxt(
        filepath, delimiter=",", dtype=object, usecols=expected_columns
    )
    headers = loaded_data[0].tolist()

    # Z refers to [something], only present in cluster data, a common mistake.
    if "z" in loaded_data:
        raise InvalidFileFormatException(
            f"{filepath} appears to be cluster data, SAUSC only operates on state "
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
                Got {exposures[chunk_indexes]}, expected {expected_exposure_format[~exposures_in_order]}.
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


pretty_string_for: dict[enum.Enum, str] = {
    DataForVisualisation.UPTAKE_DIFFERENCE: "Uptake difference (Da)",
    DataForVisualisation.RELATIVE_UPTAKE_DIFFERENCE: """Relative uptake difference (%)""",
    DataForVisualisation.P_VALUE: "P value",
    DataForVisualisation.NEG_LOG_P: "-log(p)",
    NormalisationMode.INDIVIDUAL: "Normalised by timepoint",
    NormalisationMode.ACROSS_EXPOSURES: "Normalised across timepoints",
    NormalisationMode.GLOBAL: "Normalised globally",
    StatisticalTestType.T_TEST: "Welch's T-test",
    StatisticalTestType.HYBRID: "Hybrid test",
}


def multi_line(string: str) -> str:
    n_spaces = len(string.split(" ")) - 1
    # This is because the last one would be the unit
    return string.replace(" ", "\n", n_spaces - 1)


Colour: TypeAlias = tuple[float, float, float]

# Nice bright purple to represent any errors visually
ERROR_MAGENTA: Colour = (204.0, 0.0, 153.0)

# Making a new diverging colormap only really works if the pairs are both sequential types
SEQUENTIAL_COLORMAPS: list[str] = [
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
    error: Colour = ERROR_MAGENTA

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

    if protection_colormap not in SEQUENTIAL_COLORMAPS:
        raise ValueError(
            "Protection colormap {protection_colormap} must be one of {ALLOWED_COLORMAPS}"
        )

    if deprotection_colormap not in SEQUENTIAL_COLORMAPS:
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


def get_strongest_magnitude_of_type(
    comparisons: list[Comparison], data_type: DataForVisualisation
) -> float:
    values = [abs(comparison.request(data_type)) for comparison in comparisons]
    return max(values)


def find_normalisation_value(
    comparisons: dict[str, list[Comparison]],
    data_type: DataForVisualisation,
    normalisation_mode: NormalisationMode,
) -> float:
    if normalisation_mode == NormalisationMode.GLOBAL:
        rel_data = [
            comparison
            for collection in comparisons.values()
            for comparison in collection
        ]
    elif normalisation_mode == NormalisationMode.ACROSS_EXPOSURES:
        rel_data = [
            comparison
            for (exposure, collection) in comparisons.items()
            for comparison in collection
            if exposure != CUMULATIVE_EXPOSURE_KEY
        ]
    else:
        raise ValueError(f"Cannot interpret normalisation mode {normalisation_mode}")

    return get_strongest_magnitude_of_type(rel_data, data_type)


@dataclasses.dataclass(frozen=True)
class FullSAUSCAnalysis:
    user_params: UserParameters
    experimental_params: ExperimentalParameters
    sequence_comparisons: dict[str, list[Comparison]]
    residue_comparisons: dict[str, list[SingleResidueComparison]]
    colouring: ColourScheme
    global_standard_error_mean: float
    full_sequence: str
    filepath: pathlib.Path


def run_SAUSC_from_path(
    filepath: str,
    n_repeats: int,
    confidence_interval: float,
    statistical_test: str,
    protection_colourmap: str,
    deprotection_colourmap: str,
    insignificant_colour: Colour,
    no_coverage_colour: Colour,
    normalisation_mode: str,
) -> FullSAUSCAnalysis:

    user_params = UserParameters(
        n_repeats=n_repeats,
        confidence_interval=confidence_interval,
        normalisation_type=find_matching_enum_from_meta(
            NormalisationMode, query=normalisation_mode
        ),
        statistical_test=find_matching_enum_from_meta(
            StatisticalTestType, query=statistical_test
        ),
    )
    colour_scheme = ColourScheme(
        uptake_colourmap=make_diverging_colormap(
            protection_colormap=protection_colourmap,
            deprotection_colormap=deprotection_colourmap,
        ),
        insignificant=insignificant_colour,
        no_coverage=no_coverage_colour,
    )
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
    default_states, other_states = np.reshape(np.array(loaded_results), (-1, 2)).T

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
        [residue.amino_acid for residue in list(single_residue_comparisons.values())[0]]
    )

    full_analysis = FullSAUSCAnalysis(
        user_params=user_params,
        experimental_params=experimental_params,
        sequence_comparisons=comparisons_by_exposure,
        residue_comparisons=single_residue_comparisons,
        colouring=colour_scheme,
        full_sequence=protein_sequence,
        global_standard_error_mean=global_sem,
        filepath=pathlib.Path(filepath),
    )
    return full_analysis


@dataclasses.dataclass(frozen=True)
class BaseFigure:
    fig: plt.Figure
    axes: Sequence[plt.Axes]
    colourmaps: Sequence[ScalarMappable]

    def __post_init__(self) -> None:
        assert len(self.axes) == len(
            self.colourmaps
        ), "Must have a colourmap deifned for each axis."

    def save(
        self, description: str, extension: str, analysis_filepath: pathlib.Path
    ) -> None:
        assert extension.startswith(
            "."
        ), f"{extension} is an unsuitable file extension, must begin with ."
        figure_folder = analysis_filepath.parent / "SAUSC Figures"
        figure_folder.mkdir(parents=True, exist_ok=True)
        filepath_for_saving = (
            figure_folder
            / f"{analysis_filepath.stem} {description} {datetime.now().strftime('%Y_%m_%d %H_%M')}{extension}"
        )
        if filepath_for_saving.exists():
            raise Exception(
                "Warning: {filepath_for_saving} already exists, overwriting..."
            )

        self.fig.savefig(str(filepath_for_saving))
        print(
            f"Successfully saved {description.lower()} next to file at {filepath_for_saving}"
        )


def set_up_base_figure(
    analysis: FullSAUSCAnalysis,
    plotting_params: BaseVisualisationOptions,
    over_rows: bool,
    xspan: float = 1,
    yspan: float = 1,
) -> BaseFigure:
    """
    This shared utility sorts out the arrangment of the plot, with proper scaling and colourbars
    """

    n_plots_needed = (
        len(analysis.experimental_params.exposures) + 1
    )  # Extra for the cumulative
    nrows = n_plots_needed if over_rows else 1
    ncols = 1 if over_rows else n_plots_needed

    figsize = np.multiply([xspan * ncols, yspan * nrows], plotting_params.scale)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        dpi=plotting_params.dpi,
        layout="constrained",
        figsize=figsize,
    )
    axes_colourmaps: list[ScalarMappable] = []

    global_cmap: Optional[ScalarMappable] = None

    if analysis.user_params.normalisation_type != NormalisationMode.INDIVIDUAL:
        global_normalisation_value = find_normalisation_value(
            analysis.sequence_comparisons,
            data_type=plotting_params.colour_data,
            normalisation_mode=analysis.user_params.normalisation_type,
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
        )

        colour_map = (
            analysis.colouring.uptake_colourmap_with_symmetrical_normalisation(
                get_strongest_magnitude_of_type(
                    sequence_comparisons, plotting_params.colour_data
                )
            )
            if (global_cmap is None) or exposure == CUMULATIVE_EXPOSURE_KEY
            else global_cmap
        )

        axes_colourmaps.append(colour_map)
        ax.set(
            ylabel=multi_line(pretty_string_for[plotting_params.y_data]),
        )

    # Draw colourmaps when required:
    for index in range(len(axes)):
        if (
            over_rows  # Arranged vertically, every axis needs a colourbar
            or index == len(axes) - 1  # Last one always has a colourbar
            or (
                axes_colourmaps[index].norm.vmin != axes_colourmaps[index + 1].norm.vmin
                and axes_colourmaps[index].norm.vmax
                != axes_colourmaps[index + 1].norm.vmax
            )  # Can't be merged with the one to the right
        ):
            fig.colorbar(
                axes_colourmaps[index],
                ax=axes[index],
                label=multi_line(pretty_string_for[plotting_params.colour_data]),
            )

    return BaseFigure(fig=fig, axes=axes, colourmaps=axes_colourmaps)


def draw_woods_plot(analysis: FullSAUSCAnalysis, save: bool) -> None:
    base_figure = set_up_base_figure(
        analysis=analysis,
        plotting_params=WOODS_PLOT_PARAMS,
        over_rows=True,
        xspan=3,
        yspan=1,
    )

    for index, exposure in enumerate(
        (*analysis.experimental_params.exposures, CUMULATIVE_EXPOSURE_KEY)
    ):

        sequence_comparisons = analysis.sequence_comparisons[exposure]

        base_figure.axes[index].set(
            xlabel="Residue",
            xlim=(0, analysis.experimental_params.max_residue + 1),
        )

        patches: list[mpl_patches.Rectangle] = []

        total_axis_yspan = abs(np.diff(base_figure.axes[index].get_ylim())[0])

        for comparison in sequence_comparisons:

            effective_box_thickness = WOODS_PLOT_PARAMS.box_thickness * total_axis_yspan

            # Middle of the box should be aligned to the correct value
            y_position = comparison.request(WOODS_PLOT_PARAMS.y_data) - (
                effective_box_thickness / 2
            )

            colour_data = comparison.request(WOODS_PLOT_PARAMS.colour_data)

            colour = (
                base_figure.colourmaps[index].to_rgba(np.array([colour_data]))
                if comparison.is_significant
                else analysis.colouring.insignificant
            )

            rectangle_patch = mpl_patches.Rectangle(
                xy=(comparison.start_residue, y_position),
                width=len(comparison.sequence),
                height=effective_box_thickness,
                facecolor=colour,
                edgecolor="black",
                linewidth=0.1,
            )
            patches.append(rectangle_patch)

        base_figure.axes[index].add_collection(
            mpl_collections.PatchCollection(patches, match_original=True)
        )
        base_figure.axes[index].autoscale_view(scalex=False, scaley=True)
        base_figure.axes[index].axhline(0, linewidth=0.3, color="black", alpha=1)

    if save:
        for extension in FIGURE_SAVING_FORMATS:
            base_figure.save(
                description="Woods plot",
                extension=extension,
                analysis_filepath=analysis.filepath,
            )

    plt.show()


def draw_volcano_plot(analysis: FullSAUSCAnalysis, annotate: bool, save: bool) -> None:

    base_figure = set_up_base_figure(
        analysis=analysis,
        plotting_params=VOLCANO_PLOT_PARAMS,
        over_rows=False,
    )

    statistical_boundary_params = {"color": "black", "linestyle": "--", "zorder": -1}

    for index, exposure in enumerate(
        (*analysis.experimental_params.exposures, CUMULATIVE_EXPOSURE_KEY)
    ):

        sequence_comparisons = analysis.sequence_comparisons[exposure]

        base_figure.axes[index].axhline(
            -np.log10(1 - analysis.user_params.confidence_interval),
            **statistical_boundary_params,
        )
        if VOLCANO_PLOT_PARAMS.x_data == DataForVisualisation.UPTAKE_DIFFERENCE:
            base_figure.axes[index].axvline(
                analysis.global_standard_error_mean, **statistical_boundary_params
            )
            base_figure.axes[index].axvline(
                -analysis.global_standard_error_mean, **statistical_boundary_params
            )

        x_values = [s.request(VOLCANO_PLOT_PARAMS.x_data) for s in sequence_comparisons]
        y_values = [s.request(VOLCANO_PLOT_PARAMS.y_data) for s in sequence_comparisons]
        base_figure.axes[index].scatter(
            x=x_values,
            y=y_values,
            c=[
                (
                    base_figure.colourmaps[index].to_rgba(
                        s.request(VOLCANO_PLOT_PARAMS.colour_data)
                    )
                    if s.is_significant
                    else analysis.colouring.insignificant
                )
                for s in sequence_comparisons
            ],
            s=VOLCANO_PLOT_PARAMS.circle_size,
            alpha=VOLCANO_PLOT_PARAMS.circle_transparency,
        )
        # Volcano plot should be symmetrical about x
        largest_xscale = np.abs(base_figure.axes[index].get_xlim()).max()
        base_figure.axes[index].set(
            xlabel=pretty_string_for[VOLCANO_PLOT_PARAMS.x_data],
            xlim=(-largest_xscale, largest_xscale),
        )

        if annotate:
            annotations = [
                f"{s.start_residue} - {s.end_residue}" for s in sequence_comparisons
            ]
            for seq_index, annotation in enumerate(annotations):
                if sequence_comparisons[seq_index].is_significant:
                    base_figure.axes[index].annotate(
                        annotation,
                        (x_values[seq_index], y_values[seq_index]),
                        fontsize=VOLCANO_PLOT_PARAMS.annotation_fontsize,
                    )

    if save:
        for extension in FIGURE_SAVING_FORMATS:
            base_figure.save(
                description="Volcano plot",
                extension=extension,
                analysis_filepath=analysis.filepath,
            )

    plt.show()


GLOBAL_CUSTOM_COLOUR_INDEX = 0


if __name__ == "pymol":

    # Allow the user to run a function after preloading SAUSC, but warn in the case
    # the function hasn't actually been run
    @cmd.extend
    def woods_plot(save: PymolBool = "False") -> None:
        warn_SAUSC_not_run()

    @cmd.extend
    def volcano_plot(save: PymolBool = "False", annotate: PymolBool = "True") -> None:
        warn_SAUSC_not_run()

    def register_colours(colours: list[Colour]) -> dict[Colour, str]:
        global GLOBAL_CUSTOM_COLOUR_INDEX
        colour_to_name: dict[Colour, str] = {}

        for colour in colours:
            colour_to_name[colour] = f"SAUSC custom colour {GLOBAL_CUSTOM_COLOUR_INDEX}"
            GLOBAL_CUSTOM_COLOUR_INDEX += 1

        for colour, name in colour_to_name.items():
            cmd.set_color(name=name, rgb=list(colour))

        return colour_to_name

    def draw_uptake_on_scenes(analysis: FullSAUSCAnalysis) -> None:

        global_cmap: Optional[ScalarMappable] = None

        # Even though we are colouring by single residues, we can still define the colourmap
        # from the sequences, as this is averaged data and can't ever go higher
        if analysis.user_params.normalisation_type != NormalisationMode.INDIVIDUAL:
            global_normalisation_value = find_normalisation_value(
                analysis.sequence_comparisons,
                data_type=DataForVisualisation.UPTAKE_DIFFERENCE,
                normalisation_mode=analysis.user_params.normalisation_type,
            )
            global_cmap = (
                analysis.colouring.uptake_colourmap_with_symmetrical_normalisation(
                    global_normalisation_value
                )
            )

        for index, exposure in enumerate(
            (*analysis.experimental_params.exposures, CUMULATIVE_EXPOSURE_KEY)
        ):

            residue_comparisons = analysis.residue_comparisons[exposure]
            sequence_comparisons = analysis.sequence_comparisons[exposure]
            colour_map = (
                analysis.colouring.uptake_colourmap_with_symmetrical_normalisation(
                    get_strongest_magnitude_of_type(
                        sequence_comparisons, DataForVisualisation.UPTAKE_DIFFERENCE
                    )
                )
                if (global_cmap is None) or exposure == CUMULATIVE_EXPOSURE_KEY
                else global_cmap
            )

            colours = [ERROR_MAGENTA] * len(analysis.full_sequence)

            for index, residue_data in enumerate(residue_comparisons):
                assert (
                    residue_data.residue == index
                ), "Residue number must also be position in the array."
                match residue_data.residue_type:
                    case ResidueType.AVERAGED:
                        colours[index] = colour_map.to_rgba(
                            residue_data.uptake_difference
                        )[
                            :-1
                        ]  # No alpha
                    case ResidueType.ALL_INSIGNIFICANT:
                        colours[index] = analysis.colouring.insignificant
                    case ResidueType.NOT_COVERED:
                        colours[index] = analysis.colouring.no_coverage

            colour_to_name = register_colours(colours)

            for index, colour in enumerate(colours):
                cmd.color(colour_to_name[colour], selection=f"res {index}")

            exposure_desc = (
                "Cumulative exposure"
                if exposure == CUMULATIVE_EXPOSURE_KEY
                else f"Exposure {exposure} minutes"
            )
            scene_description = f"""
            {exposure_desc}
            {pretty_string_for[analysis.user_params.normalisation_type]}
            Confidence interval = {analysis.user_params.confidence_interval*100}%
            Statistical test = {pretty_string_for[analysis.user_params.statistical_test]}
            """

            cmd.scene(
                key="(SAUSC) " + exposure_desc,
                action="store",
                message=scene_description,
                color=1,
            )

    @cmd.extend
    def SAUSC(
        filepath: Optional[str] = None,
        n_repeats: PymolInt = "3",
        confidence_interval: PymolFloat = "0.95",
        statistical_test: str = "HYBRID",
        protection_colourmap: str = "Blues",
        deprotection_colourmap: str = "Reds",
        insignificant_colour: PymolTupleFloat = "(1.0, 1.0, 1.0)",
        no_coverage_colour: PymolTupleFloat = "(0.1, 0.1, 0.1)",
        normalisation_mode: PymolBool = "ACROSS_EXPOSURES",
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

        if filepath is None:
            filepath = file_browser()
        else:
            check_valid_hdx_file(filepath)

        # Essentially just mypy juggling, this is already handled internally
        assert filepath is not None, "File checker didn't work."

        full_analysis = run_SAUSC_from_path(
            filepath=filepath,
            n_repeats=convert_from_pymol(n_repeats, int),
            confidence_interval=convert_percentage_if_necessary(
                convert_from_pymol(confidence_interval, float)
            ),
            statistical_test=statistical_test,
            protection_colourmap=protection_colourmap,
            deprotection_colourmap=deprotection_colourmap,
            insignificant_colour=cast_to_colour(
                convert_from_pymol(insignificant_colour, tuple[float])
            ),
            no_coverage_colour=cast_to_colour(
                convert_from_pymol(no_coverage_colour, tuple[float])
            ),
            normalisation_mode=normalisation_mode,
        )

        draw_uptake_on_scenes(full_analysis)

        @cmd.extend
        def woods_plot(save: PymolBool = "False"):
            draw_woods_plot(full_analysis, save=convert_from_pymol(save, bool))

        @cmd.extend
        def volcano_plot(annotate: PymolBool = "True", save: PymolBool = "False"):
            draw_volcano_plot(
                full_analysis,
                annotate=convert_from_pymol(annotate, bool),
                save=convert_from_pymol(save, bool),
            )


if __name__ == "__main__":
    full_analysis = run_SAUSC_from_path(
        filepath=r".\example_data\Cdstate.csv",
        n_repeats=3,
        confidence_interval=0.95,
        statistical_test="HYBRID",
        protection_colourmap="Blues",
        deprotection_colourmap="Reds",
        insignificant_colour=(0.9, 0.9, 0.9),
        no_coverage_colour=(0.1, 0.1, 0.1),
        normalisation_mode="ACROSS_EXPOSURES",
    )
    draw_woods_plot(full_analysis, save=False)
    draw_volcano_plot(full_analysis, annotate=True, save=False)
