"""
Sequence Averaged Uptake State Comparison (SAUSC)

This module is designed for the analysis and visualization of Hydrogen-Deuterium Exchange Mass Spectrometry (HDX-MS) data. It provides
functionalities to process HDX-MS data, perform statistical tests, and generate visual representations such as Woods plots and Volcano plots.

Features include:
- Loading and validating HDX-MS data from CSV files.
- Statistical comparisons between different states of proteins using tests like T-tests and global threshold tests.
- Visualization tools integrated with PyMOL for generating plots directly from the HDX-MS analysis results within PyMOL.
- Customizable plotting options and color schemes to highlight significant differences in deuterium uptake.

This module is structured to run both as a standalone script and within the PyMOL environment, enabling users to leverage PyMOL's 
capabilities for molecular visualization alongside HDX-MS data analysis.

Usage:
- As a standalone script, configure the required parameters and run the desired analysis functions.
- Within PyMOL, load the script as a plugin and use the provided commands to interact with the analysis functions directly.

Author: Josh Bishop
Contact: joshualiambishop@gmail.com
"""

# Standard library imports
import dataclasses
from datetime import datetime
import enum
import pathlib
from tkinter import filedialog
from typing import Any, Callable, Sequence, Type, TypeAlias, TypeVar, Optional, cast

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Try importing third-party libraries
# Non standard libraries will need to be installed within
# PyMOL itself, so that this script can properly function.
missing_libraries = []
try:
    import numpy as np
    import numpy.typing as npt
except ImportError:
    missing_libraries.append("numpy")
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpl_patches
    import matplotlib.collections as mpl_collections
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.cm import ScalarMappable
except ImportError:
    missing_libraries.append("matplotlib")
try:
    from scipy import stats
except ImportError:
    missing_libraries.append("scipy")

if len(missing_libraries) > 0:
    logging.error(
        "Missing required third-party libraries: %s", ", ".join(missing_libraries)
    )
    raise ImportError(f"Missing libraries: {', '.join(missing_libraries)}.")

# Try importing PyMOL specific module
try:
    from pymol import cmd
except ImportError:
    cmd = None
    logging.warning("PyMOL is not installed. Some functionalities will be unavailable.")


class InvalidFileFormatException(Exception):
    """Specific exception for if something is wrong with the user's results file."""


class AnalysisNotRunException(Exception):
    """
    Performing the analysis requires some user input, for functionality where analysis is
    a prerequisite, this exception can be thrown.
    """


Colour: TypeAlias = tuple[float, float, float]
# For simplicity, a cumulative operation on all exposures are included as if it were a timepoint.
# I've made this a constant simply for sanity.
CUMULATIVE_EXPOSURE_KEY = "Cumulative"
# If saving a figure, all the following formats will be saved together
FIGURE_SAVING_FORMATS: list[str] = [".png", ".svg"]
# These global variables ensure that PyMOL
# scenes and colours are never overwritten
GLOBAL_CUSTOM_COLOUR_INDEX = 1
GLOBAL_SCENE_INDEX = 1
# Nice bright purple to represent any errors visually
ERROR_MAGENTA: Colour = (204.0, 0.0, 153.0)
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

SameAsInput = TypeVar("SameAsInput")
SameAsInputEnum = TypeVar("SameAsInputEnum", bound=enum.Enum)


class DataForVisualisation(enum.Enum):
    """
    Defines the possible attributes represented visually.
    """

    UPTAKE_DIFFERENCE = enum.auto()
    RELATIVE_UPTAKE_DIFFERENCE = enum.auto()
    P_VALUE = enum.auto()
    NEG_LOG_P = enum.auto()


class NormalisationMode(enum.Enum):
    """
    With various states, these attributes define the colour normalisation between them
    """

    INDIVIDUAL = enum.auto()  # Colourmaps are normalised to each individual exposure.
    ACROSS_EXPOSURES = (
        enum.auto()
    )  # Colourmaps are normalised across the maximum for each exposure (not including cumulative).
    GLOBAL = (
        enum.auto()
    )  # A single universal colourmap, not recommended with cumulative as that'll naturally skew the colours, but the option is there.


class StatisticalTestType(enum.IntFlag):
    """
    Type of statistical test used to perform comparisons, IntFlag for bitmasking operations
    and combinations.
    """

    T_TEST = 1
    GLOBAL_THRESHOLD = 2
    HYBRID = 3


class ResidueType(enum.Enum):
    AVERAGED = enum.auto()
    ALL_INSIGNIFICANT = enum.auto()
    NOT_COVERED = enum.auto()


def require_nonnegative(value: float) -> None:
    if value < 0:
        raise ValueError(f"Value {value} must be non-negative.")


def require_all_nonnegative(*values: float) -> None:
    for value in values:
        require_nonnegative(value)


def enforce_between_0_and_1_inclusive(value: float) -> None:
    if not (0 <= value <= 1):
        raise ValueError(f"Value {value} must be between 0 and 1")


def convert_percentage_if_necessary(value: float) -> float:
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


def without_zeros(array: npt.NDArray) -> npt.NDArray:
    """
    HDX data often contains a 0 timepoint reference providing arrays
    with 0 values, when doing any combinations or averaging, we'll want
    to explicitly remove them.
    """
    is_zero = array == 0
    if is_zero.all():
        logging.warn(f"Array is all zero.")
    return array[~is_zero]


def find_matching_enum_from_meta(
    meta: Type[SameAsInputEnum], query: str
) -> SameAsInputEnum:
    for possible_match in meta:
        name: str = possible_match.name
        if name.lower() == query.lower():
            return possible_match
    raise ValueError(f"{query} not found in {enum}.")


def warn_SAUSC_not_run() -> None:
    raise AnalysisNotRunException(
        "Cannot perform requested action - SAUSC has not been run on any data."
    )


@dataclasses.dataclass(frozen=True)
class GenericFigureOptions:
    dpi: float
    scale: float

    def __post_init__(self) -> None:
        require_all_nonnegative(self.dpi, self.scale)


@dataclasses.dataclass(frozen=True)
class BaseVisualisationOptions(GenericFigureOptions):
    y_data: DataForVisualisation
    colour_data: DataForVisualisation
    statistical_linewidth: float
    statistical_linecolour: str

    def __post_init__(self) -> None:
        super().__post_init__()
        require_nonnegative(self.statistical_linewidth)


@dataclasses.dataclass(frozen=True)
class WoodsPlotOptions(BaseVisualisationOptions):
    box_thickness: float  # As a percentage of the y axes

    def __post_init__(self) -> None:
        super().__post_init__()
        enforce_between_0_and_1_inclusive(self.box_thickness)


@dataclasses.dataclass(frozen=True)
class VolcanoPlotOptions(BaseVisualisationOptions):
    x_data: DataForVisualisation
    circle_size: float
    circle_transparency: float
    annotation_fontsize: float

    def __post_init__(self) -> None:
        super().__post_init__()
        require_nonnegative(self.circle_size)
        enforce_between_0_and_1_inclusive(self.circle_transparency)


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
    statistical_linewidth=0.1,
    statistical_linecolour="black",
)

VOLCANO_PLOT_PARAMS = VolcanoPlotOptions(
    dpi=100.0,
    scale=4.0,
    x_data=DataForVisualisation.UPTAKE_DIFFERENCE,
    y_data=DataForVisualisation.NEG_LOG_P,
    colour_data=DataForVisualisation.RELATIVE_UPTAKE_DIFFERENCE,
    circle_size=13.0,
    circle_transparency=0.8,
    annotation_fontsize=5.0,
    statistical_linewidth=0.5,
    statistical_linecolour="black",
)


def check_valid_hdx_file(path: Optional[str]) -> None:
    """
    Performs a soft check that a filepath has been supplied,
    the file exists (for when users input a path directly),
    and the file is a csv.
    """
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
    # In PyMOL this gets directly injected to QtWidgets.QFileDialog.getOpenFileName
    results_file = filedialog.askopenfilename(
        title="(SAUSC) Select HDX results file (state data not cluster data)",
        initialdir=str(pathlib.Path.home()),
        filetypes=[("CSV", "*.csv"), ("All files", "*")],
    )

    check_valid_hdx_file(results_file)
    return results_file


# PyMOL passes all arguments to functions as strings
PyMOLBool: TypeAlias = str
PyMOLTupleFloat: TypeAlias = str
PyMOLInt: TypeAlias = str
PyMOLFloat: TypeAlias = str


def _str_to_bool(string: PyMOLBool) -> bool:
    # There is no guarantee someone operating PyMOL will be familiar with python
    # so .lower() is used to ensure "true" is as valid as the correct "True"
    # (and as valid as "tRuE")
    if string.lower() == "true":
        return True
    if string.lower() == "false":
        return False
    raise TypeError(f"Input {string} must be True or False")


def _str_to_tuple_float(string: PyMOLTupleFloat) -> tuple[float, ...]:
    # Assuming input is of the form "(1.0, 1.0, 1.0)"
    stripped = string.strip("()")
    components = stripped.split(",")
    if not all([is_floatable(possible_float.strip()) for possible_float in components]):
        raise ValueError(f"Components {components} are not all floatable.")
    return tuple([float(number) for number in components])


def _str_to_float(string: str) -> float:
    if is_floatable(string):
        return float(string)
    else:
        raise ValueError(f"Argument {string} could not be interpreted as a float.")


def _str_to_int(string: str) -> int:
    if is_floatable(string):
        return int(string)
    else:
        raise ValueError(f"Argument {string} could not be interpreted as an int.")


PYMOL_CONVERTORS: dict[type, Callable] = {
    bool: _str_to_bool,
    int: _str_to_int,
    float: _str_to_float,
    tuple[float]: _str_to_tuple_float,
}


def convert_from_pymol(argument: str, requested_type: Type[SameAsInput]) -> SameAsInput:
    if requested_type not in PYMOL_CONVERTORS:
        raise ValueError(f"No conversion for type {requested_type}.")
    convertor = PYMOL_CONVERTORS[requested_type]
    return convertor(argument)


### Mathematical formulas ###
def pooled_stdev(stdevs: npt.NDArray[np.float_]) -> float:
    """
    Combines standard deviations into a single number, simplified since within HDX results we
    will always be pooling standard deviations with the same number of observations.
    """
    variance = without_zeros(stdevs) ** 2
    avg_variance = variance.mean()
    return np.sqrt(avg_variance)


def pooled_standard_error_mean(stdevs: npt.NDArray[np.float_], n_repeats: int) -> float:
    pooled_sem = np.sqrt((2 * (pooled_stdev(stdevs) ** 2)) / ((n_repeats * 2) - 2))
    return pooled_sem


@dataclasses.dataclass(frozen=True)
class Uptake:
    """
    Simple wrapper for an observation of a fragment, represents the distribution
    of observed deuterium uptakes across a range of repeats (universal across an experiment)
    """

    mean: float
    stdev: float
    cumulative: bool = False

    def __post_init__(self):
        require_all_nonnegative(self.mean, self.stdev)


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
        require_all_nonnegative(
            self.start_residue, self.end_residue, self.max_deuterium_uptake
        )
        if self.start_residue <= self.end_residue:
            raise ValueError("End residue must be after the start residue.")

        expected_sequence_length = self.end_residue - self.start_residue + 1
        if len(self.sequence) == expected_sequence_length:
            raise ValueError("Sequence must have {expected_sequence_length} residues.")

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
    Represents the difference between a state and reference of a given fragment,
    at a given exposure.
    """

    uptake_difference: float
    p_value: float
    is_significant: bool
    exposure: str

    def __post_init__(self) -> None:
        super().__post_init__()
        # Uptake difference can of course be negative
        require_nonnegative(self.p_value)

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
    """
    As part of the colouring, we will need to split the comparisons over a single residue.
    """

    amino_acid: str
    residue: int
    uptake_difference: float
    uptake_stdev: float
    exposure: str
    residue_type: ResidueType

    def __post_init__(self) -> None:
        require_all_nonnegative(self.uptake_stdev, self.residue)
        if not len(self.amino_acid) == 1:
            raise ValueError("Amino acid must be a single character.")

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
    Simple container for the parameters of the experiment; the states (assuming one is the default),
    and the number of deuterium exposure durations (in minutes)
    """

    states: tuple[str, str]
    exposures: tuple[str, ...]
    max_residue: int

    def __post_init__(self):
        require_nonnegative(self.max_residue)
        if not len(self.states) == 2:
            raise ValueError("SAUSC only supports datafiles with two states.")

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
        enforce_between_0_and_1_inclusive(self.confidence_interval)
        require_nonnegative(self.n_repeats)


def load_state_data(
    filepath: str,
) -> tuple[list[StateData], ExperimentalParameters]:
    """
    This is specifically designed to load a .csv containing state data.
    There are several checks to ensure data is in the correct format and layout.
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
    # Ensure the headers are arranged in the expected layout, for confidence that
    # we are assigning the correct information.
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

    # This complexity exists only to give the user a helpful targeted
    # error message if there is a missing column
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
    StatisticalTestType.GLOBAL_THRESHOLD: "Global threshold",  # TODO: better name
    StatisticalTestType.HYBRID: "Hybrid test",
}


def multi_line(string: str) -> str:
    n_spaces = len(string.split(" ")) - 1
    # This is because the last one would be the unit
    return string.replace(" ", "\n", n_spaces - 1)


@dataclasses.dataclass(frozen=True)
class ColourScheme:
    """
    Represents all the colours for structures and visualisations
    """

    uptake_colourmap: ListedColormap
    insignificant: Colour
    no_coverage: Colour
    error: Colour = ERROR_MAGENTA

    def make_uptake_colourmap_with_symmetrical_normalisation(
        self, max_value: float
    ) -> ScalarMappable:
        require_nonnegative(max_value)
        normalisation = Normalize(vmin=-max_value, vmax=max_value)
        return ScalarMappable(norm=normalisation, cmap=self.uptake_colourmap)


def cast_to_colour(values: tuple[float, ...]) -> Colour:
    if len(values) != 3:
        raise ValueError("Colour must be formed of 3 values (RGB).")
    return cast(Colour, values)


def make_diverging_colormap(
    protection_colormap: str, deprotection_colormap: str
) -> ListedColormap:
    """
    Combine two sequential colormaps into a diverging one, giving a user flexibility for their (de)protection colors.
    Only makes sense for HDX results to have white in the middle.
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

    # Protection colour (i.e. negative values) is reversed so that the colour intensity
    # increases with more negative values.
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
    user_params: UserParameters,
    experimental_params: ExperimentalParameters,
    global_threshold: float,
) -> tuple[float, float, bool]:
    """
    Compares two uptakes and returns the difference and significance
    """
    uptake_difference = other.mean - default.mean

    if default.cumulative != other.cumulative:
        raise ValueError("Cannot compare cumulative uptakes to standard uptakes.")

    significant = True

    if user_params.statistical_test | StatisticalTestType.T_TEST:

        p_value = stats.ttest_ind_from_stats(
            mean1=default.mean,
            std1=default.stdev,
            nobs1=user_params.n_repeats,
            mean2=other.mean,
            std2=other.stdev,
            nobs2=user_params.n_repeats,
        ).pvalue

        significant &= p_value < (1 - user_params.confidence_interval)

    if user_params.statistical_test | StatisticalTestType.GLOBAL_THRESHOLD:
        if default.cumulative and other.cumulative:
            globally_significant = abs(uptake_difference) > global_threshold * len(
                experimental_params.exposures
            )  # TODO: Not sure this is the correct thing to do...
        else:
            globally_significant = abs(uptake_difference) > global_threshold
        significant &= globally_significant

    return uptake_difference, p_value, significant


def compare_states(
    default: StateData,
    other: StateData,
    user_params: UserParameters,
    experimental_params: ExperimentalParameters,
    global_threshold: float,
) -> dict[str, Comparison]:
    """
    Compares two states together and returns a comparison for each exposure.
    """
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
            user_params=user_params,
            experimental_params=experimental_params,
            global_threshold=global_threshold,
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
    """
    With all the comparisons performed across the dataset, go residue by residue, determine
    what sequences the residue is present in and average together the uptake differences on those
    that are statistically significant.
    """
    if len(set([c.exposure for c in comparisons])) > 1:
        raise ValueError("All comparisons must be from the same exposure time.")

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
        relevant_data = [
            comparison
            for collection in comparisons.values()
            for comparison in collection
        ]
    elif normalisation_mode == NormalisationMode.ACROSS_EXPOSURES:
        relevant_data = [
            comparison
            for (exposure, collection) in comparisons.items()
            for comparison in collection
            if exposure != CUMULATIVE_EXPOSURE_KEY
        ]
    else:
        raise ValueError(f"Cannot interpret normalisation mode {normalisation_mode}")

    return get_strongest_magnitude_of_type(relevant_data, data_type)


@dataclasses.dataclass(frozen=True)
class FullSAUSCAnalysis:
    user_params: UserParameters
    experimental_params: ExperimentalParameters
    sequence_comparisons: dict[str, list[Comparison]]
    residue_comparisons: dict[str, list[SingleResidueComparison]]
    colouring: ColourScheme
    global_threshold: float
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
    """
    Perform the full data analysis procedure.
    """
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
    # Keep the user explicitly updated with their full choices
    # (also helps to see defaults that may go unnoticed)
    logging.info(
        f"""
        User selected options:
        filepath: {filepath}
        n_repeats: {user_params.n_repeats}
        confidence_interval: {user_params.confidence_interval}
        statistical_test: {user_params.statistical_test}
        protection_colourmap: {protection_colourmap}
        deprotection_colourmap: {deprotection_colourmap}
        insignificant_colour: {insignificant_colour}
        no_coverage_colour: {no_coverage_colour}
        normalisation_mode: {user_params.normalisation_type}
        """
    )
    loaded_results, experimental_params = load_state_data(filepath)
    logging.info(
        f"""
        Successfully loaded file, determined parameters:
        Unique states: {' & '.join([state for state in experimental_params.states])}
        Exposure lengths: {', '.join([exposure for exposure in experimental_params.exposures])}
        """
    )
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
    degrees_of_freedom = (n_repeats * 2) - 2
    t_critical = stats.t.ppf(confidence_interval, degrees_of_freedom)
    global_threshold = global_sem * t_critical

    # Hopefully a user will call the default state "default"...
    # TODO: Check with luke if that is generally the case.
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
        compare_states(
            default,
            other,
            user_params=user_params,
            experimental_params=experimental_params,
            global_threshold=global_threshold,
        )
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
        global_threshold=global_threshold,
        filepath=pathlib.Path(filepath),
    )
    return full_analysis


@dataclasses.dataclass(frozen=True)
class BaseFigure:
    fig: plt.Figure
    axes: Sequence[plt.Axes]
    colourmaps: Sequence[ScalarMappable]

    def __post_init__(self) -> None:
        if not len(self.axes) == len(self.colourmaps):
            raise ValueError("Must provide 1 colourmap for each axis.")

    def save(
        self, description: str, extension: str, analysis_filepath: pathlib.Path
    ) -> None:
        """
        Save the current figure into a folder labelled "SAUSC Figures" next to the results file.
        Will save a copy with each given file extension.
        """
        if not extension.startswith("."):
            raise ValueError(
                f"{extension} is an unsuitable file extension, must begin with ."
            )
        figure_folder = analysis_filepath.parent / "SAUSC Figures"
        figure_folder.mkdir(parents=True, exist_ok=True)
        filepath_for_saving = (
            figure_folder
            / f"{analysis_filepath.stem} {description} {datetime.now().strftime('%Y_%m_%d %H_%M')}{extension}"
        )

        # While there is a very low probability a user will save the figure twice in the same minute, we
        # choose to just overwrite as this is more likely to be a result of a minor tweak to the figure.
        if filepath_for_saving.exists():
            raise Exception(
                "Warning: {filepath_for_saving} already exists, overwriting..."
            )

        self.fig.savefig(str(filepath_for_saving))
        logging.info(
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
    This shared utility sorts out the arrangment of the plots, with proper scaling and colourbars
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
            analysis.colouring.make_uptake_colourmap_with_symmetrical_normalisation(
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
            analysis.colouring.make_uptake_colourmap_with_symmetrical_normalisation(
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

    statistical_boundary_params = {
        "color": VOLCANO_PLOT_PARAMS.statistical_linecolour,
        "linewidth": VOLCANO_PLOT_PARAMS.statistical_linewidth,
        "linestyle": "--",
        "zorder": -1,
    }

    for index, exposure in enumerate(
        (*analysis.experimental_params.exposures, CUMULATIVE_EXPOSURE_KEY)
    ):

        sequence_comparisons = analysis.sequence_comparisons[exposure]

        base_figure.axes[index].axhline(
            -np.log10(1 - analysis.user_params.confidence_interval),
            **statistical_boundary_params,
        )

        # Draw statistical boundaries

        if analysis.user_params.statistical_test | StatisticalTestType.T_TEST:

            if VOLCANO_PLOT_PARAMS.y_data == DataForVisualisation.NEG_LOG_P:
                threshold = -np.log10(1 - analysis.user_params.confidence_interval)
                base_figure.axes[index].axhline(
                    threshold, **statistical_boundary_params
                )

            elif VOLCANO_PLOT_PARAMS.y_data == DataForVisualisation.P_VALUE:
                threshold = 1 - analysis.user_params.confidence_interval
                base_figure.axes[index].axhline(
                    threshold, **statistical_boundary_params
                )

        if (
            VOLCANO_PLOT_PARAMS.x_data == DataForVisualisation.UPTAKE_DIFFERENCE
            and analysis.user_params.statistical_test
            | StatisticalTestType.GLOBAL_THRESHOLD
        ):
            threshold = (
                analysis.global_threshold
                if exposure != CUMULATIVE_EXPOSURE_KEY
                else analysis.global_threshold
                * len(
                    analysis.experimental_params.exposures
                )  # TODO: Should this account for 0 timestep?
            )
            base_figure.axes[index].axvline(threshold, **statistical_boundary_params)
            base_figure.axes[index].axvline(-threshold, **statistical_boundary_params)

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


if __name__ == "pymol":

    # Allow the user to run a function after preloading SAUSC, but warn in the case
    # the function hasn't actually been run
    @cmd.extend
    def woods_plot(save: PyMOLBool = "False") -> None:
        warn_SAUSC_not_run()

    @cmd.extend
    def volcano_plot(save: PyMOLBool = "False", annotate: PyMOLBool = "True") -> None:
        warn_SAUSC_not_run()

    def register_colours_to_PyMOL(colours: list[Colour]) -> dict[Colour, str]:
        global GLOBAL_CUSTOM_COLOUR_INDEX
        colour_to_name: dict[Colour, str] = {}

        for colour in colours:
            colour_to_name[colour] = f"SAUSC custom colour {GLOBAL_CUSTOM_COLOUR_INDEX}"
            GLOBAL_CUSTOM_COLOUR_INDEX += 1

        for colour, name in colour_to_name.items():
            cmd.set_color(name=name, rgb=list(colour))

        return colour_to_name

    def draw_uptake_on_scenes(analysis: FullSAUSCAnalysis) -> None:
        global GLOBAL_SCENE_INDEX
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
                analysis.colouring.make_uptake_colourmap_with_symmetrical_normalisation(
                    global_normalisation_value
                )
            )

        for index, exposure in enumerate(
            (*analysis.experimental_params.exposures, CUMULATIVE_EXPOSURE_KEY)
        ):

            residue_comparisons = analysis.residue_comparisons[exposure]
            sequence_comparisons = analysis.sequence_comparisons[exposure]
            colour_map = (
                analysis.colouring.make_uptake_colourmap_with_symmetrical_normalisation(
                    get_strongest_magnitude_of_type(
                        sequence_comparisons, DataForVisualisation.UPTAKE_DIFFERENCE
                    )
                )
                if (global_cmap is None) or exposure == CUMULATIVE_EXPOSURE_KEY
                else global_cmap
            )

            colours = [ERROR_MAGENTA] * len(analysis.full_sequence)

            for index, residue_data in enumerate(residue_comparisons):
                if residue_data.residue != index:
                    raise ValueError(
                        "Residue number must also be position in the array."
                    )
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

            colour_to_name = register_colours_to_PyMOL(colours)

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
                key=f"(SAUSC {GLOBAL_SCENE_INDEX}) " + exposure_desc,
                action="store",
                message=scene_description,
                color=1,
            )

            GLOBAL_SCENE_INDEX += 1

    @cmd.extend
    def SAUSC(
        filepath: Optional[str] = None,
        n_repeats: PyMOLInt = "3",
        confidence_interval: PyMOLFloat = "0.95",
        statistical_test: str = "HYBRID",
        protection_colourmap: str = "Blues",
        deprotection_colourmap: str = "Reds",
        insignificant_colour: PyMOLTupleFloat = "(0.9, 0.9, 0.9)",
        no_coverage_colour: PyMOLTupleFloat = "(0.1, 0.1, 0.1)",
        normalisation_mode: PyMOLBool = "ACROSS_EXPOSURES",
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
        if filepath is None:
            raise ValueError("File checker didn't work.")

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
        def woods_plot(save: PyMOLBool = "False"):
            draw_woods_plot(full_analysis, save=convert_from_pymol(save, bool))

        @cmd.extend
        def volcano_plot(annotate: PyMOLBool = "True", save: PyMOLBool = "False"):
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
