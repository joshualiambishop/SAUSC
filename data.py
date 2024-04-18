
import dataclasses
import enum
from typing import Any, cast

import numpy as np

import formulas
import utils
import colouring_utils
import numpy.typing as npt

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




class StatisticalTestType(enum.Enum):
    """
    Type of statistical test used to perform comparisons
    """

    T_TEST = enum.auto()
    HYBRID = enum.auto()


# For simplicity, a cumulative operation on all exposures are included as if it were a timepoint.
# I've made this a constant simply for sanity.
CUMULATIVE_EXPOSURE_KEY = "Cumulative"




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
        combined_stdev = formulas.pooled_stdev(
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
        utils.enforce_between_0_and_1(self.confidence_interval)



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



@dataclasses.dataclass(frozen=True)
class FullSAUSCAnalysis:
    user_params: UserParameters
    experimental_params: ExperimentalParameters
    sequence_comparisons: dict[str, list[Comparison]]
    residue_comparisons: dict[str, list[SingleResidueComparison]]
    colouring: colouring_utils.ColourScheme
    global_standard_error_mean: float
    full_sequence: str