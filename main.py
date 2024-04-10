# -*- coding: utf-8 -*-
"""
Written by Josh Bishop and Luke Smith

University of Oxford

07/03/2021
"""


from typing import Mapping
import numpy as np

import copy
import numpy.typing as npt
from tkinter import filedialog
import tkinter as tk
import dataclasses

try:
    from scipy import stats
except ImportError as exc:
    raise ImportError("Please install scipy via 'pip install scipy'.") from exc
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError as exc:
    raise ImportError(
        "Please install matplotlib via 'pip install matplotlib'."
    ) from exc


class InvalidFileFormatException(Exception):
    pass


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


# Pymol unfortunately passes all arguments as strings
def _str_to_bool(string: str) -> bool:
    if string.lower() == "true":
        return True
    elif string.lower() == "false":
        return False
    else:
        raise TypeError(f"Input {string} must be True or False")


@dataclasses.dataclass(frozen=True)
class ExperimentalParameters:
    states: tuple[str]
    exposures: tuple[str]

    @property
    def sequence_parse_length(self) -> int:
        return len(self.states) * len(self.exposures)


@dataclasses.dataclass(frozen=True)
class Uptake:
    mean: float
    stdev: float


@dataclasses.dataclass(frozen=True)
class StateData:
    sequence: str
    start_residue: int
    end_residue: int
    state: str
    max_deuterium_uptake: float
    exposures: Mapping[str, Uptake]

    def residue_present(self, residue: int) -> bool:
        return self.start_residue <= residue <= self.end_residue


def load_state_data(
    filepath: str,
):  # -> tuple[list[StateData], ExperimentalParameters]:
    """
    This is specifically designed to load a .csv containing state data.
    """
    expected_columns = np.arange(len(EXPECTED_STATE_DATA_HEADERS))
    data = np.loadtxt(filepath, delimiter=",", dtype=object, usecols=expected_columns)
    headers = data[0].tolist()

    # Z refers to [something], only present in cluster data, a common mistake.
    if "z" in data:
        raise InvalidFileFormatException(
            f"{filepath} appears to be cluster data, SAUSC only operates on state data."
        )

    if not np.all(headers == EXPECTED_STATE_DATA_HEADERS):
        raise InvalidFileFormatException(
            f"headers {headers} is not the expected {EXPECTED_STATE_DATA_HEADERS}."
        )

    start_residues = data[1:, 1].astype(int)
    end_residues = data[1:, 2].astype(int)
    sequences = data[1:, 3].astype(str)
    max_deuterium_uptakes = data[1:, 6].astype(float)
    states = data[1:, 8].astype(str)
    exposures = data[1:, 9].astype(str)
    uptake_means = data[1:, 12].astype(float)
    uptake_stdevs = data[1:, 13].astype(float)

    # fromkeys acts like a set but preserves order
    unique_states: tuple[str] = tuple(dict.fromkeys(states).keys())
    unique_exposures: tuple[str] = tuple(dict.fromkeys(exposures).keys())

    experimental_parameters = ExperimentalParameters(
        states=unique_states, exposures=unique_exposures
    )

    # This complexity exists only to give the user a helpful targeted error message if there is a missing column
    indexes = np.arange(len(data) - 1)
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

        for state in unique_states:
            state_data.append(
                StateData(
                    sequence=sequences[chunk_indexes[0]],
                    state=state,
                    start_residue=start_residues[chunk_indexes[0]],
                    end_residue=end_residues[chunk_indexes[0]],
                    max_deuterium_uptake=max_deuterium_uptakes[chunk_indexes[0]],
                    exposures={
                        exposures[chunk_indexes[i % len(unique_states)]]: uptakes[
                            i % len(unique_states)
                        ]
                        for i in range(len(unique_exposures))
                    },
                )
            )

    return (state_data, experimental_parameters)


if __name__ == "__main__":
    data, params = load_state_data(r"C:\Users\joshu\Downloads\Cdstate.csv")
