"""
Correspondence: joshualiambishop@gmail.com
"""

from typing import Optional, TypeAlias, Callable, TypeVar, Any, Type, cast

import numpy as np
import numpy.typing as npt
import dataclasses

import formulas
import analysis_tools
from pymol_interface import PymolBool, PymolFloat, PymolInt, PymolTupleFloat, convert_from_pymol
import utils

import figures

import colouring_utils
import data

def colour_psb_structure_by_uptake_difference() -> None:
    pass


if __name__ == "__main__":
    # from pymol import cmd

    # @cmd.extend
    def SAUSC(
        filepath: Optional[str] = None,
        n_repeats: PymolInt = "3",
        confidence_interval: PymolFloat = "0.95",
        hybrid_test: PymolBool = "True",
        protection_colourmap: str = "Blues",
        deprotection_colourmap: str = "Reds",
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

        if filepath is None:
            filepath = utils.file_browser()
        else:
            utils.check_valid_hdx_file(filepath)
        
        # Essentially just mypy juggling, this is already handled internally
        assert (
            filepath is not None
        ), "File checker didn't work."  

        full_analysis = analysis_tools.run_SAUSC_from_path(
            filepath=filepath,
            n_repeats = convert_from_pymol(n_repeats, int),
            confidence_interval=utils.convert_percentage_if_necessary(
                convert_from_pymol(confidence_interval, float)
            ),
            hybrid_test=convert_from_pymol(hybrid_test, bool),
            protection_colourmap=protection_colourmap,
            deprotection_colourmap=deprotection_colourmap,
            insignificant_colour=colouring_utils.cast_to_colour(
                convert_from_pymol(insignificant_colour, tuple[float])
            ),
            no_coverage_colour=colouring_utils.cast_to_colour(
                convert_from_pymol(no_coverage_colour, tuple[float])
            ),
            global_normalisation=convert_from_pymol(global_normalisation, bool)

        )
        return full_analysis



if __name__ == "__main__":
    # data, params = load_state_data()
    full_analysis = SAUSC(r".\example_data\Cdstate.csv")
    figures.draw_woods_plot(full_analysis)
    figures.draw_volcano_plot(full_analysis, annotate=False)
