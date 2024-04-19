from typing import Optional
from matplotlib.cm import ScalarMappable
import numpy as np
from colouring_utils import ERROR_MAGENTA, Colour, ColourScheme
from data import (
    CUMULATIVE_EXPOSURE_KEY,
    DataForVisualisation,
    FullSAUSCAnalysis,
    NormalisationMode,
    ResidueType,
)
from pymol import cmd
import analysis_tools
from figures import pretty_string_for

GLOBAL_CUSTOM_COLOUR_INDEX = 0


def register_colours(colours: list[Colour]) -> dict[Colour, str]:

    colour_to_name: dict[Colour, str] = {}

    for colour in colours:
        colour_to_name[colour] = f"SAUSC custom colour {GLOBAL_CUSTOM_COLOUR_INDEX}"
        GLOBAL_CUSTOM_COLOUR_INDEX += 1

    for colour, name in colour_to_name.items():
        cmd.set_color(name=name, rgb=colour)

    return colour_to_name


def draw_uptake_on_scenes(analysis: FullSAUSCAnalysis) -> None:

    global_cmap: Optional[ScalarMappable] = None

    # Even though we are colouring by single residues, we can still define the colourmap
    # from the sequences, as this is averaged data and can't ever go higher
    if analysis.user_params.normalisation_type != NormalisationMode.INDIVIDUAL:
        global_normalisation_value = analysis_tools.find_normalisation_value(
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
                analysis_tools.get_strongest_magnitude_of_type(
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
                    colours[index] = colour_map.to_rgba(residue_data.uptake_difference)
                case ResidueType.ALL_INSIGNIFICANT:
                    colours[index] = analysis.colouring.insignificant
                case ResidueType.NOT_COVERED:
                    colours[index] = analysis.colouring.no_coverage

        colour_to_name = register_colours(colours)

        for index, colour in enumerate(colours):
            cmd.color(colour_to_name[colour], selection=f"res {index}")

        scene_description = f"""
        Exposure {exposure}
        {pretty_string_for[analysis.user_params.normalisation_type]}
        Confidence interval = {analysis.user_params.confidence_interval*100}%
        Statistical test = {pretty_string_for[analysis.user_params.statistical_test]}
        """

        cmd.scene(key="new", action='store', message=scene_description, color=1)
      


