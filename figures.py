from typing import Any, Optional

import numpy as np
import analysis_tools
from data import CUMULATIVE_EXPOSURE_KEY, DataForVisualisation, FullSAUSCAnalysis, NormalisationMode
from plotting_options import WOODS_PLOT_PARAMS

try:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    import matplotlib.patches as mpl_patches
    import matplotlib.collections as mpl_collections
except ImportError as exc:
    raise ImportError(
        "Please install matplotlib via 'pip install matplotlib'."
    ) from exc



pretty_string_for: dict[Any, str] = {
    DataForVisualisation.UPTAKE_DIFFERENCE: "Uptake difference (Da)",
    DataForVisualisation.RELATIVE_UPTAKE_DIFFERENCE: """Relative uptake difference (%)""",
    NormalisationMode.INDIVIDUAL: "Normalised by timepoint",
    NormalisationMode.ACROSS_EXPOSURES: "Normalised across timepoints",
    NormalisationMode.GLOBAL: "Normalised globally",
}


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
    if WOODS_PLOT_PARAMS.colour_normalisation != NormalisationMode.INDIVIDUAL:
        global_normalisation_value = analysis_tools.find_normalisation_value(
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
                analysis_tools.get_strongest_magnitude_of_type(
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

