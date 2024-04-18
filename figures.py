import dataclasses
import enum
from typing import Any, Optional, Sequence
import numpy as np
import analysis_tools
from data import (
    CUMULATIVE_EXPOSURE_KEY,
    DataForVisualisation,
    FullSAUSCAnalysis,
    NormalisationMode,
)
from plotting_options import (
    VOLCANO_PLOT_PARAMS,
    WOODS_PLOT_PARAMS,
    BaseVisualisationOptions,
)

try:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    import matplotlib.patches as mpl_patches
    import matplotlib.collections as mpl_collections
except ImportError as exc:
    raise ImportError(
        "Please install matplotlib via 'pip install matplotlib'."
    ) from exc


pretty_string_for: dict[enum.Enum, str] = {
    DataForVisualisation.UPTAKE_DIFFERENCE: "Uptake difference (Da)",
    DataForVisualisation.RELATIVE_UPTAKE_DIFFERENCE: """Relative uptake difference (%)""",
    DataForVisualisation.P_VALUE: "P value",
    DataForVisualisation.NEG_LOG_P: "-log(p)",
    NormalisationMode.INDIVIDUAL: "Normalised by timepoint",
    NormalisationMode.ACROSS_EXPOSURES: "Normalised across timepoints",
    NormalisationMode.GLOBAL: "Normalised globally",
}


def multi_line(string: str) -> str:
    n_spaces = len(string.split(" ")) - 1
    # This is because the last one would be the unit
    return string.replace(" ", "\n", n_spaces - 1)


@dataclasses.dataclass(frozen=True)
class BaseFigure:
    fig: plt.Figure
    axes: Sequence[plt.Axes]
    colourmaps: Sequence[ScalarMappable]

    def __post_init__(self) -> None:
        assert len(self.axes) == len(
            self.colourmaps
        ), "Must have a colourmap deifned for each axis."


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

    if plotting_params.colour_normalisation != NormalisationMode.INDIVIDUAL:
        global_normalisation_value = analysis_tools.find_normalisation_value(
            analysis.sequence_comparisons,
            data_type=plotting_params.colour_data,
            normalisation_mode=plotting_params.colour_normalisation,
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

        colour_map = (
            analysis.colouring.uptake_colourmap_with_symmetrical_normalisation(
                analysis_tools.get_strongest_magnitude_of_type(
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
            or index == len(axes)-1  # Last one always has a colourbar
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


def draw_woods_plot(analysis: FullSAUSCAnalysis) -> None:
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
    plt.show()


def draw_volcano_plot(analysis: FullSAUSCAnalysis, annotate: bool) -> None:

    base_figure = set_up_base_figure(
        analysis=analysis,
        plotting_params=VOLCANO_PLOT_PARAMS,
        over_rows=False,
    )

    for index, exposure in enumerate(
        (*analysis.experimental_params.exposures, CUMULATIVE_EXPOSURE_KEY)
    ):

        sequence_comparisons = analysis.sequence_comparisons[exposure]

        base_figure.axes[index].set(
            xlabel=pretty_string_for[VOLCANO_PLOT_PARAMS.x_data],
        )

        base_figure.axes[index].scatter(
            x=[s.request(VOLCANO_PLOT_PARAMS.x_data) for s in sequence_comparisons],
            y=[s.request(VOLCANO_PLOT_PARAMS.y_data) for s in sequence_comparisons],
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
            alpha= VOLCANO_PLOT_PARAMS.circle_transparency
        )
    plt.show()