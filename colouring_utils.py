
import dataclasses
from typing import cast

import numpy as np


try:
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib
except ImportError as exc:
    raise ImportError(
        "Please install matplotlib via 'pip install matplotlib'."
    ) from exc


Colour = tuple[float, float, float]

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


