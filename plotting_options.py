
# If for any reason you want to customise the fine details of the plots, please do so in the code here:
import dataclasses
import data

@dataclasses.dataclass(frozen=True)
class GenericFigureOptions:
    dpi: float = 150
    scale: float = 8


@dataclasses.dataclass(frozen=True)
class WoodsPlotOptions(GenericFigureOptions):
    y_data = data.DataForVisualisation.UPTAKE_DIFFERENCE
    colour_data = data.DataForVisualisation.RELATIVE_UPTAKE_DIFFERENCE
    colour_normalisation = data.NormalisationMode.ACROSS_EXPOSURES
    box_thickness: float = 0.07


WOODS_PLOT_PARAMS = WoodsPlotOptions()
