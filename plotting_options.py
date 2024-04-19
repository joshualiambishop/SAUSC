

import dataclasses
import utils
from data import DataForVisualisation

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
    box_thickness: float # As a percentage of the y axes
    
    def __post_init__(self) -> None:
        utils.enforce_between_0_and_1(self.box_thickness)

@dataclasses.dataclass(frozen=True)
class VolcanoPlotOptions(BaseVisualisationOptions):
    x_data: DataForVisualisation
    circle_size: float
    circle_transparency: float 


# If for any reason you want to customise the fine details of the plots
# please do so in the code here:

WOODS_PLOT_PARAMS = WoodsPlotOptions(
    dpi = 150.0,
    scale = 6.0,
    y_data = DataForVisualisation.UPTAKE_DIFFERENCE,
    colour_data = DataForVisualisation.RELATIVE_UPTAKE_DIFFERENCE,
    box_thickness = 0.07
)
VOLCANO_PLOT_PARAMS = VolcanoPlotOptions(
    dpi = 150.0,
    scale = 3.0,
    x_data = DataForVisualisation.UPTAKE_DIFFERENCE,
    y_data = DataForVisualisation.NEG_LOG_P,
    colour_data= DataForVisualisation.RELATIVE_UPTAKE_DIFFERENCE,
    circle_size = 1.0,
    circle_transparency= 0.7
)
    
