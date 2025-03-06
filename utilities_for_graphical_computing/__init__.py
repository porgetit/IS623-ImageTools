from .ImageProcessor import (
    Imagen,
    ColorConverter,
    FiltroFactory,
    FiltroStrategy,
    FiltroContraste,
    FiltroIntensidad,
    FiltroIdentity,
)

from .SimpleImageViewer import SimpleImageViewer

__all__ = [
    "Imagen",
    "ColorConverter",
    "FiltroFactory",
    "FiltroStrategy",
    "FiltroContraste",
    "FiltroIntensidad",
    "FiltroIdentity",
    "SimpleImageViewer",
]
__version__ = "1.0.0"

