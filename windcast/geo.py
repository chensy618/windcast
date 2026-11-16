"""Working plane: Lambert azimuthal equal-area centred at 20°E, 68°N (WGS84), in km.

The same plane as the terrain products and the 1 km reference grid, so distances and
areas are in km and positions are comparable across sources.
"""
import numpy as np
from pyproj import Transformer

PROJ4 = "+proj=laea +lat_0=68 +lon_0=20 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
_fwd = Transformer.from_crs("EPSG:4326", PROJ4, always_xy=True)


def to_plane(lon, lat) -> np.ndarray:
    """Longitude / latitude in degrees, any shape -> plane x / y in km, shape (..., 2)."""
    x, y = _fwd.transform(np.asarray(lon), np.asarray(lat))
    return np.stack([x, y], axis=-1) / 1000.0
