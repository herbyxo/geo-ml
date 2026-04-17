"""
Spatial feature engineering for mineral prospectivity.

Converts raw raster grids into a flat feature matrix aligned to a common
pixel grid. Each pixel becomes a row with ~15 features.

Feature groups:
  Magnetic   : TMI, first vertical derivative, analytic signal, tilt angle
  Gravity    : Bouguer anomaly, horizontal gradient magnitude
  Radiometric: K, Th, U, K/Th ratio, Th/U ratio, F-parameter
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import rowcol
from scipy.ndimage import gaussian_gradient_magnitude

log = logging.getLogger(__name__)


def load_raster(path: Path) -> tuple[np.ndarray, rasterio.Affine, dict]:
    """Load a single-band raster. Returns (array, transform, profile)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        # GA grids use -99999 as nodata even when not declared in metadata
        arr[arr <= -99990] = np.nan
        return arr, src.transform, dict(src.profile)


def compute_derivatives(tmi: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute magnetic and spatial derivatives from a TMI grid.

    Returns dict with keys:
      fvd            - First vertical derivative (shallow structure)
      analytic_signal - Edge detection regardless of remanence
      tilt_angle     - Structural boundary mapping
      hgm            - Horizontal gradient magnitude
    """
    dx = np.gradient(tmi, axis=1)
    dy = np.gradient(tmi, axis=0)
    dz = gaussian_gradient_magnitude(tmi, sigma=1.0)

    hgm = np.sqrt(dx**2 + dy**2)
    analytic_signal = np.sqrt(dx**2 + dy**2 + dz**2)
    tilt_angle = np.arctan2(dz, hgm + 1e-10)

    return {
        "fvd": dz,
        "analytic_signal": analytic_signal,
        "tilt_angle": tilt_angle,
        "hgm": hgm,
    }


def compute_radiometric_ratios(
    k: np.ndarray, th: np.ndarray, u: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Radiometric ratio transforms.

    K/Th  : potassic alteration intensity (strong Ni indicator)
    Th/U  : metamorphic grade proxy
    F-param (K*Th/U): felsic contamination / sulfur source index
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        k_th = np.where(th > 0, k / th, np.nan)
        th_u = np.where(u > 0, th / u, np.nan)
        f_param = np.where(u > 0, k * th / u, np.nan)

    return {"k_th_ratio": k_th, "th_u_ratio": th_u, "f_parameter": f_param}


def compute_gravity_derivatives(gravity: np.ndarray) -> dict[str, np.ndarray]:
    """Horizontal gradient magnitude of gravity — highlights density contrasts."""
    gx = np.gradient(gravity, axis=1)
    gy = np.gradient(gravity, axis=0)
    return {"grav_hgm": np.sqrt(gx**2 + gy**2)}


def build_feature_stack(
    raster_paths: dict[str, Path],
) -> tuple[dict[str, np.ndarray], rasterio.Affine, tuple[int, int]]:
    """
    Load all rasters, compute derivatives, and return a dict of named feature arrays.
    All arrays are aligned to the first raster's grid.

    Args:
        raster_paths: Dict mapping layer key to .tif file path.
                      Expected keys: 'tmi', 'gravity_bouguer', 'rad_k', 'rad_th', 'rad_u'

    Returns:
        (features_dict, transform, shape) where features_dict maps feature names
        to 2D arrays all of the same shape.
    """
    features: dict[str, np.ndarray] = {}
    ref_transform = None
    ref_shape = None

    # Load raw layers
    for key, path in raster_paths.items():
        if not path.exists():
            log.warning(f"Missing raster: {path}")
            continue
        arr, transform, _ = load_raster(path)
        features[key] = arr

        if ref_transform is None:
            ref_transform = transform
            ref_shape = arr.shape
        else:
            # Resize to reference grid if needed
            if arr.shape != ref_shape:
                from skimage.transform import resize

                arr = resize(arr, ref_shape, preserve_range=True).astype(np.float32)
                features[key] = arr

    # Compute magnetic derivatives if TMI available
    if "tmi" in features:
        derivs = compute_derivatives(features["tmi"])
        for name, arr in derivs.items():
            features[f"mag_{name}"] = arr

    # Compute radiometric ratios if all three available
    if all(k in features for k in ("rad_k", "rad_th", "rad_u")):
        ratios = compute_radiometric_ratios(
            features["rad_k"], features["rad_th"], features["rad_u"]
        )
        for name, arr in ratios.items():
            features[f"rad_{name}"] = arr

    # Compute gravity derivatives if gravity available
    if "gravity_bouguer" in features:
        grav_derivs = compute_gravity_derivatives(features["gravity_bouguer"])
        features.update(grav_derivs)

    log.info(f"Built {len(features)} features, shape={ref_shape}")
    return features, ref_transform, ref_shape


def build_feature_matrix(
    features: dict[str, np.ndarray],
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Stack feature arrays into (N_pixels, N_features) matrix.

    Args:
        features: Dict of {name: 2D array}, all same shape.
        mask: Boolean 2D mask. True = valid pixel. Defaults to non-NaN in all layers.

    Returns:
        (X, feature_names, mask) where X has shape (N_valid, N_features).
    """
    names = sorted(features.keys())
    stack = np.stack([features[n] for n in names], axis=-1)

    if mask is None:
        mask = np.all(np.isfinite(stack), axis=-1)

    X = stack[mask]
    log.info(f"Feature matrix: {X.shape[0]} pixels x {X.shape[1]} features")
    return X, names, mask


def extract_deposit_features(
    features: dict[str, np.ndarray],
    transform: rasterio.Affine,
    deposits: gpd.GeoDataFrame,
) -> np.ndarray:
    """
    Extract feature values at known deposit locations.

    Args:
        features: Dict of named feature arrays.
        transform: Rasterio affine transform of the grid.
        deposits: GeoDataFrame with point geometries.

    Returns:
        Feature matrix (N_deposits, N_features) for valid deposits.
    """
    names = sorted(features.keys())
    shape = features[names[0]].shape
    rows, cols = [], []

    for _, dep in deposits.iterrows():
        r, c = rowcol(transform, dep.geometry.x, dep.geometry.y)
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            rows.append(r)
            cols.append(c)

    if not rows:
        log.warning("No deposits fall within the grid extent")
        return np.empty((0, len(names)))

    X = np.column_stack(
        [features[n][rows, cols] for n in names]
    )

    # Remove rows with NaN
    valid = np.all(np.isfinite(X), axis=1)
    X = X[valid]

    log.info(f"Extracted features for {X.shape[0]} deposits ({len(rows) - X.shape[0]} dropped for NaN)")
    return X
