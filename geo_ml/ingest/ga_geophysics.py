"""
Geoscience Australia geophysical grid ingestion via WCS.

Endpoint: http://services.ga.gov.au/gis/geophysical-grids/wcs (v1.0.0)

Verified layer names (2019 compilations):
  Magnetic:     geophys:magmap_v7_2019_TMI
                geophys:magmap_v7_2019_RTP
                geophys:magmap_v7_2019_1VD
                geophys:magmap_v7_2019_VRTP_AS (analytic signal)
  Gravity:      geophys:Gravmap2019-grid-grv_cscba (Bouguer CSCBA)
                geophys:Gravmap2019-grid-grv_fa (free air)
  Radiometric:  geophys:radmap_v4_2019_filtered_pctk (potassium %)
                geophys:radmap_v4_2019_filtered_ppmth (thorium ppm)
                geophys:radmap_v4_2019_filtered_ppmu (uranium ppm)
                geophys:radmap_v4_2019_filtered_dose (total dose)
"""

from __future__ import annotations

import logging
from pathlib import Path

import rasterio
from owslib.wcs import WebCoverageService

log = logging.getLogger(__name__)

WCS_URL = "http://services.ga.gov.au/gis/geophysical-grids/wcs"

YILGARN_BBOX = (119.0, -33.0, 124.0, -27.0)
KALGOORLIE_BBOX = (120.5, -32.0, 122.5, -29.5)
PILBARA_BBOX = (114.0, -24.0, 122.0, -18.0)

LAYERS = {
    "tmi": "geophys:magmap_v7_2019_TMI",
    "tmi_rtp": "geophys:magmap_v7_2019_RTP",
    "tmi_1vd": "geophys:magmap_v7_2019_1VD",
    "mag_analytic_signal": "geophys:magmap_v7_2019_VRTP_AS",
    "gravity_bouguer": "geophys:Gravmap2019-grid-grv_cscba",
    "gravity_freeair": "geophys:Gravmap2019-grid-grv_fa",
    "rad_k": "geophys:radmap_v4_2019_filtered_pctk",
    "rad_th": "geophys:radmap_v4_2019_filtered_ppmth",
    "rad_u": "geophys:radmap_v4_2019_filtered_ppmu",
    "rad_dose": "geophys:radmap_v4_2019_filtered_dose",
}

# Core layers needed for prospectivity mapping
CORE_LAYERS = ["tmi", "gravity_bouguer", "rad_k", "rad_th", "rad_u"]

_wcs_client: WebCoverageService | None = None


def get_wcs_client(url: str = WCS_URL) -> WebCoverageService:
    global _wcs_client
    if _wcs_client is None:
        _wcs_client = WebCoverageService(url, version="1.0.0", timeout=60)
    return _wcs_client


def download_grid(
    layer_key: str,
    bbox: tuple[float, float, float, float] = KALGOORLIE_BBOX,
    output_path: Path | None = None,
    resx: float = 0.005,
    resy: float = 0.005,
    crs: str = "EPSG:4326",
) -> Path:
    """
    Download a geophysical grid from GA WCS and save as GeoTIFF.

    Args:
        layer_key: Key from LAYERS dict (e.g. 'tmi', 'gravity_bouguer').
        bbox: (minx, miny, maxx, maxy) in WGS84.
        output_path: Save path. Defaults to data/raw/<layer_key>.tif.
        resx, resy: Resolution in degrees (~0.005 deg ≈ 500m at this latitude).
        crs: Coordinate reference system.

    Returns:
        Path to saved GeoTIFF.
    """
    layer_id = LAYERS.get(layer_key, layer_key)

    if output_path is None:
        raw_dir = Path(__file__).parents[2] / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        bbox_tag = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
        output_path = raw_dir / f"{layer_key}_{bbox_tag}.tif"

    if output_path.exists():
        log.info(f"{layer_key}: already downloaded at {output_path}")
        return output_path

    log.info(f"Downloading {layer_key} ({layer_id})...")
    wcs = get_wcs_client()
    response = wcs.getCoverage(
        identifier=layer_id,
        bbox=bbox,
        resx=resx,
        resy=resy,
        crs=crs,
        format="GeoTIFF",
    )

    data = response.read()
    output_path.write_bytes(data)

    with rasterio.open(output_path) as src:
        arr = src.read(1)
        log.info(
            f"  Saved {output_path.name}: shape={arr.shape}, "
            f"range=[{arr.min():.1f}, {arr.max():.1f}]"
        )

    return output_path


def download_all_layers(
    bbox: tuple[float, float, float, float] = KALGOORLIE_BBOX,
    output_dir: Path | None = None,
    layers: list[str] | None = None,
) -> dict[str, Path]:
    """
    Download multiple geophysical layers.

    Args:
        bbox: Bounding box in WGS84.
        output_dir: Output directory. Defaults to data/raw/.
        layers: List of layer keys. Defaults to CORE_LAYERS.

    Returns:
        Dict mapping layer key to saved file path.
    """
    if layers is None:
        layers = CORE_LAYERS
    if output_dir is None:
        output_dir = Path(__file__).parents[2] / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox_tag = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
    results = {}
    for key in layers:
        try:
            path = download_grid(
                key, bbox=bbox,
                output_path=output_dir / f"{key}_{bbox_tag}.tif",
            )
            results[key] = path
        except Exception as e:
            log.error(f"Failed to download {key}: {e}")
    return results
