"""
Australian Mineral Deposits database ingestion.

Source: data.gov.au — "Australian Mineral Deposits" (Geoscience Australia)
  https://data.gov.au/data/dataset/308bd1c4-f495-4d82-b5e8-dabae98dcf69

Dataset: File Geodatabase with 18,719 records across all Australian states.
Format: ESRI File GDB (.gdb), single layer 'MineralDeposits'.
CRS: EPSG:4283 (GDA94).

Key columns:
  - NAME               : deposit name
  - STATE              : 'WA', 'NSW', etc.
  - COMMODID           : commodity code ('Ni', 'Cu', 'Li2O', 'Au', etc.)
  - COMMOD_NAME        : full commodity description
  - OPERATING_STATUS   : 'operating mine', 'historic mine', 'mineral deposit'
  - LONGITUDE, LATITUDE: WGS84-equivalent coordinates
  - TYPE               : resource classification
  - ORE, AV_GRADE      : tonnage and grade where available
  - geometry           : Point geometry

WA coverage:
  - Ni:   458 deposits (komatiite-hosted sulfide + lateritic)
  - Cu:   349 deposits
  - Li2O: 26 deposits (Greenbushes, Mt Cattlin, Pilgangoora, Mt Marion)
  - Au:   3,312 deposits
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

GDB_SUBPATH = (
    "aus_mineral_deposits/Mineral_Deposits_v01_20130729/Mineral_Deposits.gdb"
)

TARGET_COMMODITIES = {
    "nickel": ["Ni"],
    "copper": ["Cu"],
    "lithium": ["Li2O"],
    "gold": ["Au"],
    "cobalt": ["Co"],
    "pge": ["PGE", "Pt", "Pd"],
}


def load_deposits(gdb_path: Path | str) -> gpd.GeoDataFrame:
    """Load the full mineral deposits geodatabase."""
    gdf = gpd.read_file(str(gdb_path), layer="MineralDeposits")
    gdf = gdf.to_crs("EPSG:4326")
    return gdf


def filter_wa_deposits(
    gdf: gpd.GeoDataFrame,
    commodities: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """
    Filter to WA deposits for given commodity groups.

    Args:
        gdf: Full mineral deposits GeoDataFrame.
        commodities: Keys from TARGET_COMMODITIES (e.g. ['nickel', 'copper']).
                     None = nickel + copper (primary targets).

    Returns:
        GeoDataFrame with deduplicated deposit locations.
    """
    if commodities is None:
        commodities = ["nickel", "copper"]

    wa = gdf[gdf["STATE"].str.upper().str.strip() == "WA"].copy()

    commodity_codes: list[str] = []
    for c in commodities:
        commodity_codes.extend(TARGET_COMMODITIES.get(c, [c]))

    mask = wa["COMMODID"].isin(commodity_codes)
    wa = wa[mask].copy()

    # Deduplicate: same deposit appears multiple times with different resource estimates
    wa = wa.drop_duplicates(subset=["NAME", "LONGITUDE", "LATITUDE"])

    return wa


def get_positive_labels(
    gdb_path: Path | str,
    commodities: list[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> gpd.GeoDataFrame:
    """
    Return known deposit locations as positive training labels.

    Args:
        gdb_path: Path to .gdb directory.
        commodities: Target commodity groups.
        bbox: Optional spatial filter (minx, miny, maxx, maxy) in WGS84.

    Returns:
        GeoDataFrame of deposit points with label=1.
    """
    gdf = load_deposits(gdb_path)
    filtered = filter_wa_deposits(gdf, commodities)

    if bbox is not None:
        filtered = filtered.cx[bbox[0] : bbox[2], bbox[1] : bbox[3]]

    filtered["label"] = 1

    cols = ["NAME", "COMMODID", "COMMOD_NAME", "OPERATING_STATUS", "geometry", "label"]
    available = [c for c in cols if c in filtered.columns]
    return filtered[available].reset_index(drop=True)
