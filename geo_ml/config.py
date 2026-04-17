"""
Project configuration — single place to change region, targets, and parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"

GDB_PATH = (
    DATA_RAW
    / "aus_mineral_deposits"
    / "Mineral_Deposits_v01_20130729"
    / "Mineral_Deposits.gdb"
)

BBOXES = {
    "yilgarn": (119.0, -33.0, 124.0, -27.0),
    "kalgoorlie": (120.5, -32.0, 122.5, -29.5),
    "pilbara": (114.0, -24.0, 122.0, -18.0),
}


@dataclass
class PipelineConfig:
    """Configuration for a single prospectivity mapping run."""

    # Region
    region_name: str = "kalgoorlie"
    bbox: tuple[float, float, float, float] = (120.5, -32.0, 122.5, -29.5)

    # Target commodities (keys from ozmin.TARGET_COMMODITIES)
    commodities: list[str] = field(default_factory=lambda: ["nickel", "copper"])

    # Geophysical layers to use
    layers: list[str] = field(
        default_factory=lambda: ["tmi", "gravity_bouguer", "rad_k", "rad_th", "rad_u"]
    )

    # Grid resolution in degrees (~0.005 = 500m)
    resolution: float = 0.005

    # ML parameters
    n_estimators: int = 50  # number of PU bags (increase for production)
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.05
    unlabeled_ratio: float = 1.0  # ratio of unlabeled to positive per bag
    random_state: int = 42

    # Output
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)
    output_crs: str = "EPSG:4326"

    def __post_init__(self):
        if isinstance(self.bbox, list):
            self.bbox = tuple(self.bbox)
        if self.region_name in BBOXES and self.bbox == (120.5, -32.0, 122.5, -29.5):
            self.bbox = BBOXES[self.region_name]
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
