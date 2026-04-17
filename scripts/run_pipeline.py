"""
geo-ml: End-to-end mineral prospectivity mapping pipeline.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --region kalgoorlie --commodities nickel copper
    python scripts/run_pipeline.py --region yilgarn --n-bags 100

This script:
  1. Downloads geophysical grids from Geoscience Australia WCS
  2. Loads known deposit locations from Australian Mineral Deposits database
  3. Engineers spatial features (magnetic derivatives, radiometric ratios, etc.)
  4. Trains a PU-bagging XGBoost model
  5. Runs spatial cross-validation
  6. Generates a prospectivity map + SHAP importance + HTML report
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from geo_ml.config import (
    BBOXES,
    DATA_RAW,
    GDB_PATH,
    PipelineConfig,
)
from geo_ml.ingest.ga_geophysics import download_all_layers
from geo_ml.ingest.ozmin import get_positive_labels
from geo_ml.features.spatial import (
    build_feature_matrix,
    build_feature_stack,
    extract_deposit_features,
)
from geo_ml.models.train import (
    bagging_pu_train,
    compute_shap_importance,
    make_base_estimator,
    predict_map,
    spatial_cross_validate,
)
from geo_ml.viz.maps import (
    export_geotiff,
    generate_html_report,
    generate_interactive_map,
    plot_concentration_area,
    plot_input_layers,
    plot_prospectivity,
    plot_shap_importance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Mineral prospectivity mapping pipeline")
    parser.add_argument("--region", default="kalgoorlie", choices=list(BBOXES.keys()),
                        help="Target region (default: kalgoorlie)")
    parser.add_argument("--commodities", nargs="+", default=["nickel", "copper"],
                        help="Target commodities (default: nickel copper)")
    parser.add_argument("--resolution", type=float, default=0.005,
                        help="Grid resolution in degrees (default: 0.005 ≈ 500m)")
    parser.add_argument("--n-bags", type=int, default=50,
                        help="Number of PU bagging iterations (default: 50)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download (use existing files)")
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip spatial cross-validation")
    args = parser.parse_args()

    config = PipelineConfig(
        region_name=args.region,
        bbox=BBOXES[args.region],
        commodities=args.commodities,
        resolution=args.resolution,
        n_estimators=args.n_bags,
    )
    config._skip_download = args.skip_download
    config._skip_cv = args.skip_cv
    return config


def main():
    import numpy as np

    t0 = time.time()
    config = parse_args()

    log.info("=" * 60)
    log.info(f"geo-ml Mineral Prospectivity Pipeline")
    log.info(f"Region: {config.region_name} {config.bbox}")
    log.info(f"Targets: {config.commodities}")
    log.info(f"Resolution: {config.resolution}° (~{config.resolution * 111:.0f}km)")
    log.info(f"PU bags: {config.n_estimators}")
    log.info("=" * 60)

    # ── Step 1: Download geophysical grids ──
    log.info("\n[Step 1/6] Downloading geophysical grids...")
    if not getattr(config, "_skip_download", False):
        raster_paths = download_all_layers(
            bbox=config.bbox,
            output_dir=DATA_RAW,
            layers=config.layers,
        )
    else:
        bbox_tag = f"{config.bbox[0]}_{config.bbox[1]}_{config.bbox[2]}_{config.bbox[3]}"
        raster_paths = {k: DATA_RAW / f"{k}_{bbox_tag}.tif" for k in config.layers}
        missing = [k for k, p in raster_paths.items() if not p.exists()]
        if missing:
            log.error(f"Missing rasters (remove --skip-download): {missing}")
            sys.exit(1)
    log.info(f"  Available layers: {list(raster_paths.keys())}")

    # ── Step 2: Load deposit locations ──
    log.info("\n[Step 2/6] Loading deposit locations...")
    if not GDB_PATH.exists():
        log.error(f"Mineral deposits database not found: {GDB_PATH}")
        log.error("Run the download script first (see README)")
        sys.exit(1)

    deposits = get_positive_labels(GDB_PATH, config.commodities, config.bbox)
    log.info(f"  Found {len(deposits)} deposits in region")

    if len(deposits) < 10:
        log.warning(f"Only {len(deposits)} deposits — results may be unreliable")

    # ── Step 3: Feature engineering ──
    log.info("\n[Step 3/6] Engineering features...")
    features, transform, grid_shape = build_feature_stack(raster_paths)
    log.info(f"  Grid shape: {grid_shape}")

    # Plot input layers
    plot_input_layers(features, config.bbox, config.output_dir / "feature_layers.png")

    # Build feature matrix for all valid pixels
    X_all, feature_names, mask = build_feature_matrix(features)
    log.info(f"  Feature names: {feature_names}")

    # Extract features at deposit locations
    X_pos = extract_deposit_features(features, transform, deposits)
    log.info(f"  Positive samples: {X_pos.shape[0]}")

    if X_pos.shape[0] < 5:
        log.error("Too few deposits with valid features. Check data overlap.")
        sys.exit(1)

    # ── Step 4: Spatial cross-validation ──
    cv_results = None
    if not getattr(config, "_skip_cv", False):
        log.info("\n[Step 4/6] Spatial cross-validation...")
        from rasterio.transform import rowcol

        coords = []
        for _, dep in deposits.iterrows():
            r, c = rowcol(transform, dep.geometry.x, dep.geometry.y)
            if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
                coords.append([dep.geometry.x, dep.geometry.y])
        pos_coords = np.array(coords[: X_pos.shape[0]])

        # Sample unlabeled for CV (subset for speed)
        rng = np.random.default_rng(config.random_state)
        n_sample = min(50000, X_all.shape[0])
        sample_idx = rng.choice(X_all.shape[0], size=n_sample, replace=False)
        X_unlabeled_sample = X_all[sample_idx]

        cv_results = spatial_cross_validate(
            X_pos, X_unlabeled_sample, pos_coords,
            n_folds=5, n_bags=20, random_state=config.random_state,
        )
    else:
        log.info("\n[Step 4/6] Skipping cross-validation (--skip-cv)")

    # ── Step 5: Train final model ──
    log.info("\n[Step 5/6] Training final model...")

    rng = np.random.default_rng(config.random_state)
    n_sample = min(100000, X_all.shape[0])
    sample_idx = rng.choice(X_all.shape[0], size=n_sample, replace=False)
    X_unlabeled = X_all[sample_idx]

    base_est = make_base_estimator(
        n_estimators=config.xgb_n_estimators,
        max_depth=config.xgb_max_depth,
        learning_rate=config.xgb_learning_rate,
        random_state=config.random_state,
    )
    estimators = bagging_pu_train(
        X_pos, X_unlabeled,
        base_estimator=base_est,
        n_bags=config.n_estimators,
        unlabeled_ratio=config.unlabeled_ratio,
        random_state=config.random_state,
    )

    # SHAP importance
    log.info("  Computing SHAP importance...")
    importance, shap_values = compute_shap_importance(
        estimators[0], X_pos, feature_names
    )
    log.info("  Feature importance:")
    for name, val in importance.items():
        log.info(f"    {name}: {val:.4f}")

    plot_shap_importance(importance, config.output_dir / "shap_importance.png")

    # ── Step 6: Generate prospectivity map ──
    log.info("\n[Step 6/6] Generating prospectivity map...")
    prob_map = predict_map(estimators, X_all, mask, grid_shape)

    valid_probs = prob_map[np.isfinite(prob_map)]
    log.info(f"  Probability range: [{valid_probs.min():.3f}, {valid_probs.max():.3f}]")
    log.info(f"  Mean: {valid_probs.mean():.3f}, Median: {np.median(valid_probs):.3f}")

    high_pct = np.sum(valid_probs > 0.5) / len(valid_probs) * 100
    log.info(f"  Area with P > 0.5: {high_pct:.1f}%")

    # Save outputs
    plot_prospectivity(
        prob_map, config.bbox, deposits,
        title=f"Prospectivity: {', '.join(config.commodities).title()} — {config.region_name.title()}",
        output_path=config.output_dir / "prospectivity_map.png",
    )

    export_geotiff(
        prob_map, transform, config.output_crs,
        config.output_dir / "prospectivity.tif",
    )

    # Interactive map
    generate_interactive_map(
        prob_map, config.bbox, deposits,
        config.output_dir / "interactive_map.html",
    )

    # Concentration-area curve
    ca_fig, ca_metrics = plot_concentration_area(
        prob_map, deposits, config.bbox, transform,
        config.output_dir / "concentration_area.png",
    )

    # HTML report
    config_summary = {
        "region": config.region_name,
        "bbox": str(config.bbox),
        "commodities": ", ".join(config.commodities),
        "resolution": f"{config.resolution}° (~{config.resolution * 111:.0f}km)",
        "n_deposits": X_pos.shape[0],
        "n_features": len(feature_names),
        "n_bags": config.n_estimators,
        "grid_shape": str(grid_shape),
        "high_probability_area": f"{high_pct:.1f}%",
    }

    report_path = generate_html_report(
        config.output_dir,
        config_summary,
        cv_results=cv_results,
        importance=importance,
        ca_metrics=ca_metrics,
    )

    elapsed = time.time() - t0
    log.info("\n" + "=" * 60)
    log.info(f"Pipeline complete in {elapsed:.0f}s")
    log.info(f"Output directory: {config.output_dir}")
    log.info(f"  prospectivity.tif   — GeoTIFF for GIS")
    log.info(f"  prospectivity_map.png — static map")
    log.info(f"  shap_importance.png — feature importance")
    log.info(f"  concentration_area.png — C-A curve")
    log.info(f"  interactive_map.html — interactive browser map")
    log.info(f"  report.html         — full report")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
