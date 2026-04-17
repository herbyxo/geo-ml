"""
Prospectivity map visualisation and report generation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import Affine

log = logging.getLogger(__name__)


def plot_prospectivity(
    prob_map: np.ndarray,
    bbox: tuple[float, float, float, float],
    deposits: gpd.GeoDataFrame | None = None,
    title: str = "Mineral Prospectivity Map",
    output_path: Path | None = None,
    cmap: str = "hot_r",
) -> plt.Figure:
    """Plot prospectivity probability map with deposit overlays."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    extent = [bbox[0], bbox[2], bbox[1], bbox[3]]
    im = ax.imshow(prob_map, cmap=cmap, vmin=0, vmax=1, extent=extent, origin="upper")
    plt.colorbar(im, ax=ax, label="Prospectivity probability", shrink=0.8)

    if deposits is not None and len(deposits) > 0:
        local = deposits.cx[bbox[0] : bbox[2], bbox[1] : bbox[3]]
        if len(local) > 0:
            ax.scatter(
                local.geometry.x, local.geometry.y,
                c="cyan", s=40, marker="^", edgecolors="black",
                linewidths=0.5, zorder=5, label=f"Known deposits (n={len(local)})",
            )
            ax.legend(loc="upper right")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        log.info(f"Saved map: {output_path}")

    plt.close(fig)
    return fig


def plot_input_layers(
    features: dict[str, np.ndarray],
    bbox: tuple[float, float, float, float],
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot all input feature layers as a grid of subplots."""
    names = sorted(features.keys())
    n = len(names)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_2d(axes)
    extent = [bbox[0], bbox[2], bbox[1], bbox[3]]

    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        arr = features[name]
        vmin, vmax = np.nanpercentile(arr[np.isfinite(arr)], [2, 98])
        ax.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax, extent=extent, origin="upper")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)

    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_visible(False)

    plt.suptitle("Input Feature Layers", fontsize=14)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved feature overview: {output_path}")

    plt.close(fig)
    return fig


def plot_shap_importance(
    importance: dict[str, float],
    output_path: Path | None = None,
) -> plt.Figure:
    """Bar chart of mean |SHAP| feature importance."""
    names = list(importance.keys())
    values = list(importance.values())

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.4)))
    y_pos = range(len(names))
    ax.barh(y_pos, values, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (SHAP)")
    ax.invert_yaxis()

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved SHAP importance: {output_path}")

    plt.close(fig)
    return fig


def plot_concentration_area(
    prob_map: np.ndarray,
    deposits: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float],
    transform: Affine,
    output_path: Path | None = None,
) -> tuple[plt.Figure, dict]:
    """
    Concentration-area (C-A) curve: what % of area is needed to capture
    what % of known deposits. Key metric for prospectivity map quality.
    """
    from rasterio.transform import rowcol

    valid_probs = prob_map[np.isfinite(prob_map)]
    thresholds = np.linspace(0, 1, 100)

    local = deposits.cx[bbox[0] : bbox[2], bbox[1] : bbox[3]]
    deposit_probs = []
    for _, dep in local.iterrows():
        r, c = rowcol(transform, dep.geometry.x, dep.geometry.y)
        if 0 <= r < prob_map.shape[0] and 0 <= c < prob_map.shape[1]:
            p = prob_map[r, c]
            if np.isfinite(p):
                deposit_probs.append(p)
    deposit_probs = np.array(deposit_probs)

    pct_area = []
    pct_deposits = []
    for t in thresholds:
        area_frac = np.sum(valid_probs >= t) / len(valid_probs)
        dep_frac = np.sum(deposit_probs >= t) / len(deposit_probs) if len(deposit_probs) > 0 else 0
        pct_area.append(area_frac * 100)
        pct_deposits.append(dep_frac * 100)

    # Key metrics
    metrics = {}
    for target_pct in [50, 70, 90]:
        for i, pd_val in enumerate(pct_deposits):
            if pd_val < target_pct:
                metrics[f"area_for_{target_pct}pct_deposits"] = pct_area[max(0, i - 1)]
                break

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pct_area, pct_deposits, "b-", linewidth=2)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="Random baseline")
    ax.set_xlabel("% of total area (predicted prospective)")
    ax.set_ylabel("% of known deposits captured")
    ax.set_title("Concentration-Area Curve")
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)

    for key, val in metrics.items():
        pct = key.split("_")[2].replace("pct", "")
        ax.axhline(y=int(pct), color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved C-A curve: {output_path}")

    plt.close(fig)
    return fig, metrics


def export_geotiff(
    prob_map: np.ndarray,
    transform: Affine,
    crs: str,
    output_path: Path,
) -> None:
    """Save prospectivity map as GeoTIFF for GIS use."""
    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=prob_map.shape[0], width=prob_map.shape[1],
        count=1, dtype=np.float32, crs=crs, transform=transform,
    ) as dst:
        dst.write(prob_map.astype(np.float32), 1)
    log.info(f"Saved GeoTIFF: {output_path}")


def generate_html_report(
    output_dir: Path,
    config_summary: dict,
    cv_results: dict | None = None,
    importance: dict[str, float] | None = None,
    ca_metrics: dict | None = None,
) -> Path:
    """Generate a standalone HTML report with embedded images and stats."""
    import base64

    def img_to_b64(path: Path) -> str:
        if path.exists():
            data = path.read_bytes()
            return base64.b64encode(data).decode()
        return ""

    prospectivity_img = img_to_b64(output_dir / "prospectivity_map.png")
    shap_img = img_to_b64(output_dir / "shap_importance.png")
    ca_img = img_to_b64(output_dir / "concentration_area.png")
    features_img = img_to_b64(output_dir / "feature_layers.png")

    auc_str = ""
    if cv_results:
        auc_str = f"{cv_results['mean_auc']:.3f} &pm; {cv_results['std_auc']:.3f}"

    importance_rows = ""
    if importance:
        for name, val in importance.items():
            importance_rows += f"<tr><td>{name}</td><td>{val:.4f}</td></tr>\n"

    ca_rows = ""
    if ca_metrics:
        for key, val in ca_metrics.items():
            label = key.replace("_", " ").title()
            ca_rows += f"<tr><td>{label}</td><td>{val:.1f}%</td></tr>\n"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Mineral Prospectivity Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 1200px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }}
    h1 {{ color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 10px; }}
    h2 {{ color: #16213e; margin-top: 40px; }}
    .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                  gap: 15px; margin: 20px 0; }}
    .stat-card {{ background: white; padding: 20px; border-radius: 8px;
                  box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    .stat-card .value {{ font-size: 24px; font-weight: bold; color: #e94560; }}
    .stat-card .label {{ font-size: 13px; color: #666; margin-top: 5px; }}
    img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); margin: 15px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
    th, td {{ padding: 10px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
    th {{ background: #16213e; color: white; }}
    tr:nth-child(even) {{ background: #f5f5f5; }}
    .config {{ background: #f0f0f0; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 13px; }}
</style></head><body>
<h1>Mineral Prospectivity Report</h1>
<p>Generated by <strong>geo-ml</strong> — ML-based mineral prospectivity mapping</p>

<div class="stat-grid">
    <div class="stat-card">
        <div class="value">{config_summary.get('region', 'N/A')}</div>
        <div class="label">Region</div>
    </div>
    <div class="stat-card">
        <div class="value">{config_summary.get('n_deposits', 'N/A')}</div>
        <div class="label">Known deposits</div>
    </div>
    <div class="stat-card">
        <div class="value">{config_summary.get('n_features', 'N/A')}</div>
        <div class="label">Features</div>
    </div>
    <div class="stat-card">
        <div class="value">{auc_str or 'N/A'}</div>
        <div class="label">Spatial CV AUC</div>
    </div>
</div>

<h2>Prospectivity Map</h2>
{"<img src='data:image/png;base64," + prospectivity_img + "'/>" if prospectivity_img else "<p>Not generated</p>"}

<h2>Feature Importance (SHAP)</h2>
{"<img src='data:image/png;base64," + shap_img + "'/>" if shap_img else "<p>Not generated</p>"}
{('<table><tr><th>Feature</th><th>Mean |SHAP|</th></tr>' + importance_rows + '</table>') if importance_rows else ''}

<h2>Concentration-Area Curve</h2>
{"<img src='data:image/png;base64," + ca_img + "'/>" if ca_img else "<p>Not generated</p>"}
{('<table><tr><th>Metric</th><th>Value</th></tr>' + ca_rows + '</table>') if ca_rows else ''}

<h2>Input Feature Layers</h2>
{"<img src='data:image/png;base64," + features_img + "'/>" if features_img else "<p>Not generated</p>"}

<h2>Configuration</h2>
<div class="config"><pre>{_dict_to_str(config_summary)}</pre></div>

</body></html>"""

    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    log.info(f"Generated report: {report_path}")
    return report_path


def generate_interactive_map(
    prob_map: np.ndarray,
    bbox: tuple[float, float, float, float],
    deposits: gpd.GeoDataFrame | None = None,
    output_path: Path | None = None,
) -> Path | None:
    """
    Generate an interactive HTML map with prospectivity overlay using folium.
    """
    import folium
    from folium.raster_layers import ImageOverlay

    center_lat = (bbox[1] + bbox[3]) / 2
    center_lon = (bbox[0] + bbox[2]) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles="CartoDB positron",
    )

    # Prospectivity overlay
    import matplotlib.cm as cm
    colored = cm.hot_r(prob_map)
    colored[..., 3] = np.where(np.isfinite(prob_map), 0.6, 0.0)  # transparency
    colored = (colored * 255).astype(np.uint8)

    from PIL import Image
    import io
    import base64

    img = Image.fromarray(colored, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()

    bounds = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]]
    ImageOverlay(
        image=f"data:image/png;base64,{img_b64}",
        bounds=bounds,
        opacity=0.7,
        name="Prospectivity",
    ).add_to(m)

    # Deposit markers
    if deposits is not None and len(deposits) > 0:
        local = deposits.cx[bbox[0] : bbox[2], bbox[1] : bbox[3]]
        deposit_group = folium.FeatureGroup(name="Known Deposits")
        for _, row in local.iterrows():
            name = row.get("NAME", "Deposit")
            comm = row.get("COMMODID", "")
            status = row.get("OPERATING_STATUS", "")
            popup = f"<b>{name}</b><br>{comm}<br>{status}"
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=6,
                color="cyan",
                fill=True,
                fill_color="cyan",
                fill_opacity=0.9,
                popup=folium.Popup(popup, max_width=200),
            ).add_to(deposit_group)
        deposit_group.add_to(m)

    folium.LayerControl().add_to(m)

    if output_path:
        m.save(str(output_path))
        log.info(f"Saved interactive map: {output_path}")
        return output_path
    return None


def _dict_to_str(d: dict, indent: int = 0) -> str:
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{'  ' * indent}{k}:")
            lines.append(_dict_to_str(v, indent + 1))
        else:
            lines.append(f"{'  ' * indent}{k}: {v}")
    return "\n".join(lines)
