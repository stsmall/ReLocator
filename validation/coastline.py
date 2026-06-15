"""Coastline-projected error metric for the barnacle test set.

Barnacles are intertidal — predictions inland of the coast are biologically
meaningless and the standard great-circle haversine error overstates the
real prediction error for any prediction that lands in the right latitude
band but offshore or inland. This module models the coast as a piecewise-
linear polyline through the 12 sample sites (N→S) and projects each
prediction onto that polyline; the meaningful error is the along-polyline
distance from the projection to the true site.
"""

from __future__ import annotations

import numpy as np

from validation.common import haversine

# 12 sample sites in N→S order, from test_data/samples_locations.tsv.
# Format: (site_code, lat, lon).
BARNACLE_COAST: list[tuple[str, float, float]] = [
    ("01-AFB-AK", 60.1210, -149.3701),
    ("02-DIG-BC", 54.2833, -130.4167),
    ("03-FHL-WA", 48.5454, -123.0133),
    ("04-RIC-WA", 47.7635, -122.3862),
    ("05-WES-WA", 46.9123, -124.1103),
    ("06-MEA-OR", 45.4859, -123.9746),
    ("07-CPE-OR", 44.2804, -124.1119),
    ("08-BOB-OR", 44.2438, -124.1141),
    ("09-OMB-OR", 43.3447, -124.3224),
    ("10-PAR-CA", 38.9557, -123.7414),
    ("11-HOP-CA", 36.6002, -121.8947),
    ("12-GOL-CA", 34.4170, -119.8320),
]


def build_coast_index(sites=BARNACLE_COAST):
    """Return (cum_km, points).

    cum_km[i] = total polyline distance (km) from sites[0] to sites[i],
    measured as the sum of great-circle distances along consecutive segments.

    points: ndarray of shape (n, 2), rows are (lat, lon).
    """
    if not sites:
        raise ValueError("sites must be non-empty")
    pts = np.array([[lat, lon] for _, lat, lon in sites], dtype=np.float64)
    cum = np.zeros(len(pts), dtype=np.float64)
    for i in range(1, len(pts)):
        cum[i] = cum[i - 1] + float(
            haversine(pts[i - 1, 0], pts[i - 1, 1], pts[i, 0], pts[i, 1])
        )
    return cum, pts


def project_to_coast(pred_lat, pred_lon, cum_km, points):
    """Project (pred_lat, pred_lon) onto the coast polyline.

    Returns (coast_pos_km, offshore_km).

    coast_pos_km: along-polyline km position of the projected point (0 at
    the first site, cum_km[-1] at the last).
    offshore_km: great-circle distance from the prediction to its
    projection on the polyline.

    Uses planar (lat-lon-as-Cartesian) projection per segment — accurate
    enough for prediction errors of <1000 km.
    """
    best_d = float("inf")
    best_pos = 0.0
    for i in range(len(points) - 1):
        a_lat, a_lon = points[i]
        b_lat, b_lon = points[i + 1]
        ab_lat = b_lat - a_lat
        ab_lon = b_lon - a_lon
        ap_lat = pred_lat - a_lat
        ap_lon = pred_lon - a_lon
        denom = ab_lat * ab_lat + ab_lon * ab_lon
        if denom == 0:
            t = 0.0
        else:
            t = (ap_lat * ab_lat + ap_lon * ab_lon) / denom
            t = max(0.0, min(1.0, t))
        proj_lat = a_lat + t * ab_lat
        proj_lon = a_lon + t * ab_lon
        d = float(haversine(pred_lat, pred_lon, proj_lat, proj_lon))
        if d < best_d:
            best_d = d
            seg_len = float(haversine(a_lat, a_lon, b_lat, b_lon))
            best_pos = float(cum_km[i] + t * seg_len)
    return best_pos, best_d


def along_coast_distance(pred_lat, pred_lon, true_site_idx, cum_km, points):
    """|coast_pos_km - cum_km[true_site_idx]| in km."""
    coast_pos, _ = project_to_coast(pred_lat, pred_lon, cum_km, points)
    return float(abs(coast_pos - cum_km[true_site_idx]))
