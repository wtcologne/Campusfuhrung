import math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import streamlit as st
import plotly.graph_objects as go

@dataclass
class PointSource:
    position: Tuple[float, float]
    intensity: float = 1.0

@dataclass
class ExtendedSource:
    center: Tuple[float, float]
    radius: float
    samples: int = 32
    intensity: float = 1.0

@dataclass
class Obstacle:
    x: float
    width: float
    height: float
    bottom: float = 0.0


def _ray_hit_obstacle(p0: np.ndarray, p1: np.ndarray, obstacle: Obstacle) -> bool:
    # Segment p0->p1 intersects rectangle [x, x+width] x [bottom, bottom+height]
    x1, y1 = p0
    x2, y2 = p1
    xmin, xmax = obstacle.x, obstacle.x + obstacle.width
    ymin, ymax = obstacle.bottom, obstacle.bottom + obstacle.height

    # Liang-Barsky algorithm for line clipping
    dx = x2 - x1
    dy = y2 - y1

    p = np.array([-dx, dx, -dy, dy], dtype=float)
    q = np.array([x1 - xmin, xmax - x1, y1 - ymin, ymax - y1], dtype=float)

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
        else:
            t = -qi / pi
            if pi < 0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)
            if u1 > u2:
                return False
    return True


def _sample_extended_source(src: ExtendedSource) -> List[PointSource]:
    cx, cy = src.center
    angles = np.linspace(0, 2 * math.pi, src.samples, endpoint=False)
    points = [
        PointSource((cx + src.radius * math.cos(a), cy + src.radius * math.sin(a)), src.intensity / src.samples)
        for a in angles
    ]
    return points


def _compute_intensity_grid(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    point_sources: List[PointSource],
    obstacle: Obstacle,
    falloff: float = 2.0,
) -> np.ndarray:
    h, w = grid_y.shape
    intensity = np.zeros((h, w), dtype=float)
    points = np.stack([grid_x, grid_y], axis=-1)

    for src in point_sources:
        src_pos = np.array(src.position, dtype=float)
        to_points = points - src_pos
        distances = np.linalg.norm(to_points, axis=-1)
        directions = to_points / np.maximum(distances[..., None], 1e-6)
        targets = src_pos + directions * distances[..., None]

        # Check visibility to each point
        visible = np.ones((h, w), dtype=bool)
        # Vectorized obstacle check by sampling along the ray (coarse for speed)
        num_steps = 20
        ts = np.linspace(0.0, 1.0, num_steps)[None, None, :]
        seg_points = src_pos + (targets - src_pos)[..., None, :] * ts
        xmin, xmax = obstacle.x, obstacle.x + obstacle.width
        ymin, ymax = obstacle.bottom, obstacle.bottom + obstacle.height
        within_x = (seg_points[..., 0] >= xmin) & (seg_points[..., 0] <= xmax)
        within_y = (seg_points[..., 1] >= ymin) & (seg_points[..., 1] <= ymax)
        hits = (within_x & within_y).any(axis=-1)
        visible &= ~hits

        # Inverse power falloff
        intensity += visible * (src.intensity / np.maximum(distances**falloff, 1e-3))

    # Normalize for display
    intensity = intensity / (np.max(intensity) + 1e-6)
    return intensity


def light_shadow_app():
    st.sidebar.subheader("Lichtquellen")
    source_mode = st.sidebar.selectbox(
        "Quelle", ["Punktquelle", "Zwei Punktquellen", "Ausgedehnte Quelle"], index=0
    )

    st.sidebar.subheader("Hindernis")
    obs_x = st.sidebar.slider("x-Position", -2.0, 2.0, 0.2, step=0.01)
    obs_w = st.sidebar.slider("Breite", 0.05, 1.5, 0.4, step=0.01)
    obs_h = st.sidebar.slider("Höhe", 0.05, 2.5, 1.0, step=0.01)

    st.sidebar.subheader("Darstellung")
    falloff = st.sidebar.select_slider("Lichtabfall (r^-n)", options=[1.0, 1.5, 2.0, 3.0], value=2.0)
    grid_res = st.sidebar.select_slider("Auflösung", options=[128, 192, 256, 320], value=256)

    # Touch-friendly layout
    col_src, col_canvas = st.columns([1, 3])

    with col_src:
        st.markdown("### Szene")
        sx = st.slider("Quelle x", -3.0, 3.0, -1.5, step=0.01)
        sy = st.slider("Quelle y", -1.5, 1.5, 0.0, step=0.01)
        if source_mode == "Zwei Punktquellen":
            dx = st.slider("Abstand Δx", 0.0, 2.0, 0.6, step=0.01)
        elif source_mode == "Ausgedehnte Quelle":
            rad = st.slider("Quellenradius", 0.05, 1.0, 0.3, step=0.01)

        st.caption("Ziehe Regler – optimiert für Touch.")

    with col_canvas:
        # Coordinate system
        xlim = (-3.5, 3.5)
        ylim = (-2.0, 2.0)
        xs = np.linspace(xlim[0], xlim[1], grid_res)
        ys = np.linspace(ylim[0], ylim[1], grid_res)
        grid_x, grid_y = np.meshgrid(xs, ys)

        obstacle = Obstacle(x=obs_x, width=obs_w, height=obs_h)

        # Build sources
        sources: List[PointSource] = []
        if source_mode == "Punktquelle":
            sources = [PointSource((sx, sy), 1.0)]
        elif source_mode == "Zwei Punktquellen":
            sources = [PointSource((sx - dx/2, sy), 1.0), PointSource((sx + dx/2, sy), 1.0)]
        else:
            ext = ExtendedSource(center=(sx, sy), radius=rad, samples=32, intensity=1.0)
            sources = _sample_extended_source(ext)

        intensity = _compute_intensity_grid(grid_x, grid_y, sources, obstacle, float(falloff))

        # Render with Plotly Heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=intensity,
                x=xs,
                y=ys,
                colorscale="gray",
                showscale=False,
                zsmooth="best",
            )
        )

        # Draw obstacle
        rect_x = [obstacle.x, obstacle.x + obstacle.width, obstacle.x + obstacle.width, obstacle.x, obstacle.x]
        rect_y = [obstacle.bottom, obstacle.bottom, obstacle.bottom + obstacle.height, obstacle.bottom + obstacle.height, obstacle.bottom]
        fig.add_trace(go.Scatter(x=rect_x, y=rect_y, mode="lines", line=dict(color="royalblue", width=3)))

        # Draw source markers (approx)
        if source_mode == "Punktquelle":
            fig.add_trace(go.Scatter(x=[sx], y=[sy], mode="markers", marker=dict(size=12, color="gold")))
        elif source_mode == "Zwei Punktquellen":
            fig.add_trace(go.Scatter(x=[sx - dx/2, sx + dx/2], y=[sy, sy], mode="markers", marker=dict(size=10, color="gold")))
        else:
            circle_t = np.linspace(0, 2*np.pi, 60)
            fig.add_trace(go.Scatter(
                x=sx + rad*np.cos(circle_t),
                y=sy + rad*np.sin(circle_t),
                mode="lines",
                line=dict(color="gold", width=2)
            ))

        fig.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1, range=xlim, showgrid=False, zeroline=False, visible=False),
            yaxis=dict(range=ylim, showgrid=False, zeroline=False, visible=False),
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#f8fafc",
            plot_bgcolor="#f8fafc",
            height=600,
            dragmode=False,
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
