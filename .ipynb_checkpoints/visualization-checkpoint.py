"""
visualization.py
================
Generación de figuras para la bicapa lipídica.

Cada función plot_*() recibe un BicapaCryoET ya construido, genera
la figura y la guarda. No devuelve objetos de matplotlib para evitar
memory leaks en generación masiva de datasets.

FIG_DPI controla la resolución de salida (300 = publicación,
150 = previsualización rápida).

Referencias visualizadas:
  [7]  Piggot 2017 – S_CH
  [9]  Chaisson 2025 – interdigitación
  [14] Dubochet 1988 – contraste cryo-ET
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator

import analysis
from lipid_types import LIPID_TYPES

if TYPE_CHECKING:
    from builder import BicapaCryoET
    from geometry import LipidInstance


FIG_DPI = 300


CMAP_THICK = LinearSegmentedColormap.from_list(
    "thick",
    ["#053061", "#2166ac", "#92c5de", "#f7f7f7",
     "#f4a582", "#d6604d", "#67001f"],
)
CMAP_RAFT = LinearSegmentedColormap.from_list(
    "raft", ["#0d1117", "#0f3460", "#533483", "#e94560", "#ff6b6b"],
)
CMAP_ORDER = LinearSegmentedColormap.from_list(
    "order",
    ["#f7f7f7", "#d9f0d3", "#5aae61", "#1b7837", "#00441b"],
)
CMAP_INTERDIG = LinearSegmentedColormap.from_list(
    "interdig",
    ["#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#67001f"],
)

PLT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.color": "#e8e8e8",
    "grid.linewidth": 0.5,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.labelcolor": "#333333",
    "text.color": "#1a1a1a",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
}


def _save(membrane: "BicapaCryoET", fig, filename: str):
    path = os.path.join(membrane.seed_dir(), filename)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  -> %s" % filename)


def _cb(fig, im, ax, label: str, fmt: str = "%.1f"):
    cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.02, format=fmt)
    cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def _decorate(ax, title, subtitle="", xlabel="X (nm)", ylabel="Y (nm)"):
    ax.set_title(
        title + (("\n%s" % subtitle) if subtitle else ""),
        fontsize=9, fontweight="bold", pad=5,
    )
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def _draw_rafts(ax, rafts: List[List["LipidInstance"]], color: str = "#e07b00"):
    for raft in rafts:
        cx = np.mean([l.head_pos[0] for l in raft]) / 10
        cy = np.mean([l.head_pos[1] for l in raft]) / 10
        r = (
            np.std([l.head_pos[0] for l in raft])
            + np.std([l.head_pos[1] for l in raft])
        ) / 20 + 0.8
        ax.add_patch(plt.Circle(
            (cx, cy), r, color=color, fill=False, lw=1.4, ls="--", alpha=0.85
        ))
        ax.text(cx, cy, "⬡", fontsize=8, ha="center",
                va="center", color=color, alpha=0.8)


def plot_cryoET(membrane: "BicapaCryoET"):
    """Fig 1: proyección XY + sección XZ."""
    ext = [0, membrane.Lx / 10, 0, membrane.Ly / 10]
    g = membrane.geometry
    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(
            1, 2, figsize=(18, 8),
            gridspec_kw={"width_ratios": [1, 1.6]},
        )
        fig.suptitle(
            "Bicapa lipidica — Membrana plasmatica (seed=%d)\n"
            "%.0fx%.0f nm | Grosor %.1f A | kc=%.0f kBT nm2"
            % (membrane.seed, membrane.Lx / 10, membrane.Ly / 10,
               g.total_thick, membrane.bending_modulus),
            fontsize=11, fontweight="bold",
        )
        D = (analysis.density_map(membrane, membrane.outer_leaflet)
             + analysis.density_map(membrane, membrane.inner_leaflet))
        im = axes[0].imshow(D.T, origin="lower", extent=ext,
                            cmap="gray", aspect="equal")
        _cb(fig, im, axes[0], "Da/A2")
        _decorate(axes[0], "Proyeccion XY", "(simula cryo-ET)")

        Hxz, xe, ze = analysis.xz_projection(membrane, bx=280, bz=140)
        im2 = axes[1].imshow(Hxz.T, origin="lower",
                             extent=[xe[0], xe[-1], ze[0], ze[-1]],
                             cmap="gray", aspect="auto")
        _cb(fig, im2, axes[1], "Densidad (u.a.)")
        mid_z = (g.z_outer + g.z_inner) / 20
        for zl, col, lbl in [
            (g.z_outer / 10, "#2dc653", "Cabezas sup"),
            (g.z_inner / 10, "#e63946", "Cabezas inf"),
            (0.0, "#3a86ff", "Plano medio"),
        ]:
            axes[1].axhline(zl, color=col, lw=1.0, ls="--",
                            alpha=0.85, label=lbl)
        x0a = xe[-1] * 0.03
        axes[1].annotate(
            "", xy=(x0a, g.z_outer / 10), xytext=(x0a, g.z_inner / 10),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.2),
        )
        axes[1].text(x0a + xe[-1] * 0.015, mid_z, "%.0f A" % g.total_thick,
                     fontsize=9, va="center", fontweight="bold")
        axes[1].legend(fontsize=8, loc="upper right")
        _decorate(axes[1], "Seccion XZ",
                  "Banda densa sup + nucleo claro + banda densa inf")
        fig.tight_layout()
        _save(membrane, fig, "fig1_cryoET.png")


def plot_domains(membrane: "BicapaCryoET"):
    """Fig 2: balsas lipídicas, grosor local y rugosidad."""
    ext = [0, membrane.Lx / 10, 0, membrane.Ly / 10]
    g = membrane.geometry
    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(
            "Dominios lipidicos (seed=%d)" % membrane.seed,
            fontsize=12, fontweight="bold",
        )
        for ax, lips, rafts, titulo, sub in [
            (axes[0, 0], membrane.outer_leaflet, membrane.rafts_outer,
             "Balsas externas", "SM/CHOL/GM1"),
            (axes[0, 1], membrane.inner_leaflet, membrane.rafts_inner,
             "Balsas internas", "PS/PI enriquecido"),
        ]:
            im = ax.imshow(
                analysis.raft_fraction_map(membrane, lips).T,
                origin="lower", extent=ext, cmap=CMAP_RAFT,
                aspect="equal", vmin=0, vmax=1,
            )
            _cb(fig, im, ax, "Fraccion raft")
            _draw_rafts(ax, rafts)
            _decorate(ax, titulo, sub)

        G = analysis.thickness_map(membrane)
        im3 = axes[1, 0].imshow(G.T, origin="lower", extent=ext,
                                 cmap=CMAP_THICK, aspect="equal")
        _cb(fig, im3, axes[1, 0], "Grosor (A)")
        _draw_rafts(axes[1, 0], membrane.rafts_outer, color="#888888")
        axes[1, 0].text(
            0.02, 0.04,
            "mu=%.1f A | hidro=%.1f A" % (G.mean(), g.hydro_thick),
            transform=axes[1, 0].transAxes, fontsize=7.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="#cccccc", alpha=0.9),
        )
        _decorate(axes[1, 0], "Grosor local", "Zonas raft = mas gruesas")

        R = analysis.roughness_map(membrane, membrane.outer_leaflet)
        im4 = axes[1, 1].imshow(R.T, origin="lower", extent=ext,
                                 cmap="YlOrRd", aspect="equal")
        _cb(fig, im4, axes[1, 1], "sigma_z (A)")
        axes[1, 1].text(
            0.02, 0.96, "Fluido ~1.8 A | Gel ~0.6 A",
            transform=axes[1, 1].transAxes, fontsize=7.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="#cccccc", alpha=0.9),
        )
        _decorate(axes[1, 1], "Rugosidad externa", "Fluctuaciones termicas")
        fig.tight_layout()
        _save(membrane, fig, "fig2_dominios.png")


def plot_pips(membrane: "BicapaCryoET"):
    """Fig 3: densidad de fosfoinosítidos."""
    pip_types = ["PI", "PI3P", "PI4P", "PI5P", "PI34P2", "PIP2", "PIP3"]
    cnt_pip = {
        t: sum(1 for l in membrane.inner_leaflet
               if l.lipid_type == LIPID_TYPES[t])
        for t in pip_types
    }
    cnt_pip = {k: v for k, v in cnt_pip.items() if v > 0}
    if not cnt_pip:
        print("  Sin PIPs en monocapa interna")
        return

    ext = [0, membrane.Lx / 10, 0, membrane.Ly / 10]
    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("Fosfoinositoles (seed=%d)" % membrane.seed,
                     fontsize=12, fontweight="bold")

        im = axes[0].imshow(analysis.pip_density_map(membrane).T,
                            origin="lower", extent=ext,
                            cmap="magma", aspect="equal")
        _cb(fig, im, axes[0], "Da/A2 (PIPs)")
        for pc in membrane.pip_clusters:
            cx = np.mean([l.head_pos[0] for l in pc]) / 10
            cy = np.mean([l.head_pos[1] for l in pc]) / 10
            r = (np.std([l.head_pos[0] for l in pc])
                 + np.std([l.head_pos[1] for l in pc])) / 20 + 0.6
            axes[0].add_patch(plt.Circle(
                (cx, cy), r, color="black", fill=False, lw=1.5, ls=":"
            ))
        _decorate(axes[0], "Densidad PIPs", "Circulos = clusters")

        tipos_ = sorted(cnt_pip, key=lambda t: LIPID_TYPES[t].pip_order)
        bars = axes[1].bar(
            [LIPID_TYPES[t].name for t in tipos_],
            [cnt_pip[t] for t in tipos_],
            color=[LIPID_TYPES[t].color for t in tipos_],
            edgecolor="#333333", lw=0.8,
        )
        axes[1].bar_label(bars, fontsize=9, padding=3)
        axes[1].set_xlabel("Especie PIP")
        axes[1].set_ylabel("N lipidos")
        axes[1].tick_params(axis="x", rotation=35)
        axes[1].set_title("Recuento PIPs (monocapa interna)",
                          fontsize=9, fontweight="bold")
        fig.tight_layout()
        _save(membrane, fig, "fig3_pips.png")


def plot_composition(membrane: "BicapaCryoET"):
    """Fig 4: composición lipídica y perfil Z."""
    import matplotlib.gridspec as mgrid
    g = membrane.geometry
    ext = [0, membrane.Lx / 10, 0, membrane.Ly / 10]
    with plt.rc_context(PLT_STYLE):
        fig = plt.figure(figsize=(20, 7))
        fig.suptitle("Composicion y perfil Z (seed=%d)" % membrane.seed,
                     fontsize=12, fontweight="bold")
        gs = mgrid.GridSpec(1, 3, wspace=0.38,
                            left=0.06, right=0.97, top=0.88, bottom=0.14)

        ax = fig.add_subplot(gs[0])
        todos = membrane.outer_leaflet + membrane.inner_leaflet
        tipos = sorted({l.lipid_type for l in todos}, key=lambda x: x.name)
        n_s, n_i = len(membrane.outer_leaflet), len(membrane.inner_leaflet)
        cs = {lt: sum(1 for l in membrane.outer_leaflet if l.lipid_type == lt) / n_s * 100 for lt in tipos}
        ci = {lt: sum(1 for l in membrane.inner_leaflet if l.lipid_type == lt) / n_i * 100 for lt in tipos}
        x_pos = np.arange(len(tipos))
        w = 0.38
        ax.bar(x_pos - w / 2, [cs[lt] for lt in tipos], w,
               color=[lt.color for lt in tipos],
               alpha=0.95, edgecolor="#333333", lw=0.5, label="Superior")
        ax.bar(x_pos + w / 2, [ci[lt] for lt in tipos], w,
               color=[lt.color for lt in tipos],
               alpha=0.55, edgecolor="#333333", lw=0.5,
               hatch="//", label="Inferior")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([lt.name for lt in tipos], rotation=40, ha="right")
        ax.set_ylabel("% en monocapa")
        ax.legend(fontsize=8)
        ax.set_title("Composicion lipidica\nSuperior vs Inferior",
                     fontsize=10, fontweight="bold")

        ax2 = fig.add_subplot(gs[1])
        zc, hz = analysis.z_profile(membrane)
        hn = hz / hz.max()
        ax2.fill_betweenx(zc / 10, 0, hn, alpha=0.40, color="#3a86ff")
        ax2.plot(hn, zc / 10, color="#1a5fbf", lw=1.8)
        for zl, col, lbl in [
            (g.z_outer / 10, "#2dc653", "Cabezas sup (%.0f A)" % g.z_outer),
            (g.z_inner / 10, "#e63946", "Cabezas inf (%.0f A)" % g.z_inner),
        ]:
            ax2.axhline(zl, color=col, lw=1.1, ls="--", label=lbl)
        ax2.axhspan(g.z_inner / 10, g.z_outer / 10, color="#eeee88", alpha=0.3)
        ax2.legend(fontsize=7.5, loc="upper left")
        from matplotlib.ticker import AutoMinorLocator as AML
        ax2.xaxis.set_minor_locator(AML())
        ax2.yaxis.set_minor_locator(AML())
        _decorate(ax2, "Perfil densidad Z", "Patron pico-valle-pico",
                  "Densidad norm.", "Z (nm)")

        ax3 = fig.add_subplot(gs[2])
        Asim = (analysis.density_map(membrane, membrane.outer_leaflet, sigma=2.0)
                - analysis.density_map(membrane, membrane.inner_leaflet, sigma=2.0))
        vabs = np.percentile(np.abs(Asim), 98)
        im3 = ax3.imshow(Asim.T, origin="lower", extent=ext,
                         cmap="RdBu_r", vmin=-vabs, vmax=vabs)
        _cb(fig, im3, ax3, "rho_sup - rho_inf")
        _decorate(ax3, "Asimetria sup - inf",
                  "Rojo = ext mayor | Azul = int mayor")
        _save(membrane, fig, "fig4_composicion.png")


def plot_order_chains(membrane: "BicapaCryoET"):
    """Fig 6: S_CH [7,8] e interdigitación [9]."""
    ext = [0, membrane.Lx / 10, 0, membrane.Ly / 10]
    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(
            "Orden de cadenas acil e interdigitacion (seed=%d)\n"
            "Piggot et al. JCTC 2017 [7] . "
            "Chaisson, Heberle, Doktorova JCIM 2025 [9]"
            % membrane.seed,
            fontsize=11, fontweight="bold",
        )
        S = analysis.order_parameter_map(membrane)
        im = axes[0].imshow(S.T, origin="lower", extent=ext,
                            cmap=CMAP_ORDER, aspect="equal")
        _cb(fig, im, axes[0], "S_CH (orden)")
        _draw_rafts(axes[0], membrane.rafts_outer)
        axes[0].text(
            0.02, 0.04, "Lo > Ld\nBalsas = mayor orden",
            transform=axes[0].transAxes, fontsize=7.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="#cccccc", alpha=0.9),
        )
        _decorate(axes[0], "Parametro de orden S_CH",
                  "<(3cos2theta-1)/2> por celda")

        ID = analysis.interdigitation_map(membrane)
        vmax_id = max(ID.mean() + 2 * ID.std(), ID.max() * 0.5)
        im2 = axes[1].imshow(ID.T, origin="lower", extent=ext,
                             cmap=CMAP_INTERDIG, aspect="equal",
                             vmin=0, vmax=vmax_id)
        _cb(fig, im2, axes[1], "Penetracion norm. (u.a.)", "%.2f")
        axes[1].text(
            0.02, 0.04, "score = profundidad / longitud_cadena\nLo > Ld",
            transform=axes[1].transAxes, fontsize=7.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="#cccccc", alpha=0.9),
        )
        _decorate(axes[1], "Interdigitacion trans-leaflet",
                  "Chaisson et al. 2025 [9]")

        todos = membrane.outer_leaflet + membrane.inner_leaflet
        s_gel = [l.order_param for l in todos if l.lipid_type.phase == "gel"]
        s_fluid = [l.order_param for l in todos if l.lipid_type.phase == "fluid"]
        axes[2].hist(s_fluid, bins=40, density=True, alpha=0.6,
                     label="Fluido (Ld)", color="#3a86ff",
                     histtype="stepfilled", ec="#1a5fbf")
        axes[2].hist(s_gel, bins=40, density=True, alpha=0.6,
                     label="Gel (Lo)", color="#2dc653",
                     histtype="stepfilled", ec="#0a6e2d")
        axes[2].axvline(np.mean(s_fluid) if s_fluid else 0,
                        color="#1a5fbf", lw=1.5, ls="--")
        axes[2].axvline(np.mean(s_gel) if s_gel else 0,
                        color="#0a6e2d", lw=1.5, ls="--")
        axes[2].set_xlabel("S_CH")
        axes[2].set_ylabel("Densidad")
        axes[2].legend(fontsize=8)
        axes[2].set_title("Distribucion S_CH\nGel vs Fluido",
                          fontsize=9, fontweight="bold")
        fig.tight_layout()
        _save(membrane, fig, "fig6_orden_cadenas.png")


def plot_density_3d(membrane: "BicapaCryoET"):
    """Fig 7: slices ortogonales del volumen 3D de densidad [14, 16]."""
    H, edges = analysis.volumetric_density(membrane, bins_xy=55, bins_z=40)
    cx = 0.5 * (edges[0][:-1] + edges[0][1:])
    cy = 0.5 * (edges[1][:-1] + edges[1][1:])
    cz = 0.5 * (edges[2][:-1] + edges[2][1:])
    g = membrane.geometry

    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(
            "Densidad volumetrica 3D — slices ortogonales (seed=%d)\n"
            "Bicapa: banda densa (cabezas) + nucleo claro (colas) [14]"
            % membrane.seed,
            fontsize=11, fontweight="bold",
        )
        Hxz = H[:, len(cy) // 2, :]
        im = axes[0].imshow(Hxz.T, origin="lower",
                            extent=[cx[0], cx[-1], cz[0], cz[-1]],
                            cmap="gray_r", aspect="auto")
        _cb(fig, im, axes[0], "Densidad")
        for zl, col in [(g.z_outer / 10, "#2dc653"),
                        (g.z_inner / 10, "#e63946"), (0.0, "#3a86ff")]:
            axes[0].axhline(zl, color=col, lw=0.9, ls="--", alpha=0.7)
        _decorate(axes[0], "Slice XZ (Y=centro)",
                  "Z relativo al plano medio", "X (nm)", "Z relativo (nm)")

        Hyz = H[len(cx) // 2, :, :]
        im2 = axes[1].imshow(Hyz.T, origin="lower",
                             extent=[cy[0], cy[-1], cz[0], cz[-1]],
                             cmap="gray_r", aspect="auto")
        _cb(fig, im2, axes[1], "Densidad")
        for zl, col in [(g.z_outer / 10, "#2dc653"),
                        (g.z_inner / 10, "#e63946"), (0.0, "#3a86ff")]:
            axes[1].axhline(zl, color=col, lw=0.9, ls="--", alpha=0.7)
        _decorate(axes[1], "Slice YZ (X=centro)",
                  "Segunda vista transversal", "Y (nm)", "Z relativo (nm)")

        z_lo = g.z_outer / 10 * 0.5
        z_hi = g.z_outer / 10 * 1.3
        iz_lo = max(0, np.searchsorted(cz, z_lo) - 1)
        iz_hi = min(len(cz) - 1, np.searchsorted(cz, z_hi))
        Hxy = H[:, :, iz_lo:iz_hi + 1].max(axis=2)
        im3 = axes[2].imshow(Hxy.T, origin="lower",
                             extent=[cx[0], cx[-1], cy[0], cy[-1]],
                             cmap="gray_r", aspect="equal")
        _cb(fig, im3, axes[2], "Densidad (MIP)")
        _draw_rafts(axes[2], membrane.rafts_outer)
        _decorate(axes[2],
                  "Proyeccion maxima XY (cabezas externas)",
                  "MIP z=[%.1f, %.1f] nm" % (z_lo, z_hi))
        fig.tight_layout()
        _save(membrane, fig, "fig7_densidad3d.png")


def plot_geometry_detailed(membrane: "BicapaCryoET"):
    """Fig 5: vista lateral coherente de la bicapa."""
    g = membrane.geometry
    Lx_nm = membrane.Lx / 10
    width_nm = 15.0
    x_start = Lx_nm / 2 - width_nm / 2
    x_end = Lx_nm / 2 + width_nm / 2
    mid_y = (membrane.Ly / 10) / 2
    y_depth_nm = 2.0

    def in_slice(lip):
        return (x_start * 10 <= lip.head_pos[0] <= x_end * 10
                and abs(lip.head_pos[1] / 10 - mid_y) < y_depth_nm / 2)

    lipids_slice = [l for l in membrane.outer_leaflet + membrane.inner_leaflet
                    if in_slice(l)]

    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(18, 11))
        fig.suptitle(
            "Estructura Molecular de la Bicapa — Vista Lateral (seed=%d)\n"
            "Franja: %.1f-%.1f nm | Grosor: %.0f A | kc=%.0f kBT nm2"
            % (membrane.seed, x_start, x_end,
               g.total_thick, membrane.bending_modulus),
            fontsize=13, fontweight="bold", y=0.98,
        )
        ax.axhspan(g.z_inner / 10, g.z_outer / 10,
                   color="#f5f5dc", alpha=0.5, zorder=0)
        ax.axhspan(g.z_outer / 10, g.z_outer / 10 + 1.5,
                   color="#e8f5e8", alpha=0.6, zorder=0)
        ax.axhspan(g.z_inner / 10 - 1.5, g.z_inner / 10,
                   color="#f5e8e8", alpha=0.6, zorder=0)
        ax.axhline(g.z_outer / 10, color="#2dc653", lw=2.0,
                   ls="-", alpha=0.9, label="Cabezas externas")
        ax.axhline(g.z_inner / 10, color="#e63946", lw=2.0,
                   ls="-", alpha=0.9, label="Cabezas internas")
        ax.axhline(0, color="#666666", lw=1.5, ls="--",
                   alpha=0.7, label="Plano medio")
        y_range = max(y_depth_nm * 10, 1.0)

        for lip in sorted(lipids_slice, key=lambda l: l.head_pos[1]):
            alpha = 0.45 + 0.50 * abs(lip.head_pos[1] - mid_y * 10) / y_range
            ltype = lip.lipid_type
            ax.plot(
                [lip.head_pos[0] / 10, lip.glycerol_pos[0] / 10],
                [lip.head_pos[2] / 10, lip.glycerol_pos[2] / 10],
                color="#333333", lw=3.5, alpha=alpha, zorder=4,
                solid_capstyle="round",
            )
            ax.scatter(lip.head_pos[0] / 10, lip.head_pos[2] / 10,
                       s=(ltype.hg_radius * 4.5) ** 2, c=ltype.color,
                       alpha=alpha, zorder=5, edgecolors="white", linewidths=1.0)
            ax.scatter(lip.glycerol_pos[0] / 10, lip.glycerol_pos[2] / 10,
                       s=30, c="#1a1a1a", marker="s", alpha=alpha, zorder=4,
                       edgecolors="white", linewidths=0.8)

            z_lo = (-0.3 if lip.leaflet == "sup" else g.z_inner / 10 - 0.5)
            z_hi = (g.z_outer / 10 + 0.5 if lip.leaflet == "sup" else 0.3)

            for tail, color_t in [
                (lip.tail1, ltype.color_tail1),
                (lip.tail2, ltype.color_tail2) if lip.tail2 else (None, None),
            ]:
                if not tail:
                    continue
                pts = np.array(tail) / 10
                mask = (pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)
                if np.any(mask):
                    ax.plot(pts[mask, 0], pts[mask, 2],
                            color=color_t, lw=2.0, alpha=alpha, zorder=3,
                            solid_capstyle="round")

        tipos_presentes = {lip.lipid_type for lip in lipids_slice}
        if tipos_presentes:
            handles = [
                mpatches.Patch(facecolor=t.color, edgecolor="black",
                               label=t.name, alpha=0.9)
                for t in sorted(tipos_presentes, key=lambda x: x.name)
            ]
            leg1 = ax.legend(handles=handles, loc="upper right", fontsize=9,
                             framealpha=0.95, title="Tipos de lipidos")
            ax.add_artist(leg1)

        mid_z = (g.z_outer + g.z_inner) / 20
        ax.annotate(
            "", xy=(x_start + 0.5, g.z_outer / 10),
            xytext=(x_start + 0.5, g.z_inner / 10),
            arrowprops=dict(arrowstyle="<->", color="black", lw=2.5),
        )
        ax.text(x_start + 0.85, mid_z, "%.0f A" % g.total_thick,
                fontsize=12, va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="black", alpha=0.95))
        ax.set_xlabel("X (nm)", fontsize=12)
        ax.set_ylabel("Z (nm)", fontsize=12)
        ax.set_xlim(x_start, x_end)
        ax.set_ylim(g.z_inner / 10 - 2.0, g.z_outer / 10 + 2.0)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(membrane, fig, "fig5_geometria.png")


def plot_all(membrane: "BicapaCryoET"):
    """Genera las 7 figuras de una semilla."""
    print("  Figuras para seed=%d..." % membrane.seed)
    plot_cryoET(membrane)
    plot_domains(membrane)
    plot_pips(membrane)
    plot_composition(membrane)
    plot_geometry_detailed(membrane)
    plot_order_chains(membrane)
    plot_density_3d(membrane)
