"""
curved_geometry.py
==================
Geometrias de membrana curva: esfera, cilindro y silla de montar.

El modelo base genera parches PLANOS con ondulaciones Helfrich termicas.
Este modulo anade curvatura INTRINSECA real, permitiendo simular:
  - Fragmentos de vesiculas (esfera, R ~ 50 nm)
  - Reticulo endoplasmatico (cilindro, R ~ 25 nm)
  - Membrana plasmatica (plano, R > 1000 nm)
  - Cuellos de fusion (silla, curvatura gaussiana negativa)

RESULTADO CIENTIFICO del TFM:
  curvature_stability_scan() explora el espacio (R, geometria, composicion)
  y determina que radios de curvatura son fisicamente accesibles.

Referencias:
  Helfrich 1973 [4], Pinigin 2022 [5], Simons & Ikonen 1997 [12]
  McMahon & Gallop, Nature 2005
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from builder import BicapaCryoET


@dataclass
class CurvedPatch:
    geometry: str
    radius_nm: float
    radius2_nm: Optional[float]
    patch_size_nm: Tuple[float, float]
    mean_curvature: float
    gaussian_curvature: float
    bending_energy_kBT: float
    spontaneous_curvature: float = 0.0

    def stability_report(self, kc: float) -> str:
        area = 0.65
        E = 2.0 * kc * (self.mean_curvature - self.spontaneous_curvature)**2 * area
        s = ("ESTABLE" if E < 0.5 else ("LIMITE" if E < 2.0 else "INESTABLE"))
        return "%s | R=%.0f nm | %s (E=%.3f kBT/lip)" % (
            self.geometry, self.radius_nm, s, E)


def flat_patch(size_nm=(50.0, 50.0)):
    return CurvedPatch("flat", float("inf"), None, tuple(size_nm), 0.0, 0.0, 0.0)


def spherical_patch(radius_nm, size_nm=(50.0, 50.0)):
    H = 1.0 / radius_nm
    return CurvedPatch("sphere", radius_nm, radius_nm, tuple(size_nm), H, H**2, 2.0*H**2)


def cylindrical_patch(radius_nm, size_nm=(50.0, 50.0)):
    H = 1.0 / (2.0 * radius_nm)
    return CurvedPatch("cylinder", radius_nm, float("inf"), tuple(size_nm), H, 0.0, 2.0*H**2)


def saddle_patch(radius_nm, size_nm=(50.0, 50.0)):
    K = -1.0 / radius_nm**2
    return CurvedPatch("saddle", radius_nm, radius_nm, tuple(size_nm), 0.0, K, 0.0)


def spontaneous_curvature_from_composition(comp_outer, comp_inner):
    """H0 (nm^-1) de la asimetria composicional. POPE/CHOL → positivo. PIPs → negativo."""
    SHAPE = {
        "POPC": 0.0, "POPE": +0.10, "POPS": +0.05, "SM": +0.08,
        "CHOL": +0.15, "GM1": -0.05, "PI": -0.08, "PI3P": -0.12,
        "PI4P": -0.12, "PI5P": -0.10, "PI34P2": -0.15,
        "PIP2": -0.18, "PIP3": -0.22,
    }
    H0_o = sum(SHAPE.get(k, 0.0) * f for k, f in comp_outer.items())
    H0_i = sum(SHAPE.get(k, 0.0) * f for k, f in comp_inner.items())
    return (H0_o - H0_i) * 0.1


def project_positions_on_surface(x_flat, y_flat, z_flat, patch, leaflet,
                                  half_thick_nm=2.0):
    """Proyecta coordenadas planas (A) sobre la superficie curva. Retorna (x3,y3,z3,normals) en nm."""
    x_nm = x_flat / 10.0
    y_nm = y_flat / 10.0
    z_loc = z_flat / 10.0
    Lx, Ly = patch.patch_size_nm
    sign = -1.0 if leaflet == "sup" else 1.0

    if patch.geometry == "flat" or patch.radius_nm == float("inf"):
        normals = np.tile([0.0, 0.0, 1.0], (len(x_nm), 1))
        return x_nm, y_nm, z_loc, normals

    R = patch.radius_nm

    if patch.geometry == "sphere":
        theta = (x_nm / Lx) * (Lx / R)
        phi   = (y_nm / Ly) * (Ly / R)
        r_eff = R + sign * half_thick_nm
        x3 = r_eff * np.sin(theta) * np.cos(phi)
        y3 = r_eff * np.sin(theta) * np.sin(phi)
        z3 = r_eff * np.cos(theta) - R + z_loc * 0.3
        normals = np.column_stack([
            np.sin(theta)*np.cos(phi),
            np.sin(theta)*np.sin(phi),
            np.cos(theta)
        ])

    elif patch.geometry == "cylinder":
        theta = (x_nm / Lx) * (Lx / R)
        r_eff = R + sign * half_thick_nm
        x3 = r_eff * np.cos(theta)
        y3 = y_nm
        z3 = r_eff * np.sin(theta) + z_loc * 0.3
        normals = np.column_stack([
            np.cos(theta), np.zeros_like(theta), np.sin(theta)
        ])

    elif patch.geometry == "saddle":
        c = 1.0 / R
        x3 = x_nm
        y3 = y_nm
        z3 = 0.5*c*(x_nm**2 - y_nm**2) + z_loc + sign * half_thick_nm
        dzdx, dzdy = c * x_nm, -c * y_nm
        n = np.column_stack([-dzdx, -dzdy, np.ones_like(x_nm)])
        normals = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)

    else:
        raise ValueError("Geometria: %s" % patch.geometry)

    return x3, y3, z3, normals


def generate_curved_membrane(membrane, patch):
    """
    Proyecta todos los lipidos sobre la superficie curva con ondulaciones Helfrich.
    Retorna dict con outer_xyz/inner_xyz en nm, normales, E_bend y H0.
    """
    g = membrane.geometry
    thick_nm = g.total_thick / 10.0 / 2.0
    H0 = spontaneous_curvature_from_composition(membrane.comp_outer, membrane.comp_inner)
    patch.spontaneous_curvature = H0

    def process(lipids, leaflet):
        x = np.array([l.head_pos[0] for l in lipids])
        y = np.array([l.head_pos[1] for l in lipids])
        z = np.array([l.head_pos[2] for l in lipids])
        x3, y3, z3, normals = project_positions_on_surface(
            x, y, z, patch, leaflet, thick_nm)
        if membrane.curvature_map is not None:
            bins = membrane.curvature_map.shape[0]
            Lx, Ly = patch.patch_size_nm
            ix = np.clip((x/10.0/Lx*bins).astype(int), 0, bins-1)
            iy = np.clip((y/10.0/Ly*bins).astype(int), 0, bins-1)
            h = membrane.curvature_map[ix, iy] / 10.0
            z3 = z3 + normals[:, 2] * h
        return np.column_stack([x3, y3, z3]), normals

    outer_xyz, outer_n = process(membrane.outer_leaflet, "sup")
    inner_xyz, inner_n = process(membrane.inner_leaflet, "inf")

    kc = membrane.bending_modulus
    H  = patch.mean_curvature
    area_nm2 = (membrane.Lx / 10.0) * (membrane.Ly / 10.0)
    E_bend = 2.0 * kc * (H - H0)**2 * area_nm2

    return {
        "outer_xyz":     outer_xyz,
        "inner_xyz":     inner_xyz,
        "outer_normals": outer_n,
        "inner_normals": inner_n,
        "patch":         patch,
        "kc":            kc,
        "E_bend_total":  E_bend,
        "H0":            H0,
    }


def curvature_stability_scan(membrane, radii_nm=None):
    """
    Escanea radios y geometrias calculando E por lipido.
    Resultado principal: espacio de curvatura accesible para esta composicion.
    """
    if radii_nm is None:
        radii_nm = [10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 500, 1000]

    kc   = membrane.bending_modulus
    H0   = spontaneous_curvature_from_composition(membrane.comp_outer, membrane.comp_inner)
    area = 0.65

    results = []
    for R in radii_nm:
        for geom, H, K in [
            ("sphere",   1.0 / R,        1.0 / R**2),
            ("cylinder", 1.0 / (2.0*R),  0.0),
            ("saddle",   0.0,            -1.0 / R**2),
        ]:
            E = 2.0 * kc * (H - H0)**2 * area
            results.append({
                "radius_nm":       R,
                "geometry":        geom,
                "H_nm-1":          round(H, 5),
                "K_nm-2":          round(K, 6),
                "E_per_lipid_kBT": round(E, 5),
                "stable":          "ESTABLE" if E < 0.5 else ("LIMITE" if E < 2.0 else "INESTABLE"),
                "kc":              round(kc, 2),
                "H0":              round(H0, 5),
            })
    return results


def plot_curvature_analysis(membrane, save_dir="CryoET/validation"):
    """
    Genera el panel completo de analisis de curvatura (5 subplots).
    Llamado con --curvature-analysis en terminal o desde Jupyter.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from lipid_types import COMP_OUTER_BASE, COMP_INNER_BASE

    os.makedirs(save_dir, exist_ok=True)

    radii = [10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 500, 1000]
    scan  = curvature_stability_scan(membrane, radii)
    kc    = membrane.bending_modulus
    H0    = spontaneous_curvature_from_composition(membrane.comp_outer, membrane.comp_inner)

    PLT = {
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#333333", "axes.linewidth": 1.0,
        "axes.grid": True, "grid.color": "#e8e8e8",
        "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold",
    }
    GCOLS = {"sphere": "#e63946", "cylinder": "#3a86ff", "saddle": "#9b5de5"}

    with plt.rc_context(PLT):
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle(
            "Analisis de curvatura — seed=%d | kc=%.0f kBT·nm² | H0=%.4f nm⁻¹"
            % (membrane.seed, kc, H0),
            fontsize=13, fontweight="bold",
        )

        ax1 = fig.add_subplot(2, 3, (1, 2))
        for geom, col in GCOLS.items():
            r_v = [d["radius_nm"]       for d in scan if d["geometry"] == geom]
            e_v = [d["E_per_lipid_kBT"] for d in scan if d["geometry"] == geom]
            ax1.loglog(r_v, e_v, "-o", color=col, lw=2.2, ms=6,
                       label=geom, zorder=5)
        ax1.axhline(0.5, color="#fb8500", lw=1.8, ls="--",
                    label="Limite estable (0.5 kBT)")
        ax1.axhline(2.0, color="#e63946", lw=1.8, ls=":",
                    label="Inestable (2.0 kBT)")
        ax1.axhline(2.0*kc*(1.0/25)**2*0.65, color="#2dc653",
                    lw=1.5, ls="-.", alpha=0.9, label="ER cilindro R=25 nm")
        ax1.axhline(2.0*kc*(1.0/50)**2*0.65, color="#adb5bd",
                    lw=1.5, ls="-.", alpha=0.9, label="Vesicula esfera R=50 nm")
        ax1.set_xlabel("Radio de curvatura R (nm)", fontsize=10)
        ax1.set_ylabel("Energia por lipido (kBT)", fontsize=10)
        ax1.set_title("Estabilidad vs curvatura — Espacio de R fisicamente accesibles")
        ax1.legend(fontsize=8, loc="upper right")
        ax1.set_xlim(8, 1200)

        ax2 = fig.add_subplot(2, 3, 3, projection="3d")
        ps = spherical_patch(50.0, membrane.size_nm)
        rs = generate_curved_membrane(membrane, ps)
        idx = np.arange(0, len(membrane.outer_leaflet), 20)
        xyz_s = rs["outer_xyz"][idx]
        cols_s = ["#e63946" if membrane.outer_leaflet[i].in_raft else "#3a86ff"
                  for i in idx]
        ax2.scatter(xyz_s[:,0], xyz_s[:,1], xyz_s[:,2],
                    c=cols_s, s=14, alpha=0.8, depthshade=True)
        ax2.set_title("Vista 3D — Esfera R=50 nm\nRojo=raft  Azul=fluido", fontsize=9)
        ax2.set_xlabel("X (nm)", fontsize=8)
        ax2.set_ylabel("Y (nm)", fontsize=8)
        ax2.set_zlabel("Z (nm)", fontsize=8)

        ax3 = fig.add_subplot(2, 3, 4)
        pc = cylindrical_patch(25.0, membrane.size_nm)
        rc = generate_curved_membrane(membrane, pc)
        xyz_c = rc["outer_xyz"]
        mid_mask = np.abs(xyz_c[:,1] - xyz_c[:,1].mean()) < 2.5
        idx_m = np.where(mid_mask)[0]
        ax3.scatter(
            xyz_c[idx_m, 0], xyz_c[idx_m, 2],
            c=["#e63946" if membrane.outer_leaflet[i].in_raft else "#3a86ff"
               for i in idx_m],
            s=14, alpha=0.85,
        )
        ax3.set_xlabel("X (nm)", fontsize=9)
        ax3.set_ylabel("Z (nm)", fontsize=9)
        ax3.set_title("Seccion transversal — Cilindro R=25 nm (ER)\nRojo=raft  Azul=fluido",
                      fontsize=9)
        ax3.set_aspect("equal")

        ax4 = fig.add_subplot(2, 3, 5)
        chol_arr = np.linspace(0.10, 0.50, 28)
        sm_arr   = np.linspace(0.10, 0.40, 28)
        H0_grid  = np.full((28, 28), np.nan)
        for i, chol in enumerate(chol_arr):
            for j, sm in enumerate(sm_arr):
                if chol + sm > 0.90:
                    continue
                co = dict(COMP_OUTER_BASE)
                co["CHOL"] = chol
                co["SM"]   = sm
                s_ = sum(co.values())
                co = {k: v/s_ for k, v in co.items()}
                H0_grid[i, j] = spontaneous_curvature_from_composition(co, COMP_INNER_BASE)
        vabs = np.nanmax(np.abs(H0_grid))
        im = ax4.contourf(chol_arr*100, sm_arr*100,
                          np.ma.masked_invalid(H0_grid).T,
                          levels=20, cmap="RdBu_r", vmin=-vabs, vmax=vabs)
        plt.colorbar(im, ax=ax4, label="H0 (nm⁻¹)", shrink=0.9)
        chol_act = membrane.comp_outer.get("CHOL", 0) * 100
        sm_act   = membrane.comp_outer.get("SM",   0) * 100
        ax4.plot(chol_act, sm_act, "k*", ms=16, zorder=10, label="Esta semilla")
        ax4.set_xlabel("CHOL en monocapa ext. (%)", fontsize=9)
        ax4.set_ylabel("SM en monocapa ext. (%)", fontsize=9)
        ax4.set_title("Curvatura espontanea H0 vs composicion\nResultado TFM: espacio CHOL-SM",
                      fontsize=9)
        ax4.legend(fontsize=8)

        ax5 = fig.add_subplot(2, 3, 6)
        y_pos = {"sphere": 0, "cylinder": 1, "saddle": 2}
        y_labels = ["sphere", "cylinder", "saddle"]
        for geom, col in GCOLS.items():
            r_estable = sorted(set(
                d["radius_nm"] for d in scan
                if d["geometry"] == geom and d["stable"] == "ESTABLE"
            ))
            r_limite = sorted(set(
                d["radius_nm"] for d in scan
                if d["geometry"] == geom and d["stable"] == "LIMITE"
            ))
            yp = y_pos[geom]
            if r_estable:
                ax5.barh(yp, r_estable[-1] - r_estable[0],
                         left=r_estable[0], color=col, alpha=0.80, height=0.4)
                ax5.text(r_estable[-1]*1.1, yp,
                         "R=%d-%d nm" % (r_estable[0], r_estable[-1]),
                         va="center", fontsize=8)
            if r_limite:
                ax5.barh(yp, r_limite[-1] - r_limite[0],
                         left=r_limite[0], color=col, alpha=0.35, height=0.4, hatch="//")

        ax5.set_yticks([0, 1, 2])
        ax5.set_yticklabels(y_labels)
        ax5.axvline(25,   color="#333333", ls="--", lw=1.5, label="ER ~25 nm")
        ax5.axvline(50,   color="#888888", ls="--", lw=1.5, label="Vesicula ~50 nm")
        ax5.axvline(1000, color="#adb5bd", ls=":",  lw=1.0, label="Plasma ~1000 nm")
        ax5.set_xlabel("Radio de curvatura (nm)", fontsize=9)
        ax5.set_title("Rango de R estables (E < 0.5 kBT/lip)\nkc=%.0f kBT·nm²  H0=%.4f nm⁻¹"
                      % (kc, H0), fontsize=9)
        ax5.set_xscale("log")
        ax5.legend(fontsize=8, loc="lower right")
        ax5.set_xlim(8, 2000)

        plt.tight_layout()
        path = os.path.join(save_dir, "curvature_analysis_seed%04d.png" % membrane.seed)
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("  -> %s" % path)
    return path
