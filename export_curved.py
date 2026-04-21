"""
export_curved.py
================
Exportacion completa de membranas con curvatura intrinseca.

Integra curved_geometry.py con el pipeline de exportacion,
produciendo:

  1. PDB curvo    — posiciones 3D de todos los granos sobre la
                    superficie curva (esfera/cilindro/silla).
                    Visible en PyMOL, ChimeraX, VMD.

  2. MRC curvo    — volumen 3D rasterizado de la membrana curva,
                    compatible con PolNet como membrane model.

  3. YAML PolNet  — configuracion actualizada con radio de curvatura,
                    energia de deformacion y curvatura espontanea.

Diferencia con el pipeline plano:
  - export_positions.py:  posiciones en parche plano 50x50 nm
  - export_mrc.py:        volumen plano 55x55x40 voxels
  - export_curved.py:     posiciones en superficie 3D real
                          + volumen 3D de la membrana curva

Uso desde Jupyter:
    from curved_geometry import spherical_patch, cylindrical_patch
    from export_curved import export_curved_all

    b = BicapaCryoET(size_nm=(50,50), seed=27).build()

    patch = cylindrical_patch(radius_nm=25.0, patch_size_nm=(50,50))
    export_curved_all(b, patch)

Uso desde terminal:
    python main.py --seeds 27 --curved sphere --radius 50
    python main.py --seeds 27 --curved cylinder --radius 25

Referencia PolNet:
  Martinez-Sanchez et al. IEEE Trans. Med. Imaging 2024
  doi:10.1109/TMI.2024.3398401
"""

from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING, Optional

import numpy as np
import mrcfile
from scipy.ndimage import gaussian_filter

from builder import OUTPUT_DIR
from curved_geometry import (
    CurvedPatch,
    curvature_stability_scan,
    generate_curved_membrane,
    spontaneous_curvature_from_composition,
)

if TYPE_CHECKING:
    from builder import BicapaCryoET

CURVED_DIR = os.path.join(OUTPUT_DIR, "curved")


def _curved_dir():
    os.makedirs(CURVED_DIR, exist_ok=True)
    return CURVED_DIR


def _lipid_residue_name(lipid_name: str) -> str:
    mapping = {
        "POPC": "PPC", "POPE": "PPE", "POPS": "PPS",
        "PI":   "PPI", "PI3P": "P3P", "PI4P": "P4P",
        "PI5P": "P5P", "PI34P2": "P34", "PIP2": "PP2",
        "PIP3": "PP3", "SM":   "SPM", "CHOL": "CHL",
        "GM1":  "GM1",
    }
    return mapping.get(lipid_name, lipid_name[:3].upper())


def export_curved_pdb(
    membrane: "BicapaCryoET",
    patch: CurvedPatch,
    result: Optional[dict] = None,
    path: Optional[str] = None,
) -> str:
    """
    Exporta las posiciones 3D curvadas en formato PDB.

    Las coordenadas estan en Angstrom sobre la superficie curva.
    La cadena A es monocapa externa, B es interna.
    B-factor = S_CH * 100. Occupancy = 1.0 si raft, 0.5 si no.

    Incluye cabezas polares y gliceroles en la superficie curva.
    Las colas se omiten por claridad (el PDB plano las tiene).

    Parametros
    ----------
    membrane : BicapaCryoET
    patch : CurvedPatch
    result : dict, opcional
        Resultado previo de generate_curved_membrane().
        Si None, se calcula internamente.
    path : str, opcional
        Ruta de salida. Si None, auto-generada.
    """
    if result is None:
        result = generate_curved_membrane(membrane, patch)

    if path is None:
        geom_tag = "%s_R%dnm" % (patch.geometry,
                                  int(patch.radius_nm) if patch.radius_nm != float("inf") else 0)
        path = os.path.join(
            _curved_dir(),
            "curved_%s_seed%04d.pdb" % (geom_tag, membrane.seed)
        )

    outer_xyz = result["outer_xyz"]
    inner_xyz = result["inner_xyz"]
    outer_n   = result["outer_normals"]
    inner_n   = result["inner_normals"]

    lines = []
    lines.append(
        "REMARK  BicapaCryoET curved — %s R=%.0f nm — seed %d"
        % (patch.geometry,
           patch.radius_nm if patch.radius_nm != float("inf") else 0,
           membrane.seed)
    )
    lines.append(
        "REMARK  Curvatura media H=%.4f nm-1 | E_bend=%.1f kBT"
        % (patch.mean_curvature, result["E_bend_total"])
    )
    lines.append(
        "REMARK  Curvatura espontanea H0=%.4f nm-1"
        % result["H0"]
    )
    lines.append(
        "REMARK  Estabilidad: %s"
        % patch.stability_report(result["kc"])
    )
    lines.append(
        "REMARK  Coordenadas en Angstrom sobre superficie %s"
        % patch.geometry
    )

    all_xyz = np.vstack([outer_xyz, inner_xyz]) * 10.0
    bbox = all_xyz.max(axis=0) - all_xyz.min(axis=0)
    lines.append(
        "CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00 P 1           1"
        % (bbox[0] + 10, bbox[1] + 10, bbox[2] + 10)
    )

    atom_idx = 1
    res_idx = 1

    def write_atom(name, res_name, chain, res_num, x, y, z, occ, bf):
        nonlocal atom_idx
        line = (
            "ATOM  %5d %-4s %3s %1s%4d    "
            "%8.3f%8.3f%8.3f%6.2f%6.2f          %2s  "
            % (
                atom_idx % 99999, name, res_name, chain,
                res_num % 9999,
                x, y, z, occ, bf, name[:1],
            )
        )
        lines.append(line)
        atom_idx += 1

    for leaflet_lips, xyz_arr, chain in [
        (membrane.outer_leaflet, outer_xyz, "A"),
        (membrane.inner_leaflet, inner_xyz, "B"),
    ]:
        for i, lip in enumerate(leaflet_lips):
            if i >= len(xyz_arr):
                break
            res_name = _lipid_residue_name(lip.lipid_type.name)
            occ = 1.0 if lip.in_raft else 0.5
            bf  = round(lip.order_param * 100.0, 2)

            x, y, z = xyz_arr[i] * 10.0
            write_atom("HD  ", res_name, chain, res_idx, x, y, z, occ, bf)

            glyc_offset_nm = lip.lipid_type.glyc_offset / 10.0
            if glyc_offset_nm > 0 and i < len(xyz_arr):
                if chain == "A":
                    normal = outer_n[i] if i < len(outer_n) else np.array([0,0,1])
                else:
                    normal = inner_n[i] if i < len(inner_n) else np.array([0,0,-1])
                gx = x - normal[0] * glyc_offset_nm * 10.0
                gy = y - normal[1] * glyc_offset_nm * 10.0
                gz = z - normal[2] * glyc_offset_nm * 10.0
                write_atom("GL  ", res_name, chain, res_idx, gx, gy, gz, occ, bf)

            res_idx += 1

        lines.append("TER")

    lines.append("END")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("  -> %s  (%d atomos CG en %s)" % (
        os.path.basename(path), atom_idx - 1, patch.geometry
    ))
    return path


def export_curved_mrc(
    membrane: "BicapaCryoET",
    patch: CurvedPatch,
    result: Optional[dict] = None,
    voxel_angstrom: float = 10.0,
    path: Optional[str] = None,
) -> str:
    """
    Rasteriza la membrana curva en un volumen MRC 3D.

    Cada cabeza polar y cada grano de cola contribuye a la celda
    de la grilla 3D que le corresponde, con peso proporcional a su
    masa. El resultado es un volumen MRC listo para PolNet.

    La grilla 3D se dimensiona automaticamente para contener la
    membrana completa con 10 A de margen en cada lado.

    Parametros
    ----------
    voxel_angstrom : float
        Tamano de voxel en Å. PolNet usa 10 Å.
    """
    if result is None:
        result = generate_curved_membrane(membrane, patch)

    outer_xyz = result["outer_xyz"] * 10.0
    inner_xyz = result["inner_xyz"] * 10.0

    all_xyz = np.vstack([outer_xyz, inner_xyz])
    margin = 15.0
    x_min = all_xyz[:, 0].min() - margin
    y_min = all_xyz[:, 1].min() - margin
    z_min = all_xyz[:, 2].min() - margin
    x_max = all_xyz[:, 0].max() + margin
    y_max = all_xyz[:, 1].max() + margin
    z_max = all_xyz[:, 2].max() + margin

    nx = max(20, int((x_max - x_min) / voxel_angstrom) + 1)
    ny = max(20, int((y_max - y_min) / voxel_angstrom) + 1)
    nz = max(20, int((z_max - z_min) / voxel_angstrom) + 1)

    vol = np.zeros((nx, ny, nz), dtype=np.float32)
    labels = np.zeros((nx, ny, nz), dtype=np.uint8)

    def rasterize_leaflet(lipids, xyz_arr, normals, label_head, label_tail):
        for i, lip in enumerate(lipids):
            if i >= len(xyz_arr):
                break
            x, y, z = xyz_arr[i]
            ix = int((x - x_min) / voxel_angstrom)
            iy = int((y - y_min) / voxel_angstrom)
            iz = int((z - z_min) / voxel_angstrom)

            if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                vol[ix, iy, iz] += lip.lipid_type.mass * 1.5
                labels[ix, iy, iz] = label_head

            normal = normals[i] if i < len(normals) else np.array([0, 0, 1])
            if lip.tail1:
                for seg in lip.tail1:
                    tx = x - normal[0] * lip.lipid_type.tail_length * 0.5
                    ty = y - normal[1] * lip.lipid_type.tail_length * 0.5
                    tz = z - normal[2] * lip.lipid_type.tail_length * 0.5
                    itx = int((tx - x_min) / voxel_angstrom)
                    ity = int((ty - y_min) / voxel_angstrom)
                    itz = int((tz - z_min) / voxel_angstrom)
                    if 0 <= itx < nx and 0 <= ity < ny and 0 <= itz < nz:
                        vol[itx, ity, itz] += lip.lipid_type.mass * 0.5
                        if labels[itx, ity, itz] == 0:
                            labels[itx, ity, itz] = label_tail

    rasterize_leaflet(
        membrane.outer_leaflet, outer_xyz, result["outer_normals"], 1, 2
    )
    rasterize_leaflet(
        membrane.inner_leaflet, inner_xyz, result["inner_normals"], 3, 2
    )

    vol_smooth = gaussian_filter(vol, sigma=[1.2, 1.2, 1.2])
    if vol_smooth.max() > 0:
        vol_norm = (vol_smooth / vol_smooth.max() * 255.0).astype(np.float32)
    else:
        vol_norm = vol_smooth

    if path is None:
        geom_tag = "%s_R%dnm" % (
            patch.geometry,
            int(patch.radius_nm) if patch.radius_nm != float("inf") else 0
        )
        path = os.path.join(
            _curved_dir(),
            "curved_%s_seed%04d.mrc" % (geom_tag, membrane.seed)
        )

    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(vol_norm.T)
        mrc.voxel_size = voxel_angstrom

    label_path = path.replace(".mrc", "_labels.mrc")
    with mrcfile.new(label_path, overwrite=True) as mrc:
        mrc.set_data(labels.T.astype(np.float32))
        mrc.voxel_size = voxel_angstrom

    print("  -> %s  (%dx%dx%d voxels, %.0f A/voxel)" % (
        os.path.basename(path), nx, ny, nz, voxel_angstrom
    ))
    print("  -> %s  (labels: 1=ext 2=hidro 3=int)" % os.path.basename(label_path))
    return path


def export_curved_polnet_yaml(
    membrane: "BicapaCryoET",
    patch: CurvedPatch,
    result: Optional[dict] = None,
    mrc_path: Optional[str] = None,
    tomo_shape: tuple = (400, 400, 300),
    voxel_angstrom: float = 10.0,
    snr: float = 0.1,
    tilt_range: tuple = (-60, 60, 3),
) -> str:
    """
    Genera YAML de PolNet para membrana curva.

    Incluye los parametros de curvatura calculados por el modelo
    para que PolNet sepa exactamente que geometria colocar en el
    tomograma sintetico.

    Los parametros clave para PolNet son:
      mb_type: file      — membrana desde archivo MRC
      mb_file: ...       — ruta al MRC curvo
      mb_thick: ...      — grosor en nm
      mb_radius: ...     — radio de curvatura en nm (nuevo)
      mb_geometry: ...   — tipo de geometria (nuevo)
    """
    if result is None:
        result = generate_curved_membrane(membrane, patch)

    scan = curvature_stability_scan(
        membrane,
        [int(patch.radius_nm)] if patch.radius_nm != float("inf") else [1000]
    )
    stability = next(
        (d for d in scan if d["geometry"] == patch.geometry), {}
    )

    if mrc_path is None:
        geom_tag = "%s_R%dnm" % (
            patch.geometry,
            int(patch.radius_nm) if patch.radius_nm != float("inf") else 0
        )
        mrc_path = os.path.abspath(os.path.join(
            _curved_dir(),
            "curved_%s_seed%04d.mrc" % (geom_tag, membrane.seed)
        ))

    yaml_content = (
        "metadata:\n"
        "  description: BicapaCryoET curved seed%(seed)d → PolNet\n"
        "  geometry: %(geometry)s\n"
        "  radius_nm: %(radius)g\n"
        "  mean_curvature_nm-1: %(H).5f\n"
        "  gaussian_curvature_nm-2: %(K).6f\n"
        "  bending_energy_kBT: %(E_bend).2f\n"
        "  spontaneous_curvature_nm-1: %(H0).5f\n"
        "  stability: %(stable)s\n"
        "  E_per_lipid_kBT: %(E_lip).4f\n"
        "\n"
        "folders:\n"
        "  root: ./\n"
        "  input: ./polnet_input/\n"
        "  output: ./polnet_output/curved_%(geometry)s_seed%(seed)04d/\n"
        "\n"
        "global:\n"
        "  ntomos: 1\n"
        "  seed: %(seed)d\n"
        "\n"
        "sample:\n"
        "  voi_shape: [%(sx)d, %(sy)d, %(sz)d]\n"
        "  voxel_size: %(vox)g\n"
        "  membranes:\n"
        "    - mb_type: file\n"
        "      mb_file: %(mrc)s\n"
        "      mb_thick: %(thick)g\n"
        "      mb_geometry: %(geometry)s\n"
        "      mb_radius: %(radius)g\n"
        "\n"
        "tem:\n"
        "  snr: %(snr)g\n"
        "  tilt_range: [%(tmin)d, %(tmax)d, %(tstep)d]\n"
        "\n"
        "# Parametros de curvatura calculados por BicapaCryoET:\n"
        "#   kc = %(kc).1f kBT*nm2  (modulo de bending de la composicion)\n"
        "#   H0 = %(H0).5f nm-1      (curvatura espontanea por asimetria)\n"
        "#   E_deformacion = %(E_bend).1f kBT total para el parche\n"
        "#   E_por_lipido  = %(E_lip).4f kBT  → %(stable)s\n"
        "#\n"
        "# Para usar:\n"
        "#   pip install polnet\n"
        "#   polnet --config %(yaml_name)s\n"
        "#\n"
        "# Referencia:\n"
        "#   Martinez-Sanchez et al. IEEE Trans. Med. Imaging 2024\n"
    ) % {
        "seed":     membrane.seed,
        "geometry": patch.geometry,
        "radius":   patch.radius_nm if patch.radius_nm != float("inf") else 0,
        "H":        patch.mean_curvature,
        "K":        patch.gaussian_curvature,
        "E_bend":   result["E_bend_total"],
        "H0":       result["H0"],
        "kc":       result["kc"],
        "stable":   stability.get("stable", "N/A"),
        "E_lip":    stability.get("E_per_lipid_kBT", 0.0),
        "sx":       tomo_shape[0],
        "sy":       tomo_shape[1],
        "sz":       tomo_shape[2],
        "vox":      voxel_angstrom,
        "mrc":      mrc_path,
        "thick":    membrane.geometry.total_thick / 10.0,
        "snr":      snr,
        "tmin":     tilt_range[0],
        "tmax":     tilt_range[1],
        "tstep":    tilt_range[2],
        "yaml_name": "curved_%s_seed%04d.yaml" % (patch.geometry, membrane.seed),
    }

    geom_tag = "%s_R%dnm" % (
        patch.geometry,
        int(patch.radius_nm) if patch.radius_nm != float("inf") else 0
    )
    yaml_name = "curved_%s_seed%04d.yaml" % (geom_tag, membrane.seed)
    yaml_path = os.path.join(_curved_dir(), yaml_name)
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print("  -> %s  (PolNet YAML curvo)" % yaml_name)
    return yaml_path


def export_curved_all(
    membrane: "BicapaCryoET",
    patch: CurvedPatch,
    voxel_angstrom: float = 10.0,
    snr: float = 0.1,
) -> dict:
    """
    Exporta PDB curvo + MRC curvo + YAML PolNet en una sola llamada.

    Uso:
        from curved_geometry import spherical_patch, cylindrical_patch
        from export_curved import export_curved_all

        b = BicapaCryoET(size_nm=(50,50), seed=27).build()

        export_curved_all(b, cylindrical_patch(25.0, (50,50)))
        export_curved_all(b, spherical_patch(50.0, (50,50)))

    Retorna
    -------
    dict con 'pdb', 'mrc', 'labels', 'yaml' y 'stability'
    """
    print("  Exportando membrana curva (%s R=%.0f nm) para seed=%d..." % (
        patch.geometry,
        patch.radius_nm if patch.radius_nm != float("inf") else 0,
        membrane.seed,
    ))

    result = generate_curved_membrane(membrane, patch)

    print("  E_bend=%.1f kBT | H0=%.4f nm-1 | %s" % (
        result["E_bend_total"], result["H0"],
        patch.stability_report(result["kc"])
    ))

    pdb_path   = export_curved_pdb(membrane, patch, result)
    mrc_path   = export_curved_mrc(membrane, patch, result, voxel_angstrom)
    yaml_path  = export_curved_polnet_yaml(
        membrane, patch, result, mrc_path, snr=snr
    )

    scan = curvature_stability_scan(
        membrane,
        [int(patch.radius_nm)] if patch.radius_nm != float("inf") else [1000]
    )

    return {
        "pdb":       pdb_path,
        "mrc":       mrc_path,
        "labels":    mrc_path.replace(".mrc", "_labels.mrc"),
        "yaml":      yaml_path,
        "stability": next(
            (d for d in scan if d["geometry"] == patch.geometry), {}
        ),
        "E_bend_kBT": result["E_bend_total"],
        "H0":          result["H0"],
    }
