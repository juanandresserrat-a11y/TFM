"""
main.py — Generador de bicapas lipidicas para cryo-ET

Uso rapido:
    python main.py                                    # figuras + training seeds 27,42
    python main.py --seeds 1 2 3 --validate           # con benchmarks cuantitativos
    python main.py --seeds 27 --curved cylinder --radius 25   # membrana curva ER
    python main.py --seeds 27 --all                   # TODOS los outputs de una vez

Flags disponibles:
    --seeds N [N ...]       semillas (default: 27 42)
    --size X Y              tamano en nm (default: 50 50)
    --solo-training         sin figuras, solo arrays .npy
    --mrc                   volumenes MRC planos para PolNet
    --positions             PDB + CSV + PolNet particle list (plano)
    --validate              6 benchmarks cuantitativos + panel PNG
    --ctf-compare           comparativa CTF vs densidad de masa
    --curved GEOM           membrana curva: sphere | cylinder | saddle
    --radius R              radio de curvatura en nm (default: 50)
    --curvature-analysis    panel completo de estabilidad de curvatura
    --stats                 estadisticas del dataset (requiere >1 semilla)
    --dpi N                 resolucion figuras (default: 300)
    --all                   activa validate + mrc + positions + curvature-analysis
"""

import argparse
import os
import sys

sys.path.insert(0, "/home/alumno25/.local/lib/python3.6/site-packages")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from builder import BicapaCryoET
from export import export_training
from visualization import plot_all


def parse_args():
    p = argparse.ArgumentParser(
        description="Generador sintetico de bicapas lipidicas para cryo-ET",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py
  python main.py --seeds 1 2 3 --validate --mrc --positions
  python main.py --seeds 27 --curved cylinder --radius 25 --curvature-analysis
  python main.py --seeds 27 --all
  python main.py --seeds $(seq 0 49) --solo-training --validate --stats
        """,
    )
    p.add_argument("--seeds",    type=int, nargs="+", default=[27, 42], metavar="N")
    p.add_argument("--size",     type=float, nargs=2, default=[50.0, 50.0], metavar=("X","Y"))
    p.add_argument("--solo-training",  action="store_true")
    p.add_argument("--mrc",            action="store_true",
                   help="MRC plano para PolNet")
    p.add_argument("--positions",      action="store_true",
                   help="PDB + CSV + PolNet particle list (plano)")
    p.add_argument("--validate",       action="store_true",
                   help="6 benchmarks cuantitativos")
    p.add_argument("--ctf-compare",    action="store_true", dest="ctf_compare",
                   help="Figura comparativa CTF")
    p.add_argument("--curved",
                   choices=["sphere", "cylinder", "saddle"], default=None,
                   help="Geometria curva: sphere | cylinder | saddle")
    p.add_argument("--radius",  type=float, default=50.0,
                   help="Radio de curvatura en nm (usar con --curved)")
    p.add_argument("--curvature-analysis", action="store_true", dest="curvature_analysis",
                   help="Panel completo de estabilidad de curvatura")
    p.add_argument("--stats",          action="store_true",
                   help="Estadisticas del dataset (requiere >1 semilla)")
    p.add_argument("--dpi",    type=int, default=None)
    p.add_argument("--all",    action="store_true",
                   help="Activa validate + mrc + positions + curvature-analysis")
    return p.parse_args()


def run_seed(seed, size_nm, args):
    b = BicapaCryoET(size_nm=tuple(size_nm), seed=seed)
    b.build()
    export_training(b)

    if not args.solo_training:
        plot_all(b)

    if args.mrc or args.all:
        from export_mrc import export_mrc, generate_polnet_yaml
        export_mrc(b)
        generate_polnet_yaml(b)

    if args.positions or args.all:
        from export_positions import export_all_positions
        export_all_positions(b)

    if args.validate or args.all:
        from validation import run_all_benchmarks, plot_validation_panel, save_benchmark_json
        results = run_all_benchmarks(b)
        plot_validation_panel(b, results)
        save_benchmark_json(results, b)

    if args.ctf_compare:
        from dataset_stats import plot_ctf_comparison, plot_mrc_comparison
        plot_ctf_comparison(b)
        plot_mrc_comparison(b)

    if args.curvature_analysis or args.all:
        from curved_geometry import plot_curvature_analysis
        plot_curvature_analysis(b)

    if args.curved:
        from curved_geometry import spherical_patch, cylindrical_patch, saddle_patch
        from export_curved import export_curved_all
        patches = {
            "sphere":   spherical_patch(args.radius, tuple(size_nm)),
            "cylinder": cylindrical_patch(args.radius, tuple(size_nm)),
            "saddle":   saddle_patch(args.radius, tuple(size_nm)),
        }
        export_curved_all(b, patches[args.curved])


def main():
    args = parse_args()

    if args.dpi is not None:
        import visualization
        visualization.FIG_DPI = args.dpi

    seeds   = list(args.seeds)
    size_nm = list(args.size)

    print("Dataset: %d semilla(s) de %.0fx%.0f nm" % (
        len(seeds), size_nm[0], size_nm[1]))

    for seed in seeds:
        run_seed(seed, size_nm, args)

    if args.stats and len(seeds) > 1:
        from dataset_stats import compute_dataset_stats, plot_dataset_summary
        stats = compute_dataset_stats(seeds, size_nm=tuple(size_nm),
                                      run_validation=args.validate)
        plot_dataset_summary(stats)

    print("\nListo. Outputs en: CryoET/")


if __name__ == "__main__":
    main()
