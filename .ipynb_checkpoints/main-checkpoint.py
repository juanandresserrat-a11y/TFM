"""
main.py
=======
Punto de entrada del generador de bicapas lipídicas.

Uso
---
Semilla única (figuras + training):
    python main.py

Dataset de múltiples semillas:
    python main.py --seeds 1 2 3 4 5 --size 50 50

Solo datos de training (sin figuras, más rápido):
    python main.py --seeds 0 1 2 3 --solo-training

Ver todas las opciones:
    python main.py --help
"""

import argparse
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from builder import BicapaCryoET
from export import export_training
from visualization import plot_all


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generador sintético de bicapas lipídicas para cryo-ET",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py
  python main.py --seeds 27 42 100 --size 50 50
  python main.py --seeds $(seq 0 99) --solo-training
        """,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[27, 42],
        metavar="N",
        help="Lista de semillas (default: 27 42)",
    )
    parser.add_argument(
        "--size",
        type=float,
        nargs=2,
        default=[50.0, 50.0],
        metavar=("X", "Y"),
        help="Tamaño de la membrana en nm (default: 50 50)",
    )
    parser.add_argument(
        "--solo-training",
        action="store_true",
        help="Solo exportar datos de training, sin generar figuras",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Resolución de las figuras en DPI (default: 300)",
    )
    return parser.parse_args()


def run_seed(seed: int, size_nm, solo_training: bool):
    """Construye y exporta una semilla individual."""
    b = BicapaCryoET(size_nm=tuple(size_nm), seed=seed)
    b.build()
    export_training(b)
    if not solo_training:
        plot_all(b)


def main():
    args = parse_args()


    if args.dpi is not None:
        import visualization
        visualization.FIG_DPI = args.dpi
        print("DPI configurado a %d" % args.dpi)

    seeds = list(args.seeds)
    size_nm = tuple(args.size)

    print("Dataset: %d instantaneas de %.0fx%.0f nm" % (
        len(seeds), size_nm[0], size_nm[1]
    ))

    for seed in seeds:
        run_seed(seed, size_nm, args.solo_training)

    print("\nListo en: CryoET/")


if __name__ == "__main__":
    main()
