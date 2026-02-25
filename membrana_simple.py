#!/usr/bin/env python3
"""
Membrana lipídica simple (demo rápida).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

OUTPUT_DIR = "membrana_simple_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


LIPID_DATA = {
    'POPC': {'tail_length': 14.5, 'color': '#FFA500'},
    'CHOL': {'tail_length': 17.0, 'color': '#FFFFFF'},
    'SM': {'tail_length': 16.5, 'color': '#FFD700'},
    'DOPS': {'tail_length': 14.2, 'color': '#FF4500'},
    'PIP2': {'tail_length': 14.8, 'color': '#FFFF00'}
}


class LipidMembraneMini:
    def __init__(self, size=(50, 50), num_lipids=200):
        self.size = size
        self.num_lipids = num_lipids
        self.lipids_per_leaflet = num_lipids // 2
        self.composition = {
            'upper': {'POPC': 0.50, 'CHOL': 0.30, 'SM': 0.20},
            'lower': {'POPC': 0.45, 'DOPS': 0.25, 'PIP2': 0.05, 'CHOL': 0.25}
        }
        self.upper_leaflet = []
        self.lower_leaflet = []

    def _distribuir_lipidos(self, leaflet):
        comp = self.composition[leaflet]
        num_total = self.lipids_per_leaflet
        items = sorted(comp.items(), key=lambda x: x[1], reverse=True)
        conteo = {}
        restantes = num_total
        for i, (tipo, frac) in enumerate(items):
            if i < len(items) - 1:
                cant = int(num_total * frac)
                conteo[tipo] = cant
                restantes -= cant
            else:
                conteo[tipo] = restantes
        return conteo

    def generate_lipid_positions(self, leaflet='upper', z_position=0):
        lipids = []
        conteo = self._distribuir_lipidos(leaflet)
        for lip_type, num in conteo.items():
            tail_length = LIPID_DATA[lip_type]['tail_length']
            color = LIPID_DATA[lip_type]['color']
            for _ in range(num):
                x = np.random.uniform(0, self.size[0])
                y = np.random.uniform(0, self.size[1])
                z_head = z_position
                if leaflet == 'upper':
                    z_tail = z_head - tail_length
                else:
                    z_tail = z_head + tail_length
                lipids.append({
                    'type': lip_type,
                    'head': np.array([x, y, z_head]),
                    'tail': np.array([x, y, z_tail]),
                    'color': color
                })
        return lipids

    def build_membrane(self):
        self.upper_leaflet = self.generate_lipid_positions('upper', z_position=20)
        self.lower_leaflet = self.generate_lipid_positions('lower', z_position=-20)
        print("Leaflet superior: {} lípidos".format(len(self.upper_leaflet)))
        print("Leaflet inferior: {} lípidos".format(len(self.lower_leaflet)))

    def save_to_pdb(self, filename="membrane_mini.pdb"):
        ruta = os.path.join(OUTPUT_DIR, filename)
        with open(ruta, 'w', encoding='utf-8') as f:
            f.write("REMARK   Membrana lipídica simple\n")
            total = len(self.upper_leaflet) + len(self.lower_leaflet)
            f.write("REMARK   Total lípidos: {}\n".format(total))
            atom_id = 1
            for i, lip in enumerate(self.upper_leaflet, 1):
                f.write(("ATOM  {:5d}  P   {:4s}A{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00           P\n"
                         .format(atom_id, lip['type'], i, lip['head'][0], lip['head'][1], lip['head'][2])))
                atom_id += 1
                f.write(("ATOM  {:5d}  C   {:4s}A{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00           C\n"
                         .format(atom_id, lip['type'], i, lip['tail'][0], lip['tail'][1], lip['tail'][2])))
                atom_id += 1
            for i, lip in enumerate(self.lower_leaflet, 1):
                f.write(("ATOM  {:5d}  P   {:4s}B{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00           P\n"
                         .format(atom_id, lip['type'], i, lip['head'][0], lip['head'][1], lip['head'][2])))
                atom_id += 1
                f.write(("ATOM  {:5d}  C   {:4s}B{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00           C\n"
                         .format(atom_id, lip['type'], i, lip['tail'][0], lip['tail'][1], lip['tail'][2])))
                atom_id += 1
            f.write("END\n")
        print("PDB guardado en:", ruta)

    def save_composition_report(self, filename="composition_report.txt"):
        ruta = os.path.join(OUTPUT_DIR, filename)
        with open(ruta, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE COMPOSICIÓN\n")
            f.write("Tamaño: {:.1f} x {:.1f} nm\n".format(self.size[0]/10, self.size[1]/10))
            f.write("Total lípidos: {}\n\n".format(len(self.upper_leaflet) + len(self.lower_leaflet)))
            f.write("LEAFLET SUPERIOR:\n")
            upper_counts = {}
            for lip in self.upper_leaflet:
                upper_counts[lip['type']] = upper_counts.get(lip['type'], 0) + 1
            for t, c in sorted(upper_counts.items()):
                f.write("  {}: {} ({:.1f}%)\n".format(t, c, 100*c/len(self.upper_leaflet)))
            f.write("\nLEAFLET INFERIOR:\n")
            lower_counts = {}
            for lip in self.lower_leaflet:
                lower_counts[lip['type']] = lower_counts.get(lip['type'], 0) + 1
            for t, c in sorted(lower_counts.items()):
                f.write("  {}: {} ({:.1f}%)\n".format(t, c, 100*c/len(self.lower_leaflet)))
        print("Reporte guardado en:", ruta)

    def visualize_3d(self):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        for lip in self.upper_leaflet:
            ax.scatter(*lip['head'], c=lip['color'], s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
            ax.plot([lip['head'][0], lip['tail'][0]],
                    [lip['head'][1], lip['tail'][1]],
                    [lip['head'][2], lip['tail'][2]],
                    c='gray', alpha=0.3, linewidth=1)
        for lip in self.lower_leaflet:
            ax.scatter(*lip['head'], c=lip['color'], s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
            ax.plot([lip['head'][0], lip['tail'][0]],
                    [lip['head'][1], lip['tail'][1]],
                    [lip['head'][2], lip['tail'][2]],
                    c='gray', alpha=0.3, linewidth=1)
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title('Vista 3D')
        archivo = os.path.join(OUTPUT_DIR, 'membrane_3d.png')
        plt.savefig(archivo, dpi=300)
        plt.close()
        print("Vista 3D guardada en:", archivo)

    def visualize_top_view(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.set_title('Leaflet Superior')
        for lip in self.upper_leaflet:
            circle = Circle((lip['head'][0], lip['head'][1]), radius=2,
                            color=lip['color'], alpha=0.7, edgecolor='black', linewidth=0.5)
            ax1.add_patch(circle)
        ax1.set_xlim(0, self.size[0])
        ax1.set_ylim(0, self.size[1])
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        ax2.set_title('Leaflet Inferior')
        for lip in self.lower_leaflet:
            circle = Circle((lip['head'][0], lip['head'][1]), radius=2,
                            color=lip['color'], alpha=0.7, edgecolor='black', linewidth=0.5)
            ax2.add_patch(circle)
        ax2.set_xlim(0, self.size[0])
        ax2.set_ylim(0, self.size[1])
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        archivo = os.path.join(OUTPUT_DIR, 'membrane_top_view.png')
        plt.savefig(archivo, dpi=300)
        plt.close()
        print("Vista superior guardada en:", archivo)

    def visualize_side_view(self):
        fig, ax = plt.subplots(figsize=(14, 8))
        for lip in self.upper_leaflet + self.lower_leaflet:
            ax.plot([lip['head'][0], lip['tail'][0]],
                    [lip['head'][2], lip['tail'][2]],
                    c='gray', alpha=0.2, linewidth=1)
            ax.scatter(lip['head'][0], lip['head'][2],
                       c=lip['color'], s=100, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=3)
            ax.scatter(lip['tail'][0], lip['tail'][2],
                       c='gray', s=30, alpha=0.5, zorder=2)
        ax.axhspan(-15, 15, alpha=0.1, color='yellow', label='Región hidrofóbica')
        ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Headgroups superiores')
        ax.axhline(y=-20, color='blue', linestyle='--', alpha=0.5, label='Headgroups inferiores')
        ax.axhline(y=0, color='black', linestyle=':', alpha=0.3, label='Centro')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Z (Å)')
        ax.set_title('Vista lateral')
        ax.set_xlim(0, self.size[0])
        ax.set_ylim(-35, 35)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        archivo = os.path.join(OUTPUT_DIR, 'membrane_side_view.png')
        plt.savefig(archivo, dpi=300)
        plt.close()
        print("Vista lateral guardada en:", archivo)


def main():
    membrane = LipidMembraneMini(size=(50, 50), num_lipids=200)
    membrane.build_membrane()
    membrane.save_to_pdb("membrane_mini.pdb")
    membrane.save_composition_report("composition_report.txt")
    membrane.visualize_top_view()
    membrane.visualize_side_view()
    membrane.visualize_3d()
    print("Proceso completado. Archivos en:", OUTPUT_DIR)


if __name__ == "__main__":
    main()