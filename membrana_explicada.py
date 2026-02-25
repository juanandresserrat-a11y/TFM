#!/usr/bin/env python3
"""
Simulación de membrana lipídica con parámetros físicos reales.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

OUTPUT_DIR = "membrana_explicada_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class PropiedadesLipidos:
    """Propiedades físicas de lípidos (datos experimentales)."""
    def __init__(self):
        self.areas = {
            'POPC': 64.3, 'DOPS': 59.7, 'PIP2': 75.0,
            'CHOL': 38.5, 'SM': 47.0
        }
        self.volumenes = {
            'POPC': 1230, 'DOPS': 1179, 'PIP2': 1450,
            'CHOL': 630, 'SM': 1100
        }
        self.longitud_colas = {
            'POPC': 14.5, 'DOPS': 14.2, 'PIP2': 14.8,
            'CHOL': 17.0, 'SM': 16.5
        }
        self.grosor_cabezas = {
            'POPC': 9.0, 'DOPS': 9.5, 'PIP2': 12.0,
            'CHOL': 4.0, 'SM': 10.0
        }
        self.colores = {
            'POPC': '#FFA500', 'CHOL': '#FFFFFF', 'SM': '#FFD700',
            'DOPS': '#FF4500', 'PIP2': '#FFFF00'
        }


class MembranaRealista:
    """Construye una membrana lipídica físicamente realista."""
    def __init__(self, tamano_caja_nm=(15, 15)):
        self.tamano = (tamano_caja_nm[0] * 10, tamano_caja_nm[1] * 10)
        self.props = PropiedadesLipidos()
        self.composicion = {
            'superior': {'POPC': 0.40, 'SM': 0.30, 'CHOL': 0.30},
            'inferior': {'POPC': 0.45, 'DOPS': 0.20, 'PIP2': 0.05, 'CHOL': 0.30}
        }
        self.lado_superior = []
        self.lado_inferior = []
        self.grosor_final = None

    def calcular_numero_lipidos(self, lado='superior'):
        composicion = self.composicion[lado]
        area_total = self.tamano[0] * self.tamano[1]
        area_promedio = sum(self.props.areas[t] * f for t, f in composicion.items())
        num_total = int(area_total / area_promedio)
        conteo = {}
        restantes = num_total
        items = sorted(composicion.items(), key=lambda x: x[1], reverse=True)
        for i, (tipo, frac) in enumerate(items):
            if i < len(items) - 1:
                cantidad = int(num_total * frac)
                conteo[tipo] = cantidad
                restantes -= cantidad
            else:
                conteo[tipo] = restantes
        return conteo, num_total

    def empaquetar_lipidos_hexagonal(self, num_lipidos, tipo_lipido, z_posicion, lado='superior'):
        lipidos = []
        area_lipido = self.props.areas[tipo_lipido]
        d = np.sqrt(area_lipido * 2 / np.sqrt(3))
        nx = int(self.tamano[0] / d) + 1
        ny = int(self.tamano[1] / (d * np.sqrt(3) / 2)) + 1
        posiciones = []
        for i in range(nx):
            for j in range(ny):
                x = i * d
                y = j * d * np.sqrt(3) / 2
                if j % 2 == 1:
                    x += d / 2
                jitter = 0.1
                x += np.random.uniform(-d * jitter, d * jitter)
                y += np.random.uniform(-d * jitter, d * jitter)
                # Solo añadir si está dentro de los límites
                if 0 <= x < self.tamano[0] and 0 <= y < self.tamano[1]:
                    if len(posiciones) < num_lipidos:
                        posiciones.append((x, y))
        longitud_cola = self.props.longitud_colas[tipo_lipido]
        for x, y in posiciones[:num_lipidos]:
            z_cabeza = z_posicion
            if lado == 'superior':
                z_cola = z_cabeza - longitud_cola
            else:
                z_cola = z_cabeza + longitud_cola
            angulo = np.random.uniform(10, 15) * np.pi / 180
            dx = longitud_cola * np.sin(angulo) * np.cos(np.random.uniform(0, 2 * np.pi))
            dy = longitud_cola * np.sin(angulo) * np.sin(np.random.uniform(0, 2 * np.pi))
            lipidos.append({
                'tipo': tipo_lipido,
                'cabeza': np.array([x, y, z_cabeza]),
                'cola': np.array([x + dx, y + dy, z_cola]),
                'area': area_lipido,
                'volumen': self.props.volumenes[tipo_lipido],
                'angulo': np.degrees(angulo),
                'color': self.props.colores[tipo_lipido]
            })
        return lipidos

    def calcular_grosor_membrana(self):
        cola_sup = sum(self.props.longitud_colas[t] * f for t, f in self.composicion['superior'].items())
        cola_inf = sum(self.props.longitud_colas[t] * f for t, f in self.composicion['inferior'].items())
        grosor_hidrofobo = cola_sup + cola_inf
        cabeza_sup = sum(self.props.grosor_cabezas[t] * f for t, f in self.composicion['superior'].items())
        cabeza_inf = sum(self.props.grosor_cabezas[t] * f for t, f in self.composicion['inferior'].items())
        grosor_total = grosor_hidrofobo + cabeza_sup + cabeza_inf
        return {
            'hidrofobo': grosor_hidrofobo,
            'total': grosor_total,
            'cabeza_superior': cabeza_sup,
            'cabeza_inferior': cabeza_inf
        }

    def construir_membrana_completa(self):
        self.grosor_final = self.calcular_grosor_membrana()
        z_superior = self.grosor_final['hidrofobo'] / 2
        z_inferior = -self.grosor_final['hidrofobo'] / 2
        conteo_sup, _ = self.calcular_numero_lipidos('superior')
        for tipo, cant in conteo_sup.items():
            self.lado_superior.extend(self.empaquetar_lipidos_hexagonal(cant, tipo, z_superior, 'superior'))
        conteo_inf, _ = self.calcular_numero_lipidos('inferior')
        for tipo, cant in conteo_inf.items():
            self.lado_inferior.extend(self.empaquetar_lipidos_hexagonal(cant, tipo, z_inferior, 'inferior'))
        return self.grosor_final

    def guardar_pdb(self, nombre_archivo="membrana_realista.pdb"):
        ruta = os.path.join(OUTPUT_DIR, nombre_archivo)
        with open(ruta, 'w', encoding='utf-8') as f:
            f.write("REMARK   Membrana lipídica físicamente realista\n")
            if self.grosor_final:
                f.write("REMARK   Grosor: {:.1f} Å\n".format(self.grosor_final['total']))
            total_lip = len(self.lado_superior) + len(self.lado_inferior)
            f.write("REMARK   Lípidos totales: {}\n".format(total_lip))
            atom_id = 1
            for i, lip in enumerate(self.lado_superior, 1):
                f.write(("ATOM  {:5d}  P   {:4s}A{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00           P\n"
                         .format(atom_id, lip['tipo'], i, lip['cabeza'][0], lip['cabeza'][1], lip['cabeza'][2])))
                atom_id += 1
                f.write(("ATOM  {:5d}  C   {:4s}A{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00           C\n"
                         .format(atom_id, lip['tipo'], i, lip['cola'][0], lip['cola'][1], lip['cola'][2])))
                atom_id += 1
            for i, lip in enumerate(self.lado_inferior, 1):
                f.write(("ATOM  {:5d}  P   {:4s}B{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00           P\n"
                         .format(atom_id, lip['tipo'], i, lip['cabeza'][0], lip['cabeza'][1], lip['cabeza'][2])))
                atom_id += 1
                f.write(("ATOM  {:5d}  C   {:4s}B{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00           C\n"
                         .format(atom_id, lip['tipo'], i, lip['cola'][0], lip['cola'][1], lip['cola'][2])))
                atom_id += 1
            f.write("END\n")
        print("Archivo PDB guardado en:", ruta)

    # --- Visualizaciones ---
    def visualizar_top(self):
        """Vista superior (headgroups)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.set_title('Lado superior')
        for lip in self.lado_superior:
            circle = Circle((lip['cabeza'][0], lip['cabeza'][1]), radius=2,
                            color=lip['color'], alpha=0.7, edgecolor='black', linewidth=0.5)
            ax1.add_patch(circle)
        ax1.set_xlim(0, self.tamano[0])
        ax1.set_ylim(0, self.tamano[1])
        ax1.set_aspect('equal')
        ax1.set_xlabel('X (Å)')
        ax1.set_ylabel('Y (Å)')
        ax1.grid(True, alpha=0.3)

        ax2.set_title('Lado inferior')
        for lip in self.lado_inferior:
            circle = Circle((lip['cabeza'][0], lip['cabeza'][1]), radius=2,
                            color=lip['color'], alpha=0.7, edgecolor='black', linewidth=0.5)
            ax2.add_patch(circle)
        ax2.set_xlim(0, self.tamano[0])
        ax2.set_ylim(0, self.tamano[1])
        ax2.set_aspect('equal')
        ax2.set_xlabel('X (Å)')
        ax2.set_ylabel('Y (Å)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        archivo = os.path.join(OUTPUT_DIR, 'top_view.png')
        plt.savefig(archivo, dpi=300, bbox_inches='tight')
        plt.close()
        print("Vista superior guardada en:", archivo)

    def visualizar_side(self):
        """Vista lateral (perfil)"""
        fig, ax = plt.subplots(figsize=(14, 8))
        todos = self.lado_superior + self.lado_inferior
        for lip in todos:
            ax.plot([lip['cabeza'][0], lip['cola'][0]],
                    [lip['cabeza'][2], lip['cola'][2]],
                    c='gray', alpha=0.2, linewidth=1)
            ax.scatter(lip['cabeza'][0], lip['cabeza'][2],
                       c=lip['color'], s=100, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=3)
            ax.scatter(lip['cola'][0], lip['cola'][2],
                       c='gray', s=30, alpha=0.5, zorder=2)
        if self.grosor_final:
            gh = self.grosor_final['hidrofobo']
            ax.axhspan(-gh/2, gh/2, alpha=0.1, color='yellow', label='Región hidrofóbica')
            ax.axhline(y=gh/2, color='red', linestyle='--', alpha=0.5, label='Headgroups superiores')
            ax.axhline(y=-gh/2, color='blue', linestyle='--', alpha=0.5, label='Headgroups inferiores')
            ax.axhline(y=0, color='black', linestyle=':', alpha=0.3, label='Centro')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Z (Å)')
        ax.set_title('Vista lateral')
        ax.set_xlim(0, self.tamano[0])
        ax.set_ylim(-40, 40)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        archivo = os.path.join(OUTPUT_DIR, 'side_view.png')
        plt.savefig(archivo, dpi=300, bbox_inches='tight')
        plt.close()
        print("Vista lateral guardada en:", archivo)

    def visualizar_3d(self):
        """Vista 3D"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        todos = self.lado_superior + self.lado_inferior
        for lip in self.lado_superior:
            ax.scatter(*lip['cabeza'], c=lip['color'], s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
            ax.plot([lip['cabeza'][0], lip['cola'][0]],
                    [lip['cabeza'][1], lip['cola'][1]],
                    [lip['cabeza'][2], lip['cola'][2]],
                    c='gray', alpha=0.3, linewidth=1)
        for lip in self.lado_inferior:
            ax.scatter(*lip['cabeza'], c=lip['color'], s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
            ax.plot([lip['cabeza'][0], lip['cola'][0]],
                    [lip['cabeza'][1], lip['cola'][1]],
                    [lip['cabeza'][2], lip['cola'][2]],
                    c='gray', alpha=0.3, linewidth=1)
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title('Vista 3D')
        z_min = min(lip['cola'][2] for lip in todos) - 5
        z_max = max(lip['cabeza'][2] for lip in todos) + 5
        ax.set_zlim(z_min, z_max)
        plt.tight_layout()
        archivo = os.path.join(OUTPUT_DIR, '3d_view.png')
        plt.savefig(archivo, dpi=300, bbox_inches='tight')
        plt.close()
        print("Vista 3D guardada en:", archivo)


def main():
    membrana = MembranaRealista(tamano_caja_nm=(15, 15))
    membrana.construir_membrana_completa()
    membrana.guardar_pdb("membrana_realista.pdb")
    membrana.visualizar_top()
    membrana.visualizar_side()
    membrana.visualizar_3d()
    print("Proceso completado. Archivos en la carpeta:", OUTPUT_DIR)


if __name__ == "__main__":
    main()