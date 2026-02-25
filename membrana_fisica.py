#!/usr/bin/env python3
"""
Membrana lipídica con parámetros físicos experimentales.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

OUTPUT_DIR = "membrana_fisica_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class PropiedadesFisicas:
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


class MembranaFisica:
    def __init__(self, tamano_nm=(15, 15)):
        self.tamano = (tamano_nm[0] * 10, tamano_nm[1] * 10)
        self.props = PropiedadesFisicas()
        self.composicion = {
            'superior': {'POPC': 0.40, 'SM': 0.30, 'CHOL': 0.30},
            'inferior': {'POPC': 0.45, 'DOPS': 0.20, 'PIP2': 0.05, 'CHOL': 0.30}
        }
        self.lado_superior = []
        self.lado_inferior = []

    def calcular_numero_lipidos(self, lado='superior'):
        comp = self.composicion[lado]
        area_total = self.tamano[0] * self.tamano[1]
        area_prom = sum(self.props.areas[t] * f for t, f in comp.items())
        num_total = int(area_total / area_prom)
        conteo = {}
        restantes = num_total
        items = sorted(comp.items(), key=lambda x: x[1], reverse=True)
        for i, (tipo, frac) in enumerate(items):
            if i < len(items) - 1:
                cant = int(num_total * frac)
                conteo[tipo] = cant
                restantes -= cant
            else:
                conteo[tipo] = restantes
        return conteo, num_total, area_prom

    def empaquetar_hexagonal(self, num_lipidos, tipo_lipido, z_pos, lado='superior'):
        lipidos = []
        area = self.props.areas[tipo_lipido]
        d = np.sqrt(area * 2 / np.sqrt(3))
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
        long_cola = self.props.longitud_colas[tipo_lipido]
        for x, y in posiciones[:num_lipidos]:
            z_head = z_pos
            z_tail = z_head - long_cola if lado == 'superior' else z_head + long_cola
            ang = np.random.uniform(10, 15) * np.pi / 180
            dx = long_cola * np.sin(ang) * np.cos(np.random.uniform(0, 2 * np.pi))
            dy = long_cola * np.sin(ang) * np.sin(np.random.uniform(0, 2 * np.pi))
            lipidos.append({
                'tipo': tipo_lipido,
                'cabeza': np.array([x, y, z_head]),
                'cola': np.array([x + dx, y + dy, z_tail]),
                'area': area,
                'volumen': self.props.volumenes[tipo_lipido],
                'angulo': np.degrees(ang),
                'color': self.props.colores[tipo_lipido]
            })
        return lipidos

    def calcular_grosor(self):
        cola_sup = sum(self.props.longitud_colas[t] * f for t, f in self.composicion['superior'].items())
        cola_inf = sum(self.props.longitud_colas[t] * f for t, f in self.composicion['inferior'].items())
        hidrofobo = cola_sup + cola_inf
        cabeza_sup = sum(self.props.grosor_cabezas[t] * f for t, f in self.composicion['superior'].items())
        cabeza_inf = sum(self.props.grosor_cabezas[t] * f for t, f in self.composicion['inferior'].items())
        total = hidrofobo + cabeza_sup + cabeza_inf
        return {'hidrofobo': hidrofobo, 'total': total}

    def construir(self):
        grosor = self.calcular_grosor()
        z_sup = grosor['hidrofobo'] / 2
        z_inf = -grosor['hidrofobo'] / 2
        conteo_sup, _, _ = self.calcular_numero_lipidos('superior')
        for t, c in conteo_sup.items():
            self.lado_superior.extend(self.empaquetar_hexagonal(c, t, z_sup, 'superior'))
        conteo_inf, _, _ = self.calcular_numero_lipidos('inferior')
        for t, c in conteo_inf.items():
            self.lado_inferior.extend(self.empaquetar_hexagonal(c, t, z_inf, 'inferior'))
        return grosor

    def guardar_pdb(self, nombre="membrane_physical.pdb", grosor=None):
        ruta = os.path.join(OUTPUT_DIR, nombre)
        with open(ruta, 'w', encoding='utf-8') as f:
            f.write("REMARK   Membrana físicamente correcta\n")
            if grosor:
                f.write("REMARK   Grosor: {:.1f} Å\n".format(grosor['total']))
            total_lip = len(self.lado_superior) + len(self.lado_inferior)
            f.write("REMARK   Lípidos: {}\n".format(total_lip))
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
        print("PDB guardado en:", ruta)

    def guardar_propiedades(self, nombre="physical_properties.txt", grosor=None):
        ruta = os.path.join(OUTPUT_DIR, nombre)
        with open(ruta, 'w', encoding='utf-8') as f:
            f.write("PROPIEDADES DE LA MEMBRANA\n")
            f.write("Tamaño: {:.1f} x {:.1f} nm\n".format(self.tamano[0]/10, self.tamano[1]/10))
            if grosor:
                f.write("Grosor hidrofóbico: {:.1f} Å\n".format(grosor['hidrofobo']))
                f.write("Grosor total: {:.1f} Å\n".format(grosor['total']))
            total = len(self.lado_superior) + len(self.lado_inferior)
            area_nm2 = (self.tamano[0] * self.tamano[1]) / 100
            f.write("Total lípidos: {}\n".format(total))
            f.write("Densidad: {:.2f} lípidos/nm²\n".format(total / area_nm2))
        print("Propiedades guardadas en:", ruta)

    # Visualizaciones
    def visualizar_top(self):
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
        grosor = self.calcular_grosor()
        gh = grosor['hidrofobo']
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

    def visualizar_perfil(self):
        """Perfil de densidad (opcional)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        todos = self.lado_superior + self.lado_inferior
        for lip in todos:
            ax1.plot([lip['cabeza'][0], lip['cola'][0]],
                     [lip['cabeza'][2], lip['cola'][2]],
                     c='gray', alpha=0.2, linewidth=0.5)
            ax1.scatter(lip['cabeza'][0], lip['cabeza'][2],
                        s=30, alpha=0.6, c=lip['color'], edgecolors='none')
        ax1.axhline(y=0, color='black', linestyle=':', alpha=0.5)
        ax1.set_xlabel('X (Å)')
        ax1.set_ylabel('Z (Å)')
        ax1.set_title('Estructura')
        ax1.grid(True, alpha=0.3)
        z_coords = []
        for lip in todos:
            z_coords.append(lip['cabeza'][2])
            z_coords.append(lip['cola'][2])
        counts, edges = np.histogram(z_coords, bins=100, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax2.fill_between(centers, counts, alpha=0.3, color='blue')
        ax2.plot(centers, counts, 'b-', linewidth=2)
        ax2.axvline(x=0, color='black', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Z (Å)')
        ax2.set_ylabel('Densidad (u.a.)')
        ax2.set_title('Perfil de densidad')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        archivo = os.path.join(OUTPUT_DIR, 'perfil_densidad.png')
        plt.savefig(archivo, dpi=300, bbox_inches='tight')
        plt.close()
        print("Perfil de densidad guardado en:", archivo)


def main():
    membrana = MembranaFisica(tamano_nm=(15, 15))
    grosor = membrana.construir()
    membrana.guardar_pdb("membrane_physical.pdb", grosor)
    membrana.guardar_propiedades("physical_properties.txt", grosor)
    membrana.visualizar_top()
    membrana.visualizar_side()
    membrana.visualizar_3d()
    membrana.visualizar_perfil()  # opcional
    print("Proceso completado. Archivos en:", OUTPUT_DIR)


if __name__ == "__main__":
    main()