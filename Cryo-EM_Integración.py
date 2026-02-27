#!/usr/bin/env python3
"""
Integración Cryo-EM con modelos de membrana.
Requiere:
    - membrana_fisica.py (o membrana_explicada.py) en el mismo directorio.
    - Opcional: un archivo .mrc con mapa experimental.
"""

from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')

try:
    from membrana_fisica import MembranaFisica
except ImportError:
    print("Error: No se encontró membrana_fisica.py")
    sys.exit(1)

OUTPUT_DIR = "Cryo_EM_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CRYO_EM_PARAMS = {
    'pixel_size': 1.6,
    'resolution': 3.5,
    'b_factor': 50.0,
    'gaussian_sigma': 2.0,
    'density_threshold': 0.2,
    'correlation_weight': 1.0
}


class CryoEMDensityMap:
    """Maneja mapas de densidad cryo-EM."""

    def __init__(self, filepath=None):
        self.density = None
        self.header = {}
        self.origin = (0, 0, 0)
        self.voxel_size = CRYO_EM_PARAMS['pixel_size']

        if filepath and os.path.exists(filepath):
            self.load_mrc(filepath)

    def load_mrc(self, filepath):
        """Carga mapa MRC/CCP4 usando mrcfile si está disponible."""
        try:
            import mrcfile  # type: ignore
            with mrcfile.open(filepath, mode='r') as mrc:  # type: ignore
                self.density = mrc.data.copy()  # type: ignore
                self.voxel_size = float(mrc.voxel_size.x)  # type: ignore
                self.origin = (mrc.header.origin.x,  # type: ignore
                               mrc.header.origin.y,  # type: ignore
                               mrc.header.origin.z)  # type: ignore
        except ImportError:
            self._load_mrc_manual(filepath)

    def _load_mrc_manual(self, filepath):
        """Carga manual de MRC sin librería externa."""
        with open(filepath, 'rb') as f:
            header = np.frombuffer(f.read(1024), dtype=np.int32)
            nx, ny, nz = header[0], header[1], header[2]
            mode = header[3]
            dtype_map = {0: np.int8, 1: np.int16, 2: np.float32}
            dtype = dtype_map.get(mode, np.float32)
            data = np.frombuffer(f.read(), dtype=dtype)
            self.density = data.reshape((nz, ny, nx))
            self.voxel_size = 1.0

    def create_from_pdb(self, pdb_coords, box_size, atom_radius=2.0):
        """
        Simula mapa de densidad desde coordenadas atómicas.

        Args:
            pdb_coords: array Nx3 con coordenadas (x,y,z) en Å
            box_size: tupla (nx, ny, nz) en voxels
            atom_radius: radio gaussiano del átomo en Å
        """
        self.density = np.zeros(box_size, dtype=np.float32)
        sigma = atom_radius / self.voxel_size
        r = int(3 * sigma)

        grid_x, grid_y, grid_z = np.meshgrid(
            np.arange(-r, r+1),
            np.arange(-r, r+1),
            np.arange(-r, r+1),
            indexing='ij'
        )
        dist_grid = np.sqrt(grid_x**2 + grid_y**2 + grid_z**2)
        gauss_kernel = np.exp(-(dist_grid**2) / (2 * sigma**2))

        for coord in pdb_coords:
            vox = (coord / self.voxel_size).astype(int)

            if not (0 <= vox[0] < box_size[0] and
                    0 <= vox[1] < box_size[1] and
                    0 <= vox[2] < box_size[2]):
                continue

            x_start = max(0, vox[0] - r)
            x_end = min(box_size[0], vox[0] + r + 1)
            y_start = max(0, vox[1] - r)
            y_end = min(box_size[1], vox[1] + r + 1)
            z_start = max(0, vox[2] - r)
            z_end = min(box_size[2], vox[2] + r + 1)

            kx_start = x_start - (vox[0] - r)
            kx_end = kx_start + (x_end - x_start)
            ky_start = y_start - (vox[1] - r)
            ky_end = ky_start + (y_end - y_start)
            kz_start = z_start - (vox[2] - r)
            kz_end = kz_start + (z_end - z_start)

            self.density[x_start:x_end, y_start:y_end, z_start:z_end] += \
                gauss_kernel[kx_start:kx_end, ky_start:ky_end, kz_start:kz_end]

        self.density = gaussian_filter(self.density, sigma=sigma)

    def normalize(self):
        """Normaliza densidad a media=0, desviación estándar=1."""
        if self.density is not None:
            mean_val = np.mean(self.density)
            std_val = np.std(self.density)
            if std_val > 0:
                self.density = (self.density - mean_val) / std_val

    def apply_threshold(self, threshold=None):
        """Aplica umbral de densidad."""
        if self.density is None:
            return
        if threshold is None:
            threshold = CRYO_EM_PARAMS['density_threshold']
        self.density[self.density < threshold] = 0

    def get_slice(self, z_slice):
        """Obtiene slice 2D en Z."""
        if self.density is not None:
            return self.density[z_slice, :, :]
        return None

    def save_mrc(self, filepath):
        """Guarda el mapa como archivo MRC."""
        if self.density is None:
            return
        try:
            import mrcfile
            with mrcfile.new(filepath, overwrite=True) as mrc:
                mrc.set_data(self.density.astype(np.float32))
                mrc.voxel_size = self.voxel_size
        except ImportError:
            pass


class CryoEMFitter:
    """
    Ajusta modelo molecular a densidad cryo-EM experimental.
    Utiliza Monte Carlo + gradiente descendente.
    """

    def __init__(self, experimental_map, initial_structure, output_dir=OUTPUT_DIR):
        self.exp_map = experimental_map
        self.structure = initial_structure
        self.sim_map = None
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.history = {
            'correlations': [],
            'rmsd': [],
            'energies': []
        }

    def calculate_correlation(self):
        """Calcula correlación cruzada entre mapas experimental y simulado."""
        if self.sim_map is None:
            self.simulate_density()

        if self.exp_map.density is None or self.sim_map is None or self.sim_map.density is None:
            return 0.0

        exp_density = self.exp_map.density
        sim_density = self.sim_map.density

        exp_flat = exp_density.ravel()
        sim_flat = sim_density.ravel()

        exp_mean = exp_flat.mean()
        sim_mean = sim_flat.mean()

        numerator = np.sum((exp_flat - exp_mean) * (sim_flat - sim_mean))
        denominator = np.sqrt(np.sum((exp_flat - exp_mean)**2)
                              * np.sum((sim_flat - sim_mean)**2))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def simulate_density(self):
        """Genera mapa simulado desde la estructura actual."""
        all_lips = self.structure.lado_superior + self.structure.lado_inferior
        coords = np.empty((len(all_lips) * 2, 3), dtype=np.float32)

        idx = 0
        for lip in all_lips:
            coords[idx] = lip['cabeza']
            coords[idx + 1] = lip['cola']
            idx += 2

        box_size = self.exp_map.density.shape

        self.sim_map = CryoEMDensityMap()
        self.sim_map.voxel_size = self.exp_map.voxel_size
        self.sim_map.create_from_pdb(coords, box_size)
        self.sim_map.normalize()

    def optimize_lipid_positions(self, max_iterations=1000, temperature=300):
        """
        Optimiza posiciones mediante simulated annealing.

        Args:
            max_iterations: número de pasos Monte Carlo
            temperature: temperatura inicial
        """
        print(f"\nOptimizando {max_iterations} iteraciones...")
        best_correlation = self.calculate_correlation()
        best_structure = self._copy_structure()
        print(f"Correlación inicial: {best_correlation:.4f}")

        T = temperature
        for iteration in range(max_iterations):
            self._perturb_structure(amplitude=2.0)

            self.simulate_density()
            new_correlation = self.calculate_correlation()

            delta_E = -(new_correlation - best_correlation)

            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
                best_correlation = new_correlation
                best_structure = self._copy_structure()
            else:
                self.structure = best_structure

            T *= 0.995

            if iteration % 100 == 0:
                print(
                    f"  Iter {iteration:4d}: Corr={best_correlation:.4f}, T={T:.1f}")
                self.history['correlations'].append(best_correlation)

        print(f"Correlación final: {best_correlation:.4f}")
        return best_correlation

    def _perturb_structure(self, amplitude=2.0):
        """Perturba posiciones aleatoriamente."""
        for lip in self.structure.lado_superior + self.structure.lado_inferior:
            lip['cabeza'] += np.random.randn(3) * amplitude
            lip['cola'] += np.random.randn(3) * amplitude

    def _copy_structure(self):
        """Copia profunda de la estructura."""
        import copy
        return copy.deepcopy(self.structure)

    def refine_with_gradient(self, learning_rate=0.1, steps=100):
        """Refinamiento fino con gradiente descendente."""
        for step in range(steps):
            grad = self._calculate_gradient()

            for i, lip in enumerate(
                    self.structure.lado_superior + self.structure.lado_inferior):
                lip['cabeza'] -= learning_rate * grad[i * 2]
                lip['cola'] -= learning_rate * grad[i * 2 + 1]

            if step % 20 == 0:
                correlation = self.calculate_correlation()

    def _calculate_gradient(self):
        """Calcula gradiente numérico de la correlación."""
        epsilon = 0.05
        gradients = []
        base_corr = self.calculate_correlation()

        all_lips = self.structure.lado_superior + self.structure.lado_inferior

        for lip in all_lips:
            for pos_type in ['cabeza', 'cola']:
                grad = np.zeros(3, dtype=np.float32)
                original = lip[pos_type].copy()

                for dim in range(3):
                    lip[pos_type][dim] += epsilon
                    self.sim_map = None
                    corr_plus = self.calculate_correlation()
                    grad[dim] = (corr_plus - base_corr) / epsilon
                    lip[pos_type][dim] = original[dim]

                gradients.append(grad)

        return gradients

    def plot_results(self):
        """Visualiza resultados del ajuste y guarda en output_dir."""
        if self.exp_map.density is None or self.sim_map is None or self.sim_map.density is None:
            print("Error: mapas no disponibles para graficar")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        z_mid = self.exp_map.density.shape[0] // 2

        exp_slice = self.exp_map.get_slice(z_mid)
        sim_slice = self.sim_map.get_slice(z_mid)

        if exp_slice is not None:
            axes[0, 0].imshow(exp_slice, cmap='gray')
            axes[0, 0].set_title('Experimental Map')

        if sim_slice is not None:
            axes[0, 1].imshow(sim_slice, cmap='gray')
            axes[0, 1].set_title('Simulated Map')

        if exp_slice is not None and sim_slice is not None:
            diff = exp_slice - sim_slice
            axes[0, 2].imshow(diff, cmap='bwr', vmin=-2, vmax=2)
            axes[0, 2].set_title('Difference')

        axes[1, 0].plot(self.history['correlations'])
        axes[1, 0].set_xlabel('Iteration (×100)')
        axes[1, 0].set_ylabel('Correlation')
        axes[1, 0].set_title('Optimization Progress')
        axes[1, 0].grid(True)

        axes[1, 1].hist(self.exp_map.density.flatten(),
                        bins=50, alpha=0.5, label='Exp')
        axes[1, 1].hist(self.sim_map.density.flatten(),
                        bins=50, alpha=0.5, label='Sim')
        axes[1, 1].set_xlabel('Density')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        axes[1, 1].set_title('Density Distribution')

        correlation = self.calculate_correlation()
        axes[1, 2].text(0.5, 0.5,
                        'Final Correlation:\n{:.4f}'.format(correlation),
                        ha='center', va='center', fontsize=20)
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fit_results.png'), dpi=300)
        plt.close()


def integrate_with_membrana_fisica(membrana, cryo_em_map, optimize=True):
    """
    Integración con el módulo membrana_fisica.

    Args:
        membrana: instancia de MembranaFisica ya construida.
        cryo_em_map: CryoEMDensityMap con datos experimentales.
        optimize: si True, optimiza posiciones.

    Returns:
        fitter: CryoEMFitter con resultados.
    """
    fitter = CryoEMFitter(cryo_em_map, membrana, output_dir=OUTPUT_DIR)

    initial_corr = fitter.calculate_correlation()

    if optimize:
        final_corr = fitter.optimize_lipid_positions(max_iterations=1000)
        fitter.refine_with_gradient(steps=50)

    fitter.plot_results()
    membrana.guardar_pdb(os.path.join(OUTPUT_DIR, 'fitted_structure.pdb'))
    if fitter.sim_map is not None:
        fitter.sim_map.save_mrc(os.path.join(OUTPUT_DIR, 'fitted_density.mrc'))
    return fitter


def ejemplo_uso():
    """Ejemplo de uso del módulo."""
    # 1. Crear o cargar un mapa experimental
    mapa_path = "ejemplo.mrc"
    if os.path.exists(mapa_path):
        exp_map = CryoEMDensityMap(mapa_path)
    else:
        # Crear una membrana temporal para simular un mapa
        temp_membrana = MembranaFisica(tamano_nm=(20, 20))
        temp_membrana.construir()
        coords = []
        for lip in temp_membrana.lado_superior + temp_membrana.lado_inferior:
            coords.append(lip['cabeza'])
            coords.append(lip['cola'])
        coords = np.array(coords)
        box_size = (40, 40, 40)
        exp_map = CryoEMDensityMap()
        exp_map.voxel_size = 2.0
        exp_map.create_from_pdb(coords, box_size)
        exp_map.normalize()
        exp_map.save_mrc(os.path.join(
            OUTPUT_DIR, "simulated_experimental.mrc"))

    exp_map.normalize()

    # 2. Crear modelo inicial de membrana
    membrana = MembranaFisica(tamano_nm=(20, 20))
    membrana.construir()

    # 3. Ajustar modelo a densidad
    fitter = integrate_with_membrana_fisica(membrana, exp_map, optimize=True)


if __name__ == "__main__":
    ejemplo_uso()
