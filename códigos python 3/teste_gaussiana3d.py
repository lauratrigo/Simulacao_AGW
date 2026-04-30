"""
Script para visualizar a fonte gaussiana do GEOSTIDS em 3D.
Executa apenas a criação da grade e da fonte, sem rodar a simulação completa.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

# =============================================================================
# Configuração (parâmetros extraídos do código original)
# =============================================================================

class GaussianSourceVisualizer:
    def __init__(self):
        # Parâmetros da simulação
        self.earth_radius_km = 6371.0
        self.hemisphere = 1.0  # +1 south
        
        # Parâmetros da fonte gaussiana
        self.alt_o = 105.0  # altitude central (km)
        self.sigma_p = 15.0  # desvio padrão na altitude (km)
        
        # Fator de amplitude (como no seu código: amplitude_factor = 1000)
        self.amplitude_factor = 0.01
        
        # Parâmetros da grade
        self.nf = 101  # número de longitudes
        self.nq = 101  # número de latitudes
        
        # Constrói a grade
        self._build_grid()
        
        # Calcula a fonte gaussiana
        self._compute_gaussian_source()
        
    def _build_grid(self):
        """Constrói a grade de altitudes, latitudes e longitudes"""
        
        # ----- Grade de altitudes -----
        alt = [0.0]
        alt_max = 0.0
        while alt_max <= 6000.0:
            dr = 15.0
            if alt_max >= 600.0:
                dr = 60.0
            if alt_max >= 1200.0:
                dr = 120.0
            alt.append(alt_max + dr)
            alt_max = max(alt)
        
        self.alt = np.asarray(alt, dtype=float)
        self.np_ = len(self.alt)
        
        # ----- Grade de longitudes -----
        dr = 15.0
        dphi = np.round(np.arctan(3.0 * dr / (self.earth_radius_km + 300.0)), 5)
        lon_in = -np.radians(82.21)
        lon = lon_in + np.arange(self.nf) * dphi
        self.lon = np.degrees(lon)
        
        # ----- Grade de latitudes -----
        dtheta = dphi
        lat_in = -np.radians(67.17)
        lat = self.hemisphere * (lat_in + np.arange(self.nq) * dtheta)
        self.lat = np.degrees(lat)
        
        # Centro da fonte
        self.iphi_s = self.nf // 2
        self.iq_s = self.nq // 4
        self.lon_o = self.lon[self.iphi_s]
        self.lat_o = self.lat[self.iq_s]
        
        # Largura da gaussiana em longitude/latitude
        sigma_x = 5.0 * np.degrees(np.arctan(3.0 * 15.0 / (self.earth_radius_km + 300.0))) / 2.0
        self.sigma_x = sigma_x
        self.sigma_z = sigma_x
        
    def _compute_gaussian_source(self):
        """Calcula a fonte gaussiana 3D"""
        
        # Distribuição vertical (gaussiana em altitude)
        var_p = np.exp(-((self.alt - self.alt_o) ** 2) / self.sigma_p**2)
        
        # Distribuição em longitude
        var_f = np.exp(-((self.lon - self.lon_o) ** 2) / self.sigma_x**2)
        
        # Distribuição em latitude
        var_q = np.exp(-((self.lat - self.lat_o) ** 2) / self.sigma_z**2)
        
        # Fonte 3D (produto das três gaussianas)
        self.source_3d = var_p[:, None, None] * var_f[None, :, None] * var_q[None, None, :]
        
        # Aplica fator de amplitude
        self.source_3d = self.source_3d * self.amplitude_factor
        
        # Máscara de borda (atenuação nas bordas)
        self.mask_bound = np.ones((self.np_, self.nf, self.nq), dtype=float)
        j_m = self.nf // 2
        k_m = self.nq // 2
        rad_m = np.sqrt(j_m**2 + k_m**2)
        
        for j in range(self.nf):
            for k in range(self.nq):
                rad_o = np.sqrt((j - j_m) ** 2 + (k - k_m) ** 2)
                if rad_o >= 0.6 * rad_m:
                    self.mask_bound[:, j, k] = np.exp(-((2.0 * rad_o / rad_m) ** 2))
        
        # Aplica máscara
        self.source_masked = self.source_3d * self.mask_bound
        
    def plot_3d_volume(self, threshold=0.01, save_path=None):
        """
        Plota a fonte gaussiana em 3D usando scatter plot com cores.
        Mostra apenas pontos com amplitude > threshold para não sobrecarregar.
        """
        print("Gerando visualização 3D (isso pode levar alguns segundos)...")
        
        # Encontra pontos com amplitude significativa
        mask = self.source_masked > threshold * np.max(self.source_masked)
        
        # Coordenadas dos pontos
        alt_grid, lat_grid, lon_grid = np.meshgrid(self.alt, self.lat, self.lon, indexing='ij')
        
        x = lon_grid[mask].flatten()
        y = lat_grid[mask].flatten()
        z = alt_grid[mask].flatten()
        colors = self.source_masked[mask].flatten()
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot scatter colorido
        scatter = ax.scatter(x, y, z, c=colors, cmap='hot', s=10, alpha=0.7, 
                            norm=Normalize(vmin=0, vmax=np.max(colors)))
        
        # Marca o centro da fonte
        ax.scatter([self.lon_o], [self.lat_o], [self.alt_o], 
                  c='red', s=100, marker='*', label='Centro da Fonte')
        
        ax.set_xlabel('Longitude (graus)', fontsize=12)
        ax.set_ylabel('Latitude (graus)', fontsize=12)
        ax.set_zlabel('Altitude (km)', fontsize=12)
        ax.set_title(f'Fonte Gaussiana 3D\nAmplitude máxima = {np.max(self.source_masked):.2f}', fontsize=14)
        
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Amplitude', fontsize=10)
        
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figura salva: {save_path}")
        
        plt.show()
        
    def plot_3d_surface(self, altitude_idx=None, save_path=None):
        """
        Plota a fonte em 3D como superfície em uma altitude fixa.
        """
        if altitude_idx is None:
            altitude_idx = np.argmin(np.abs(self.alt - self.alt_o))
        
        data = self.source_masked[altitude_idx, :, :]
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
        
        surf = ax.plot_surface(lon_grid, lat_grid, data.T, cmap='hot', 
                              linewidth=0, antialiased=True, alpha=0.8)
        
        ax.set_xlabel('Longitude (graus)', fontsize=12)
        ax.set_ylabel('Latitude (graus)', fontsize=12)
        ax.set_zlabel('Amplitude', fontsize=12)
        ax.set_title(f'Fonte Gaussiana - Altitude = {self.alt[altitude_idx]:.1f} km\n'
                    f'Amplitude máxima = {np.max(data):.2f}', fontsize=14)
        
        cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Amplitude', fontsize=10)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figura salva: {save_path}")
        
        plt.show()
        
    def plot_contour_slices(self, save_path=None):
        """
        Plota cortes 2D da fonte gaussiana em diferentes altitudes e posições.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Cortes em diferentes altitudes
        alt_indices = [0, np.argmin(np.abs(self.alt - 105)), np.argmin(np.abs(self.alt - 300))]
        alt_labels = ['Superfície (0 km)', 'Ionosfera (105 km)', 'Alta altitude (300 km)']
        
        for i, (idx, label) in enumerate(zip(alt_indices, alt_labels)):
            ax = axes[0, i]
            data = self.source_masked[idx, :, :]
            im = ax.contourf(self.lon, self.lat, data.T, levels=20, cmap='hot')
            ax.set_title(f'{label}\nMax = {np.max(data):.2f}')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.colorbar(im, ax=ax)
        
        # Corte vertical em latitude fixa (no centro)
        ax = axes[1, 0]
        lat_idx = self.iq_s
        data = self.source_masked[:, :, lat_idx]
        im = ax.contourf(self.lon, self.alt, data, levels=20, cmap='hot')
        ax.set_title(f'Corte vertical - Latitude = {self.lat[lat_idx]:.1f}°')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Altitude (km)')
        plt.colorbar(im, ax=ax)
        
        # Corte vertical em longitude fixa (no centro)
        ax = axes[1, 1]
        lon_idx = self.iphi_s
        data = self.source_masked[:, lon_idx, :]
        im = ax.contourf(self.lat, self.alt, data, levels=20, cmap='hot')
        ax.set_title(f'Corte vertical - Longitude = {self.lon[lon_idx]:.1f}°')
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Altitude (km)')
        plt.colorbar(im, ax=ax)
        
        # Perfil vertical no centro
        ax = axes[1, 2]
        profile = self.source_masked[:, self.iphi_s, self.iq_s]
        ax.plot(profile, self.alt, 'b-', linewidth=2)
        ax.axhline(y=self.alt_o, color='r', linestyle='--', label=f'Altitude central = {self.alt_o} km')
        ax.set_xlabel('Amplitude')
        ax.set_ylabel('Altitude (km)')
        ax.set_title('Perfil Vertical no Centro')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Visualização da Fonte Gaussiana - Cortes 2D', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figura salva: {save_path}")
        
        plt.show()
        
    def plot_2d_map(self, altitude_idx=None, save_path=None):
        """
        Plota mapa 2D da fonte em uma altitude específica.
        """
        if altitude_idx is None:
            altitude_idx = np.argmin(np.abs(self.alt - self.alt_o))
        
        data = self.source_masked[altitude_idx, :, :]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        im = ax.pcolormesh(self.lon, self.lat, data.T, cmap='hot', shading='auto')
        
        # Marca o centro
        ax.plot(self.lon_o, self.lat_o, 'r*', markersize=15, 
               label=f'Centro\nAmplitude = {data.max():.2f}')
        
        ax.set_xlabel('Longitude (graus)', fontsize=12)
        ax.set_ylabel('Latitude (graus)', fontsize=12)
        ax.set_title(f'Fonte Gaussiana - Altitude = {self.alt[altitude_idx]:.1f} km\n'
                    f'Amplitude máxima = {data.max():.4f}', fontsize=14)
        
        divider = plt.axes([0.92, 0.1, 0.02, 0.8])
        cbar = plt.colorbar(im, cax=divider)
        cbar.set_label('Amplitude', fontsize=10)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figura salva: {save_path}")
        
        plt.show()
        
    def print_info(self):
        """Imprime informações sobre a fonte gaussiana"""
        print("\n" + "=" * 60)
        print("INFORMAÇÕES DA FONTE GAUSSIANA")
        print("=" * 60)
        
        print(f"\n--- Parâmetros da Grade ---")
        print(f"  Número de altitudes: {self.np_}")
        print(f"  Número de longitudes: {self.nf}")
        print(f"  Número de latitudes: {self.nq}")
        print(f"  Altitude mínima: {self.alt[0]:.0f} km")
        print(f"  Altitude máxima: {self.alt[-1]:.0f} km")
        print(f"  Longitude mínima: {self.lon[0]:.2f}°")
        print(f"  Longitude máxima: {self.lon[-1]:.2f}°")
        print(f"  Latitude mínima: {self.lat[0]:.2f}°")
        print(f"  Latitude máxima: {self.lat[-1]:.2f}°")
        
        print(f"\n--- Parâmetros da Gaussiana ---")
        print(f"  Altitude central: {self.alt_o} km")
        print(f"  σ altitude: {self.sigma_p} km")
        print(f"  Longitude central: {self.lon_o:.3f}°")
        print(f"  Latitude central: {self.lat_o:.3f}°")
        print(f"  σ horizontal: {self.sigma_x:.4f}° ({self.sigma_x*111:.1f} km)")
        print(f"  Fator de amplitude: {self.amplitude_factor}")
        
        print(f"\n--- Estatísticas ---")
        print(f"  Amplitude máxima: {np.max(self.source_masked):.2f}")
        print(f"  Amplitude média: {np.mean(self.source_masked):.4f}")
        print(f"  Volume total (integral): {np.sum(self.source_masked):.2f}")
        
        print("=" * 60 + "\n")


# =============================================================================
# Execução principal
# =============================================================================

if __name__ == "__main__":
    # Cria visualizador
    viz = GaussianSourceVisualizer()
    
    # Imprime informações
    viz.print_info()
    
    # 1. Mapa 2D na altitude central (105 km)
    print("\n1. Plotando mapa 2D na altitude central (105 km)...")
    viz.plot_2d_map()
    
    # 2. Corte 2D na superfície (0 km)
    print("\n2. Plotando mapa 2D na superfície...")
    viz.plot_2d_map(altitude_idx=0)
    
    # 3. Cortes 2D (contornos) em diferentes altitudes e posições
    print("\n3. Plotando cortes 2D...")
    viz.plot_contour_slices()
    
    # 4. Superfície 3D na altitude central
    print("\n4. Plotando superfície 3D...")
    viz.plot_3d_surface()
    
    # 5. Volume 3D (scatter)
    print("\n5. Plotando volume 3D (pode levar alguns segundos)...")
    viz.plot_3d_volume(threshold=0.05)
    
    print("\n" + "=" * 60)
    print("Visualização concluída!")
    print("=" * 60)