# """
# Script para visualizar a fonte gaussiana (forçante) do código GEOSTIDS.
# Mostra as distribuições em altitude, latitude, longitude e a fonte 3D completa.
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.mplot3d import Axes3D

# # =============================================================================
# # Configuração (mesmos parâmetros do código original)
# # =============================================================================

# class GaussianSourceVisualizer:
#     def __init__(self):
#         # Parâmetros da simulação
#         self.earth_radius_km = 6371.0
#         self.hemisphere = 1.0  # +1 south
        
#         # Parâmetros da fonte gaussiana
#         self.alt_o = 105.0  # altitude central (km)
#         self.sigma_p = 15.0  # desvio padrão na altitude (km)
        
#         # Parâmetros da grade
#         self.nf = 101  # número de longitudes
#         self.nq = 101  # número de latitudes
        
#         # Constrói a grade (similar ao código original)
#         self._build_grid()
        
#         # Calcula os parâmetros da fonte
#         self._compute_source_parameters()
        
#         # Calcula as distribuições
#         self._compute_distributions()
        
#     def _build_grid(self):
#         """Constrói a grade de altitudes, latitudes e longitudes"""
        
#         # ----- Grade de altitudes -----
#         alt = [0.0]
#         alt_max = 0.0
#         while alt_max <= 6000.0:
#             dr = 15.0
#             if alt_max >= 600.0:
#                 dr = 60.0
#             if alt_max >= 1200.0:
#                 dr = 120.0
#             alt.append(alt_max + dr)
#             alt_max = max(alt)
        
#         self.alt = np.asarray(alt, dtype=float)
#         self.np_ = len(self.alt)
        
#         # ----- Grade de longitudes -----
#         dr = 15.0
#         dphi = np.round(np.arctan(3.0 * dr / (self.earth_radius_km + 300.0)), 5)
#         lon_in = -np.radians(82.21)
#         lon = lon_in + np.arange(self.nf) * dphi
#         self.lon = np.degrees(lon)
        
#         # ----- Grade de latitudes -----
#         dtheta = dphi
#         lat_in = -np.radians(67.17)
#         lat = self.hemisphere * (lat_in + np.arange(self.nq) * dtheta)
#         self.lat = np.degrees(lat)
        
#     def _compute_source_parameters(self):
#         """Calcula os parâmetros da fonte gaussiana"""
        
#         # Índice central em longitude
#         self.iphi_s = self.nf // 2
#         self.lon_o = self.lon[self.iphi_s]
        
#         # Largura da gaussiana em longitude
#         sigma_x = 5.0 * np.degrees(np.arctan(3.0 * 15.0 / (self.earth_radius_km + 300.0))) / 2.0
#         self.sigma_x = sigma_x
        
#         # Índice central em latitude (1/4 da grade)
#         self.iq_s = self.nq // 4
#         self.lat_o = self.lat[self.iq_s]
        
#         # Largura da gaussiana em latitude (mesma da longitude)
#         self.sigma_z = sigma_x
        
#     def _compute_distributions(self):
#         """Calcula as distribuições gaussianas"""
        
#         # ----- Distribuição vertical (altitude) -----
#         self.var_p = np.exp(-((self.alt - self.alt_o) ** 2) / self.sigma_p**2)
        
#         # ----- Distribuição em longitude -----
#         self.var_f = np.exp(-((self.lon - self.lon_o) ** 2) / self.sigma_x**2)
        
#         # ----- Distribuição em latitude -----
#         self.var_q = np.exp(-((self.lat - self.lat_o) ** 2) / self.sigma_z**2)
        
#         # ----- Fonte 3D (produto das três gaussianas) -----
#         self.source_3d = (self.var_p[:, None, None] * 
#                          self.var_f[None, :, None] * 
#                          self.var_q[None, None, :])
        
#         # ----- Máscara de borda (atenuação) -----
#         self.mask_bound = np.ones((self.np_, self.nf, self.nq), dtype=float)
        
#         j_m = self.nf // 2
#         k_m = self.nq // 2
#         rad_m = np.sqrt(j_m**2 + k_m**2)
        
#         for j in range(self.nf):
#             for k in range(self.nq):
#                 rad_o = np.sqrt((j - j_m) ** 2 + (k - k_m) ** 2)
#                 if rad_o >= 0.6 * rad_m:
#                     self.mask_bound[:, j, k] = np.exp(-((2.0 * rad_o / rad_m) ** 2))
        
#         # Fonte com máscara aplicada
#         self.source_masked = self.source_3d * self.mask_bound
    
#     def plot_vertical_distribution(self, save_path=None):
#         """Plota a distribuição gaussiana na altitude"""
#         fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
#         ax.plot(self.var_p, self.alt, 'b-', linewidth=2)
#         ax.axhline(y=self.alt_o, color='r', linestyle='--', 
#                    label=f'Altitude central = {self.alt_o} km')
#         ax.axvline(x=np.exp(-0.5), color='g', linestyle=':', 
#                    label=f'σ = {self.sigma_p} km (1/√e)')
        
#         ax.set_xlabel('Amplitude relativa', fontsize=12)
#         ax.set_ylabel('Altitude (km)', fontsize=12)
#         ax.set_title('Distribuição Vertical da Fonte Gaussiana\n'
#                     f'Altitude central = {self.alt_o} km, σ = {self.sigma_p} km',
#                     fontsize=14)
#         ax.legend()
#         ax.grid(True, alpha=0.3)
        
#         if save_path:
#             plt.savefig(save_path, dpi=150, bbox_inches='tight')
#             print(f"Figura salva: {save_path}")
        
#         plt.show()
        
#     def plot_longitude_distribution(self, save_path=None):
#         """Plota a distribuição gaussiana na longitude"""
#         fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
#         ax.plot(self.lon, self.var_f, 'b-', linewidth=2)
#         ax.axvline(x=self.lon_o, color='r', linestyle='--', 
#                    label=f'Longitude central = {self.lon_o:.2f}°')
        
#         # Marca a largura (σ)
#         half_max = np.exp(-0.5)
#         indices = np.where(self.var_f >= half_max)[0]
#         if len(indices) > 0:
#             lon_min = self.lon[indices[0]]
#             lon_max = self.lon[indices[-1]]
#             ax.axvspan(lon_min, lon_max, alpha=0.3, color='g',
#                       label=f'Largura (2σ) = {lon_max - lon_min:.2f}°')
        
#         ax.set_xlabel('Longitude (graus)', fontsize=12)
#         ax.set_ylabel('Amplitude relativa', fontsize=12)
#         ax.set_title('Distribuição Longitudinal da Fonte Gaussiana\n'
#                     f'Longitude central = {self.lon_o:.2f}°, σ = {self.sigma_x:.3f}°',
#                     fontsize=14)
#         ax.legend()
#         ax.grid(True, alpha=0.3)
        
#         if save_path:
#             plt.savefig(save_path, dpi=150, bbox_inches='tight')
#             print(f"Figura salva: {save_path}")
        
#         plt.show()
        
#     def plot_latitude_distribution(self, save_path=None):
#         """Plota a distribuição gaussiana na latitude"""
#         fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
#         ax.plot(self.lat, self.var_q, 'b-', linewidth=2)
#         ax.axvline(x=self.lat_o, color='r', linestyle='--', 
#                    label=f'Latitude central = {self.lat_o:.2f}°')
        
#         # Marca a largura (σ)
#         half_max = np.exp(-0.5)
#         indices = np.where(self.var_q >= half_max)[0]
#         if len(indices) > 0:
#             lat_min = self.lat[indices[0]]
#             lat_max = self.lat[indices[-1]]
#             ax.axvspan(lat_min, lat_max, alpha=0.3, color='g',
#                       label=f'Largura (2σ) = {lat_max - lat_min:.2f}°')
        
#         ax.set_xlabel('Latitude (graus)', fontsize=12)
#         ax.set_ylabel('Amplitude relativa', fontsize=12)
#         ax.set_title('Distribuição Latitudinal da Fonte Gaussiana\n'
#                     f'Latitude central = {self.lat_o:.2f}°, σ = {self.sigma_z:.3f}°',
#                     fontsize=14)
#         ax.legend()
#         ax.grid(True, alpha=0.3)
        
#         if save_path:
#             plt.savefig(save_path, dpi=150, bbox_inches='tight')
#             print(f"Figura salva: {save_path}")
        
#         plt.show()
    
#     def plot_2d_map(self, altitude_idx=None, save_path=None):
#         """Plota um mapa 2D da fonte em uma altitude específica"""
#         if altitude_idx is None:
#             # Pega o índice mais próximo da altitude central
#             altitude_idx = np.argmin(np.abs(self.alt - self.alt_o))
        
#         # Dados na altitude selecionada
#         data = self.source_masked[altitude_idx, :, :]
        
#         fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
#         # Plot
#         im = ax.pcolormesh(self.lon, self.lat, data.T, 
#                           cmap='hot', shading='auto')
        
#         # Marca o centro
#         ax.plot(self.lon_o, self.lat_o, 'r*', markersize=15, 
#                label=f'Centro\n({self.lat_o:.1f}°, {self.lon_o:.1f}°)')
        
#         # Contornos
#         levels = [0.1, 0.3, 0.5, 0.7, 0.9]
#         contour = ax.contour(self.lon, self.lat, data.T, levels=levels, 
#                             colors='white', linewidths=0.5, alpha=0.5)
#         ax.clabel(contour, inline=True, fontsize=8)
        
#         ax.set_xlabel('Longitude (graus)', fontsize=12)
#         ax.set_ylabel('Latitude (graus)', fontsize=12)
#         ax.set_title(f'Fonte Gaussiana - Altitude = {self.alt[altitude_idx]:.0f} km\n'
#                     f'Centro: ({self.lat_o:.1f}°, {self.lon_o:.1f}°)',
#                     fontsize=14)
        
#         # Barra de cores
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="3%", pad=0.08)
#         cbar = plt.colorbar(im, cax=cax)
#         cbar.set_label('Amplitude relativa', fontsize=10)
        
#         ax.legend()
#         ax.grid(True, alpha=0.3)
        
#         if save_path:
#             plt.savefig(save_path, dpi=150, bbox_inches='tight')
#             print(f"Figura salva: {save_path}")
        
#         plt.show()
        
#     def plot_mask_boundary(self, save_path=None):
#         """Plota a máscara de borda em uma altitude"""
#         altitude_idx = self.np_ // 2  # altitude média
        
#         fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
#         # Plot da máscara
#         mask_data = self.mask_bound[altitude_idx, :, :]
#         im = ax.pcolormesh(self.lon, self.lat, mask_data.T, 
#                           cmap='viridis', shading='auto')
        
#         ax.set_xlabel('Longitude (graus)', fontsize=12)
#         ax.set_ylabel('Latitude (graus)', fontsize=12)
#         ax.set_title(f'Máscara de Borda - Altitude = {self.alt[altitude_idx]:.0f} km\n'
#                     f'Atenuação nas bordas do domínio',
#                     fontsize=14)
        
#         # Barra de cores
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="3%", pad=0.08)
#         cbar = plt.colorbar(im, cax=cax)
#         cbar.set_label('Fator de atenuação', fontsize=10)
        
#         ax.grid(True, alpha=0.3)
        
#         if save_path:
#             plt.savefig(save_path, dpi=150, bbox_inches='tight')
#             print(f"Figura salva: {save_path}")
        
#         plt.show()
    
#     def plot_vertical_slice(self, save_path=None):
#         """Plota um corte vertical (altitude × longitude) na latitude central"""
#         # Índice da latitude central
#         lat_idx = self.iq_s
        
#         # Dados no corte vertical
#         data = self.source_masked[:, :, lat_idx]
        
#         fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
#         # Plot
#         im = ax.pcolormesh(self.lon, self.alt, data, 
#                           cmap='hot', shading='auto')
        
#         # Marca a altitude central
#         ax.axhline(y=self.alt_o, color='c', linestyle='--', linewidth=2,
#                    label=f'Altitude central = {self.alt_o} km')
        
#         # Marca a longitude central
#         ax.axvline(x=self.lon_o, color='r', linestyle='--', linewidth=2,
#                    label=f'Longitude central = {self.lon_o:.1f}°')
        
#         ax.set_xlabel('Longitude (graus)', fontsize=12)
#         ax.set_ylabel('Altitude (km)', fontsize=12)
#         ax.set_title(f'Corte Vertical - Latitude = {self.lat[self.iq_s]:.1f}°\n'
#                     f'Centro: ({self.alt_o} km, {self.lon_o:.1f}°)',
#                     fontsize=14)
        
#         # Barra de cores
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="3%", pad=0.08)
#         cbar = plt.colorbar(im, cax=cax)
#         cbar.set_label('Amplitude relativa', fontsize=10)
        
#         ax.legend()
#         ax.grid(True, alpha=0.3)
        
#         if save_path:
#             plt.savefig(save_path, dpi=150, bbox_inches='tight')
#             print(f"Figura salva: {save_path}")
        
#         plt.show()
    
#     def plot_statistics(self, save_path=None):
#         """Plota estatísticas da fonte"""
#         fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
#         # 1. Perfil vertical na fonte central
#         ax = axes[0, 0]
#         lon_idx = self.iphi_s
#         lat_idx = self.iq_s
#         vertical_profile = self.source_masked[:, lon_idx, lat_idx]
#         ax.plot(vertical_profile, self.alt, 'b-', linewidth=2)
#         ax.axhline(y=self.alt_o, color='r', linestyle='--')
#         ax.set_xlabel('Amplitude')
#         ax.set_ylabel('Altitude (km)')
#         ax.set_title('Perfil Vertical (no centro)')
#         ax.grid(True, alpha=0.3)
        
#         # 2. Perfil horizontal na superfície
#         ax = axes[0, 1]
#         surface_data = self.source_masked[0, :, :]
#         im = ax.pcolormesh(self.lon, self.lat, surface_data.T, cmap='hot', shading='auto')
#         ax.plot(self.lon_o, self.lat_o, 'r*', markersize=10)
#         ax.set_xlabel('Longitude (graus)')
#         ax.set_ylabel('Latitude (graus)')
#         ax.set_title('Fonte na Superfície (altitude = 0 km)')
#         plt.colorbar(im, ax=ax, label='Amplitude')
#         ax.grid(True, alpha=0.3)
        
#         # 3. Histograma da amplitude
#         ax = axes[1, 0]
#         ax.hist(self.source_masked.flatten(), bins=50, alpha=0.7, color='b')
#         ax.set_xlabel('Amplitude')
#         ax.set_ylabel('Frequência')
#         ax.set_title('Distribuição das Amplitudes')
#         ax.set_yscale('log')
#         ax.grid(True, alpha=0.3)
        
#         # 4. Amplitude máxima por altitude
#         ax = axes[1, 1]
#         max_per_alt = np.max(self.source_masked, axis=(1, 2))
#         ax.plot(self.alt, max_per_alt, 'b-', linewidth=2)
#         ax.axvline(x=self.alt_o, color='r', linestyle='--')
#         ax.set_xlabel('Altitude (km)')
#         ax.set_ylabel('Amplitude Máxima')
#         ax.set_title('Amplitude Máxima por Altitude')
#         ax.grid(True, alpha=0.3)
        
#         plt.suptitle('Estatísticas da Fonte Gaussiana', fontsize=16)
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, dpi=150, bbox_inches='tight')
#             print(f"Figura salva: {save_path}")
        
#         plt.show()
    
#     def print_source_info(self):
#         """Imprime informações sobre a fonte"""
#         print("\n" + "=" * 60)
#         print("INFORMAÇÕES DA FONTE GAUSSIANA")
#         print("=" * 60)
#         print(f"\n--- Distribuição Vertical ---")
#         print(f"  Altitude central: {self.alt_o} km")
#         print(f"  Desvio padrão (σ): {self.sigma_p} km")
#         print(f"  Largura (2σ): {2*self.sigma_p} km")
#         print(f"  Altura da grade: {self.alt[0]:.0f} a {self.alt[-1]:.0f} km")
        
#         print(f"\n--- Distribuição Longitudinal ---")
#         print(f"  Longitude central: {self.lon_o:.3f}°")
#         print(f"  Desvio padrão (σ): {self.sigma_x:.4f}°")
#         print(f"  Largura (2σ): {2*self.sigma_x:.4f}°")
#         print(f"  Em km (aproximado): {2*self.sigma_x * 111:.1f} km")
        
#         print(f"\n--- Distribuição Latitudinal ---")
#         print(f"  Latitude central: {self.lat_o:.3f}°")
#         print(f"  Desvio padrão (σ): {self.sigma_z:.4f}°")
#         print(f"  Largura (2σ): {2*self.sigma_z:.4f}°")
#         print(f"  Em km (aproximado): {2*self.sigma_z * 111:.1f} km")
        
#         print(f"\n--- Estatísticas da Fonte ---")
#         print(f"  Amplitude máxima: {np.max(self.source_masked):.6f}")
#         print(f"  Amplitude média: {np.mean(self.source_masked):.6f}")
#         print(f"  Volume total (integral): {np.sum(self.source_masked):.2f}")
        
#         print(f"\n--- Máscara de Borda ---")
#         print(f"  Atenuação mínima: {np.min(self.mask_bound):.6f}")
#         print(f"  Pontos com atenuação < 0.5: {np.sum(self.mask_bound < 0.5)}")
        
#         print("=" * 60 + "\n")


# # =============================================================================
# # Execução principal
# # =============================================================================

# if __name__ == "__main__":
#     # Cria visualizador
#     viz = GaussianSourceVisualizer()
    
#     # Imprime informações
#     viz.print_source_info()
    
#     # 1. Plot das distribuições 1D
#     print("\n1. Plotando distribuição vertical...")
#     viz.plot_vertical_distribution()
    
#     print("\n2. Plotando distribuição longitudinal...")
#     viz.plot_longitude_distribution()
    
#     print("\n3. Plotando distribuição latitudinal...")
#     viz.plot_latitude_distribution()
    
#     # 2. Plot 2D
#     print("\n4. Plotando mapa 2D na altitude central...")
#     viz.plot_2d_map()
    
#     print("\n5. Plotando mapa 2D na superfície...")
#     viz.plot_2d_map(altitude_idx=0)
    
#     # 3. Corte vertical
#     print("\n6. Plotando corte vertical...")
#     viz.plot_vertical_slice()
    
#     # 4. Máscara de borda
#     print("\n7. Plotando máscara de borda...")
#     viz.plot_mask_boundary()
    
#     # 5. Estatísticas
#     print("\n8. Plotando estatísticas...")
#     viz.plot_statistics()
    
#     print("\n" + "=" * 60)
#     print("Visualização concluída!")
#     print("=" * 60)

"""
Script para visualizar a fonte gaussiana modificada para ter amplitude e período 
de um terremoto de magnitude 7.4 com profundidade de 10 km.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# =============================================================================
# Configuração (parâmetros do terremoto M7.4)
# =============================================================================

class SeismicGaussianSourceVisualizer:
    def __init__(self):
        # Parâmetros do terremoto M7.4
        self.earthquake_magnitude = 7.4
        self.earthquake_depth_km = 10.0
        self.seismic_period = 20.0  # período da onda (s) - para M7.4
        self.seismic_velocity = 3500.0  # velocidade (m/s)
        
        # Calcula amplitude baseada na magnitude
        # Deslocamento máximo (cm) = 10^(M - 6.0)
        displacement_cm = 10 ** (self.earthquake_magnitude - 6.0)  # 25.1 cm
        displacement_m = displacement_cm / 100.0  # 0.251 m
        omega = 2 * np.pi / self.seismic_period  # 0.314 rad/s
        velocity_max = omega * displacement_m  # 0.079 m/s
        
        # Amplitude na escala do modelo (referência 1e-3)
        self.seismic_amplitude = velocity_max / 1e-3  # ~79
        
        # Parâmetros da simulação
        self.earth_radius_km = 6371.0
        self.hemisphere = 1.0  # +1 south
        
        # Parâmetros da fonte gaussiana (modificados para o terremoto)
        self.alt_o = 10.0  # altitude central (km) - agora na superfície para o terremoto
        self.sigma_p = 5.0  # desvio padrão na altitude (km) - mais estreito na superfície
        
        # Parâmetros da grade
        self.nf = 101  # número de longitudes
        self.nq = 101  # número de latitudes
        
        # Constrói a grade
        self._build_grid()
        
        # Calcula os parâmetros da fonte
        self._compute_source_parameters()
        
        # Calcula as distribuições
        self._compute_distributions()
        
    def _build_grid(self):
        """Constrói a grade de altitudes, latitudes e longitudes"""
        
        # ----- Grade de altitudes (modificada para incluir superfície) -----
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
        
    def _compute_source_parameters(self):
        """Calcula os parâmetros da fonte gaussiana sísmica"""
        
        # Índice central em longitude
        self.iphi_s = self.nf // 2
        self.lon_o = self.lon[self.iphi_s]
        
        # Largura da gaussiana em longitude (baseada no período da onda)
        # Para período de 20s, comprimento de onda ~ 70 km
        wavelength_km = self.seismic_velocity * self.seismic_period / 1000  # 70 km
        sigma_x = wavelength_km / 4  # ~17.5 km em termos de distância
        # Converte para graus (1° ~ 111 km)
        self.sigma_x = sigma_x / 111.0  # ~0.158° em graus
        
        # Índice central em latitude
        self.iq_s = self.nq // 4
        self.lat_o = self.lat[self.iq_s]
        
        # Largura da gaussiana em latitude (mesma da longitude)
        self.sigma_z = self.sigma_x
        
    def _compute_distributions(self):
        """Calcula as distribuições gaussianas com amplitude sísmica"""
        
        # ----- Distribuição vertical (gaussiana na superfície) -----
        # Para o terremoto, a fonte é máxima na superfície (altitude 0 km)
        self.var_p = np.exp(-((self.alt - self.alt_o) ** 2) / self.sigma_p**2)
        
        # Normaliza para que o máximo seja 1
        self.var_p = self.var_p / np.max(self.var_p)
        
        # ----- Distribuição em longitude (gaussiana) -----
        self.var_f = np.exp(-((self.lon - self.lon_o) ** 2) / self.sigma_x**2)
        
        # ----- Distribuição em latitude (gaussiana) -----
        self.var_q = np.exp(-((self.lat - self.lat_o) ** 2) / self.sigma_z**2)
        
        # ----- Fonte 3D (produto das três gaussianas) -----
        self.source_shape = (self.var_p[:, None, None] * 
                            self.var_f[None, :, None] * 
                            self.var_q[None, None, :])
        
        # Aplica a amplitude sísmica
        self.source_amplitude = self.source_shape * self.seismic_amplitude
        
        # ----- Máscara de borda (atenuação) -----
        self.mask_bound = np.ones((self.np_, self.nf, self.nq), dtype=float)
        
        j_m = self.nf // 2
        k_m = self.nq // 2
        rad_m = np.sqrt(j_m**2 + k_m**2)
        
        for j in range(self.nf):
            for k in range(self.nq):
                rad_o = np.sqrt((j - j_m) ** 2 + (k - k_m) ** 2)
                if rad_o >= 0.6 * rad_m:
                    self.mask_bound[:, j, k] = np.exp(-((2.0 * rad_o / rad_m) ** 2))
        
        # Fonte com máscara aplicada
        self.source_masked = self.source_amplitude * self.mask_bound
    
    def plot_vertical_distribution(self, save_path=None):
        """Plota a distribuição gaussiana na altitude"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.plot(self.var_p, self.alt, 'b-', linewidth=2)
        ax.axhline(y=self.alt_o, color='r', linestyle='--', 
                   label=f'Altitude central = {self.alt_o} km (superfície)')
        ax.axvline(x=np.exp(-0.5), color='g', linestyle=':', 
                   label=f'σ = {self.sigma_p} km')
        
        # Marca a profundidade do hipocentro
        ax.axhline(y=self.earthquake_depth_km, color='orange', linestyle='-.', 
                   label=f'Profundidade do hipocentro = {self.earthquake_depth_km} km')
        
        ax.set_xlabel('Amplitude relativa', fontsize=12)
        ax.set_ylabel('Altitude (km)', fontsize=12)
        ax.set_title(f'Distribuição Vertical da Fonte - Terremoto M{self.earthquake_magnitude}\n'
                    f'Fonte na superfície (0 km), hipocentro a {self.earthquake_depth_km} km',
                    fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.1)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figura salva: {save_path}")
        
        plt.show()
        
    def plot_longitude_distribution(self, save_path=None):
        """Plota a distribuição gaussiana na longitude"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        ax.plot(self.lon, self.var_f, 'b-', linewidth=2)
        ax.axvline(x=self.lon_o, color='r', linestyle='--', 
                   label=f'Longitude central = {self.lon_o:.2f}°')
        
        # Marca a largura (σ)
        half_max = np.exp(-0.5)
        indices = np.where(self.var_f >= half_max)[0]
        if len(indices) > 0:
            lon_min = self.lon[indices[0]]
            lon_max = self.lon[indices[-1]]
            width_deg = lon_max - lon_min
            width_km = width_deg * 111  # 1° ~ 111 km
            ax.axvspan(lon_min, lon_max, alpha=0.3, color='g',
                      label=f'Largura (2σ) = {width_deg:.2f}° ({width_km:.1f} km)')
        
        ax.set_xlabel('Longitude (graus)', fontsize=12)
        ax.set_ylabel('Amplitude relativa', fontsize=12)
        ax.set_title(f'Distribuição Longitudinal - Terremoto M{self.earthquake_magnitude}\n'
                    f'Período = {self.seismic_period} s, Comprimento de onda ~ {self.seismic_velocity*self.seismic_period/1000:.0f} km',
                    fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figura salva: {save_path}")
        
        plt.show()
        
    def plot_latitude_distribution(self, save_path=None):
        """Plota a distribuição gaussiana na latitude"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        ax.plot(self.lat, self.var_q, 'b-', linewidth=2)
        ax.axvline(x=self.lat_o, color='r', linestyle='--', 
                   label=f'Latitude central = {self.lat_o:.2f}°')
        
        # Marca a largura (σ)
        half_max = np.exp(-0.5)
        indices = np.where(self.var_q >= half_max)[0]
        if len(indices) > 0:
            lat_min = self.lat[indices[0]]
            lat_max = self.lat[indices[-1]]
            width_deg = lat_max - lat_min
            width_km = width_deg * 111
            ax.axvspan(lat_min, lat_max, alpha=0.3, color='g',
                      label=f'Largura (2σ) = {width_deg:.2f}° ({width_km:.1f} km)')
        
        ax.set_xlabel('Latitude (graus)', fontsize=12)
        ax.set_ylabel('Amplitude relativa', fontsize=12)
        ax.set_title(f'Distribuição Latitudinal - Terremoto M{self.earthquake_magnitude}',
                    fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figura salva: {save_path}")
        
        plt.show()
    
    def plot_2d_map(self, altitude_idx=None, save_path=None):
        """Plota um mapa 2D da fonte em uma altitude específica"""
        if altitude_idx is None:
            # Pega o índice da superfície (altitude 0)
            altitude_idx = 0
        
        # Dados na altitude selecionada
        data = self.source_masked[altitude_idx, :, :]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Plot
        im = ax.pcolormesh(self.lon, self.lat, data.T, 
                          cmap='hot', shading='auto')
        
        # Marca o centro
        ax.plot(self.lon_o, self.lat_o, 'r*', markersize=15, 
               label=f'Epicentro\n({self.lat_o:.1f}°, {self.lon_o:.1f}°)')
        
        # Contornos
        levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        max_val = np.max(data)
        contour_levels = [l * max_val for l in levels]
        contour = ax.contour(self.lon, self.lat, data.T, levels=contour_levels, 
                            colors='white', linewidths=0.5, alpha=0.5)
        ax.clabel(contour, inline=True, fontsize=8)
        
        # Adiciona círculo indicando o comprimento de onda
        wavelength_km = self.seismic_velocity * self.seismic_period / 1000
        wavelength_deg = wavelength_km / 111
        from matplotlib.patches import Circle
        circle = Circle((self.lon_o, self.lat_o), wavelength_deg/2, 
                       fill=False, color='cyan', linestyle='--', linewidth=1,
                       label=f'Comprimento de onda = {wavelength_km:.0f} km')
        ax.add_patch(circle)
        
        ax.set_xlabel('Longitude (graus)', fontsize=12)
        ax.set_ylabel('Latitude (graus)', fontsize=12)
        ax.set_title(f'Fonte Gaussiana Sísmica - Altitude = {self.alt[altitude_idx]:.0f} km\n'
                    f'Magnitude M{self.earthquake_magnitude}, Período = {self.seismic_period} s\n'
                    f'Amplitude máxima = {np.max(data):.2f} (escala do modelo)',
                    fontsize=12)
        
        # Barra de cores
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.08)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Amplitude (escala do modelo)', fontsize=10)
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figura salva: {save_path}")
        
        plt.show()
    
    def plot_amplitude_spectrum(self, save_path=None):
        """Plota o espectro de amplitude da fonte"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Calcula amplitude máxima em cada altitude
        altitudes = self.alt
        max_amplitude = np.max(self.source_masked, axis=(1, 2))
        
        ax.semilogy(altitudes, max_amplitude, 'b-', linewidth=2)
        ax.axvline(x=0, color='r', linestyle='--', label='Superfície')
        ax.axvline(x=self.earthquake_depth_km, color='orange', linestyle='-.', 
                   label=f'Hipocentro ({self.earthquake_depth_km} km)')
        
        ax.set_xlabel('Altitude (km)', fontsize=12)
        ax.set_ylabel('Amplitude Máxima (escala do modelo)', fontsize=12)
        ax.set_title(f'Espectro Vertical da Fonte - Terremoto M{self.earthquake_magnitude}\n'
                    f'Amplitude máxima na superfície = {max_amplitude[0]:.2f}',
                    fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figura salva: {save_path}")
        
        plt.show()
    
    def print_source_info(self):
        """Imprime informações sobre a fonte sísmica"""
        print("\n" + "=" * 60)
        print(f"FONTE GAUSSIANA PARA TERREMOTO M{self.earthquake_magnitude}")
        print("=" * 60)
        
        print(f"\n--- Parâmetros do Terremoto ---")
        print(f"  Magnitude: {self.earthquake_magnitude}")
        print(f"  Profundidade do hipocentro: {self.earthquake_depth_km} km")
        print(f"  Período da onda: {self.seismic_period} s")
        print(f"  Velocidade da onda: {self.seismic_velocity/1000:.1f} km/s")
        print(f"  Comprimento de onda: {self.seismic_velocity * self.seismic_period / 1000:.1f} km")
        
        print(f"\n--- Amplitude ---")
        print(f"  Deslocamento máximo: {10**(self.earthquake_magnitude - 6.0):.1f} cm")
        print(f"  Velocidade máxima: {(2*np.pi/self.seismic_period)*10**(self.earthquake_magnitude - 6.0)/100:.3f} m/s")
        print(f"  Amplitude no modelo: {self.seismic_amplitude:.1f}")
        
        print(f"\n--- Distribuição Vertical ---")
        print(f"  Altitude central: {self.alt_o} km (superfície)")
        print(f"  Desvio padrão (σ): {self.sigma_p} km")
        print(f"  Largura (2σ): {2*self.sigma_p} km")
        
        print(f"\n--- Distribuição Horizontal ---")
        print(f"  Longitude central: {self.lon_o:.3f}°")
        print(f"  Latitude central: {self.lat_o:.3f}°")
        print(f"  σ (longitude): {self.sigma_x:.4f}° ({self.sigma_x*111:.1f} km)")
        print(f"  σ (latitude): {self.sigma_z:.4f}° ({self.sigma_z*111:.1f} km)")
        print(f"  Largura 2σ: {2*self.sigma_x*111:.1f} km")
        
        print(f"\n--- Estatísticas da Fonte ---")
        print(f"  Amplitude máxima: {np.max(self.source_masked):.2f}")
        print(f"  Amplitude na superfície: {np.max(self.source_masked[0, :, :]):.2f}")
        print(f"  Volume total (integral): {np.sum(self.source_masked):.2f}")
        
        print("=" * 60 + "\n")


# =============================================================================
# Execução principal
# =============================================================================

if __name__ == "__main__":
    # Cria visualizador
    viz = SeismicGaussianSourceVisualizer()
    
    # Imprime informações
    viz.print_source_info()
    
    # 1. Plot das distribuições 1D
    print("\n1. Plotando distribuição vertical...")
    viz.plot_vertical_distribution()
    
    print("\n2. Plotando distribuição longitudinal...")
    viz.plot_longitude_distribution()
    
    print("\n3. Plotando distribuição latitudinal...")
    viz.plot_latitude_distribution()
    
    # 2. Plot 2D
    print("\n4. Plotando mapa 2D na superfície...")
    viz.plot_2d_map(altitude_idx=0)
    
    # 3. Espectro vertical
    print("\n5. Plotando espectro vertical...")
    viz.plot_amplitude_spectrum()
    
    print("\n" + "=" * 60)
    print("Visualização concluída!")
    print("=" * 60)