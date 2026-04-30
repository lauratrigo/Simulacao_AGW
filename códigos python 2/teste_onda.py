"""
Script para visualizar a onda sísmica gerada pelo código GEOSTIDS.
Este script não executa a simulação completa, apenas plota a fonte sísmica.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# =============================================================================
# Configuração (mesmos parâmetros do seu código)
# =============================================================================

class SeismicSourceVisualizer:
    def __init__(self):
        # Parâmetros da onda sísmica (igual ao seu SimulationConfig)
        self.epicenter_lat = -57.17      # latitude do epicentro (graus)
        self.epicenter_lon = -67.21      # longitude do epicentro (graus)
        self.seismic_amplitude = 1e-4    # amplitude máxima
        self.seismic_period = 20.0      # período da onda (s)
        self.seismic_velocity = 3400.0   # velocidade de propagação (m/s)
        self.earth_radius_km = 6371.0    # raio da Terra (km)
        
        # Cria grade de latitudes e longitudes (similar ao seu código)
        self.nf = 101  # número de longitudes
        self.nq = 101  # número de latitudes
        
        # Define limites da grade (baseado no seu código original)
        lon_in = -82.21  # longitude inicial (graus)
        dphi = 0.5      # espaçamento aproximado
        self.lon = lon_in + np.arange(self.nf) * dphi
        
        lat_in = -67.17  # latitude inicial (graus)
        dtheta = dphi    # mesmo espaçamento
        self.lat = lat_in + np.arange(self.nq) * dtheta
        
    def haversine_distance(self, lat, lon):
        """Calcula distância do epicentro usando fórmula de Haversine"""
        # Converte para radianos
        epi_lat_rad = np.radians(self.epicenter_lat)
        epi_lon_rad = np.radians(self.epicenter_lon)
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Fórmula de Haversine
        dlat = lat_rad - epi_lat_rad
        dlon = lon_rad - epi_lon_rad
        a = np.sin(dlat/2)**2 + np.cos(epi_lat_rad) * np.cos(lat_rad) * np.sin(dlon/2)**2
        angular_dist = 2 * np.arcsin(np.sqrt(a))
        
        # Distância linear (metros)
        distance = self.earth_radius_km * 1000 * angular_dist
        return distance
    
    def seismic_wave(self, t, lat, lon):
        """Calcula amplitude da onda sísmica no tempo t e posição (lat, lon)"""
        # Calcula distância do epicentro
        distance = self.haversine_distance(lat, lon)
        
        # Tempo de chegada da onda
        arrival_time = distance / self.seismic_velocity
        
        # Parâmetros da onda
        omega = 2 * np.pi / self.seismic_period
        
        # Máscara: só ativa onde a onda já chegou e ainda não passou (3 períodos)
        mask = (t >= arrival_time) & (t < arrival_time + 3 * self.seismic_period)
        
        # Amplitude decai com a distância
        amplitude = self.seismic_amplitude / (1 + distance / 100000)
        
        # Forma da onda (senoidal com envelope gaussiano)
        wave = np.sin(omega * (t - arrival_time)) * np.exp(-((t - arrival_time) / self.seismic_period)**2)
        
        return amplitude * wave * mask
    
    def plot_spatial_pattern(self, t, save_path=None):
        """Plota o padrão espacial da onda em um determinado tempo"""
        # Cria grade 2D
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
        
        # Calcula amplitude da onda
        amplitude = self.seismic_wave(t, lat_grid, lon_grid)
        
        # Cria figura
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Plot
        im = ax.pcolormesh(lon_grid, lat_grid, amplitude, 
                          cmap='seismic', shading='auto')
        
        # Marca o epicentro
        ax.plot(self.epicenter_lon, self.epicenter_lat, 'r*', 
                markersize=15, label=f'Epicentro\n({self.epicenter_lat}°, {self.epicenter_lon}°)')
        
        # Configurações do gráfico
        ax.set_xlabel('Longitude (graus)', fontsize=12)
        ax.set_ylabel('Latitude (graus)', fontsize=12)
        ax.set_title(f'Onda Sísmica - t = {t:.0f} s\n'
                    f'Período = {self.seismic_period} s, Velocidade = {self.seismic_velocity/1000:.1f} km/s',
                    fontsize=14)
        
        # Barra de cores
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.08)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Amplitude', fontsize=10)
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figura salva: {save_path}")
        
        plt.show()
        return amplitude
    
    def plot_time_series(self, lat, lon, t_max=2000, save_path=None):
        """Plota a série temporal da onda em um ponto específico"""
        # Cria array de tempos
        times = np.linspace(0, t_max, 500)
        
        # Calcula amplitude em cada tempo
        amplitudes = [self.seismic_wave(t, lat, lon) for t in times]
        
        # Cria figura
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot
        ax.plot(times, amplitudes, 'b-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        
        # Calcula distância e tempo de chegada
        distance = self.haversine_distance(lat, lon)
        arrival_time = distance / self.seismic_velocity
        
        # Marca tempo de chegada
        ax.axvline(x=arrival_time, color='r', linestyle='--', 
                  label=f'Tempo de chegada = {arrival_time:.0f} s')
        
        # Configurações
        ax.set_xlabel('Tempo (s)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title(f'Série Temporal da Onda Sísmica\n'
                    f'Posição: ({lat}°, {lon}°) - Distância do epicentro = {distance/1000:.1f} km',
                    fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figura salva: {save_path}")
        
        plt.show()
        return times, amplitudes
    
    def plot_animation_frames(self, times=[200, 400, 600, 800, 1000, 1200], save_dir=None):
        """Gera múltiplos frames para animação"""
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
        
        for t in times:
            print(f"Plotando t = {t} s...")
            save_path = None
            if save_dir:
                save_path = f"{save_dir}/seismic_wave_t_{t:04d}s.png"
            self.plot_spatial_pattern(t, save_path)
    
    def plot_distance_attenuation(self, max_distance_km=5000, save_path=None):
        """Plota a atenuação da amplitude com a distância"""
        # Cria array de distâncias
        distances_km = np.linspace(0, max_distance_km, 500)
        distances_m = distances_km * 1000
        
        # Calcula amplitude máxima para cada distância
        amplitude = self.seismic_amplitude / (1 + distances_m / 100000)
        
        # Cria figura
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot
        ax.plot(distances_km, amplitude, 'b-', linewidth=2)
        
        # Configurações
        ax.set_xlabel('Distância do Epicentro (km)', fontsize=12)
        ax.set_ylabel('Amplitude Máxima', fontsize=12)
        ax.set_title('Atenuação da Onda Sísmica com a Distância', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figura salva: {save_path}")
        
        plt.show()


# =============================================================================
# Execução principal
# =============================================================================

if __name__ == "__main__":
    # Cria visualizador
    viz = SeismicSourceVisualizer()
    
    print("=" * 60)
    print("VISUALIZADOR DE ONDA SÍSMICA")
    print("=" * 60)
    print(f"Epicentro: {viz.epicenter_lat}°, {viz.epicenter_lon}°")
    print(f"Período: {viz.seismic_period} s")
    print(f"Velocidade: {viz.seismic_velocity/1000:.1f} km/s")
    print(f"Amplitude máxima: {viz.seismic_amplitude}")
    print("=" * 60)
    
    # 1. Plota atenuação com distância
    print("\n1. Plotando atenuação da amplitude com a distância...")
    viz.plot_distance_attenuation()
    
    # 2. Plota padrão espacial em diferentes tempos
    print("\n2. Plotando padrão espacial da onda...")
    viz.plot_spatial_pattern(t=600)      # 600s = 10 minutos
    
    # 3. Plota série temporal em pontos específicos
    print("\n3. Plotando séries temporais...")
    
    # Ponto próximo ao epicentro
    viz.plot_time_series(lat=-57.17, lon=-67.21, t_max=2000)
    
    # Ponto a ~500 km do epicentro
    viz.plot_time_series(lat=-60.0, lon=-70.0, t_max=2000)
    
    # 4. Gera múltiplos frames (opcional - descomente se quiser)
    # print("\n4. Gerando frames para animação...")
    # viz.plot_animation_frames(
    #     times=[200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    #     save_dir="./seismic_frames"
    # )
    
    print("\n" + "=" * 60)
    print("Visualização concluída!")
    print("=" * 60)