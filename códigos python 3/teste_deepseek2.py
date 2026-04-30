from __future__ import annotations

from dataclasses import dataclass
import os
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _save_keogram_figure(
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    outfile: Path,
    cmap: str = "gaussian",
    normalize: bool = False,
) -> None:
    fig = plt.figure(figsize=(10, 6), facecolor="w", edgecolor="k")
    ax = plt.gca()
    field = safe_normalize(data) if normalize else data
    if field.shape == (len(x), len(y)):
        field = field.T
    elif field.shape != (len(y), len(x)):
        raise ValueError(
            f"Keogram data has shape {field.shape}, but expected {(len(y), len(x))} or {(len(x), len(y))}."
        )
    im = ax.pcolormesh(x, y, field, cmap=cmap, shading="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)
    plt.colorbar(im, cax=cax)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)

try:
    from mpl_toolkits.basemap import Basemap
except Exception:  # pragma: no cover
    Basemap = None


# =============================================================================
# Configuration dataclasses
# =============================================================================

@dataclass(slots=True)
class SimulationConfig:
    hemisphere: float = 1.0            # +1 south, -1 north
    use_magnetic_coords: bool = False  # original: i_mag
    use_inertia: bool = True           # original: i_inert
    use_ambient_force: bool = False    # original: i_amb
    solve_potential: bool = True       # original: i_pot
    use_parallel_dynamics: bool = True # original: i_o
    nt: int = 400
    dt: float = 20.0
    earth_radius_km: float = 6371.0
    output_dir: str = "outputs"
    make_plots: bool = True
    source_type: str = "gaussian"  # "gaussian" ou "seismic"
    # epicenter_lat: float = -67.17  # latitude do epicentro (graus)
    # epicenter_lon: float = -82.21  # longitude do epicentro (graus)
    epicenter_lat: float = -57.17  # latitude do epicentro (graus)
    epicenter_lon: float = -67.21  # longitude do epicentro (graus)
    # seismic_amplitude: float = 1e-50
    seismic_amplitude: float = 50
    seismic_period: float = 20.0  # período da onda (s)
    seismic_velocity: float = 3400.0  # velocidade de propagação (m/s)

    enable_temporal_anisotropy: bool = True
    anisotropy_factor: float = 3.0  # Aumente para ficar mais oval
    anisotropy_growth_rate: float = 0.5
    anisotropy_max: float = 10.0
    # NOVOS PARÂMETROS:
    anisotropic_propagation: bool = True  # Propaga com anisotropia
    horizontal_damping: float = 0.95  # Amortece propagação horizontal
    vertical_amplification: float = 1.05  # Amplifica propagação vertical


@dataclass(slots=True)
class Grid:
    alt: np.ndarray
    alt_i: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    r: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    dp_m: np.ndarray
    df_m: np.ndarray
    dq_m: np.ndarray
    dp_i: np.ndarray
    df_i: np.ndarray
    dq_i: np.ndarray
    y_f: np.ndarray
    z_f: np.ndarray
    y_sph: np.ndarray
    z_sph: np.ndarray
    alt_3: np.ndarray
    lat_3: np.ndarray
    lon_3: np.ndarray
    np_: int
    np_i: int
    nf: int
    nq: int


@dataclass(slots=True)
class Source:
    shape: np.ndarray
    mask_bound: np.ndarray
    iphi_s: int
    iq_s: int


@dataclass(slots=True)
class AmbientIonosphere:
    den_amb: np.ndarray
    nu_in: np.ndarray
    nu_en: np.ndarray
    tec_fac: float
    q_charge: float
    mag_o: float
    omega_i: float
    omega_e: float


@dataclass(slots=True)
class Mobility:
    mu_p_i: np.ndarray
    mu_h_i: np.ndarray
    mu_o_i: np.ndarray
    mu_p_e: np.ndarray
    mu_h_e: np.ndarray
    mu_o_e: np.ndarray


@dataclass(slots=True)
class State:
    rho: np.ndarray
    rho_d: np.ndarray
    rho_u: np.ndarray
    temp: np.ndarray
    r_g: np.ndarray
    den_t: np.ndarray
    pot: np.ndarray
    wp_m: np.ndarray
    wf_m: np.ndarray
    wq_m: np.ndarray
    wp_o: np.ndarray
    wf_o: np.ndarray
    wq_o: np.ndarray
    wp: np.ndarray
    wf: np.ndarray
    wq: np.ndarray
    up_tot: np.ndarray
    uf_tot: np.ndarray
    uq_tot: np.ndarray


# =============================================================================
# Utilities
# =============================================================================

def safe_normalize(arr: np.ndarray) -> np.ndarray:
    max_abs = float(np.max(np.abs(arr)))
    if max_abs == 0.0 or not np.isfinite(max_abs):
        return np.zeros_like(arr)
    return arr / max_abs


def parse_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "":
        return default
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def parse_float(value: object, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return default
    return float(text)


def env_bool(name: str, default: bool) -> bool:
    return parse_bool(os.getenv(name), default)


def env_float(name: str, default: float) -> float:
    return parse_float(os.getenv(name), default)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(str(raw).strip())


def can_prompt() -> bool:
    try:
        return os.isatty(0)
    except OSError:
        return False


# =============================================================================
# Core simulator
# =============================================================================

class GEOSTIDSSimulator:
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.grid = self._build_grid()
        self.source = self._build_source()
        self.ambient = self._build_ambient_ionosphere()
        self.state = self._initialize_state()
        self.time = np.zeros(self.cfg.nt, dtype=float)

    # -------------------------------------------------------------------------
    # Grid and source
    # -------------------------------------------------------------------------
    def _build_grid(self) -> Grid:
        r_ea = self.cfg.earth_radius_km

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

        alt = np.asarray(alt, dtype=float)
        alt_i = alt[10:]
        np_ = len(alt)
        np_i = len(alt_i)

        nf_target = 101
        dr = 15.0
        dphi = np.round(np.arctan(3.0 * dr / (r_ea + 300.0)), 5)
        lon_in = -np.radians(82.21)
        lon = lon_in + np.arange(nf_target) * dphi
        lon = np.degrees(lon)
        nf = len(lon)

        nq_target = 101
        dtheta = dphi
        lat_in = -np.radians(67.17)
        lat = self.cfg.hemisphere * (lat_in + np.arange(nq_target) * dtheta)
        lat = np.degrees(lat)
        nq = len(lat)

        r = np.zeros((np_, nf, nq), dtype=float)
        theta = np.zeros((np_, nf, nq), dtype=float)
        phi = np.zeros((np_, nf, nq), dtype=float)

        for i in range(np_):
            r[i, :, :] = (alt[i] + r_ea) * 1e3
        for j in range(nf):
            phi[:, j, :] = np.radians(lon[j])
        for k in range(nq):
            theta[:, :, k] = np.radians(lat[k])

        cos_theta = np.cos(theta)
        cos_theta = np.where(np.abs(cos_theta) < 1e-12, 1e-12, cos_theta)
        r_o = r / (cos_theta ** 2)
        delta = np.sqrt(1.0 + float(self.cfg.use_magnetic_coords) * 3.0 * (np.sin(theta) ** 2))
        q = r_o**3 * np.sin(theta) / np.where(np.abs(r**2) < 1e-12, 1e-12, r**2)
        p = r / cos_theta ** 2
        hq = (r / np.where(np.abs(r_o) < 1e-12, 1e-12, r_o)) ** 3 / delta
        hphi = r * cos_theta
        hp = cos_theta ** 3 / delta

        gp = np.gradient(p)
        gq = np.gradient(q)
        dp = np.abs(gp[0] + np.abs(gp[2]))
        dq = np.abs(gq[0] + gq[2])

        dp_m = dp * hp
        dq_m = dq * hq
        df_m = dphi * hphi

        x_f = hphi * np.tan(phi) * 1e-3
        y_f = p * hp * 1e-3
        z_f = q * hq * 1e-3
        y_sph = y_f * delta
        z_sph = z_f * delta

        alt_3 = y_sph - r_ea
        alt_3 = np.where(alt_3 < 0, 0, alt_3)
        lat_3 = np.degrees(np.arctan(z_f / (y_f + 1e-12)))
        lon_3 = np.degrees(np.arctan(x_f / (y_f + 1e-12)))

        dq_m[:, :, :] = dq_m.max()
        df_m[:, :, :] = df_m.max()
        dp_i = dp_m[np_ - np_i :, :, :]
        df_i = df_m[np_ - np_i :, :, :]
        dq_i = dq_m[np_ - np_i :, :, :]

        return Grid(
            alt=alt,
            alt_i=alt_i,
            lat=lat,
            lon=lon,
            r=r,
            theta=theta,
            phi=phi,
            dp_m=dp_m,
            df_m=df_m,
            dq_m=dq_m,
            dp_i=dp_i,
            df_i=df_i,
            dq_i=dq_i,
            y_f=y_f,
            z_f=z_f,
            y_sph=y_sph,
            z_sph=z_sph,
            alt_3=alt_3,
            lat_3=lat_3,
            lon_3=lon_3,
            np_=np_,
            np_i=np_i,
            nf=nf,
            nq=nq,
        )

    # def _build_source(self) -> Source:
    #     g = self.grid
    #     alt_o = 105.0
    #     sigma_p = 15.0
    #     var_p = np.exp(-((g.alt - alt_o) ** 2) / sigma_p**2)

    #     iphi_s = g.nf // 2
    #     lon_o = g.lon[iphi_s]
    #     sigma_x = 5.0 * np.degrees(np.arctan(3.0 * 15.0 / (self.cfg.earth_radius_km + 300.0))) / 2.0
    #     var_f = np.exp(-((g.lon - lon_o) ** 2) / sigma_x**2)

    #     iq_s = g.nq // 4
    #     lat_o = g.lat[iq_s]
    #     sigma_z = sigma_x
    #     var_q = np.exp(-((g.lat - lat_o) ** 2) / sigma_z**2)

    #     mask_bound = np.ones((g.np_, g.nf, g.nq), dtype=float)
    #     source_shape = var_p[:, None, None] * var_f[None, :, None] * var_q[None, None, :]

    #     j_m = g.nf // 2
    #     k_m = g.nq // 2
    #     rad_m = np.sqrt(j_m**2 + k_m**2)
    #     for j in range(g.nf):
    #         for k in range(g.nq):
    #             rad_o = np.sqrt((j - j_m) ** 2 + (k - k_m) ** 2)
    #             if rad_o >= 0.6 * rad_m:
    #                 mask_bound[:, j, k] = np.exp(-((2.0 * rad_o / rad_m) ** 2))

    #     return Source(source_shape, mask_bound, iphi_s, iq_s)
    
    # def _build_source(self) -> Source:
    #     g = self.grid
        
    #     if self.cfg.source_type == "seismic":
    #         # Para fonte sísmica, a forma espacial será calculada dinamicamente
    #         source_shape = np.ones((g.np_, g.nf, g.nq), dtype=float)
    #         iphi_s = np.argmin(np.abs(g.lon - self.cfg.epicenter_lon))
    #         iq_s = np.argmin(np.abs(g.lat - self.cfg.epicenter_lat))
    #     else:
    #         # Código original gaussiano
    #         alt_o = 105.0
    #         sigma_p = 15.0
    #         var_p = np.exp(-((g.alt - alt_o) ** 2) / sigma_p**2)
            
    #         iphi_s = g.nf // 2
    #         lon_o = g.lon[iphi_s]
    #         sigma_x = 5.0 * np.degrees(np.arctan(3.0 * 15.0 / (self.cfg.earth_radius_km + 300.0))) / 2.0
    #         var_f = np.exp(-((g.lon - lon_o) ** 2) / sigma_x**2)
            
    #         iq_s = g.nq // 4
    #         lat_o = g.lat[iq_s]
    #         sigma_z = sigma_x
    #         var_q = np.exp(-((g.lat - lat_o) ** 2) / sigma_z**2)
            
    #         source_shape = var_p[:, None, None] * var_f[None, :, None] * var_q[None, None, :]
        
    #     amplitude_factor = 150  # Mude este valor: 1, 10, 50, 100, 500
    #     source_shape = source_shape * amplitude_factor

    #     mask_bound = np.ones((g.np_, g.nf, g.nq), dtype=float)
    #     j_m = g.nf // 2
    #     k_m = g.nq // 2
    #     rad_m = np.sqrt(j_m**2 + k_m**2)
    #     for j in range(g.nf):
    #         for k in range(g.nq):
    #             rad_o = np.sqrt((j - j_m) ** 2 + (k - k_m) ** 2)
    #             if rad_o >= 0.6 * rad_m:
    #                 mask_bound[:, j, k] = np.exp(-((2.0 * rad_o / rad_m) ** 2))
    
    #     return Source(source_shape, mask_bound, iphi_s, iq_s)

    def _build_source(self) -> Source:
        g = self.grid
        
        if self.cfg.source_type == "seismic":
            # Para fonte sísmica, a forma espacial será calculada dinamicamente
            source_shape = np.ones((g.np_, g.nf, g.nq), dtype=float)
            iphi_s = np.argmin(np.abs(g.lon - self.cfg.epicenter_lon))
            iq_s = np.argmin(np.abs(g.lat - self.cfg.epicenter_lat))
        else:
            # Código original gaussiano
            alt_o = 105.0
            sigma_p = 15.0
            var_p = np.exp(-((g.alt - alt_o) ** 2) / sigma_p**2)
            
            iphi_s = g.nf // 2
            lon_o = g.lon[iphi_s]
            sigma_x = 5.0 * np.degrees(np.arctan(3.0 * 15.0 / (self.cfg.earth_radius_km + 300.0))) / 2.0
            var_f = np.exp(-((g.lon - lon_o) ** 2) / sigma_x**2)
            
            iq_s = g.nq // 4
            lat_o = g.lat[iq_s]
            sigma_z = sigma_x
            var_q = np.exp(-((g.lat - lat_o) ** 2) / sigma_z**2)
            
            source_shape = var_p[:, None, None] * var_f[None, :, None] * var_q[None, None, :]
        
        amplitude_factor = 400
        source_shape = source_shape * amplitude_factor
        
        # Calcula máscara de borda (sem alterações)
        mask_bound = np.ones((g.np_, g.nf, g.nq), dtype=float)
        j_m = g.nf // 2
        k_m = g.nq // 2
        rad_m = np.sqrt(j_m**2 + k_m**2)
        for j in range(g.nf):
            for k in range(g.nq):
                rad_o = np.sqrt((j - j_m) ** 2 + (k - k_m) ** 2)
                if rad_o >= 0.6 * rad_m:
                    mask_bound[:, j, k] = np.exp(-((2.0 * rad_o / rad_m) ** 2))
        
        return Source(source_shape, mask_bound, iphi_s, iq_s)

    @staticmethod
    def source_t(t: float) -> float:
        omega = 2.0 * np.pi / (5.0 * 60.0)
        t_o = 2000.0
        sigma_t = 1500.0
        return np.cos(0.0 * omega * t) * np.exp(-((t - t_o) ** 2) / sigma_t**2)
    
    def seismic_source(self, t: float, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        Gera uma onda sísmica que se propaga a partir do epicentro.
        
        Args:
            t: tempo atual (s)
            lat: array de latitudes (graus) - pode ser 2D
            lon: array de longitudes (graus) - pode ser 2D
        
        Returns:
            array com amplitude da fonte em cada ponto
        """
        # Converte epicentro para radianos
        epi_lat_rad = np.radians(self.cfg.epicenter_lat)
        epi_lon_rad = np.radians(self.cfg.epicenter_lon)
        
        # Converte pontos da grade para radianos
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Calcula distância angular (fórmula de Haversine)
        dlat = lat_rad - epi_lat_rad
        dlon = lon_rad - epi_lon_rad
        a = np.sin(dlat/2)**2 + np.cos(epi_lat_rad) * np.cos(lat_rad) * np.sin(dlon/2)**2
        angular_dist = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        
        # Converte distância angular para distância linear (metros)
        R_earth = self.cfg.earth_radius_km * 1000
        distance = R_earth * angular_dist
        
        # Tempo de chegada da onda
        arrival_time = distance / self.cfg.seismic_velocity
        
        # Onda sísmica (senoidal com envelope)
        omega = 2 * np.pi / self.cfg.seismic_period
        
        # Máscara: só ativa onde a onda já chegou
        mask = (t >= arrival_time) & (t < arrival_time + 5 * self.cfg.seismic_period)
        
        # Amplitude decai com a distância
        amplitude = self.cfg.seismic_amplitude / (1 + distance / 100000)
        
        # Forma da onda
        dt_arrival = t - arrival_time
        wave = np.sin(omega * dt_arrival) * np.exp(-(dt_arrival / (2 * self.cfg.seismic_period))**2)
        
        return amplitude * wave * mask

    
    def plot_gaussian_source(self, altitude_idx: int = None, save_path: Path = None) -> None:
        """
        Plota a fonte gaussiana para verificar sua amplitude.
        
        Args:
            altitude_idx: índice da altitude para plotar (None = altitude central)
            save_path: caminho para salvar a figura
        """
        g = self.grid
        
        if altitude_idx is None:
            # Encontra o índice da altitude central (105 km)
            altitude_idx = np.argmin(np.abs(g.alt - 105.0))
        
        # Dados na altitude selecionada
        data = self.source.shape[altitude_idx, :, :]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        im = ax.pcolormesh(g.lon, g.lat, data.T, cmap='hot', shading='auto')
        
        # Marca o centro
        ax.plot(g.lon[self.source.iphi_s], g.lat[self.source.iq_s], 'r*', 
                markersize=15, label=f'Centro\nAmplitude = {data.max():.2f}')
        
        ax.set_xlabel('Longitude (graus)', fontsize=12)
        ax.set_ylabel('Latitude (graus)', fontsize=12)
        ax.set_title(f'Fonte Gaussiana - Altitude = {g.alt[altitude_idx]:.1f} km\n'
                    f'Amplitude máxima = {data.max():.4f}',
                    fontsize=14)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.08)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Amplitude', fontsize=10)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figura salva: {save_path}")
        
        plt.show()

    def get_anisotropic_source(self, t: float, lon_2d: np.ndarray, lat_2d: np.ndarray) -> np.ndarray:
        """
        Calcula a fonte com anisotropia que aumenta com o tempo.
        A bolinha fica mais oval na vertical (achatada na horizontal) conforme t aumenta.
        """
        # Posição do centro da fonte
        center_lon = self.cfg.epicenter_lon if self.cfg.source_type == "seismic" else self.grid.lon[self.source.iphi_s]
        center_lat = self.cfg.epicenter_lat if self.cfg.source_type == "seismic" else self.grid.lat[self.source.iq_s]
        
        # Calcula distâncias em longitude e latitude
        delta_lon = lon_2d - center_lon
        delta_lat = lat_2d - center_lat
        
        # Fator de anisotropia que cresce com o tempo
        if self.cfg.enable_temporal_anisotropy:
            # Crescimento exponencial suave da anisotropia
            anisotropy = 1.0 + (self.cfg.anisotropy_factor - 1.0) * (1 - np.exp(-self.cfg.anisotropy_growth_rate * t))
            anisotropy = min(anisotropy, self.cfg.anisotropy_max)
        else:
            anisotropy = 1.0
        
        # Aplica anisotropia: estica na direção vertical (latitude) e comprime na horizontal (longitude)
        # Isso faz a bolinha ficar oval verticalmente
        delta_lat_stretched = delta_lat * anisotropy  # Estica na vertical
        delta_lon_compressed = delta_lon / anisotropy  # Comprime na horizontal
        
        # Calcula a distância efetiva (elíptica)
        # Usando diferentes escalas para cada direção
        lon_scale = 15.0  # Escala em longitude (graus)
        lat_scale = 15.0  # Escala em latitude (graus)
        
        distance_effective = np.sqrt(
            (delta_lon_compressed / lon_scale)**2 + 
            (delta_lat_stretched / lat_scale)**2
        )
        
        # Forma gaussiana com corte suave
        source_amplitude = np.exp(-distance_effective**2)
        
        # Adiciona anéis concêntricos (opcional, para simular ondas secundárias)
        if t > 50:  # Após alguns segundos
            # Ondas secundárias que se propagam
            ring_distance = np.sqrt(delta_lon**2 + delta_lat**2)
            wave_number = 0.15  # Número de onda
            ring_amplitude = 0.3 * np.cos(2 * np.pi * wave_number * ring_distance) * np.exp(-ring_distance**2 / 100)
            source_amplitude += ring_amplitude
        
        return source_amplitude

    def visualize_anisotropy_evolution(self, time_steps: list = [0, 50, 100, 200, 300]):
        """
        Visualiza como a forma da fonte evolui ao longo do tempo.
        """
        g = self.grid
        lon_2d, lat_2d = np.meshgrid(g.lon, g.lat, indexing='ij')
        
        fig, axes = plt.subplots(1, len(time_steps), figsize=(15, 4))
        if len(time_steps) == 1:
            axes = [axes]
        
        for idx, t in enumerate(time_steps):
            if self.cfg.source_type == "seismic":
                source_amp = self.seismic_source(t, lat_2d, lon_2d)
                if self.cfg.enable_temporal_anisotropy:
                    anisotropic = self.get_anisotropic_source(t, lon_2d, lat_2d)
                    source_amp = source_amp * anisotropic
            else:
                source_amp = self.get_anisotropic_source(t, lon_2d, lat_2d)
            
            im = axes[idx].pcolormesh(g.lon, g.lat, source_amp.T, 
                                    cmap='Reds', shading='auto', vmin=0, vmax=1)
            axes[idx].set_title(f't = {t} s')
            axes[idx].set_xlabel('Longitude')
            axes[idx].set_ylabel('Latitude')
            axes[idx].grid(True, alpha=0.3)
            
            # Marca o centro
            center_lon = self.cfg.epicenter_lon if self.cfg.source_type == "seismic" else g.lon[self.source.iphi_s]
            center_lat = self.cfg.epicenter_lat if self.cfg.source_type == "seismic" else g.lat[self.source.iq_s]
            axes[idx].plot(center_lon, center_lat, 'b*', markersize=10)
        
        plt.colorbar(im, ax=axes, orientation='horizontal', label='Amplitude', pad=0.2)
        plt.suptitle('Evolução da Anisotropia da Fonte ao Longo do Tempo')
        plt.tight_layout()
        plt.savefig(self.output_dir / "anisotropy_evolution.png", dpi=150)
        plt.show()

    # -------------------------------------------------------------------------
    # Atmosphere and ionosphere
    # -------------------------------------------------------------------------
    def atmosphere_profiles(self, alt: np.ndarray, dr: float = 15.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        a = 0.25e25 * 32 * 1.6e-27 * 1e-8
        c, d = -3.5, -19.5
        yr = 130.0

        def rho_profile(z: np.ndarray) -> np.ndarray:
            r_y = z / yr - 1.0
            return a * (np.exp(c * r_y) + np.exp(d * r_y))

        rho_o = rho_profile(alt)
        rho_od = rho_profile(alt - dr)
        rho_ou = rho_profile(alt + dr)

        a1, b1, c1, d1 = 310, 200, -2.0, 0.5
        a2, b2, c2, d2 = 220, 250, -0.2, 10.0
        a3, b3, c3, d3 = 220, 160, -0.25, 5.5
        a4, b4, c4, d4 = 120, 250, -1.5, 18.0
        a5, b5, c5, d5 = 220, 220, -0.25, 5.0
        y1, y2, y3, y4, y5 = 10.0, 75.0, 130.0, 100.0, 230.0

        r_y1 = alt / y1 - 1.0
        r_y2 = alt / y2 - 1.0
        r_y3 = alt / y3 - 1.0
        r_y4 = alt / y4 - 1.0
        r_y5 = alt / y5 - 1.0

        first = a1 - b1 / (np.exp(c1 * r_y1) + np.exp(d1 * r_y1))
        second = -a2 + b2 / (np.exp(c2 * r_y2) + np.exp(d2 * r_y2))
        third = a3 - b3 / (np.exp(c3 * r_y3) + np.exp(d3 * r_y3))
        fourth = -a4 + b4 / (np.exp(c4 * r_y4) + np.exp(d4 * r_y4))
        fifth = a5 - b5 / (np.exp(c5 * r_y5) + np.exp(d5 * r_y5))
        sixth = -720 + 1.25 * (first + second + fourth) + 8.5 * third + 1.5 * fifth
        temp_o = 0.65 * sixth

        return rho_o, rho_od, rho_ou, temp_o

    def _build_ambient_ionosphere(self) -> AmbientIonosphere:
        g = self.grid
        q_charge = 1.6e-19
        eps_o = 8.854e-12
        m_i = 1.67e-27
        z_i = 16.0

        mag_o = 0.25e-4
        omega_i = q_charge * mag_o / (z_i * m_i)
        omega_e = -omega_i * 1837.0

        den_100 = 1e18
        sc_h = 40.0
        den_neu = den_100 * np.exp(-(g.alt_i - 100.0) / sc_h)

        yp = 3e2
        enp = 2e12
        enpe = enp * 1e-2 * 1e1
        af, bf, ae, be = 1.0, -10.0, 5.0, -60.0
        r_n = g.alt_i / yp
        r_of, r_oe, r_ov = 0.9, 0.255, 0.5

        den_f = enp / (np.exp(af * (r_n - r_of)) + np.exp(bf * (r_n - r_of)))
        den_e = enpe / (np.exp(ae * (r_n - r_oe)) + np.exp(be * (r_n - r_oe)))
        den_ef = 5e-1 * enpe * (r_n + r_ov) ** 2 / 5.0
        den_i = den_f + den_e + den_ef

        int_fac = 50.0 * 60.0 * 1e3
        tec_fac = int_fac / 1e18

        t_i = 800.0
        a = 16.0
        nu_ii = 0.22 * den_i * 1e-6 / t_i ** 1.5
        nu_in_1 = 0 * nu_ii + 2.6e-9 * 1e-6 * (den_neu + den_i) * a ** (-0.5)
        nu_ei = 34 * den_i * 1e-6 / t_i ** 1.5
        nu_en_1 = 5.4e-16 * den_neu * np.sqrt(t_i) + nu_ei

        _w_p1 = np.sqrt(den_i * q_charge**2 / (eps_o * m_i * z_i))
        _ = _w_p1

        if len(den_i) > 40:
            den_i[40:] = den_i[40]
            nu_in_1[40:] = nu_in_1[40]
            nu_en_1[40:] = nu_en_1[40]

        den_amb = np.zeros((g.np_i, g.nf, g.nq), dtype=float)
        nu_in = np.zeros((g.np_i, g.nf, g.nq), dtype=float)
        nu_en = np.zeros((g.np_i, g.nf, g.nq), dtype=float)

        for k in range(g.nq):
            for j in range(g.nf):
                den_amb[:, j, k] = den_i
                nu_in[:, j, k] = nu_in_1
                nu_en[:, j, k] = nu_en_1

        return AmbientIonosphere(
            den_amb=den_amb,
            nu_in=nu_in,
            nu_en=nu_en,
            tec_fac=tec_fac,
            q_charge=q_charge,
            mag_o=mag_o,
            omega_i=omega_i,
            omega_e=omega_e,
        )

    def mobility(self) -> Mobility:
        dt = self.cfg.dt
        omega = (1.0 if self.cfg.use_inertia else 0.0) / dt
        amb = self.ambient

        nu_eff = amb.nu_in + omega
        kappa = amb.omega_i / np.where(np.abs(nu_eff) < 1e-30, 1e-30, nu_eff)
        mu_p_i = kappa / (amb.mag_o * (1 + kappa**2))
        mu_h_i = kappa**2 / (amb.mag_o * (1 + kappa**2))
        mu_o_i = kappa / amb.mag_o

        nu_eff = amb.nu_en + omega
        kappa = amb.omega_e / np.where(np.abs(nu_eff) < 1e-30, 1e-30, nu_eff)
        mu_p_e = kappa / (amb.mag_o * (1 + kappa**2))
        mu_h_e = kappa**2 / (amb.mag_o * (1 + kappa**2))
        mu_o_e = kappa / amb.mag_o

        return Mobility(mu_p_i, mu_h_i, mu_o_i, mu_p_e, mu_h_e, mu_o_e)

    def conductivity(self, den_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mob = self.mobility()
        q = self.ambient.q_charge
        sigma_p = q * den_t * (mob.mu_p_i - mob.mu_p_e)
        sigma_h = q * den_t * (mob.mu_h_i - mob.mu_h_e)
        sigma_o = q * den_t * (mob.mu_o_i - mob.mu_o_e)
        return sigma_p, sigma_h, sigma_o

    # -------------------------------------------------------------------------
    # State initialization
    # -------------------------------------------------------------------------
    def _initialize_state(self) -> State:
        g = self.grid
        rho = np.zeros((g.np_, g.nf, g.nq), dtype=float)
        rho_d = np.zeros_like(rho)
        rho_u = np.zeros_like(rho)
        temp = np.zeros_like(rho)
        r_g = np.zeros_like(rho)

        rho_o, rho_od, rho_ou, temp_o = self.atmosphere_profiles(g.alt)
        for k in range(g.nq):
            for j in range(g.nf):
                rho[:, j, k] = rho_o
                rho_d[:, j, k] = rho_od
                rho_u[:, j, k] = rho_ou
                temp[:, j, k] = temp_o
                r_g[:, j, k] = 150.0 * (1.0 + np.sqrt(g.alt + 5.0) / 5.0)

        shape_3d = (g.np_, g.nf, g.nq)
        shape_i = (g.np_i, g.nf, g.nq)
        zeros_3d = np.zeros(shape_3d, dtype=float)

        return State(
            rho=rho,
            rho_d=rho_d,
            rho_u=rho_u,
            temp=temp,
            r_g=r_g,
            den_t=self.ambient.den_amb.copy(),
            pot=np.zeros(shape_i, dtype=float),
            wp_m=zeros_3d.copy(),
            wf_m=zeros_3d.copy(),
            wq_m=zeros_3d.copy(),
            wp_o=zeros_3d.copy(),
            wf_o=zeros_3d.copy(),
            wq_o=zeros_3d.copy(),
            wp=zeros_3d.copy(),
            wf=zeros_3d.copy(),
            wq=zeros_3d.copy(),
            up_tot=np.zeros(shape_i, dtype=float),
            uf_tot=np.zeros(shape_i, dtype=float),
            uq_tot=np.zeros(shape_i, dtype=float),
        )

    # -------------------------------------------------------------------------
    # Physics kernels
    # -------------------------------------------------------------------------
    @staticmethod
    def agw_rhs(i_axis: int, delta: np.ndarray, press: np.ndarray, rho: np.ndarray,
                rho_d: np.ndarray, rho_u: np.ndarray, div_w: np.ndarray,
                div_flux: np.ndarray, w_gr_press: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gamma = 1.4
        flux_p = gamma * press * div_w
        grad_temp = np.gradient(flux_p)
        grad_flux = grad_temp[i_axis] / np.where(np.abs(delta) < 1e-30, 1e-30, delta)
        w_1 = grad_flux / np.where(np.abs(rho) < 1e-30, 1e-30, rho)

        grad_temp = np.gradient(press)
        grad_press = np.abs(grad_temp[i_axis]) / np.where(np.abs(delta) < 1e-30, 1e-30, delta)
        rho_m = (rho_u + 2.0 * rho + rho_d) / 4.0
        w_2 = -grad_press * div_flux / np.where(np.abs(rho_m**2) < 1e-30, 1e-30, rho_m**2)

        grad_temp = np.gradient(w_gr_press)
        w_3 = grad_temp[i_axis] / np.where(np.abs(delta) < 1e-30, 1e-30, delta)
        return w_1, w_2, w_3

    def electric_field_spherical(self, wp_g: np.ndarray, wf_g: np.ndarray, wq_g: np.ndarray,
                                 den_t: np.ndarray, pot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g = self.grid
        amb = self.ambient
        angle_I = g.theta[g.np_ - g.np_i :, :, :]
        sigma_p, sigma_h, sigma_o = self.conductivity(den_t)

        _s11 = sigma_p
        _s12 = -sigma_h * np.cos(angle_I)
        _s13 = -sigma_h * np.sin(angle_I)
        _s21 = -_s12
        _s22 = sigma_p * np.cos(angle_I) ** 2 + sigma_o * np.sin(angle_I) ** 2
        _s23 = (sigma_p - sigma_o) * np.sin(angle_I) * np.cos(angle_I)
        _s31 = -_s12
        _s32 = _s23
        _s33 = sigma_o * np.cos(angle_I) ** 2 + sigma_p * np.sin(angle_I) ** 2
        _ = (_s11, _s12, _s13, _s21, _s22, _s23, _s31, _s32, _s33)

        mob = self.mobility()

        def species_velocity(mu_11: np.ndarray, mu_o: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            kappa = mu_o * amb.mag_o
            mu_12 = -mu_11 * kappa * np.cos(angle_I)
            mu_13 = -mu_11 * kappa * np.sin(angle_I)
            mu_21 = -mu_12
            mu_22 = mu_11 * (1.0 + kappa**2 * np.sin(angle_I) ** 2)
            mu_23 = -mu_11 * kappa**2 * np.cos(angle_I) * np.sin(angle_I)
            mu_31 = -mu_13
            mu_32 = mu_23
            mu_33 = mu_11 * (1.0 + kappa**2 * np.cos(angle_I) ** 2)

            ele_w_p = -wf_g * amb.mag_o
            ele_w_f = wp_g * amb.mag_o
            ele_w_q = wq_g * amb.mag_o

            up = mu_11 * ele_w_p + mu_12 * ele_w_f + mu_13 * ele_w_q
            uf = mu_22 * ele_w_f + mu_21 * ele_w_p + mu_23 * ele_w_q
            uq = mu_33 * ele_w_q + mu_31 * ele_w_p + mu_32 * ele_w_f
            return up, uf, uq

        up_i, uf_i, uq_i = species_velocity(mob.mu_p_i, mob.mu_o_i)
        up_e, uf_e, uq_e = species_velocity(mob.mu_p_e, mob.mu_o_e)
        _ = (up_i, uf_i, uq_i)

        up_ele = np.zeros_like(up_e)
        uf_ele = np.zeros_like(uf_e)
        uq_ele = np.zeros_like(uq_e)

        self.state.up_tot = up_ele + up_e
        self.state.uf_tot = uf_ele + uf_e
        self.state.uq_tot = uq_ele + uq_e

        return pot, up_ele

    def update_ion_density(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        g = self.grid
        s = self.state
        den_o = s.den_t.copy()
        den_g = s.den_t.copy()

        for _ in range(8):
            flux_p = den_g * s.up_tot
            flux_f = den_g * s.uf_tot
            flux_q = den_g * s.uq_tot
            gr_p = np.gradient(flux_p)
            gr_f = np.gradient(flux_f)
            gr_q = np.gradient(flux_q)
            div_flux = gr_p[0] / g.dp_i + gr_f[1] / g.df_i + gr_q[2] / g.dq_i
            den_g = den_o - self.cfg.dt * div_flux

        s.den_t = den_g
        delta_den = (s.den_t - self.ambient.den_amb) / (s.den_t + 1e-30)
        dtec = self.ambient.tec_fac * (s.den_t.sum(axis=0) - self.ambient.den_amb.sum(axis=0))
        return s.den_t, delta_den, dtec

    # -------------------------------------------------------------------------
    # Time integration
    # -------------------------------------------------------------------------
    def step(self, i_t: int) -> dict:
        g = self.grid
        s = self.state
        dt = self.cfg.dt
        
        gamma = 1.4
        press = s.r_g * s.rho * s.temp
        c_s = np.sqrt(gamma * press / (s.rho + 1e-30))
        
        # Pré-calcula fonte sísmica se necessário (fora do loop i_sor)
        if self.cfg.source_type == "seismic":
            # Cria grade 2D de lat/lon
            lon_2d, lat_2d = np.meshgrid(g.lon, g.lat, indexing='ij')
            
            # Calcula amplitude sísmica no tempo atual (com anisotropia)
            if self.cfg.enable_temporal_anisotropy:
                # Usa fonte com anisotropia temporal
                anisotropic_source = self.get_anisotropic_source(self.time[i_t], lon_2d, lat_2d)
                seismic_amp = self.seismic_source(self.time[i_t], lat_2d, lon_2d)
                seismic_amp = seismic_amp * anisotropic_source
            else:
                seismic_amp = self.seismic_source(self.time[i_t], lat_2d, lon_2d)
            
            # Expande para 3D (todas as altitudes) e aplica máscara de borda
            source_3d = seismic_amp[None, :, :] * self.source.mask_bound
            # Fator de escala para amplitudes físicas
            source_3d = source_3d * 1e-3

        for i_sor in range(3):
            if i_sor == 0:
                if self.cfg.source_type == "seismic":
                    # Para fonte sísmica - aplica em todos os passos
                    s.wp_m = s.wp_m + source_3d
                    s.wp_o = s.wp_o + source_3d
                    s.wp = source_3d
                else:
                    # Código original para fonte gaussiana
                    if self.cfg.enable_temporal_anisotropy:
                        # Para fonte gaussiana, também aplica anisotropia
                        lon_2d, lat_2d = np.meshgrid(g.lon, g.lat, indexing='ij')
                        anisotropic_source = self.get_anisotropic_source(self.time[i_t], lon_2d, lat_2d)
                        # Aplica anisotropia na fonte gaussiana
                        anisotropic_3d = anisotropic_source[None, :, :] * self.source.shape
                        s.wp_m = s.wp_m + 1e-3 * self.source_t(self.time[i_t] - dt) * anisotropic_3d
                        s.wp_o = s.wp_o + 1e-3 * self.source_t(self.time[i_t]) * anisotropic_3d
                        s.wp = 1e-3 * self.source_t(self.time[i_t] + dt) * anisotropic_3d
                    else:
                        s.wp_m = s.wp_m + 1e-3 * self.source_t(self.time[i_t] - dt) * self.source.shape
                        s.wp_o = s.wp_o + 1e-3 * self.source_t(self.time[i_t]) * self.source.shape
                        s.wp = 1e-3 * self.source_t(self.time[i_t] + dt) * self.source.shape
                            
            if i_t == 1 and i_sor == 0:
                wp_g, wf_g, wq_g = s.wp_o.copy(), s.wf_o.copy(), s.wq_o.copy()
            else:
                wp_g, wf_g, wq_g = s.wp.copy(), s.wf.copy(), s.wq.copy()

            wp_0 = 2.0 * s.wp_o - s.wp_m
            wf_0 = 2.0 * s.wf_o - s.wf_m
            wq_0 = 2.0 * s.wq_o - s.wq_m

            gr_p = np.gradient(wp_g)
            gr_f = np.gradient(wf_g)
            gr_q = np.gradient(wq_g)
            div_w = gr_p[0] / g.dp_m + gr_f[1] / g.df_m + gr_q[2] / g.dq_m

            flux_p = s.rho * wp_g
            flux_f = s.rho * wf_g
            flux_q = s.rho * wq_g
            gr_p = np.gradient(flux_p)
            gr_f = np.gradient(flux_f)
            gr_q = np.gradient(flux_q)
            div_flux = gr_p[0] / g.dp_m + gr_f[1] / g.df_m + gr_q[2] / g.dq_m

            gr_pr = np.gradient(press)
            w_gr_press = wp_g * gr_pr[0] / g.dp_m + wf_g * gr_pr[1] / g.df_m + wq_g * gr_pr[2] / g.dq_m

            f_p = self.agw_rhs(0, g.dp_m, press, s.rho, s.rho_d, s.rho_u, div_w, div_flux, w_gr_press)
            f_f = self.agw_rhs(1, g.df_m, press, s.rho, s.rho_d, s.rho_u, div_w, div_flux, w_gr_press)
            f_q = self.agw_rhs(2, g.dq_m, press, s.rho, s.rho_d, s.rho_u, div_w, div_flux, w_gr_press)
            wp_1, wp_2, wp_3 = f_p
            wf_1, wf_2, wf_3 = f_f
            wq_1, wq_2, wq_3 = f_q

            visc_mu = 3.563e-7 * np.maximum(s.temp, 0.0)**0.71
            visc_ki = visc_mu / (s.rho + 1e-30)

            a = 1.0 / (g.dp_m**2 + 1e-30)
            flux = s.rho * s.wp_o
            grad_temp = np.gradient(flux)
            grad_flux = grad_temp[0] / g.dp_m
            w_visc_p = a * grad_flux * visc_mu / (s.rho + 1e-30) ** 2

            a = 1.0 / (g.df_m**2 + 1e-30)
            flux = s.rho * s.wf_o
            grad_temp = np.gradient(flux)
            grad_flux = grad_temp[1] / g.df_m
            w_visc_f = a * grad_flux * visc_mu / (s.rho + 1e-30) ** 2

            a = 1.0 / (g.dq_m**2 + 1e-30)
            flux = s.rho * s.wq_o
            grad_temp = np.gradient(flux)
            grad_flux = grad_temp[2] / g.dq_m
            w_visc_q = a * grad_flux * visc_mu / (s.rho + 1e-30) ** 2

            w_visc_rho = np.abs(w_visc_p) + np.abs(w_visc_f) + np.abs(w_visc_q)
            a = 1.0 / (g.dp_m**2 + 1e-30) + 1.0 / (g.df_m**2 + 1e-30) + 1.0 / (g.dq_m**2 + 1e-30)
            w_visc_w = visc_ki * a / dt
            w_visc = np.abs(w_visc_rho) + np.abs(w_visc_w)
            w_damp = np.exp(-0.5 * dt**2 * w_visc)

            fac_nl = wp_g.max() ** 2 / (g.dp_m * dt + 1e-30)
            w_nl = np.exp(-0.1 * dt**2 * fac_nl)
            wp_amp = w_nl * w_damp * self.source.mask_bound

            fac_nl = wf_g.max() / (g.df_m * dt + 1e-30) + wq_g.max() / (g.dq_m * dt + 1e-30)
            w_nl = np.exp(-0.001 * dt**2 * fac_nl)
            w_amp = w_nl * w_damp * self.source.mask_bound

            s.wp = wp_amp * (wp_0 + dt**2 * (wp_1 + wp_2 + wp_3))
            s.wf = w_amp * (wf_0 + dt**2 * (wf_1 + wf_2 + wf_3))
            s.wq = w_amp * (wq_0 + dt**2 * (wq_1 + wq_2 + wq_3))

        # APLICAR ANISOTROPIA CONTÍNUA NA PROPAGAÇÃO (depois do loop i_sor)
        if self.cfg.enable_temporal_anisotropy:
            # Calcula fator de anisotropia baseado no tempo
            t = self.time[i_t]
            anisotropy = 1.0 + (self.cfg.anisotropy_factor - 1.0) * (1 - np.exp(-self.cfg.anisotropy_growth_rate * t))
            anisotropy = min(anisotropy, self.cfg.anisotropy_max)
            
            # Cria máscara elíptica para moldar a onda
            lon_2d, lat_2d = np.meshgrid(g.lon, g.lat, indexing='ij')
            center_lon = g.lon[self.source.iphi_s]
            center_lat = g.lat[self.source.iq_s]
            
            # Distâncias normalizadas
            delta_lon_norm = (lon_2d - center_lon) / 20.0
            delta_lat_norm = (lat_2d - center_lat) / 20.0
            
            # Máscara elíptica (mais alongada na vertical com anisotropia crescente)
            ellipse_mask = np.exp(-(delta_lon_norm**2 + (delta_lat_norm / anisotropy)**2))
            
            # Aplica máscara em todas as altitudes
            for i in range(s.wp.shape[0]):
                s.wp[i, :, :] = s.wp[i, :, :] * ellipse_mask
                s.wf[i, :, :] = s.wf[i, :, :] * ellipse_mask
                s.wq[i, :, :] = s.wq[i, :, :] * ellipse_mask

        s.wp_m = s.wp_o.copy()
        s.wp_o = s.wp.copy()
        s.wf_m = s.wf_o.copy()
        s.wf_o = s.wf.copy()
        s.wq_m = s.wq_o.copy()
        s.wq_o = s.wq.copy()

        rho_oo = s.rho.copy()
        temp_oo = s.temp.copy()

        flux_p = s.rho * s.wp
        flux_f = s.rho * s.wf
        flux_q = s.rho * s.wq
        gr_p = np.gradient(flux_p)
        gr_f = np.gradient(flux_f)
        gr_q = np.gradient(flux_q)
        div_flux = gr_p[0] / g.dp_m + gr_f[1] / g.df_m + gr_q[2] / g.dq_m

        s.rho = rho_oo - 0.0 * dt * div_flux
        if np.any(s.rho < 0):
            s.rho = rho_oo
        s.rho_d[0:g.np_ - 1, :, :] = s.rho[1:g.np_, :, :]
        s.rho_u[1:g.np_, :, :] = s.rho[0:g.np_ - 1, :, :]

        flux_p = s.temp * s.wp
        flux_f = s.temp * s.wf
        flux_q = s.temp * s.wq
        gr_p = np.gradient(flux_p)
        gr_f = np.gradient(flux_f)
        gr_q = np.gradient(flux_q)
        div_flux = gr_p[0] / g.dp_m + gr_f[1] / g.df_m + gr_q[2] / g.dq_m
        s.temp = temp_oo - 0.25 * dt * div_flux

        wp_i = s.wp[g.np_ - g.np_i :, :, :]
        wf_i = s.wf[g.np_ - g.np_i :, :, :]
        wq_i = s.wq[g.np_ - g.np_i :, :, :]

        if self.cfg.use_magnetic_coords:
            warnings.warn(
                "Magnetic-coordinate path requested, but the original routine is incomplete. Falling back to spherical coordinates.",
                RuntimeWarning,
            )
        s.pot, _up_ele = self.electric_field_spherical(wp_i, wf_i, wq_i, s.den_t, s.pot)
        _ = _up_ele

        den_t, delta_den, dtec = self.update_ion_density()
        sigma_p, sigma_h, sigma_o = self.conductivity(self.ambient.den_amb)
        _ = (sigma_p, sigma_h, sigma_o)

        self.time[i_t + 1] = self.time[i_t] + dt

        return {
            "dtec": dtec,
            "delta_den": delta_den,
            "wp_max": float(np.max(s.wp)),
            "wf_max": float(np.max(s.wf)),
            "wq_max": float(np.max(s.wq)),
            "up_max": float(np.max(s.up_tot)),
            "forcing_amplitude": float(np.max(s.wp[0, :, :])),
            "pot": s.pot,
            "den_t": den_t,
        }

    # -------------------------------------------------------------------------
    # Plotting and outputs
    # -------------------------------------------------------------------------
    def quicklook(self, dtec: np.ndarray, pot: np.ndarray, step: int) -> None:
        if not self.cfg.make_plots:
            return

        g = self.grid
        s = self.state
        vm = 0.05
        fr_gr_2 = s.wp[0, :, :]

        fig = plt.figure(figsize=(18, 12), facecolor="w", edgecolor="k")
        level_idx = min(16, s.wp.shape[0] - 1)
        lat_pl = g.lat_3[level_idx, :, :]
        lon_pl = g.lon_3[level_idx, :, :]
        data = s.wp[level_idx, :, :]
        pot_idx = min(11, pot.shape[0] - 1)
        pot_norm = safe_normalize(pot[pot_idx, :, :])

        if Basemap is not None:
            ax = plt.subplot(121)
            basemap = Basemap(projection="ortho", lat_0=lat_pl.mean(), lon_0=lon_pl.mean(), resolution="l")
            basemap.etopo(alpha=0.5)
            basemap.drawmeridians(np.arange(lon_pl.min(), lon_pl.max() + 10, 10))
            basemap.drawparallels(np.arange(lat_pl.min(), lat_pl.max() + 10, 10))
            x, y = basemap(lon_pl, lat_pl)
            plt.contour(x, y, safe_normalize(np.abs(fr_gr_2)), colors="g", linewidths=1, levels=[0.05, 0.5], alpha=0.2)
            im = plt.pcolormesh(x, y, safe_normalize(data), cmap=plt.cm.seismic, vmax=vm, vmin=-vm, alpha=0.5, shading="auto")
            plt.title(f"AGW: MAX={data.max():.3g}")
            plt.axis("off")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            plt.colorbar(im, cax=cax)

            ax = plt.subplot(122)
            basemap = Basemap(projection="ortho", lat_0=lat_pl.mean(), lon_0=lon_pl.mean(), resolution="l")
            basemap.etopo(alpha=0.5)
            basemap.drawmeridians(np.arange(lon_pl.min(), lon_pl.max() + 10, 10))
            basemap.drawparallels(np.arange(lat_pl.min(), lat_pl.max() + 10, 10))
            x, y = basemap(lon_pl, lat_pl)
            plt.contour(x, y, safe_normalize(np.abs(fr_gr_2)), colors="g", linewidths=1, levels=[0.05, 0.5], alpha=0.2)
            if np.any(np.abs(pot_norm) > 0):
                cs = plt.contour(zoom(x, 2), zoom(y, 2), zoom(pot_norm, 2), colors="k", linewidths=1, levels=[-0.5, -0.05, -0.00005, 0.00005, 0.05, 0.5])
                plt.clabel(cs, fontsize=8)
            im = plt.pcolormesh(x, y, safe_normalize(dtec), cmap=plt.cm.seismic, vmax=vm, vmin=-vm, alpha=0.5, shading="auto")
            plt.title(f"ΔTEC: MAX={dtec.max():.3g}")
            plt.axis("off")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            plt.colorbar(im, cax=cax)
        else:
            ax = plt.subplot(121)
            im = plt.pcolormesh(g.lon, g.lat, safe_normalize(data).T, cmap=plt.cm.seismic, vmax=vm, vmin=-vm, shading="auto")
            plt.contour(g.lon, g.lat, safe_normalize(np.abs(fr_gr_2)).T, colors="g", linewidths=1, levels=[0.05, 0.5], alpha=0.2)
            plt.xlabel("Longitude (deg)")
            plt.ylabel("Latitude (deg)")
            plt.title(f"AGW: MAX={data.max():.3g}")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            plt.colorbar(im, cax=cax)

            ax = plt.subplot(122)
            im = plt.pcolormesh(g.lon, g.lat, safe_normalize(dtec).T, cmap=plt.cm.seismic, vmax=vm, vmin=-vm, shading="auto")
            if np.any(np.abs(pot_norm) > 0):
                cs = plt.contour(g.lon, g.lat, pot_norm.T, colors="k", linewidths=1, levels=[-0.5, -0.05, -0.00005, 0.00005, 0.05, 0.5])
                plt.clabel(cs, fontsize=8)
            plt.xlabel("Longitude (deg)")
            plt.ylabel("Latitude (deg)")
            plt.title(f"ΔTEC: MAX={dtec.max():.3g}")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            plt.colorbar(im, cax=cax)

        fig.tight_layout()
        outfile = self.output_dir / f"quicklook_{step:05d}.png"
        fig.savefig(outfile, dpi=150)
        plt.show(block=False)
        plt.pause(0.1)
        print(f"Saved plot: {outfile}")
        plt.close(fig)

    def save_outputs(self, dtec_keo: np.ndarray, uplift_keo: np.ndarray, wp_keo: np.ndarray, wh_keo: np.ndarray) -> None:
        np.save(self.output_dir / "uplift_keo_t.npy", uplift_keo)
        np.save(self.output_dir / "wp_keo_t.npy", wp_keo)
        np.save(self.output_dir / "wh_keo_t.npy", wh_keo)
        np.save(self.output_dir / "dtec_keo_t.npy", dtec_keo)

    def save_keograms(
        self,
        time_keo: np.ndarray,
        dtec_keo: np.ndarray,
        uplift_keo: np.ndarray,
        wp_keo: np.ndarray,
        wh_keo: np.ndarray,
        pot_keo: np.ndarray,
    ) -> None:
        g = self.grid
        lat_idx = self.source.iq_s
        lon_idx = self.source.iphi_s

        # Tempo × longitude para latitude fixa
        _save_keogram_figure(
            time_keo,
            g.lon,
            dtec_keo[:, :, lat_idx].T,
            "Keograma ΔTEC (tempo × longitude)",
            "Tempo (s)",
            "Longitude (deg)",
            self.output_dir / "keogram_dtec_time_longitude.png",
            normalize=True,
        )
        
        _save_keogram_figure(
            time_keo,
            g.lon,
            wp_keo[:, :, lat_idx].T,
            "Keograma Velocidade Vertical (tempo × longitude)",
            "Tempo (s)",
            "Longitude (deg)",
            self.output_dir / "keogram_wp_time_longitude.png",
            normalize=True,
        )
        
        _save_keogram_figure(
            time_keo,
            g.lon,
            wh_keo[:, :, lat_idx].T,
            "Keograma Velocidade Horizontal (tempo × longitude)",
            "Tempo (s)",
            "Longitude (deg)",
            self.output_dir / "keogram_wh_time_longitude.png",
            normalize=True,
        )
        
        _save_keogram_figure(
            time_keo,
            g.lon,
            pot_keo[:, :, lat_idx].T,
            "Keograma Potencial Elétrico (tempo × longitude)",
            "Tempo (s)",
            "Longitude (deg)",
            self.output_dir / "keogram_pot_time_longitude.png",
            normalize=True,
        )
        
        _save_keogram_figure(
            time_keo,
            g.lon,
            uplift_keo[:, :, lat_idx].T,
            "Keograma Uplift (tempo × longitude)",
            "Tempo (s)",
            "Longitude (deg)",
            self.output_dir / "keogram_uplift_time_longitude.png",
            normalize=True,
        )

        # Tempo × latitude para longitude fixa
        _save_keogram_figure(
            time_keo,
            g.lat,
            dtec_keo[:, lon_idx, :].T,
            "Keograma ΔTEC (tempo × latitude)",
            "Tempo (s)",
            "Latitude (deg)",
            self.output_dir / "keogram_dtec_time_latitude.png",
            normalize=True,
        )

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------
    def run(self) -> dict:
        g = self.grid
        s = self.state

        dtec_keo = np.zeros((self.cfg.nt // 5, g.nf, g.nq), dtype=float)
        uplift_keo = np.zeros((self.cfg.nt // 5, g.nf, g.nq), dtype=float)
        wp_keo = np.zeros((self.cfg.nt // 5, g.nf, g.nq), dtype=float)
        wh_keo = np.zeros((self.cfg.nt // 5, g.nf, g.nq), dtype=float)
        pot_keo = np.zeros((self.cfg.nt // 5, g.nf, g.nq), dtype=float)
        time_keo = np.zeros((self.cfg.nt // 5,), dtype=float)
        i_w = 0

        wp_prop = np.zeros((self.cfg.nt, g.np_), dtype=float)
        uplift_t = np.zeros((self.cfg.nt, g.np_), dtype=float)

        i_t = 1
        while i_t < self.cfg.nt - 1 and np.max(s.wp) < 500 and np.max(s.up_tot) < 1500:
            result = self.step(i_t)
            dtec = result["dtec"]
            pot = result["pot"]

            wh = np.sqrt(s.wf**2 + s.wq**2)

            if i_t % 5 == 0 and i_w < dtec_keo.shape[0]:
                dtec_keo[i_w, :, :] = dtec
                uplift_keo[i_w, :, :] = s.wp[0, :, :]
                wp_store_idx = min(13, s.wp.shape[0] - 1)
                pot_store_idx = min(3, pot.shape[0] - 1)
                wp_keo[i_w, :, :] = s.wp[wp_store_idx, :, :]
                wh_keo[i_w, :, :] = wh[wp_store_idx, :, :]
                pot_keo[i_w, :, :] = pot[pot_store_idx, :, :]
                time_keo[i_w] = self.time[i_t]
                i_w += 1

            wp_prop[i_t, :] = s.wp[:, self.source.iphi_s, self.source.iq_s]
            uplift_t[i_t, :] = s.wp[0, self.source.iphi_s, self.source.iq_s]

            if i_t % 5 == 0:
                self.quicklook(dtec, pot, i_t)

            print("=" * 54)
            print(f"TIME STEP = {i_t:4d} | t = {self.time[i_t]:8.1f} s")
            print(f"FORCING AMPLITUDE = {result['forcing_amplitude']:.5f}")
            print(f"ATMOS W_MAX = ({result['wp_max']:.2f}, {result['wf_max']:.2f}, {result['wq_max']:.2f})")
            print(f"IONO U_MAX  = {result['up_max']:.2f}")

            i_t += 1

        valid = max(i_w, 1)
        self.save_outputs(dtec_keo[:valid], uplift_keo[:valid], wp_keo[:valid], wh_keo[:valid])
        self.save_keograms(
            time_keo[:valid],
            dtec_keo[:valid],
            uplift_keo[:valid],
            wp_keo[:valid],
            wh_keo[:valid],
            pot_keo[:valid],
        )
        # i_t = 1
        # while i_t < self.cfg.nt - 1:
        #     result = self.step(i_t)
        
        # # DEBUG: veja a posição da máxima amplitude
        #     max_pos = np.unravel_index(np.argmax(s.wp[0, :, :]), s.wp[0, :, :].shape)
        # print(f"t={self.time[i_t]:.0f}s: Max AGW em lat={self.grid.lat[max_pos[0]]:.2f}°, lon={self.grid.lon[max_pos[1]]:.2f}°")

        return {
            "time": self.time,
            "dtec_keo": dtec_keo,
            "uplift_keo": uplift_keo,
            "wp_keo": wp_keo,
            "wh_keo": wh_keo,
            "pot_keo": pot_keo,
            "time_keo": time_keo,
            "wp_prop": wp_prop,
            "uplift_t": uplift_t,
        }


# =============================================================================
# CLI helpers
# =============================================================================

def prompt_bool(prompt: str, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        raw = input(f"{prompt} {suffix}: ").strip().lower()
    except OSError:
        return default
    except EOFError:
        return default
    if not raw:
        return default
    return parse_bool(raw, default)


def prompt_float(prompt: str, default: float) -> float:
    try:
        raw = input(f"{prompt} [{default}]: ").strip()
    except OSError:
        return default
    except EOFError:
        return default
    return default if not raw else float(raw)


def build_config_from_sources(interactive: bool | None = None) -> SimulationConfig:
    if interactive is None:
        interactive = can_prompt() and not env_bool("GEOS_NONINTERACTIVE", False)

    defaults = SimulationConfig(
        hemisphere=env_float("GEOS_HEMISPHERE", 1.0),
        use_magnetic_coords=env_bool("GEOS_USE_MAGNETIC_COORDS", False),
        use_inertia=env_bool("GEOS_USE_INERTIA", True),
        use_ambient_force=env_bool("GEOS_USE_AMBIENT_FORCE", False),
        solve_potential=env_bool("GEOS_SOLVE_POTENTIAL", True),
        use_parallel_dynamics=env_bool("GEOS_USE_PARALLEL_DYNAMICS", True),
        nt=env_int("GEOS_NT", 400),
        dt=env_float("GEOS_DT", 20.0),
        earth_radius_km=env_float("GEOS_EARTH_RADIUS_KM", 6371.0),
        output_dir=os.getenv("GEOS_OUTPUT_DIR", "outputs"),
        make_plots=env_bool("GEOS_MAKE_PLOTS", True),
    )

    if not interactive:
        return defaults

    return SimulationConfig(
        hemisphere=prompt_float("HEMISPHERE (+1 south, -1 north)", defaults.hemisphere),
        use_magnetic_coords=prompt_bool("Use magnetic coordinates?", defaults.use_magnetic_coords),
        use_inertia=prompt_bool("Use inertia?", defaults.use_inertia),
        use_ambient_force=prompt_bool("Use ambient force?", defaults.use_ambient_force),
        solve_potential=prompt_bool("Solve polarization potential?", defaults.solve_potential),
        use_parallel_dynamics=prompt_bool("Use parallel dynamics?", defaults.use_parallel_dynamics),
        nt=defaults.nt,
        dt=defaults.dt,
        earth_radius_km=defaults.earth_radius_km,
        output_dir=defaults.output_dir,
        make_plots=prompt_bool("Generate quicklook plots?", defaults.make_plots),
    )


def build_config_from_cli() -> SimulationConfig:
    return build_config_from_sources(interactive=None)


# =============================================================================
# Lightweight tests
# =============================================================================

def _run_self_tests() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 5.0])
    z2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tmpfile = Path("_test_keogram.png")
    _save_keogram_figure(x, y, z2, "test", "x", "y", tmpfile, normalize=False)
    assert tmpfile.exists()
    tmpfile.unlink()

    z2_t = z2.T
    tmpfile2 = Path("_test_keogram_t.png")
    _save_keogram_figure(x, y, z2_t, "test", "x", "y", tmpfile2, normalize=False)
    assert tmpfile2.exists()
    tmpfile2.unlink()

    assert parse_bool("yes", False) is True
    assert parse_bool("No", True) is False
    assert parse_bool("", True) is True
    assert parse_float("3.5", 1.0) == 3.5
    assert parse_float("", 1.0) == 1.0

    os.environ["GEOS_HEMISPHERE"] = "-1"
    os.environ["GEOS_MAKE_PLOTS"] = "0"
    os.environ["GEOS_NT"] = "12"
    cfg = build_config_from_sources(interactive=False)
    assert cfg.hemisphere == -1.0
    assert cfg.make_plots is False
    assert cfg.nt == 12

    z = np.zeros((2, 2))
    assert np.array_equal(safe_normalize(z), z)
    a = np.array([[1.0, -2.0]])
    assert np.allclose(safe_normalize(a), np.array([[0.5, -1.0]]))

    for key in ["GEOS_HEMISPHERE", "GEOS_MAKE_PLOTS", "GEOS_NT"]:
        os.environ.pop(key, None)


# =============================================================================
# Entrypoint
# =============================================================================

# def main() -> None:
#     if env_bool("GEOS_RUN_SELF_TESTS", False):
#         _run_self_tests()
#         print("Self-tests passed.")
#         return

#     cfg = build_config_from_cli()
#     sim = GEOSTIDSSimulator(cfg)
    
#     # Se for fonte gaussiana, plota para ver a amplitude
#     if cfg.source_type == "gaussian":
#         print(f"\nAmplitude da fonte gaussiana: {np.max(sim.source.shape):.2f}")
#         sim.plot_gaussian_source()
        
#         resposta = input("\nContinuar com a simulação? (s/n): ")
#         if resposta.lower() != 's':
#             return
    
#     sim.run()
#     if cfg.make_plots:
#         plt.show()

# if __name__ == "__main__":
#     main()

def main() -> None:
    if env_bool("GEOS_RUN_SELF_TESTS", False):
        _run_self_tests()
        print("Self-tests passed.")
        return

        # Configuração direta para teste da anisotropia
    cfg = SimulationConfig(
        source_type="gaussian",
        enable_temporal_anisotropy=True,
        anisotropy_factor=5.0,  # AUMENTE para ficar mais oval
        anisotropy_growth_rate=0.2,  # MAIS LENTO para manter a forma por mais tempo
        anisotropy_max=15.0,  # MAIOR para ficar mais alongada
        nt=300,
        dt=20.0,
        make_plots=True,
        output_dir="outputs_anisotropy_v2"
    )
        
    sim = GEOSTIDSSimulator(cfg)
    
    # Visualiza a evolução da anisotropia antes de rodar
    sim.visualize_anisotropy_evolution(time_steps=[0, 50, 100, 150, 200])
    
    # Roda a simulação
    sim.run()
    
    if cfg.make_plots:
        plt.show()

if __name__ == "__main__":
    main()