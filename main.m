% ============================================================================
% GEOS_TIDS - Simulação de Ondas de Gravidade na Ionosfera (Versão MATLAB)
% ============================================================================

clear; 
clc; 
close all;

% ============================================================================
% PARTE 1: Script principal (código executável)
% ============================================================================

% ==============================================================================
% Variáveis globais de configuração da simulação
% ==============================================================================

% Entrada do usuário
i_hemi = input('HEMISPHERE, +1 FOR SOUTHERN: ');
i_mag = input('0 FOR SPHERICAL: '); 
i_inert = input('INERTIA OR COLLISIONAL, 0 FOR COLLISIONAL: ');
i_amb = input('AMBIENT FORCE, 0 FOR NO AMBIENT: ');
i_pot = input('POLARIZATION POTENTIAL, 1 FOR SOLVING POTENTIAL: ');
i_o = input('PARALLEL DYNAMICS, 1 FOR YES: ');

fprintf('Configurações: %.0f %.0f %.0f\n', i_o, i_amb, i_pot);

% ==============================================================================
% Configuração de tempo UT (Universal Time) 
% ==============================================================================

fprintf('\n=== CONFIGURAÇÃO DE TEMPO UT ===\n');
fprintf('A simulação começa em t=0s e vai até ~8000s (~2.2h)\n');
fprintf('O pico do forçante ocorre em t=2000s (~33 min)\n');

% Horário UT do terremoto
HORA_UT_TERREMOTO = 2.0;   % 2:00 UT

% Horário final desejado (6:00 UT)
HORA_UT_FINAL = 6.0;       % 6:00 UT

% Calcula o tempo total de simulação necessário
tempo_total_horas = HORA_UT_FINAL - HORA_UT_TERREMOTO;  % 4 horas
tempo_total_segundos = tempo_total_horas * 3600;        % 14400 segundos

% Define o horário UT inicial (t=0 = horário do terremoto)
hora_ut_inicial = HORA_UT_TERREMOTO;

fprintf('Horário do terremoto: %.0f:00 UT\n', HORA_UT_TERREMOTO);
fprintf('Horário final: %.0f:00 UT\n', HORA_UT_FINAL);
fprintf('Duração da simulação: %.1f horas (%.0f segundos)\n', tempo_total_horas, tempo_total_segundos);
fprintf('t=0 corresponde a %.0f:00 UT\n', hora_ut_inicial);
fprintf('=================================\n\n');


% ============================================================================
% Configurações Iniciais
% ============================================================================

% Pasta onde as figuras serão salvas
pasta_figs = 'figs_output';
if ~exist(pasta_figs, 'dir')
    mkdir(pasta_figs);
end

% Configurações de estilo dos gráficos (MATLAB usa defaults diferentes)
% Para fonte em negrito:
set(0, 'DefaultAxesFontWeight', 'bold');
set(0, 'DefaultTextFontWeight', 'bold');
set(0, 'DefaultAxesFontName', 'serif');
set(0, 'DefaultTextFontName', 'serif');
set(0, 'DefaultAxesFontSize', 14);
set(0, 'DefaultTextFontSize', 14);

fprintf('\n=== SIMULAÇÃO CONFIGURADA ===\n');
fprintf('Código em desenvolvimento - funções definidas no final do arquivo\n');
fprintf('=============================\n');

% ===== CHAMADA DAS FUNÇÕES =====
[r_ea, np, nf, nq, np_i, alt, alt_i, lat, lon, r, theta, phi, ...
 dp_m, df_m, dq_m, dp_i, df_i, dq_i, y_f, z_f, y_sph, z_sph, ...
 alt_3, lon_3, lat_3, dr, dphi, dtheta] = sim_vol(i_hemi, i_mag);
[source_shape, mask_bound, iphi_s, iq_s] = source_space(alt, lon, lat, np, nf, nq, dphi, dtheta, alt_3);
[nu_in, nu_en, mag_o, q_charge, omega_i, omega_e, tec_fac, den_amb] = iono_amb(alt_i, np_i, nf, nq);

% ============================================================================
% INICIALIZAÇÃO DOS ESTADOS
% ============================================================================

% global t
t = zeros(400, 1); 

% Inicializa arrays 3D
rho = zeros(np, nf, nq);      % Densidade neutra (kg/m³)
rho_d = zeros(np, nf, nq);    % Densidade no ponto abaixo
rho_u = zeros(np, nf, nq);    % Densidade no ponto acima
temp = zeros(np, nf, nq);     % Temperatura (K)
r_g = zeros(np, nf, nq);      % Constante dos gases * temperatura (ajuste empírico)

% ===== PERFIS ATMOSFÉRICOS =====
[rho_o, rho_od, rho_ou, temp_o] = atmos(alt, dr);

% Preenche os arrays 3D com os perfis 1D
for k = 1:nq
    for j = 1:nf
        rho(:, j, k) = rho_o;
        rho_d(:, j, k) = rho_od;
        rho_u(:, j, k) = rho_ou;
        temp(:, j, k) = temp_o;
        r_g(:, j, k) = 150.0 * (1.0 + sqrt(alt + 5.0) / 5.0);
    end
end

% ===== CONFIGURAÇÃO DO LOOP TEMPORAL =====
nt = 400;                      % Número máximo de passos de tempo
t = zeros(nt, 1);              % Vetor de tempo

% Arrays para armazenar campos nos passos de tempo
wp_m = zeros(np, nf, nq);      % Velocidade vertical (-1 passo)
wf_m = zeros(np, nf, nq);      % Velocidade horizontal (-1 passo)
wq_m = zeros(np, nf, nq);      % Velocidade paralela (-1 passo)

wp_o = zeros(np, nf, nq);      % Velocidade vertical (passo anterior)
wf_o = zeros(np, nf, nq);      % Velocidade horizontal (passo anterior)
wq_o = zeros(np, nf, nq);      % Velocidade paralela (passo anterior)

wp = zeros(np, nf, nq);        % Velocidade vertical (passo atual)
wf = zeros(np, nf, nq);        % Velocidade horizontal (passo atual)
wq = zeros(np, nf, nq);        % Velocidade paralela (passo atual)

up_tot = zeros(np_i, nf, nq);  % Velocidade total na ionosfera

i_t = 2;  % Contador de tempo (MATLAB usa 1-indexado, então começamos em 2)

while (i_t < nt) && (max(wp(:)) < 500) && (max(up_tot(:)) < 1500)
    
    w_g = 1.0;  % Fator de sub-relaxação
    
    % Loop SOR (Successive Over-Relaxation) para convergência
    for i_sor = 1:3
        
        % ===== PASSO DE TEMPO =====
        dt = 20.0;               % Passo de tempo fixo (segundos)
        t(2) = t(1) + dt;        % Inicializa t(2)
        
        % ===== CÁLCULO DA PRESSÃO E VELOCIDADE DO SOM =====
        gamma = 1.4;
        press = r_g .* rho .* temp;
        c_s = sqrt(gamma * press ./ rho);
        
        % ===== ATUALIZAÇÃO DA FONTE (FORÇANTE) =====
        if i_sor == 1
            % Passo anterior (t - dt)
            s_t = source_t(t(i_t) - dt);
            wp_m = wp_m + 1.0e-3 * s_t * source_shape;
            
            % Passo atual (t)
            s_t = source_t(t(i_t));
            wp_o = wp_o + 1.0e-3 * s_t * source_shape;
            
            % Passo futuro (t + dt)
            s_t = source_t(t(i_t) + dt);
            wp = 1.0e-3 * s_t * source_shape;
        end
        
        % Inicialização no primeiro passo
        if (i_t == 2) && (i_sor == 1)
            wp_g = wp_o;
            wf_g = wf_o;
            wq_g = wq_o;
        else
            wp_g = wp;
            wf_g = wf;
            wq_g = wq;
        end
        
        % ===== DIFERENÇAS FINITAS NO TEMPO (LEAPFROG) =====
        wp_0 = 2.0 * wp_o - wp_m;
        wf_0 = 2.0 * wf_o - wf_m;
        wq_0 = 2.0 * wq_o - wq_m;
        
        % ===== CÁLCULO DOS TERMOS ESPACIAIS =====
        % Divergência da velocidade
        [gr_p_x, gr_p_y, gr_p_z] = gradient(wp_g);
        [gr_f_x, gr_f_y, gr_f_z] = gradient(wf_g);
        [gr_q_x, gr_q_y, gr_q_z] = gradient(wq_g);
        
        div_w = gr_p_x ./ dp_m + gr_f_y ./ df_m + gr_q_z ./ dq_m;
        
        % Divergência do fluxo de massa
        flux_p = rho .* wp_g;
        flux_f = rho .* wf_g;
        flux_q = rho .* wq_g;
        
        [gr_fp_x, gr_fp_y, gr_fp_z] = gradient(flux_p);
        [gr_ff_x, gr_ff_y, gr_ff_z] = gradient(flux_f);
        [gr_fq_x, gr_fq_y, gr_fq_z] = gradient(flux_q);
        
        div_flux = gr_fp_x ./ dp_m + gr_ff_y ./ df_m + gr_fq_z ./ dq_m;
        
        % Gradiente de pressão
        [gr_pr_x, gr_pr_y, gr_pr_z] = gradient(press);
        
        w_gr_press = wp_g .* gr_pr_x ./ dp_m + wf_g .* gr_pr_y ./ df_m + wq_g .* gr_pr_z ./ dq_m;
        
        % ===== CÁLCULO DOS TERMOS DE FORÇA (AGW) =====
        [wp_1, wp_2, wp_3] = AGW_2(1, dp_m, press, rho, rho_d, rho_u, div_w, div_flux, w_gr_press);
        [wf_1, wf_2, wf_3] = AGW_2(2, df_m, press, rho, rho_d, rho_u, div_w, div_flux, w_gr_press);
        [wq_1, wq_2, wq_3] = AGW_2(3, dq_m, press, rho, rho_d, rho_u, div_w, div_flux, w_gr_press);
        
        % ===== VISCOSIDADE =====
        visc_mu = 3.563e-7 * temp.^0.71;      % Viscosidade molecular
        visc_ki = visc_mu ./ rho;              % Difusividade cinemática
        
        % Termos de viscosidade (direção P)
        a = 1.0 ./ (dp_m.^2);
        flux = rho .* wp_o;
        [grad_temp_x, ~, ~] = gradient(flux);
        grad_flux = grad_temp_x ./ dp_m;
        w_visc_p = a .* grad_flux .* visc_mu ./ (rho.^2);
        
        % Termos de viscosidade (direção F)
        a = 1.0 ./ (df_m.^2);
        flux = rho .* wf_o;
        [~, grad_temp_y, ~] = gradient(flux);
        grad_flux = grad_temp_y ./ df_m;
        w_visc_f = a .* grad_flux .* visc_mu ./ (rho.^2);
        
        % Termos de viscosidade (direção Q)
        a = 1.0 ./ (dq_m.^2);
        flux = rho .* wq_o;
        [~, ~, grad_temp_z] = gradient(flux);
        grad_flux = grad_temp_z ./ dq_m;
        w_visc_q = a .* grad_flux .* visc_mu ./ (rho.^2);
        
        w_visc_rho = abs(w_visc_p) + abs(w_visc_f) + abs(w_visc_q);
        
        a = 1.0 ./ (dp_m.^2) + 1.0 ./ (df_m.^2) + 1.0 ./ (dq_m.^2);
        w_visc_w = visc_ki .* a ./ dt;
        
        w_visc = abs(w_visc_rho) + abs(w_visc_w);
        w_damp = exp(-0.5 * dt^2 * w_visc);
        
        % ===== AMORTECIMENTO NÃO LINEAR =====
        fac_nl = max(wp_g(:))^2 ./ (dp_m * dt);
        w_nl = exp(-0.1 * dt^2 * fac_nl);
        wp_amp = w_nl .* w_damp .* mask_bound;
        
        fac_nl = max(wf_g(:)) ./ (df_m * dt) + max(wq_g(:)) ./ (dq_m * dt);
        w_nl = exp(-0.001 * dt^2 * fac_nl);
        w_amp = w_nl .* w_damp .* mask_bound;
        
        % ===== ATUALIZA AS VELOCIDADES =====
        wp = wp_amp .* (wp_0 + dt^2 * (wp_1 + wp_2 + wp_3));
        wf = w_amp .* (wf_0 + dt^2 * (wf_1 + wf_2 + wf_3));
        wq = w_amp .* (wq_0 + dt^2 * (wq_1 + wq_2 + wq_3));
        
        % ===== CRITÉRIO DE CONVERGÊNCIA =====
        error = max(abs(wp(:) - wp_g(:))) / max(wp(:));
        
        wp_g = wp;
        wf_g = wf;
        wq_g = wq;
        
        w_g = w_g - 0.1;
        if w_g < 1.0
            w_g = 1.0;
        end
        
        fprintf('SOR LOOP CONTINUES: i_sor=%d, w_g=%.1f, error=%.6f\n', i_sor, w_g, error);
        
    end  % fim do loop SOR
    
    
    % ===== ATUALIZA OS ESTADOS ANTERIORES =====
    wp_m = wp_o;
    wp_o = wp;
    wf_m = wf_o;
    wf_o = wf;
    wq_m = wq_o;
    wq_o = wq;
    
    % ===== ATUALIZA DENSIDADE E TEMPERATURA =====
    for i_sor = 1:1
        if i_sor == 1
            rho_oo = rho;
            temp_oo = temp;
        end
        
        flux_p = rho .* wp_g;
        flux_f = rho .* wf_g;
        flux_q = rho .* wq_g;
        
        [gr_fp_x, gr_fp_y, gr_fp_z] = gradient(flux_p);
        [gr_ff_x, gr_ff_y, gr_ff_z] = gradient(flux_f);
        [gr_fq_x, gr_fq_y, gr_fq_z] = gradient(flux_q);
        
        div_flux = gr_fp_x ./ dp_m + gr_ff_y ./ df_m + gr_fq_z ./ dq_m;
        
        rho = rho_oo - 0.0 * dt * div_flux;
        if any(rho(:) < 0)
            rho = rho_oo;
        end
        
%         % Atualiza valores nos pontos vizinhos
%         rho_d(1:np-1, :, :) = rho(2:np, :, :);
%         rho_u(2:np, :, :) = rho(1:np-1, :, :);

        % Atualiza valores nos pontos vizinhos
        if np > 1
            rho_d(1:np-1, :, :) = rho(2:np, :, :);
            rho_u(2:np, :, :) = rho(1:np-1, :, :);
        end
        
        % Atualiza temperatura
        flux_p = temp .* wp_g;
        flux_f = temp .* wf_g;
        flux_q = temp .* wq_g;
        
        [gr_fp_x, gr_fp_y, gr_fp_z] = gradient(flux_p);
        [gr_ff_x, gr_ff_y, gr_ff_z] = gradient(flux_f);
        [gr_fq_x, gr_fq_y, gr_fq_z] = gradient(flux_q);
        
        div_flux = gr_fp_x ./ dp_m + gr_ff_y ./ df_m + gr_fq_z ./ dq_m;
        
        temp = temp_oo - 0.25 * dt * div_flux;
               
        % ============================================================================
        % IONOSFERA - CONTINUIDADE IÔNICA
        % ============================================================================

        % ======================== ION_CONTINUITY ===================================
        f = iono_amb(alt_i, np_i, nf, nq);
        if i_t == 2  % MATLAB: i_t=2 corresponde ao primeiro passo (Python i_t=1)
            den_t = den_amb;
            pot = zeros(np_i, nf, nq);
        end

        % Extrai componentes da velocidade na região da ionosfera (últimos np_i pontos)
        wp_i = wp_g(np - np_i + 1:end, :, :);
        wf_i = wf_g(np - np_i + 1:end, :, :);
        wq_i = wq_g(np - np_i + 1:end, :, :);

        % Calcula campo elétrico (coordenadas esféricas ou magnéticas)
        if i_mag == 0
            % Primeiro calcula condutividades e mobilidades
            [mu_p_i, mu_h_i, mu_o_i, mu_p_e, mu_h_e, mu_o_e] = mobility(nu_in, nu_en, i_inert, dt, omega_i, omega_e, mag_o);

            [sigma_p, sigma_h, sigma_o] = conductivity(den_t, q_charge, mu_p_i, mu_h_i, mu_o_i, mu_p_e, mu_h_e, mu_o_e);

            [pot, up_ele, up_tot, uf_tot, uq_tot] = campo_ele_sph_coord(t(i_t), wp_i, wf_i, wq_i, den_t, pot, theta, np, np_i, mag_o, sigma_p, sigma_h, sigma_o, mu_p_i, mu_h_i, mu_o_i, mu_p_e, mu_h_e, mu_o_e, up_tot);        end

        if i_mag == 1
            [pot, up_ele] = campo_ele_mag_coord(t(i_t), wp_i, wf_i, wq_i, den_t, pot, theta, np, np_i, mag_o, sigma_p, sigma_h, sigma_o, mu_p_i, mu_h_i, mu_o_i, mu_p_e, mu_h_e, mu_o_e, up_tot, uf_tot, uq_tot);
        end

        % Atualiza densidade iônica
        [den_t, delta_den, dtec] = den_ion(den_t, up_tot, uf_tot, uq_tot, dp_i, df_i, dq_i, dt, den_amb, tec_fac);

        % Condutividade ambiente (não utilizada)
        [sigma_p_amb, sigma_h_amb, sigma_o_amb] = conductivity(den_amb, q_charge, mu_p_i, mu_h_i, mu_o_i, mu_p_e, mu_h_e, mu_o_e);

        % ============================================================================
        % VELOCIDADE HORIZONTAL TOTAL
        % ============================================================================

        wh = sqrt(wf.^2 + wq.^2);

        % ============================================================================
        % ARMAZENAMENTO PARA KEOGRAMAS
        % ============================================================================

        if i_t == 2  % Primeiro passo
            dtec_keo = zeros(floor(nt/5), nf, nq);
            uplift_keo = zeros(floor(nt/5), nf, nq);
            wp_keo = zeros(floor(nt/5), nf, nq);
            wh_keo = zeros(floor(nt/5), nf, nq);
            pot_keo = zeros(floor(nt/5), nf, nq);
            i_w = 1;  % MATLAB começa em 1
        end

        if mod(i_t - 1, 5) == 0  % i_t-1 porque MATLAB i_t=2 equivale a Python i_t=1
            dtec_keo(i_w, :, :) = dtec;
            uplift_keo(i_w, :, :) = wp(1, :, :);  % wp[0, :, :]
            wp_keo(i_w, :, :) = wp(14, :, :);     % wp[13, :, :] (14 em MATLAB)
            wh_keo(i_w, :, :) = wh(14, :, :);     % wh[13, :, :]
            pot_keo(i_w, :, :) = pot(4, :, :);    % pot[3, :, :]
            i_w = i_w + 1;
        end

        % ============================================================================
        % PROPAGAÇÃO VERTICAL
        % ============================================================================

        if i_t == 2
            wp_prop = zeros(nt, np);
            uplift_t = zeros(nt, np);
        end
        wp_prop(i_t, :) = wp(:, iphi_s, iq_s);
        uplift_t(i_t, :) = wp(1, iphi_s, iq_s);  % wp[0, iphi_s, iq_s]
        
    end
    
    vm = 0.05;
    fr_gr_2 = wp(1, :, :);  % wp[0, :, :] - DEFINIR ANTES DE USAR!
    
    % ============================================================================
    % PLOTAGEM DOS MAPAS (FIGURA 2) - A CADA 5 PASSOS
    % ============================================================================

%     if mod(i_t - 1, 5) == 0
% 
%         % ===== FIGURA 2: MAPAS =====
%         fig2 = figure(2);
%         set(fig2, 'Position', [100, 100, 1800, 1200], 'Color', 'white');
% 
%         % Extrai dados para plotagem
%         lat_pl = squeeze(lat_3(5+11, :, :));
%         lon_pl = squeeze(lon_3(5+11, :, :));
%         lat_center = mean(lat_pl(:));
%         lon_center = mean(lon_pl(:));
%         [X, Y] = meshgrid(lon_pl(1,:), lat_pl(:,1));
% 
%         % ===== SUBPLOT ESQUERDO: AGW =====
%         ax1 = subplot(1,2,1);
%         axesm('ortho', 'Origin', [lat_center, lon_center, 0]);
%         framem('on');
%         gridm('on');
%         load coastlines;
%         geoshow(coastlat, coastlon, 'Color', 'k', 'LineWidth', 0.5);
%         
%         data_agw = squeeze(wp(5+11, :, :));
%         data_agw_norm = data_agw ./ max(abs(data_agw(:)));
%         
%         % DEBUG depois de definir as variáveis
%         fprintf('DEBUG data_agw: min=%.6e, max=%.6e, mean=%.6e\n', ...
%             min(data_agw(:)), max(data_agw(:)), mean(data_agw(:)));
%         
%         geoshow(Y, X, data_agw_norm, 'DisplayType', 'texturemap');
%         colormap(ax1, 'jet');
%         caxis([-vm, vm]);
%         
%         % Contornos
%         C_data = squeeze(abs(fr_gr_2) ./ max(abs(fr_gr_2(:))));
% %         [~, h] = contourm(Y, X, C_data, [0.05, 0.5], 'Color', 'g', 'LineWidth', 1);
% %         clabelm(h);
%         [C, h] = contourm(Y, X, C_data, [0.05, 0.5], 'Color', 'g', 'LineWidth', 1);
%         if ~isempty(C)
%             clabelm(C, h);
%         end
% 
%         tempo_horas = hora_ut_inicial + t(i_t) / 3600.0;
%         horas_ut = floor(mod(tempo_horas, 24));
%         minutos_ut = floor(mod(tempo_horas * 60, 60));
%         title(sprintf('%02d:%02d UT', horas_ut, minutos_ut));
%         colorbar;
% 
%          % ===== SUBPLOT DIREITO: dTEC e POTENCIAL  =====
%         ax2 = subplot(1,2,2);
%         axesm('ortho', 'Origin', [lat_center, lon_center, 0]);
%         framem('on');
%         gridm('on');
%         geoshow(coastlat, coastlon, 'Color', 'k', 'LineWidth', 0.5);
%         
%         data_dtec = squeeze(dtec);
%         data_dtec_norm = data_dtec ./ max(abs(data_dtec(:)));
%         
%         fprintf('DEBUG data_dtec: min=%.6e, max=%.6e, mean=%.6e\n', ...
%         min(data_dtec(:)), max(data_dtec(:)), mean(data_dtec(:)));
%         
%         geoshow(Y, X, data_dtec_norm, 'DisplayType', 'texturemap');
%         colormap(ax2, 'jet');
%         caxis([-vm, vm]);
%         
%         % Contornos do potencial
% %         data_pot = squeeze(pot(11, :, :));
%         data_pot = squeeze(pot(1:np_i, iphi_s, :));
%         data_pot_norm = data_pot ./ max(abs(data_pot(:)));
%         con_lev = [-0.5, -0.05, -0.00005, 0.00005, 0.05, 0.5];
% %         [~, h] = contourm(Y, X, data_pot_norm, con_lev, 'Color', 'k', 'LineWidth', 1);
% %         clabelm(h);
%         [C, h] = contourm(Y, X, C_data, [0.05, 0.5], 'Color', 'g', 'LineWidth', 1);
%         if ~isempty(C)
%             clabelm(C, h);
%         end
%         
%         title(sprintf('%02d:%02d UT', horas_ut, minutos_ut));
%         colorbar;
% 
%         % Salva
%         nome_fig = fullfile(pasta_figs, sprintf('mapa_%05d.png', i_t-1));
%         saveas(fig2, nome_fig);
%         close(fig2);
%         fprintf('Mapa salvo: %s\n', nome_fig);

%         % ============================================================================
%         % PLOTAGEM DOS MAPAS (FIGURA 2) - A CADA 5 PASSOS
%         % ============================================================================
% 
%         if mod(i_t - 1, 5) == 0
% 
%             % ===== FIGURA 2: MAPAS =====
%             fig2 = figure(2);
%             set(fig2, 'Position', [100, 100, 1800, 1200], 'Color', 'white');
% 
%             % Extrai dados para plotagem
%             lat_pl = squeeze(lat_3(5+11, :, :));
%             lon_pl = squeeze(lon_3(5+11, :, :));
%             lat_center = mean(lat_pl(:));
%             lon_center = mean(lon_pl(:));
% 
%             % ===== SUBPLOT ESQUERDO: AGW =====
%             ax1 = subplot(1,2,1);
%             axesm('ortho', 'Origin', [lat_center, lon_center, 0]);
%             framem('on');
%             gridm('on');
%             load coastlines;
%             geoshow(coastlat, coastlon, 'Color', 'k', 'LineWidth', 0.5);
% 
%             data_agw = squeeze(wp(5+11, :, :));
% 
%             % === CORREÇÃO AQUI: NÃO NORMALIZAR, USAR VALORES REAIS ===
%             % Encontra o valor máximo absoluto para escala simétrica
%             max_agw = max(abs(data_agw(:)));
%             if max_agw > 0
%                 % Usa escala simétrica baseada no valor máximo
%                 clim_agw = [-max_agw, max_agw];
%             else
%                 clim_agw = [-0.05, 0.05];
%             end
% 
%             fprintf('DEBUG data_agw: min=%.6e, max=%.6e, mean=%.6e, clim=[%.6e, %.6e]\n', ...
%                 min(data_agw(:)), max(data_agw(:)), mean(data_agw(:)), clim_agw(1), clim_agw(2));
% 
%             % Plota os dados REAIS (não normalizados)
%             geoshow(lat_pl, lon_pl, data_agw, 'DisplayType', 'texturemap');
%             colormap(ax1, 'jet');
%             caxis(ax1, clim_agw);  % Usa escala baseada nos dados reais
% 
%             % Contornos (usando valores reais, não normalizados)
%             max_val = max(abs(data_agw(:)));
%             if max_val > 0
%                 % Define níveis de contorno baseados no valor máximo
%                 contour_levels = [0.05*max_val, 0.5*max_val];
%                 [C, h] = contourm(lat_pl, lon_pl, data_agw, contour_levels, ...
%                                   'Color', 'g', 'LineWidth', 1);
%                 if ~isempty(C)
%                     clabelm(C, h);
%                 end
%             end
% 
%             tempo_horas = hora_ut_inicial + t(i_t) / 3600.0;
%             horas_ut = floor(mod(tempo_horas, 24));
%             minutos_ut = floor(mod(tempo_horas * 60, 60));
%             title(sprintf('%02d:%02d UT - AGW', horas_ut, minutos_ut));
%             colorbar;
% 
%             % ===== SUBPLOT DIREITO: dTEC =====
%             ax2 = subplot(1,2,2);
%             axesm('ortho', 'Origin', [lat_center, lon_center, 0]);
%             framem('on');
%             gridm('on');
%             geoshow(coastlat, coastlon, 'Color', 'k', 'LineWidth', 0.5);
% 
%             % dtec é uma matriz 3D? Vamos verificar e ajustar
%             if ndims(dtec) == 3
%                 data_dtec = squeeze(dtec(1, :, :));  % Primeira altitude
%             else
%                 data_dtec = squeeze(dtec);
%             end
% 
%             % === CORREÇÃO AQUI: ESCALA BASEADA NOS DADOS REAIS ===
%             max_dtec = max(abs(data_dtec(:)));
%             if max_dtec > 0
%                 clim_dtec = [-max_dtec, max_dtec];
%             else
%                 clim_dtec = [-0.05, 0.05];
%             end
% 
%             fprintf('DEBUG data_dtec: min=%.6e, max=%.6e, mean=%.6e, clim=[%.6e, %.6e]\n', ...
%                 min(data_dtec(:)), max(data_dtec(:)), mean(data_dtec(:)), clim_dtec(1), clim_dtec(2));
% 
%             % Plota os dados REAIS
%             geoshow(lat_pl, lon_pl, data_dtec, 'DisplayType', 'texturemap');
%             colormap(ax2, 'jet');
%             caxis(ax2, clim_dtec);
% 
%             title(sprintf('%02d:%02d UT - dTEC', horas_ut, minutos_ut));
%             colorbar;
% 
%             % Salva
%             nome_fig = fullfile(pasta_figs, sprintf('mapa_%05d.png', i_t-1));
%             saveas(fig2, nome_fig);
%             close(fig2);
%             fprintf('Mapa salvo: %s\n', nome_fig);

        % ============================================================================
        % PLOTAGEM DOS MAPAS (FIGURA 2) - A CADA 5 PASSOS
        % ============================================================================

        if mod(i_t - 1, 5) == 0

            % ===== FIGURA 2: MAPAS =====
            fig2 = figure(2);
            set(fig2, 'Position', [100, 100, 1800, 1200], 'Color', 'white');

            % Extrai dados para plotagem
            % Usando o mesmo índice que o Python: 5+11 = 16 (16ª altitude)
            idx_alt = 16;  % 5+11 em MATLAB (1-indexado)
            lat_pl = squeeze(lat_3(idx_alt, :, :));
            lon_pl = squeeze(lon_3(idx_alt, :, :));
            
            lat_center = mean(lat_pl(:));
            lon_center  = mean(lon_pl(:));
            
            % Vetores 1D de lat/lon para surfm (usa primeira linha/coluna)
            lat_vec = lat_pl(1, :);   % 1 x nq
            lon_vec = lon_pl(:, 1)';  % 1 x nf


            % ===== SUBPLOT ESQUERDO: AGW (como no Python) =====
            ax1 = subplot(1,2,1);
            axesm('ortho', 'Origin', [lat_center, lon_center, 0]);

            % Configura projeção ortográfica
%             axesm('ortho', 'Origin', [mean(lat_pl(:)), mean(lon_pl(:)), 0]);
            framem('on');
            gridm('on');
            load coastlines;
            geoshow(coastlat, coastlon, 'Color', 'k', 'LineWidth', 0.5);

            % Dados do AGW - mesmo índice que o Python: wp[5+11, :, :]
            data_agw = squeeze(wp(idx_alt, :, :));

            % Normalização para escala simétrica (como no Python)
            max_agw = max(abs(data_agw(:)));
%             if max_agw > 0
%                 data_agw_norm = data_agw / max_agw;
%                 clim_agw = [-1, 1];
%             else
%                 data_agw_norm = data_agw;
%                 clim_agw = [-0.05, 0.05];
%             end


            if max_agw > 1e-10
                data_agw_norm = data_agw / max_agw;

                % Transparência: alpha=0 onde |valor| < threshold
                threshold = 0.01;
                alpha_agw = double(abs(data_agw_norm) >= threshold);

                % surfm aceita vetores 1D de lat/lon e matriz 2D de dados
                % lat_vec(nq), lon_vec(nf), data(nf x nq) -> transpor para (nq x nf)
                hs = surfm(lat_vec, lon_vec, data_agw_norm', 'FaceAlpha', 'flat', ...
                           'AlphaDataMapping', 'none', 'EdgeColor', 'none');
                hs.AlphaData = alpha_agw';
                colormap(ax1, 'jet');
                caxis(ax1, [-1, 1]);
            end
            
            % Mascara valores muito pequenos (ruído de fundo) como NaN
%             threshold_agw = 0.01;  % 1% do máximo
%             data_agw_masked = data_agw_norm;
%             data_agw_masked(abs(data_agw_masked) < threshold_agw) = NaN;
%             geoshow(lat_pl, lon_pl, data_agw_masked, 'DisplayType', 'texturemap');

            % Plota usando geoshow com texturemap (equivalente ao pcolormesh do Python)
%             geoshow(lat_pl, lon_pl, data_agw_norm, 'DisplayType', 'texturemap');
%             colormap(ax1, 'jet');
%             caxis(ax1, clim_agw);

            % Contornos - baseados nos mesmos níveis do Python
            % Python: levels=[0.05, 0.5] de abs(fr_gr_2)/fr_gr_2.max()
%             fr_gr_2_data = squeeze(wp(1, :, :));  % wp[0, :, :] em Python
%             max_fr = max(abs(fr_gr_2_data(:)));
%             if max_fr > 0
%                 contour_levels = [0.05 * max_fr, 0.5 * max_fr];
%                 [C, h] = contourm(lat_pl, lon_pl, abs(fr_gr_2_data), contour_levels, ...
%                                   'Color', 'g', 'LineWidth', 1, 'LineStyle', '-');
%                 if ~isempty(C)
%                     clabelm(C, h);
%                 end
%             end

            % Aplica máscara para limitar os contornos
%             fr_gr_2_data = squeeze(wp(1, :, :));
%             max_fr = max(abs(fr_gr_2_data(:)));
%             if max_fr > 0
%                 % Cria uma máscara baseada nos dados (ex: onde |dados| > 1% do máximo)
%                 mask = abs(fr_gr_2_data) > 0.01 * max_fr;
%                 fr_gr_2_masked = fr_gr_2_data;
%                 fr_gr_2_masked(~mask) = NaN;  % Torna NaN fora da região de interesse
% 
%                 contour_levels = [0.05 * max_fr, 0.5 * max_fr];
%                 [C, h] = contourm(lat_pl, lon_pl, abs(fr_gr_2_masked), contour_levels, ...
%                                   'Color', 'g', 'LineWidth', 1, 'LineStyle', '-');
%                 if ~isempty(C)
%                     clabelm(C, h);
%                 end
%             end

            tempo_horas = hora_ut_inicial + t(i_t) / 3600.0;
            horas_ut = floor(mod(tempo_horas, 24));
            minutos_ut = floor(mod(tempo_horas * 60, 60));
            title(sprintf('%02d:%02d UT', horas_ut, minutos_ut));
            colorbar;

            % ===== SUBPLOT DIREITO: dTEC (como no Python) =====
            ax2 = subplot(1,2,2);
            
            axesm('ortho', 'Origin', [lat_center, lon_center, 0]);

%             axesm('ortho', 'Origin', [mean(lat_pl(:)), mean(lon_pl(:)), 0]);
            framem('on');
            gridm('on');
            geoshow(coastlat, coastlon, 'Color', 'k', 'LineWidth', 0.5);

            % Dados do dTEC
            if ndims(dtec) == 3
                % Se dtec for 3D, pega a primeira altitude (ou a que faz sentido)
                data_dtec = squeeze(dtec(1, :, :));
            else
                data_dtec = squeeze(dtec);
            end

            % Normalização para escala simétrica
            max_dtec = max(abs(data_dtec(:)));
%             if max_dtec > 0
%                 data_dtec_norm = data_dtec / max_dtec;
%                 clim_dtec = [-1, 1];
%             else
%                 data_dtec_norm = data_dtec;
%                 clim_dtec = [-0.05, 0.05];
%             end

            if max_dtec > 1e-10
                data_dtec_norm = data_dtec / max_dtec;

                threshold = 0.01;
                alpha_dtec = double(abs(data_dtec_norm) >= threshold);

                hs2 = surfm(lat_vec, lon_vec, data_dtec_norm', 'FaceAlpha', 'flat', ...
                            'AlphaDataMapping', 'none', 'EdgeColor', 'none');
                hs2.AlphaData = alpha_dtec';
                colormap(ax2, 'jet');
                caxis(ax2, [-1, 1]);
            end

            % Plota dTEC
            
%             threshold_dtec = 0.01;
%             data_dtec_masked = data_dtec_norm;
%             data_dtec_masked(abs(data_dtec_masked) < threshold_dtec) = NaN;
% 
%             geoshow(lat_pl, lon_pl, data_dtec_norm, 'DisplayType', 'texturemap');
%             colormap(ax2, 'jet');
%             caxis(ax2, clim_dtec);

            % Contornos do potencial (se existir pot)
            if exist('pot', 'var') && ~isempty(pot)
                if ndims(pot) == 3
                    data_pot = squeeze(pot(11, :, :));  % pot[11] em Python
                else
                    data_pot = pot;
                end
                max_pot = max(abs(data_pot(:)));
                if max_pot > 0
                    data_pot_norm = data_pot / max_pot;
                    con_lev = [-0.5, -0.05, -0.00005, 0.00005, 0.05, 0.5];
                    [C, h] = contourm(lat_pl, lon_pl, data_pot_norm, con_lev, ...
                                      'Color', 'k', 'LineWidth', 1);
                    if ~isempty(C)
                        clabelm(C, h);
                    end
                end
            end

            title(sprintf('%02d:%02d UT', horas_ut, minutos_ut));
            colorbar;

            % Salva a figura
            nome_fig = fullfile(pasta_figs, sprintf('mapa_%05d.png', i_t-1));
            saveas(fig2, nome_fig);
            close(fig2);
            fprintf('Mapa salvo: %s\n', nome_fig);
        end

        % =========================================================================
        % FIGURA 3: CORTES VERTICAIS (AGW e Potencial/Densidade)
        % =========================================================================

        % Cria figura 3
        fig3 = figure('Position', [100, 100, 1800, 1200], 'Color', 'white', 'Visible', 'off');

        % Índice médio em latitude
        i_q_medio = round(nq / 2);

        % Ângulo para rotação do corte
        angulo_corte = atan2(z_f(np, 1, i_q_medio), y_f(np, 1, i_q_medio));

        % Aplica rotação à grade
        [z_grid_corte, y_grid_corte] = rotaciona_corte(squeeze(z_f(:, 1, :)), squeeze(y_f(:, 1, :)), angulo_corte);

        % ===== SUBPLOT ESQUERDO: AGW =====
        ax1 = subplot(1,2,1);

        % Dados do AGW no corte (longitude fixa = iphi_s)
        data_agw_corte = squeeze(wp(:, iphi_s, :));

        % Aplica rotação às coordenadas geográficas (TODA a altitude)
        [lat_geo_corte, alt_geo_corte] = rotaciona_corte(...
            squeeze(z_sph(:, iphi_s, :)), ...
            squeeze(y_sph(:, iphi_s, :)), ...
            angulo_corte);

        % Normaliza os dados
        data_agw_norm = data_agw_corte ./ max(abs(data_agw_corte(:)));

        % Plota o pcolormesh
        pcolor(lat_geo_corte, alt_geo_corte, data_agw_norm);
        shading interp;
        colormap(ax1, 'jet');
        caxis([-vm, vm]);
        axis tight;

        % Contornos
        con_lev = [-0.5, -0.1, -0.05, 0.05, 0.1, 0.5];
        hold on;
        [C, h] = contour(lat_geo_corte, alt_geo_corte, data_agw_norm, con_lev, 'Color', 'g', 'LineWidth', 1);
        clabel(C, h, 'FontSize', 8);
        hold off;

        % ===== LINHAS DE GRADE DO GRÁFICO ESQUERDO =====
        hold on;

        % Linhas de latitude (a cada 10 pontos)
        for i = 1:10:nq
            plot(z_grid_corte(:, i), y_grid_corte(:, i), 'Color', [0.5, 0.5, 0.5], 'LineWidth', 0.5);
        end

        % Rótulos de latitude
        labels = round(lat(1:10:nq), 0);
        ii = 1;
        for i = 1:10:nq
            text(z_grid_corte(np, i), y_grid_corte(np, i), num2str(labels(ii)), 'FontSize', 8, ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
            ii = ii + 1;
        end

        % Linhas de altitude (a cada 10 pontos)
        for i = 1:10:np
            plot(z_grid_corte(i, :), y_grid_corte(i, :), 'Color', [0.5, 0.5, 0.5], 'LineWidth', 0.5);
        end

        % Rótulos de altitude
        labels = round(alt(1:10:np), 0);
        ii = 1;
        for i = 1:10:np
            text(z_grid_corte(i, i_q_medio), y_grid_corte(i, i_q_medio), num2str(labels(ii)), 'FontSize', 8, ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
            ii = ii + 1;
        end

        % Desenha círculos representando a Terra
        r_ea = 6371;
        rectangle('Position', [-r_ea, -r_ea, 2*r_ea, 2*r_ea], ...
                  'Curvature', [1, 1], 'EdgeColor', [0.5, 0.5, 0.5], 'LineWidth', 1, ...
                  'FaceColor', 'none');
        rectangle('Position', [-(r_ea-100), -(r_ea-100), 2*(r_ea-100), 2*(r_ea-100)], ...
                  'Curvature', [1, 1], 'EdgeColor', 'w', 'LineWidth', 1, 'FaceColor', 'none');

        hold off;
        axis off;
        axis equal;

        % ===== LABELS DOS EIXOS =====
        annotation(fig3, 'textbox', [0.25, 0.02, 0.1, 0.05], ...
                   'String', 'Latitude (°)', 'FontSize', 12, 'FontWeight', 'bold', ...
                   'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                   'BackgroundColor', 'white', 'EdgeColor', 'none');
        text(0.02, 0.5, 'Altitude (km)', 'Units', 'normalized', ...
             'FontSize', 12, 'FontWeight', 'bold', 'Rotation', 90, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

        % ===== BARRA DE CORES =====
        colorbar('Position', [0.42, 0.11, 0.02, 0.8]);

        % ===== SUBPLOT DIREITO: POTENCIAL E DENSIDADE =====
        ax2 = subplot(1,2,2);

        % Dados do potencial no corte (apenas ionosfera)
        data_pot_corte = squeeze(pot(1:np_i, iphi_s, :));
        data_pot_norm = data_pot_corte ./ max(abs(data_pot_corte(:)));

        % Aplica rotação às coordenadas geográficas (apenas ionosfera)
        [lat_geo_ion, alt_geo_ion] = rotaciona_corte(...
            squeeze(z_sph(1:np_i, iphi_s, :)), ...
            squeeze(y_sph(1:np_i, iphi_s, :)), ...
            angulo_corte);

        % Plota densidade como fundo
        data_den_corte = squeeze(den_t(:, iphi_s, :));
        data_den_norm = data_den_corte ./ max(data_den_corte(:));
        pcolor(lat_geo_ion, alt_geo_ion, data_den_norm);
        shading interp;
        colormap(ax2, 'gray');
        axis tight;
        hold on;

        % Contornos do potencial
        [C, h] = contour(lat_geo_ion, alt_geo_ion, data_pot_norm, con_lev, 'Color', 'g', 'LineWidth', 1);
        clabel(C, h, 'FontSize', 8);
        hold off;

        % ===== LINHAS DE GRADE DO GRÁFICO DIREITO =====
        hold on;
        for i = 1:10:nq
            plot(z_grid_corte(:, i), y_grid_corte(:, i), 'Color', [0.5, 0.5, 0.5], 'LineWidth', 0.5);
        end
        for i = 1:10:np
            plot(z_grid_corte(i, :), y_grid_corte(i, :), 'Color', [0.5, 0.5, 0.5], 'LineWidth', 0.5);
        end
        rectangle('Position', [-r_ea, -r_ea, 2*r_ea, 2*r_ea], ...
                  'Curvature', [1, 1], 'EdgeColor', [0.5, 0.5, 0.5], 'LineWidth', 1, ...
                  'FaceColor', 'none');
        rectangle('Position', [-(r_ea-100), -(r_ea-100), 2*(r_ea-100), 2*(r_ea-100)], ...
                  'Curvature', [1, 1], 'EdgeColor', 'w', 'LineWidth', 1, 'FaceColor', 'none');
        hold off;
        axis off;
        axis equal;

        colorbar('Position', [0.92, 0.11, 0.02, 0.8]);

        % ===== SALVA A FIGURA =====
        nome_fig_corte = fullfile(pasta_figs, sprintf('corte_%05d.png', i_t-1));
        saveas(fig3, nome_fig_corte);
        fprintf('Corte vertical salvo: %s\n', nome_fig_corte);
        close(fig3);

        % =========================================================================
        % FIGURA 4: CAMPO ELÉTRICO E DENSIDADE IONOSFÉRICA
        % =========================================================================

         % Cria figura 4
        fig4 = figure('Position', [800, 100, 800, 1200], 'Color', 'white', 'Visible', 'off');

        % Dados do campo elétrico (up_tot) no corte de latitude fixa (iq_s)
        data_up = squeeze(up_tot(:, :, iq_s));

        % Cria grades para interpolação (zoom)
        lon_zoom = linspace(min(lon), max(lon), length(lon)*2);
        alt_i_zoom = linspace(min(alt_i), max(alt_i), length(alt_i)*2);

        % Interpola os dados
        [Lon_grid, Alt_grid] = meshgrid(lon, alt_i);
        [Lon_zoom, Alt_zoom] = meshgrid(lon_zoom, alt_i_zoom);

        data_up_zoom = interp2(Lon_grid, Alt_grid, data_up, Lon_zoom, Alt_zoom, 'spline');

        % Plota o campo elétrico
        pcolor(Lon_zoom, Alt_zoom, data_up_zoom);
        shading interp;
        colormap(fig4, 'jet');
        axis tight;
        hold on;

        % Dados da densidade iônica para contorno
        data_den_cont = squeeze(den_t(:, :, iq_s));
        data_den_cont_norm = data_den_cont ./ max(abs(data_den_cont(:)));
        data_den_cont_zoom = interp2(Lon_grid, Alt_grid, data_den_cont_norm, Lon_zoom, Alt_zoom, 'spline');

        % Contornos da densidade (21 níveis)
        [~, h] = contour(Lon_zoom, Alt_zoom, data_den_cont_zoom, 21, 'Color', 'g', 'LineWidth', 1);
        hold off;

        % ===== LABELS DOS EIXOS =====
        xlabel('Longitude (°)');
        ylabel('Altitude (km)');

        % ===== TÍTULO =====
        tempo_horas = hora_ut_inicial + t(i_t) / 3600.0;
        horas_ut = floor(mod(tempo_horas, 24));
        minutos_ut = floor(mod(tempo_horas * 60, 60));
        title(sprintf('Campo Elétrico e Densidade - %02d:%02d UT', horas_ut, minutos_ut));

        % ===== BARRA DE CORES =====
        colorbar;

        % ===== SALVA A FIGURA =====
        nome_fig_campo = fullfile(pasta_figs, sprintf('campo_%05d.png', i_t-1));
        saveas(fig4, nome_fig_campo);
        fprintf('Campo salvo: %s\n', nome_fig_campo);
        close(fig4);
        
        % =========================================================================
        % MENSAGENS DE DEBUG (equivalente ao print do Python)
        % =========================================================================

        fprintf('PLOTTING\n');
        fprintf('======================================================\n');
        fprintf('TIME = %d, t = %.1f s\n', i_t-1, t(i_t));
        fprintf('FORCING AMPLITUDE = %.5f\n', max(wp(1,:,:), [], 'all'));
        fprintf('ATMOS W_MAX = (%.2f, %.2f, %.2f)\n', max(wp(:)), max(wf(:)), max(wq(:)));
        fprintf('======================================================\n');

   
    
    % =========================================================================
    % ATUALIZAÇÃO DO TEMPO PARA O PRÓXIMO PASSO
    % =========================================================================

     t(i_t + 1) = t(i_t) + dt;
     i_t = i_t + 1;
    
end

% ============================================================================
% CRIAÇÃO DOS KEOGRAMAS
% ============================================================================

fprintf('\n============================================================\n');
fprintf('CRIANDO KEOGRAMAS...\n');
fprintf('============================================================\n');

if i_w > 0
    % Coordenadas desejadas para o keograma (ponto específico)
    lat_desejada = -60.319;   % 60.319°S
    lon_desejada = -61.922;   % 61.922°W

    % Encontra os índices mais próximos das coordenadas desejadas
    [~, lat_medio] = min(abs(lat - lat_desejada));
    [~, lon_medio] = min(abs(lon - lon_desejada));

    % DEBUG: Exibe informações das coordenadas
    fprintf('\n=== COORDENADAS DOS KEOGRAMAS ===\n');
    fprintf('nq = %d, nf = %d\n', nq, nf);
    fprintf('Coordenadas desejadas: %.3f°S, %.3f°W\n', abs(lat_desejada), abs(lon_desejada));
    fprintf('lat_medio (índice %d) = %.2f°\n', lat_medio, lat(lat_medio));
    fprintf('lon_medio (índice %d) = %.2f°\n', lon_medio, lon(lon_medio));
    fprintf('Range de longitude: %.2f° a %.2f°\n', lon(1), lon(end));
    fprintf('Range de latitude: %.2f° a %.2f°\n', lat(1), lat(end));
    fprintf('===================================\n\n');

    % Tempo em horas para o keograma (converte de segundos para horas)
    tempo_keo_horas = zeros(i_w, 1);
    for i = 1:i_w
        % cada keograma é a cada 5 passos (i_t = 2,7,12,...)
        tempo_keo_horas(i) = t((i+1)*5) / 3600.0;
    end

    % Configurações de rebate
    flip_lon = false;               % True se longitude precisa rebater no eixo vertical
    flip_lat = false;               % True se latitude precisa rebater no eixo vertical
    flip_time = false;              % True para inverter ordem cronológica
    flip_conteudo_horizontal = true; % Rebate a imagem da direita para esquerda

    % Dados com rebate
    if flip_lon
        lon_plot = fliplr(lon);
    else
        lon_plot = lon;
    end

    if flip_lat
        lat_plot = fliplr(lat);
    else
        lat_plot = lat;
    end

    if flip_time
        tempo_plot = flipud(tempo_keo_horas);
    else
        tempo_plot = tempo_keo_horas;
    end

    % ===== KEOGRAMA 1: ?TEC × Longitude =====
    dtec_plot = squeeze(dtec_keo(1:i_w, :, lat_medio));
    if flip_lon
        dtec_plot = fliplr(dtec_plot);
    end
    if flip_time
        dtec_plot = flipud(dtec_plot);
    end

    salvar_keograma(tempo_plot, lon_plot, dtec_plot', ...
        'Keograma ?TEC (Tempo × Longitude)', ...
        fullfile(pasta_figs, 'keograma_dtec_time_lon.png'), ...
        'Tempo (horas)', 'Longitude (°)', flip_conteudo_horizontal, false);

    % ===== KEOGRAMA 2: ?TEC × Latitude =====
    dtec_lat_plot = squeeze(dtec_keo(1:i_w, lon_medio, :));
    if flip_lat
        dtec_lat_plot = fliplr(dtec_lat_plot);
    end
    if flip_time
        dtec_lat_plot = flipud(dtec_lat_plot);
    end

    salvar_keograma(tempo_plot, lat_plot, dtec_lat_plot', ...
        'Keograma ?TEC (Tempo × Latitude)', ...
        fullfile(pasta_figs, 'keograma_dtec_time_lat.png'), ...
        'Tempo (horas)', 'Latitude (°)', flip_conteudo_horizontal, false);

    % ===== KEOGRAMA 3: WP 300 km × Longitude =====
    wp_plot = squeeze(wp_keo(1:i_w, :, lat_medio));
    if flip_lon
        wp_plot = fliplr(wp_plot);
    end
    if flip_time
        wp_plot = flipud(wp_plot);
    end

    salvar_keograma(tempo_plot, lon_plot, wp_plot', ...
        'Keograma WP (300 km) - Tempo × Longitude', ...
        fullfile(pasta_figs, 'keograma_wp300_time_lon.png'), ...
        'Tempo (horas)', 'Longitude (°)', flip_conteudo_horizontal, false);

    % ===== KEOGRAMA 4: Uplift (150 km) × Longitude =====
    uplift_plot = squeeze(uplift_keo(1:i_w, :, lat_medio));
    if flip_lon
        uplift_plot = fliplr(uplift_plot);
    end
    if flip_time
        uplift_plot = flipud(uplift_plot);
    end

    salvar_keograma(tempo_plot, lon_plot, uplift_plot', ...
        'Keograma Uplift (150 km) - Tempo × Longitude', ...
        fullfile(pasta_figs, 'keograma_uplift_time_lon.png'), ...
        'Tempo (horas)', 'Longitude (°)', flip_conteudo_horizontal, false);

    % ===== KEOGRAMA 5: Potencial × Longitude =====
    pot_plot = squeeze(pot_keo(1:i_w, :, lat_medio));
    if flip_lon
        pot_plot = fliplr(pot_plot);
    end
    if flip_time
        pot_plot = flipud(pot_plot);
    end

    salvar_keograma(tempo_plot, lon_plot, pot_plot', ...
        'Keograma Potencial - Tempo × Longitude', ...
        fullfile(pasta_figs, 'keograma_pot_time_lon.png'), ...
        'Tempo (horas)', 'Longitude (°)', flip_conteudo_horizontal, false);

    % =========================================================================
    % SALVAMENTO DOS DADOS DOS KEOGRAMAS (equivalente ao save() do Python)
    % =========================================================================
    save(fullfile(pasta_figs, 'uplift_keo_t.mat'), 'uplift_keo');
    save(fullfile(pasta_figs, 'wp_keo_t.mat'), 'wp_keo');
    save(fullfile(pasta_figs, 'wh_keo_t.mat'), 'wh_keo');
    save(fullfile(pasta_figs, 'dtec_keo_t.mat'), 'dtec_keo');

    fprintf('\nDados dos keogramas salvos em: %s\n', pasta_figs);
    fprintf('  - uplift_keo_t.mat\n');
    fprintf('  - wp_keo_t.mat\n');
    fprintf('  - wh_keo_t.mat\n');
    fprintf('  - dtec_keo_t.mat\n');
    fprintf('\nKeogramas salvos em: %s\n', pasta_figs);

else
    fprintf('Não há dados suficientes para criar keogramas.\n');
end
    

% ============================================================================
% Parte 2: Definições das funções (todas no final do código)
% ============================================================================

% Função para rotacionar o corte (coloque ANTES das outras funções)
function [x_rot, y_rot] = rotaciona_corte(x_plot, y_plot, angulo)
    % ROTACIONA_CORTE - Rotaciona coordenadas para o corte vertical
    %   [x_rot, y_rot] = rotaciona_corte(x_plot, y_plot, angulo)
    %
    %   x_plot, y_plot: coordenadas originais
    %   angulo: ângulo de rotação (radianos)
    %   x_rot, y_rot: coordenadas rotacionadas
    
    cos_ang = cos(angulo);
    sin_ang = sin(angulo);
    
    x_rot = x_plot * cos_ang - y_plot * sin_ang;
    y_rot = x_plot * sin_ang + y_plot * cos_ang;
end


% ============================================================================
% Função SIM_VOL - Define o volume de simualação e a grade
% ============================================================================

function [r_ea, np, nf, nq, np_i, alt, alt_i, lat, lon, r, theta, phi, ...
          dp_m, df_m, dq_m, dp_i, df_i, dq_i, y_f, z_f, y_sph, z_sph, ...
          alt_3, lon_3, lat_3, dr, dphi, dtheta] = sim_vol(i_hemi, i_mag)
    
    r_ea = 6371.0; % Raio da Terra em km
    
    % Cria o vetor de altitude
    alt = 0;
    alt_max = 0;
    
    while alt_max <= 1000  % Altitude máxima de 1000 km
        dr = 15.0;  % Resolução padrão de 15 km
        if alt_max >= 600.0
            dr = 60.0;  % Resolução reduzida para 60 km acima de 600 km
        end
        if alt_max >= 1200.0  % Esta condição não é atingida (max=1000 km)
            dr = 120.0;
        end
        
        alt = [alt, alt_max + dr];  % Adiciona próxima altitude
        alt_max = max(alt);
    end
    
    % Cria vetor de altitudes internas (a partir do 10º ponto)
    alt_i = alt(11:end);  
    np = length(alt);
    np_i = length(alt_i);
    
    % Exibe altitudes da ionosfera
    fprintf('Altitudes ionosfera: ');
    fprintf('%.0f ', alt_i);
    fprintf('\n');
    fprintf('Total de altitudes: %d (%.0f a %.0f km)\n', np, min(alt), max(alt));
    
    % Cria o vetor de longitude
    nf = 101;
    dr = 15.0;
    % Calcula passo angular baseado na altitude de referência (300 km)
    dphi = atan(3.0 * dr / (r_ea + 300.0));
    dphi = round(dphi, 5);  % round com 5 casas decimais
    lon_in = deg2rad(-80.0);  % Longitude inicial (-80° em radianos)
    % Cria array de longitudes em radianos
    lon = lon_in + (0:(nf-1)) * dphi;
    lon = rad2deg(lon);  % Converte para graus
    nf = length(lon);
    
    % Cria o vetor de latitude
    nq = 101;
    dtheta = 1.0 * dphi;  % Passo angular igual ao da longitude
    lat_in = deg2rad(-65.0);  % Latitude inicial (-65° em radianos)
    % Cria array de latitudes em radianos (multiplica por i_hemi para hemisfério)
    lat = i_hemi * (lat_in + (0:(nq-1)) * dtheta);
    lat = rad2deg(lat);  % Converte para graus
    nq = length(lat);
    
    fprintf('Dimensões da grade: %d latitudes, %d longitudes\n', length(lat), length(lon));
    
    % Cria as coordenadas 3D (r, theta, phi)
    % Inicializa arrays tridimensionais com zeros
    r = zeros(np, nf, nq);      % Distância radial (metros)
    theta = zeros(np, nf, nq);  % Ângulo colatitude (radianos)
    phi = zeros(np, nf, nq);    % Longitude (radianos)
    
    % Preenche r: converte altitude para distância radial (metros)
    for i = 1:np
        r(i, :, :) = (alt(i) + r_ea) * 1.0e3;
    end
    
    % Preenche phi (longitude) para todos os pontos
    for j = 1:nf
        phi(:, j, :) = deg2rad(lon(j));
    end
    
    % Preenche theta (colatitude) para todos os pontos
    for k = 1:nq
        theta(:, :, k) = deg2rad(lat(k));
    end
    
    % Cálculos das coordenadas curvilíneas e grid
    cos_theta = cos(theta);
    cos_theta = max(cos_theta, 1e-12);
    
    r_o = r ./ (cos_theta .^ 2);
    delta = sqrt(1.0 + i_mag * 3.0 * (sin(theta) .^ 2));
    
    q_curv = r_o .^ 3 .* sin(theta) ./ (r .^ 2 + 1e-12);
    p_curv = r ./ (cos_theta .^ 2);
    
    hq = (r ./ (r_o + 1e-12)) .^ 3 ./ delta;
    hphi = r .* cos_theta;
    hp = (cos_theta .^ 3) ./ delta;
    
    [dp_dx, dp_dy, dp_dz] = gradient(p_curv);
    [dq_dx, dq_dy, dq_dz] = gradient(q_curv);
    
    dp = abs(dp_dx + abs(dp_dz));
    dq = abs(dq_dx + dq_dz);
    
    dp_m = dp .* hp;
    dq_m = dq .* hq;
    df_m = dphi .* hphi;
    
    x_f = hphi .* tan(phi) .* 1.0e-3;
    y_f = p_curv .* hp .* 1.0e-3;
    z_f = q_curv .* hq .* 1.0e-3;
    
    y_sph = y_f .* delta;
    z_sph = z_f .* delta;
    
    alt_3 = y_sph - r_ea;
    alt_3(alt_3 < 0) = 0;
    
    lat_3 = rad2deg(atan(z_f ./ (y_f + 1e-12)));
    lon_3 = rad2deg(atan(x_f ./ (y_f + 1e-12)));
    
    fprintf('dp_m: min=%.6f, max=%.6f\n', min(dp_m(:)), max(dp_m(:)));
    fprintf('dq_m: min=%.6f, max=%.6f\n', min(dq_m(:)), max(dq_m(:)));
    
    dq_m(:) = max(dq_m(:));
    df_m(:) = max(df_m(:));
    
    dp_i = dp_m(np - np_i + 1:end, :, :);
    df_i = df_m(np - np_i + 1:end, :, :);
    dq_i = dq_m(np - np_i + 1:end, :, :);
    
    fprintf('=======================================================================\n');
    fprintf('GRID_SIZES:\n');
    fprintf('  dp_m: min=%.6f, max=%.6f\n', min(dp_m(:)), max(dp_m(:)));
    fprintf('  dq_m: min=%.6f, max=%.6f\n', min(dq_m(:)), max(dq_m(:)));
    fprintf('  df_m: min=%.6f, max=%.6f\n', min(df_m(:)), max(df_m(:)));
    fprintf('=======================================================================\n');
    
    % ============================================================================
    % Plotagem da grade (Figura 1)
    % ============================================================================

    % Cria figura 1 com tamanho 18x12 polegadas
    fig = figure(1);
    set(fig, 'Position', [100, 100, 1800, 1200], 'Color', 'white');

    % Cálculos auxiliares
    theta_r = 180.0 * atan2(y_f, z_f) / pi;
    alt_r = sqrt(y_f.^2 + z_f.^2);

    % ===== SUBPLOT ESQUERDO: GRADE COMPUTACIONAL =====
    ax1 = subplot(1,2,1);

    % Plota a grade com cores baseadas em y_f
    % pcolor com shading interp
%     pcolor(squeeze(z_sph(:,1,:)), squeeze(y_sph(:,1,:)), squeeze(y_f(:,1,:) - r_ea));
    shading interp;
    colorbar;
    hold on;

    % Linhas de grade (latitude) - a cada 10 pontos
    plot(squeeze(z_f(:,1,1:10:end)), squeeze(y_f(:,1,1:10:end)), 'Color', [0.5,0.5,0.5], 'LineWidth', 0.5);
    hold on;

    % Rótulos de latitude
    labels = round(lat(1:10:end), 0);
    ii = 1;
    for i = 1:10:nq
        text(z_f(np, 1, i), y_f(np, 1, i), num2str(labels(ii)), 'FontSize', 8);
        ii = ii + 1;
    end

    % Linhas de grade (altitude) - a cada 10 pontos
    plot(squeeze(z_f(1:10:end,1,:))', squeeze(y_f(1:10:end,1,:))', 'Color', [0.5,0.5,0.5], 'LineWidth', 0.5);

    % Rótulos de altitude
    labels = round(alt(1:10:end), 0);
    ii = 1;
    for i = 1:10:np
        text(z_f(i, 1, round(nq/2)), y_f(i, 1, round(nq/2)), num2str(labels(ii)), 'FontSize', 8);
        ii = ii + 1;
    end

    % Desenha círculo representando a Terra
    rectangle('Position', [-r_ea, -r_ea, 2*r_ea, 2*r_ea], ...
              'Curvature', [1,1], 'EdgeColor', 'b', 'LineWidth', 1, 'FaceColor', 'none');
    rectangle('Position', [-(r_ea-200), -(r_ea-200), 2*(r_ea-200), 2*(r_ea-200)], ...
              'Curvature', [1,1], 'EdgeColor', 'w', 'LineWidth', 1, 'FaceColor', 'none');

    % Remove eixos
    axis off;
    axis equal;
    title('Grade Computacional', 'FontSize', 14, 'FontWeight', 'bold');

    % Subplot direito: Mapa Mundi (versão simplificada sem Basemap) 
    ax2 = subplot(1,2,2);

    % Média para centralizar
    lat_center = mean(lat_3(:));
    lon_center = mean(lon_3(:));

    % Configura a projeção ortográfica
    figure(fig);
    axesm('ortho', 'Origin', [lat_center, lon_center, 0], 'Frame', 'off', 'Grid', 'off');
    
    % Carrega e plota continentes
    load coastlines;
    geoshow(coastlat, coastlon, 'Color', [0.2, 0.2, 0.2], 'LineWidth', 0.5);

    % Plota os dados como uma superfície sobre o mapa
    % (simplificado - para dados reais você precisaria interpolar)

    % Adiciona grade com rótulos
    plabel('on');   % Rótulos de latitude
    mlabel('on');   % Rótulos de longitude
    gridm('on');    % Liga a grade

    % ===== SALVA A FIGURA =====
%     nome_fig_grid = fullfile(pasta_figs, 'grid_mapa.png');
%     saveas(fig, nome_fig_grid);
%     fprintf('Grade e mapa salvos como: %s\n', nome_fig_grid);

    % Fecha a figura
    close(fig);
    
    
        % Garantir que dr, dphi, dtheta sejam retornados
    % dr já é definido dentro do while loop, mas precisamos do valor final
    % Vamos pegar o último dr usado (geralmente 15 ou 60)
    if exist('dr', 'var')
        % dr já existe
    else
        dr = 15.0;
    end
    
    % dphi e dtheta já são calculados
    if ~exist('dphi', 'var')
        dphi = atan(3.0 * 15.0 / (r_ea + 300.0));
        dphi = round(dphi, 5);
    end
    
    if ~exist('dtheta', 'var')
        dtheta = dphi;
    end
    
end

% ============================================================================
% Função Source_Space - Define a forma espacial da fonte (forçante)
% ============================================================================

function [source_shape, mask_bound, iphi_s, iq_s] = source_space(alt, lon, lat, np, nf, nq, dphi, dtheta, alt_3)
    
    % Perfil vertical (altitude)
    alt_o = 105.0;           % Altitude central da fonte (km)
    sigma_p = 15.0;          % Largura do pulso (km)
    var_p = exp(-((alt - alt_o).^2) / sigma_p^2); % Gaussiana centrada em alt_o
    
    % Perfil longitudinal
    iphi_s = floor(nf / 2.0) + 1;  % Índice central em longitude 
    lon_o = lon(iphi_s);           % Longitude central
    
    % Largura do pulso em longitude (convertendo de radianos para graus)
    sigma_x = 5.0 * rad2deg(dphi) / 2.0;
    var_f = exp(-((lon - lon_o).^2) / sigma_x^2); % Gaussiana centrada em lon_o
    
    % Perfil Longitudinal
    iq_s = floor(nq / 4.0) + 1;    % Índice a 1/4 da grade (em latitude)
    lat_o = lat(iq_s);              % Latitude central
    sigma_z = sigma_x;              % Mesma largura
    var_q = exp(-((lat - lat_o).^2) / sigma_z^2); % Gaussiana centrada em lat_o
    
    % Máscara de fronteira
    mask_bound = ones(np, nf, nq); % Inicializa a máscara como 1 (sem atenuação)
    
    % Combina os três perfis para formar a forma 3D da fonte
    source_shape = zeros(np, nf, nq);
    for i = 1:np
        for j = 1:nf
            for k = 1:nq
                source_shape(i, j, k) = var_p(i) * var_f(j) * var_q(k);
            end
        end
    end
    
    % Mask_Bound (atenuação nas bordas)
    j_m = floor(nf / 2.0) + 1;
    k_m = floor(nq / 2.0) + 1;
    rad_m = sqrt(j_m^2 + k_m^2);  % Raio máximo
    
    for j = 1:nf
        for k = 1:nq
            rad_o = sqrt((j - j_m)^2 + (k - k_m)^2);
            % Se estiver além de 60% do raio máximo, aplica atenuação
            if rad_o >= 0.6 * rad_m
                mask_bound(:, j, k) = exp(-((2.0 * rad_o / rad_m)^2));
            end
        end
    end
    
end

% ============================================================================
% Função Source_T - Define a evolução temporal da fonte
% ============================================================================

function s_t = source_t(t)
    omega = 2.0 * pi / (5.0 * 60.0);  % Frequência angular (período de 5 minutos)
    t_o = 2000.0;                      % Tempo central (segundos)
    sigma_t = 1500.0;                  % Largura do pulso (segundos)
    s_t = cos(0 * omega * t) * exp(-((t - t_o)^2) / sigma_t^2); % Pulso gaussiano
end

% ============================================================================
% Função ATMOS - Calcula perfis atmosféricos 
% ============================================================================

function [rho_o, rho_od, rho_ou, temp_o] = atmos(alt, dr)
    
    % Densidade atmosférica
    a = 0.25e25 * 32 * 1.6e-27 * 1.0e-08;  % Constante de normalização
    c = -3.5;
    d = -19.5;
    yr = 130.0;                             % Altitude de escala (km)
    
    % Função para perfil de densidade (inline)
    % rho_profile = @(z) a * (exp(c * (z/yr - 1)) + exp(d * (z/yr - 1)));
    
    % Densidade na altitude atual
    r_y = alt / yr - 1.0;
    rho_o = a * (exp(c * r_y) + exp(d * r_y));
    
    % Densidade um passo abaixo
    r_y = (alt - dr) / yr - 1.0;
    rho_od = a * (exp(c * r_y) + exp(d * r_y));
    
    % Densidade um passo acima
    r_y = (alt + dr) / yr - 1.0;
    rho_ou = a * (exp(c * r_y) + exp(d * r_y));
    
    % Perfil de temperatura 
    % Parâmetros para modelo empírico de temperatura
    a1 = 310; b1 = 200; c1 = -2.0; d1 = 0.5;
    a2 = 220; b2 = 250; c2 = -0.2; d2 = 10.0;
    a3 = 220; b3 = 160; c3 = -0.25; d3 = 5.5;
    a4 = 120; b4 = 250; c4 = -1.5; d4 = 18.0;
    a5 = 220; b5 = 220; c5 = -0.25; d5 = 5.0;
    y1 = 10.0; y2 = 75.0; y3 = 130.0; y4 = 100.0; y5 = 230.0;
    
    % Variáveis reduzidas para cada camada
    r_y1 = alt / y1 - 1.0;
    r_y2 = alt / y2 - 1.0;
    r_y3 = alt / y3 - 1.0;
    r_y4 = alt / y4 - 1.0;
    r_y5 = alt / y5 - 1.0;
    
    % Componentes do modelo de temperatura
    first = a1 - b1 ./ (exp(c1 * r_y1) + exp(d1 * r_y1));
    second = -a2 + b2 ./ (exp(c2 * r_y2) + exp(d2 * r_y2));
    third = a3 - b3 ./ (exp(c3 * r_y3) + exp(d3 * r_y3));
    fourth = -a4 + b4 ./ (exp(c4 * r_y4) + exp(d4 * r_y4));
    fifth = a5 - b5 ./ (exp(c5 * r_y5) + exp(d5 * r_y5));
    
    % Combina os componentes (duas versões - a segunda é a usada)
    % sixth = -130 + 1.0 * (first + second + fourth) + 5.2 * third + 0.5 * fifth;
    sixth = -720 + 1.25 * (first + second + fourth) + 8.5 * third + 1.5 * fifth;
    
    % Temperatura final (K)
    temp_o = 0.65 * sixth;
    
end

% ============================================================================
% FUNÇÃO IONO_AMB - CALCULA A IONOSFERA AMBIENTE
% ============================================================================

function [nu_in, nu_en, mag_o, q_charge, omega_i, omega_e, tec_fac, den_amb] = iono_amb(alt_i, np_i, nf, nq)
    
    % ===== CONSTANTES FÍSICAS =====
    q_charge = 1.6e-19;      % Carga do elétron (Coulombs)
    b_c = 1.38e-23;          % Constante de Boltzmann (J/K)
    eps_o = 8.854e-12;       % Permissividade do vácuo (F/m)
    m_i = 1.67e-27;          % Massa do próton (kg)
    z_i = 16.0;              % Número de massa do íon (Oxigênio)
    
    % Campo magnético terrestre
    mag_o = 0.25e-4;         % Intensidade (Tesla)
    
    % Frequências de giro
    omega_i = q_charge * mag_o / (z_i * m_i);   % Íons
    omega_e = -omega_i * 1837.0;                % Elétrons
    
    % ===== DENSIDADE NEUTRA =====
    den_100 = 1.0e18;        % Densidade de referência a 100 km (m^-3)
    sc_h = 40.0;             % Altitude de escala (km)
    den_neu = den_100 * exp(-(alt_i - 100.0) / sc_h);
    
    % ===== PERFIL DE DENSIDADE IÔNICA =====
    yp = 3.0e02;             % Altitude de escala (km)
    enp = 2.0e12;            % Densidade de pico (m^-3)
    enpe = enp * 1.0e-2 * 1.0e1;
    
    af = 1.0;
    bf = -10.0;
    ae = 5.0;
    be = -60.0;
    
    r_n = alt_i / yp;        % Altitude reduzida
    r_of = 0.9;
    r_oe = 0.255;
    r_ov = 0.5;
    
    % Componentes da densidade iônica
    den_f = enp ./ (exp(af * (r_n - r_of)) + exp(bf * (r_n - r_of)));
    den_e = enpe ./ (exp(ae * (r_n - r_oe)) + exp(be * (r_n - r_oe)));
    den_ef = 5.0e-1 * enpe * (r_n + r_ov).^2 / 5.0;
    den_i = den_f + den_e + den_ef;
    
    % ===== CONTEÚDO ELETRÔNICO TOTAL (TEC) =====
    int_fac = 50.0 * 60.0 * 1.0e3;
    tec_fac = int_fac / 1.0e18;
    tec = 5.0 * sum(den_i) * tec_fac;  % TEC em unidades (TECU)
    
    % ===== FREQUÊNCIAS DE COLISÃO =====
    t_i = 800.0;             % Temperatura iônica (K)
    a = 16.0;                % Massa atômica do oxigênio
    
    nu_ii = 0.22 * den_i * 1.0e-6 ./ t_i^1.5;                      % Íon-íon
    nu_in_1 = 2.6e-9 * 1.0e-6 * (den_neu + den_i) * a^(-0.5);     % Íon-neutro
    nu_ei = 34 * den_i * 1.0e-6 ./ t_i^1.5;                        % Elétron-íon
    nu_en_1 = 5.4e-16 * den_neu * sqrt(t_i) + nu_ei;               % Elétron-neutro
    
    % Frequência de plasma (não utilizada, mas calculada)
    w_p1 = sqrt(den_i * q_charge^2 / (eps_o * m_i * z_i));
    
    % ===== CORREÇÃO PARA ALTITUDES MAIORES (se necessário) =====
    % Nota: MATLAB lida automaticamente com arrays de diferentes tamanhos
    % Se precisar estender os valores para altitudes maiores (como no Python),
    % descomente as linhas abaixo:
    %
    % if length(den_i) > 40
    %     den_i(40:end) = den_i(40);
    %     nu_in_1(40:end) = nu_in_1(40);
    %     nu_en_1(40:end) = nu_en_1(40);
    % end
    
    % ===== INICIALIZA ARRAYS 3D =====
    den_amb = zeros(np_i, nf, nq);
    nu_in = zeros(np_i, nf, nq);
    nu_en = zeros(np_i, nf, nq);
    
    % Preenche arrays 3D com valores 1D
    for k = 1:nq
        for j = 1:nf
            den_amb(:, j, k) = den_i;
            nu_in(:, j, k) = nu_in_1;
            nu_en(:, j, k) = nu_en_1;
        end
    end
    
end

% ============================================================================
% FUNÇÃO FOTO - CÁLCULO DE FOTO-IONIZAÇÃO (NÃO IMPLEMENTADA)
% ============================================================================
%
% NOTA: Esta função está completamente comentada no código original,
%       indicando que nunca foi ativada/implementada.
%       Mantida apenas para referência e compatibilidade estrutural.
%
% ============================================================================

function foto()
    % A função original em Python estava totalmente comentada.
    
    % % Obtém a ionosfera ambiente (se necessário)
    % % [nu_in, nu_en, mag_o, q_charge, omega_i, omega_e, tec_fac, den_amb] = iono_amb(alt_i, np_i, nf, nq);
    % 
    % % Números de onda
    % wk_f = 2 * pi / (1200 * dphi);
    % wk_q = 2 * pi / (200 * dtheta);
    % 
    % % Inicializa matriz de TEC de foto-ionização
    % tec_t = zeros(nf, nq);
    % 
    % % Loop sobre longitudes e latitudes
    % for j = 1:nf
    %     for k = 1:nq
    %         lat_o = 13.0;
    %         sigma = 15.0 * dtheta;
    %         gauss_lat = exp(-(abs(-12.0 + lat(k)) - lat_o)^2 / sigma^2);
    %         amp = 0.5 * gauss_lat;
    %         fac_bg = (1.0 - 0.9 * cos(wk_f * (lon(j) - 35))) * (1 + amp);
    %         tec_t(j, k) = fac_bg;  % Original: f[3] * fac_bg
    %     end
    % end
    % 
    % % Inicializa matriz de foto (variação temporal)
    % foto_matrix = zeros(nf, nq);
    % 
    % % Loop sobre longitudes (variação temporal)
    % for j = 1:nf
    %     time = t(i_t) / 3600.0;
    %     t_o = -1.5 + (lon(nf) - lon(j)) * 1.0 / 15.0;
    %     sigma = 1.75;
    %     amp_t = exp(-(time - t_o)^2 / sigma^2);
    %     period = 3600.0;
    %     fac = -2.0 * (time - 1.5) / sigma^2;
    %     foto_matrix(j, :) = 0 * fac * amp_t / period;  % Sempre zero!
    % end
    
end

% ============================================================================
% FUNÇÃO MOBILITY - CALCULA MOBILIDADES DOS ÍONS E ELÉTRONS
% ============================================================================
%
% Retorna:
%   mu_p_i - Mobilidade de Pedersen dos íons
%   mu_h_i - Mobilidade de Hall dos íons
%   mu_o_i - Mobilidade paralela dos íons
%   mu_p_e - Mobilidade de Pedersen dos elétrons
%   mu_h_e - Mobilidade de Hall dos elétrons
%   mu_o_e - Mobilidade paralela dos elétrons
%
% ============================================================================

function [mu_p_i, mu_h_i, mu_o_i, mu_p_e, mu_h_e, mu_o_e] = mobility(nu_in, nu_en, i_inert, dt, omega_i, omega_e, mag_o)
    
    % ===== MOBILIDADE DOS ÍONS =====
    omega = i_inert / dt;
    nu_eff = nu_in + omega;
    kappa = omega_i ./ nu_eff;
    
    mu_p_i = kappa ./ (mag_o * (1 + kappa.^2));
    mu_h_i = (kappa.^2) ./ (mag_o * (1 + kappa.^2));
    mu_o_i = kappa ./ mag_o;
    
    % ===== MOBILIDADE DOS ELÉTRONS =====
    nu_eff = nu_en + omega;
    kappa = omega_e ./ nu_eff;
    
    mu_p_e = kappa ./ (mag_o * (1 + kappa.^2));
    mu_h_e = (kappa.^2) ./ (mag_o * (1 + kappa.^2));
    mu_o_e = kappa ./ mag_o;
    
end

% ============================================================================
% FUNÇÃO CONDUCTIVITY - CALCULA CONDUTIVIDADES
% ============================================================================
%
% Calcula as condutividades Pedersen, Hall e paralela
%
% Entradas:
%   den_t - densidade iônica (array 3D)
%   q_charge - carga do elétron (constante)
%   mu_p_i, mu_h_i, mu_o_i - mobilidades dos íons (de mobility())
%   mu_p_e, mu_h_e, mu_o_e - mobilidades dos elétrons (de mobility())
%
% Saídas:
%   sigma_p - condutividade de Pedersen
%   sigma_h - condutividade de Hall
%   sigma_o - condutividade paralela
%
% ============================================================================

function [sigma_p, sigma_h, sigma_o] = conductivity(den_t, q_charge, mu_p_i, mu_h_i, mu_o_i, mu_p_e, mu_h_e, mu_o_e)
    
    % Calcula as condutividades
    sigma_p = q_charge * den_t .* (mu_p_i - mu_p_e);
    sigma_h = q_charge * den_t .* (mu_h_i - mu_h_e);
    sigma_o = q_charge * den_t .* (mu_o_i - mu_o_e);
    
    % Exibe valores máximos
    fprintf('CONDUCTIVITY: sigma_p max=%.6e, sigma_h max=%.6e, sigma_o max=%.6e\n', ...
            max(sigma_p(:)), max(sigma_h(:)), max(sigma_o(:)));
    
end

% ============================================================================
% FUNÇÃO CAMPO_ELE_SPH_COORD - CAMPO ELÉTRICO EM COORDENADAS ESFÉRICAS
% ============================================================================
%
% Calcula o campo elétrico induzido em coordenadas esféricas
%
% Entradas:
%   t_e   - tempo (não utilizado diretamente)
%   wp_g  - velocidade vertical (3D array)
%   wf_g  - velocidade horizontal (3D array)
%   wq_g  - velocidade paralela (3D array)
%   den_t - densidade iônica
%   pot   - potencial elétrico
%
% Saídas:
%   pot    - potencial (mesmo da entrada)
%   up_ele - velocidade elétrica vertical (zeros)
%
% Variáveis globais modificadas:
%   up_tot, uf_tot, uq_tot - velocidades totais na ionosfera
%
% ============================================================================

function [pot, up_ele, up_tot, uf_tot, uq_tot] = campo_ele_sph_coord(t_e, wp_g, wf_g, wq_g, den_t, pot, theta, np, np_i, mag_o, sigma_p, sigma_h, sigma_o, mu_p_i, mu_h_i, mu_o_i, mu_p_e, mu_h_e, mu_o_e, up_tot)

    % ===== ÂNGULO DE INCLINAÇÃO MAGNÉTICA =====
    % Extrai a parte da ionosfera (últimos np_i pontos)
    angle_I = theta(np - np_i + 1:end, :, :);
    
    % ===== TENSOR DE CONDUTIVIDADE =====
    % Os parâmetros sigma_p, sigma_h, sigma_o já são calculados fora
    % e passados como argumentos
    
    % Cálculo dos componentes do tensor de condutividade
    cos_angle = cos(angle_I);
    sin_angle = sin(angle_I);
    
    s_11 = sigma_p;
    s_12 = -sigma_h .* cos_angle;
    s_13 = -sigma_h .* sin_angle;
    s_21 = -s_12;
    s_22 = sigma_p .* cos_angle.^2 + sigma_o .* sin_angle.^2;
    s_23 = (sigma_p - sigma_o) .* sin_angle .* cos_angle;
    s_31 = -s_12;
    s_32 = s_23;
    s_33 = sigma_o .* cos_angle.^2 + sigma_p .* sin_angle.^2;
    
    % ===== VELOCIDADES INDUZIDAS PELAS ONDAS - ÍONS =====
    kappa_i = mu_o_i * mag_o;
    
    mu_11_i = mu_p_i;
    mu_12_i = -mu_11_i .* kappa_i .* cos_angle;
    mu_13_i = -mu_11_i .* kappa_i .* sin_angle;
    mu_21_i = -mu_12_i;
    mu_22_i = mu_11_i .* (1.0 + kappa_i.^2 .* sin_angle.^2);
    mu_23_i = -mu_11_i .* kappa_i.^2 .* cos_angle .* sin_angle;
    mu_31_i = -mu_13_i;
    mu_32_i = mu_23_i;
    mu_33_i = mu_11_i .* (1.0 + kappa_i.^2 .* cos_angle.^2);
    
    % Campo elétrico induzido pelo movimento da onda
    ele_w_p = -wf_g * mag_o;
    ele_w_f = wp_g * mag_o;
    ele_w_q = wq_g * mag_o;
    
    % Velocidade dos íons
    up_agw_i = mu_11_i .* ele_w_p + mu_12_i .* ele_w_f + mu_13_i .* ele_w_q;
    uf_agw_i = mu_22_i .* ele_w_f + mu_21_i .* ele_w_p + mu_23_i .* ele_w_q;
    uq_agw_i = mu_33_i .* ele_w_q + mu_31_i .* ele_w_p + mu_32_i .* ele_w_f;
    
    % ===== VELOCIDADES INDUZIDAS PELAS ONDAS - ELÉTRONS =====
    kappa_e = mu_o_e * mag_o;
    
    mu_11_e = mu_p_e;
    mu_12_e = -mu_11_e .* kappa_e .* cos_angle;
    mu_13_e = -mu_11_e .* kappa_e .* sin_angle;
    mu_21_e = -mu_12_e;
    mu_22_e = mu_11_e .* (1.0 + kappa_e.^2 .* sin_angle.^2);
    mu_23_e = -mu_11_e .* kappa_e.^2 .* cos_angle .* sin_angle;
    mu_31_e = -mu_13_e;
    mu_32_e = mu_23_e;
    mu_33_e = mu_11_e .* (1.0 + kappa_e.^2 .* cos_angle.^2);
    
    % Velocidade dos elétrons
    up_agw_e = mu_11_e .* ele_w_p + mu_12_e .* ele_w_f + mu_13_e .* ele_w_q;
    uf_agw_e = mu_22_e .* ele_w_f + mu_21_e .* ele_w_p + mu_23_e .* ele_w_q;
    uq_agw_e = mu_33_e .* ele_w_q + mu_31_e .* ele_w_p + mu_32_e .* ele_w_f;
    
    % Termo elétrico (não implementado)
    up_ele = zeros(size(up_agw_e));
    uf_ele = zeros(size(uf_agw_e));
    uq_ele = zeros(size(uq_agw_e));
    
    % ===== VELOCIDADES TOTAIS =====
    up_tot = up_ele + up_agw_e;
    uf_tot = uf_ele + uf_agw_e;
    uq_tot = uq_ele + uq_agw_e;
    
    % Exibe valores máximos
    fprintf('POTENTIAL, U_MAX: up=%.2f, uf=%.2f, uq=%.2f\n', ...
            max(up_tot(:)), max(uf_tot(:)), max(uq_tot(:)));
    
end

% ============================================================================
% FUNÇÃO DEN_ION - ATUALIZA A DENSIDADE IÔNICA PELA EQUAÇÃO DA CONTINUIDADE
% ============================================================================
%
% Resolve a equação da continuidade para a densidade iônica usando
% 8 iterações do método de relaxação.
%
% Entradas:
%   den_t   - densidade iônica atual (array 3D)
%   up_tot  - velocidade total na direção P (array 3D)
%   uf_tot  - velocidade total na direção F (array 3D)
%   uq_tot  - velocidade total na direção Q (array 3D)
%   dp_i    - espaçamento na direção P na ionosfera (array 3D)
%   df_i    - espaçamento na direção F na ionosfera (array 3D)
%   dq_i    - espaçamento na direção Q na ionosfera (array 3D)
%   dt      - passo de tempo
%   den_amb - densidade ambiente (array 3D)
%   tec_fac - fator de conversão para TEC
%
% Saídas:
%   den_t    - densidade iônica atualizada
%   delta_den - variação relativa da densidade
%   dtec      - variação do TEC (Conteúdo Eletrônico Total)
%
% ============================================================================

function [den_t, delta_den, dtec] = den_ion(den_t, up_tot, uf_tot, uq_tot, ...
                                             dp_i, df_i, dq_i, dt, den_amb, tec_fac)
    
    % Salva o valor original da densidade
    den_o = den_t;
    den_g = den_t;
    
    % 8 iterações para resolver a continuidade
    for i_sor = 1:8
        % Calcula os fluxos
        flux_p = den_g .* up_tot;
        flux_f = den_g .* uf_tot;
        flux_q = den_g .* uq_tot;
        
        % Calcula os gradientes dos fluxos
        [gr_p_x, gr_p_y, gr_p_z] = gradient(flux_p);
        [gr_f_x, gr_f_y, gr_f_z] = gradient(flux_f);
        [gr_q_x, gr_q_y, gr_q_z] = gradient(flux_q);
        
        % Divergência do fluxo
        div_flux = gr_p_x ./ dp_i + gr_f_y ./ df_i + gr_q_z ./ dq_i;
        
        % Atualiza a densidade
        den_g = den_o - dt * div_flux;
    end
    
    % Atualiza a densidade final
    den_t = den_g;
    
    % Variação relativa da densidade
    delta_den = (den_t - den_amb) ./ den_t;
    
    % Variação do TEC (Conteúdo Eletrônico Total)
    % sum(den_t, 1) soma na primeira dimensão (altitude)
    dtec = tec_fac * (sum(den_t, 1) - sum(den_amb, 1));
    
    % Exibe o valor máximo da variação percentual
    fprintf('ION_DENSITY: max delta = %.2f%%\n', 100.0 * max(delta_den(:)));
    
end

% ============================================================================
% FUNÇÃO AGW_2 - VERSÃO COMPACTA
% ============================================================================

function [w_1, w_2, w_3] = AGW_2(i_axis, delta, press, rho, rho_d, rho_u, div_w, div_flux, w_gr_press)
    
    gamma = 1.4;
    
    % Função auxiliar para extrair componente do gradiente
    get_grad_component = @(arr) extract_gradient_component(arr, i_axis);
    
    % Termo 1
    flux_p = gamma * press .* div_w;
    grad_flux = get_grad_component(flux_p) ./ delta;
    w_1 = grad_flux ./ rho;
    
    % Termo 2
    grad_press = abs(get_grad_component(press)) ./ delta;
    rho_m = (rho_u + 2.0 * rho + rho_d) / 4.0;
    w_2 = -grad_press .* div_flux ./ (rho_m.^2);
    
    % Termo 3
    w_3 = get_grad_component(w_gr_press) ./ delta;
    
end

% Função auxiliar para extrair componente do gradiente
function grad_comp = extract_gradient_component(arr, i_axis)
    [grad_x, grad_y, grad_z] = gradient(arr);
    switch i_axis
        case 1
            grad_comp = grad_x;
        case 2
            grad_comp = grad_y;
        case 3
            grad_comp = grad_z;
        otherwise
            error('i_axis deve ser 1, 2 ou 3');
    end
end

% ============================================================================
%  Função para desenhar o fundo do mapa
% ============================================================================

function desenhar_fundo_mapa()
    % No MATLAB, a função equivalente ao Basemap precisa do Mapping Toolbox
    % Esta é uma versão simplificada sem Basemap
    
    % Para usar o Basemap equivalente no MATLAB:
    % load coastlines;
    % geoplot(coastlat, coastlon);
    % ou
    try
        worldmap('world');
        load coast;
        plotm(lat, long);
    catch
        fprintf('Nota: Função desenhar_fundo_mapa requer Mapping Toolbox no MATLAB\n');
        fprintf('Use geoplot ou worldmap para mapas.\n');
    end
end

% ============================================================================
% Função para salvar keogramas
% ============================================================================

function salvar_keograma(x, y, dados, titulo, nome_arquivo, xlabel_str, ylabel_str, flip_horizontal, flip_vertical)
    % Salva um keograma com x crescendo da esquerda para a direita.
    
    % Valores padrão para parâmetros opcionais
    if nargin < 6
        xlabel_str = 'Tempo (s)';
    end
    if nargin < 7
        ylabel_str = 'Longitude/Latitude';
    end
    if nargin < 8
        flip_horizontal = false;
    end
    if nargin < 9
        flip_vertical = false;
    end
    
    % ===== VERIFICA SE OS DADOS SÃO UMA MATRIZ 2D =====
    % Remove dimensões singleton se existirem
    dados = squeeze(dados);
    
    % Verifica se ainda é 3D
    if ndims(dados) > 2
        error('Os dados têm %d dimensões. Deve ser uma matriz 2D.', ndims(dados));
    end
    
    % Se for um vetor linha ou coluna, transforma em matriz 2D
    if isvector(dados)
        dados = dados(:);  % Transforma em vetor coluna
    end
    
    % Cria uma nova figura com tamanho 10x6 polegadas
    fig = figure('Position', [100, 100, 1000, 600], 'Color', 'white');
    
    % ===== VERIFICA AS DIMENSÕES DOS DADOS =====
    [n_linhas, n_colunas] = size(dados);
    
    % Se os dados têm o formato (len(x), len(y)), faz transposição
    if n_linhas == length(x) && n_colunas == length(y)
        dados = dados';
    elseif n_linhas ~= length(y) || n_colunas ~= length(x)
        fprintf('Aviso: Formato inesperado [%d, %d], tentando usar como está\n', n_linhas, n_colunas);
    end
    
    % ===== APLICA FLIPS (INVERSÕES) SE NECESSÁRIO =====
    dados_plot = dados;
    if flip_horizontal
        dados_plot = fliplr(dados_plot);  % Inverte esquerda/direita
        fprintf('  ? Flip horizontal aplicado em %s\n', nome_arquivo);
    end
    if flip_vertical
        dados_plot = flipud(dados_plot);  % Inverte cima/baixo
        fprintf('  ? Flip vertical aplicado em %s\n', nome_arquivo);
    end
    
    % ===== CRIA O GRÁFICO =====
%     % pcolor com shading interp para efeito similar ao pcolormesh
%     pcolor(x, y, dados_plot);
%     shading interp;
%     colormap('seismic');  % Mapa de cores sísmico (vermelho-azul)

    % ===== CRIA O GRÁFICO =====
    % Garante que x e y são vetores linha
    if size(x, 1) > 1
        x = x';
    end
    if size(y, 1) > 1
        y = y';
    end
    
    % Verifica se as dimensões são compatíveis
    fprintf('Debug: size(x)=[%d,%d], size(y)=[%d,%d], size(dados_plot)=[%d,%d]\n', ...
            size(x,1), size(x,2), size(y,1), size(y,2), size(dados_plot,1), size(dados_plot,2));
    
    % Usa imagesc para dados 2D (alternativa ao pcolor)
    imagesc(x, y, dados_plot);
    set(gca, 'YDir', 'normal');  % Garante orientação correta
%     colormap('seismic');
    colormap('jet');

    
    % Adiciona barra de cores
    colorbar;
    
    % Adiciona rótulos e título
    xlabel(xlabel_str);
    ylabel(ylabel_str);
    title(titulo);
    
    % Salva a figura como PNG com resolução de 150 dpi
    print(fig, nome_arquivo, '-dpng', '-r150');
    
    % Fecha a figura
    close(fig);
    fprintf('Keograma salvo: %s\n', nome_arquivo);
end