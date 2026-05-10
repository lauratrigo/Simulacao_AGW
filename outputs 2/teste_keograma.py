# =============================================================================#
# PLOTADOR DE KEOGRAMAS A PARTIR DAS IMAGENS PNG SALVAS
# =============================================================================#

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import re

# =============================================================================#
# CONFIGURAÇÕES
# =============================================================================#
pasta_imagens = "figs_output"  # Pasta onde estão as imagens PNG
pasta_saida = "keogramas_output"  # Pasta onde salvar os keogramas

# Cria pasta de saída se não existir
Path(pasta_saida).mkdir(parents=True, exist_ok=True)

# =============================================================================#
# FUNÇÃO PARA CRIAR KEOGRAMA A PARTIR DE UMA LISTA DE IMAGENS
# =============================================================================#
def criar_keograma_de_imagens(lista_imagens, nome_saida, titulo, extrair_tempo=True):
    """
    Cria um keograma a partir de uma sequência de imagens PNG.
    
    Args:
        lista_imagens: lista de caminhos das imagens ordenadas por tempo
        nome_saida: nome do arquivo de saída
        titulo: título do keograma
        extrair_tempo: se True, extrai o tempo do nome do arquivo
    """
    if not lista_imagens:
        print(f"Nenhuma imagem encontrada para {titulo}")
        return
    
    print(f"Processando {len(lista_imagens)} imagens para: {titulo}")
    
    # Carrega a primeira imagem para obter as dimensões
    primeira_imagem = Image.open(lista_imagens[0])
    largura, altura = primeira_imagem.size
    
    # Cria um array para armazenar os dados do keograma
    # Vamos pegar uma linha do meio de cada imagem
    keograma = np.zeros((len(lista_imagens), largura))
    
    tempos = []
    
    for i, img_path in enumerate(lista_imagens):
        # Carrega a imagem em escala de cinza
        img = Image.open(img_path).convert('L')  # 'L' = escala de cinza
        img_array = np.array(img)
        
        # Pega a linha do meio da imagem
        linha_meio = altura // 2
        keograma[i, :] = img_array[linha_meio, :]
        
        # Extrai o tempo do nome do arquivo (ex: mapa_00001.png)
        if extrair_tempo:
            match = re.search(r'(\d+)', img_path.name)
            if match:
                tempos.append(int(match.group(1)))
            else:
                tempos.append(i)
        else:
            tempos.append(i)
    
    # Converte tempos para horas (assumindo cada passo = 20s * 5 = 100s)
    tempos_horas = np.array(tempos) * 100 / 3600.0
    
    # Cria o keograma
    fig = plt.figure(figsize=(12, 6), facecolor="w", edgecolor="k")
    ax = plt.gca()
    
    im = ax.imshow(keograma, aspect='auto', cmap='seismic', 
                   extent=[0, largura, tempos_horas[-1], tempos_horas[0]])
    
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.set_xlabel('Coluna da imagem (longitude →)', fontsize=12)
    ax.set_ylabel('Tempo (horas)', fontsize=12)
    
    # Barra de cores
    plt.colorbar(im, label='Intensidade')
    
    plt.tight_layout()
    fig.savefig(nome_saida, dpi=150)
    plt.close(fig)
    print(f"Keograma salvo: {nome_saida}")

# =============================================================================#
# FUNÇÃO PARA EXTRAIR UMA LINHA ESPECÍFICA DA IMAGEM
# =============================================================================#
def criar_keograma_linha_especifica(lista_imagens, nome_saida, titulo, 
                                     linha_percentual=50, extrair_tempo=True):
    """
    Cria um keograma a partir de uma linha específica da imagem.
    
    Args:
        linha_percentual: porcentagem da altura da imagem (0 = topo, 100 = base)
    """
    if not lista_imagens:
        print(f"Nenhuma imagem encontrada para {titulo}")
        return
    
    print(f"Processando {len(lista_imagens)} imagens para: {titulo}")
    
    # Carrega a primeira imagem para obter as dimensões
    primeira_imagem = Image.open(lista_imagens[0])
    largura, altura = primeira_imagem.size
    
    # Calcula a linha a ser extraída
    linha = int(altura * linha_percentual / 100)
    
    # Cria um array para armazenar os dados do keograma
    keograma = np.zeros((len(lista_imagens), largura))
    
    tempos = []
    
    for i, img_path in enumerate(lista_imagens):
        # Carrega a imagem em escala de cinza
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)
        
        # Pega a linha específica
        keograma[i, :] = img_array[linha, :]
        
        # Extrai o tempo do nome do arquivo
        if extrair_tempo:
            match = re.search(r'(\d+)', img_path.name)
            if match:
                tempos.append(int(match.group(1)))
            else:
                tempos.append(i)
        else:
            tempos.append(i)
    
    # Converte tempos para horas
    tempos_horas = np.array(tempos) * 100 / 3600.0
    
    # Cria o keograma
    fig = plt.figure(figsize=(12, 6), facecolor="w", edgecolor="k")
    ax = plt.gca()
    
    im = ax.imshow(keograma, aspect='auto', cmap='seismic',
                   extent=[0, largura, tempos_horas[-1], tempos_horas[0]])
    
    ax.set_title(f"{titulo} (linha {linha_percentual}%)", fontsize=14, fontweight='bold')
    ax.set_xlabel('Coluna da imagem (longitude →)', fontsize=12)
    ax.set_ylabel('Tempo (horas)', fontsize=12)
    
    plt.colorbar(im, label='Intensidade')
    
    plt.tight_layout()
    fig.savefig(nome_saida, dpi=150)
    plt.close(fig)
    print(f"Keograma salvo: {nome_saida}")

# =============================================================================#
# FUNÇÃO PARA LISTAR E ORDENAR IMAGENS
# =============================================================================#
def listar_imagens_ordenadas(pasta, prefixo):
    """Lista e ordena imagens por número sequencial"""
    imagens = list(Path(pasta).glob(f"{prefixo}_*.png"))
    
    # Extrai o número do nome e ordena
    def extrair_numero(caminho):
        match = re.search(r'(\d+)', caminho.name)
        return int(match.group(1)) if match else 0
    
    imagens.sort(key=extrair_numero)
    return imagens

# =============================================================================#
# PRINCIPAL
# =============================================================================#
print("="*60)
print("PLOTADOR DE KEOGRAMAS - A PARTIR DAS IMAGENS PNG")
print("="*60)

# Lista as imagens disponíveis
imagens_mapa = listar_imagens_ordenadas(pasta_imagens, "mapa")
imagens_corte = listar_imagens_ordenadas(pasta_imagens, "corte")
imagens_campo = listar_imagens_ordenadas(pasta_imagens, "campo")

print(f"\nImagens encontradas:")
print(f"  - mapa: {len(imagens_mapa)} arquivos")
print(f"  - corte: {len(imagens_corte)} arquivos")
print(f"  - campo: {len(imagens_campo)} arquivos")

if len(imagens_mapa) == 0:
    print("\nERRO: Nenhuma imagem encontrada!")
    print("Certifique-se de que a simulação já foi executada e as imagens foram salvas.")
    exit(1)

# =============================================================================#
# CRIA OS KEOGRAMAS
# =============================================================================#
print("\n" + "="*60)
print("GERANDO KEOGRAMAS...")
print("="*60)

# Keograma a partir das imagens "mapa"
criar_keograma_de_imagens(
    imagens_mapa,
    f"{pasta_saida}/keograma_mapa.png",
    "Keograma - Mapas AGW/ΔTEC"
)

# Keograma a partir das imagens "corte" (linha do meio)
criar_keograma_de_imagens(
    imagens_corte,
    f"{pasta_saida}/keograma_corte.png",
    "Keograma - Cortes verticais"
)

# Keograma a partir das imagens "campo"
criar_keograma_de_imagens(
    imagens_campo,
    f"{pasta_saida}/keograma_campo.png",
    "Keograma - Campo elétrico/densidade"
)

# =============================================================================#
# CRIA KEOGRAMAS COM LINHAS ESPECÍFICAS DAS IMAGENS "corte"
# =============================================================================#
if len(imagens_corte) > 0:
    # Linha superior (topo da imagem) - alta altitude
    criar_keograma_linha_especifica(
        imagens_corte,
        f"{pasta_saida}/keograma_corte_topo.png",
        "Keograma - Corte (alta altitude)",
        linha_percentual=20  # 20% do topo
    )
    
    # Linha inferior (base da imagem) - baixa altitude
    criar_keograma_linha_especifica(
        imagens_corte,
        f"{pasta_saida}/keograma_corte_base.png",
        "Keograma - Corte (baixa altitude)",
        linha_percentual=80  # 80% do topo (próximo da base)
    )

# =============================================================================#
# FINALIZAÇÃO
# =============================================================================#
print("\n" + "="*60)
print(f"PROCESSO CONCLUÍDO!")
print(f"Keogramas salvos em: {pasta_saida}")
print("\nArquivos gerados:")
print(f"  - keograma_mapa.png")
print(f"  - keograma_corte.png")
print(f"  - keograma_campo.png")
print(f"  - keograma_corte_topo.png")
print(f"  - keograma_corte_base.png")
print("="*60)