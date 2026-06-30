# 🌎 Código de Simulação Analítica de Ondas Acústico-Gravitacionais (AGWs)

Implementação em Python da simulação analítica apresentada no artigo <br>
<b>"A New Analytical Simulation Code of Acoustic-Gravity Waves of Seismic Origin and Rapid Co-Seismic Thermospheric Disturbance Energetics"</b>.
</p>

---

# 📖 Sobre o Projeto

Este repositório reúne o código desenvolvido e utilizado no artigo científico:

> **Sanchez, S. A.; Kherani, E. A. (2024).**  
> *A New Analytical Simulation Code of Acoustic-Gravity Waves of Seismic Origin and Rapid Co-Seismic Thermospheric Disturbance Energetics.*  
> **Atmosphere**, 15(5), 592.  
> https://doi.org/10.3390/atmos15050592

A **branch `main`** preserva o **código original**, exatamente como disponibilizado pelos autores do artigo, sem qualquer modificação.

Além da versão original, este repositório também é utilizado para o desenvolvimento de novas implementações e experimentos relacionados ao modelo. As demais branches contêm adaptações, testes e novas funcionalidades que estão sendo desenvolvidas, incluindo uma implementação da simulação em **MATLAB**.

Dessa forma, o repositório serve tanto como um **registro da implementação original** quanto como um ambiente para pesquisa, reprodução e evolução do modelo apresentado no artigo.

---

# 🎯 Objetivos

O código realiza a simulação de:

- Propagação de Ondas Acústico-Gravitacionais (AGWs)
- Acoplamento Sismo-Atmosfera-Ionosfera (SAI)
- Distúrbios Termosféricos Co-Sísmicos (CSTDs)
- Perturbações de pressão atmosférica
- Perturbações de temperatura
- Velocidade horizontal das ondas
- Velocidade vertical das ondas
- Evolução temporal da propagação das ondas na atmosfera

---

# 📂 Estrutura do Repositório

```text
.
├── Code_251024_143119.py      # Código principal da simulação
├── nrlmsise_2000.py           # Interface com o modelo atmosférico NRLMSISE-00
├── signal_alam.py             # Funções auxiliares
├── agws.png                   # Figura ilustrativa do artigo
│
├── KSN.npy                    # Dados sísmicos de entrada
├── time.npy                   # Vetor temporal
├── cs3.npy                    # Dados da velocidade do som
├── pr3.npy                    # Dados de pressão
├── tn3.npy                    # Dados de temperatura
├── wx3.npy                    # Velocidade horizontal
├── wy3.npy                    # Velocidade vertical
│
└── __pycache__
```

---

# ⚙️ Requisitos

O projeto foi desenvolvido em **Python** e utiliza principalmente as seguintes bibliotecas:

- NumPy
- SciPy
- Matplotlib
- nrlmsise00

Instale as dependências com:

```bash
pip install numpy scipy matplotlib nrlmsise00
```

---

# ▶️ Como Executar

Após instalar as dependências, execute:

```bash
python Code_251024_143119.py
```

O script utiliza automaticamente os arquivos `.npy` presentes no repositório para realizar a simulação.

---

# 📊 Dados Utilizados

Os arquivos presentes neste repositório correspondem aos dados utilizados na simulação descrita no artigo.

| Arquivo | Descrição |
|----------|-----------|
| `KSN.npy` | Dados do sinal sísmico utilizado como entrada |
| `time.npy` | Vetor temporal da simulação |
| `cs3.npy` | Perfil da velocidade do som |
| `pr3.npy` | Dados de pressão atmosférica |
| `tn3.npy` | Dados de temperatura |
| `wx3.npy` | Velocidade horizontal das ondas |
| `wy3.npy` | Velocidade vertical das ondas |

Esses arquivos são carregados automaticamente pelo código principal e devem permanecer no mesmo diretório.

---

# 🌿 Organização das Branches

| Branch | Descrição |
|---------|-----------|
| `master` | Código original utilizado no artigo científico, preservado sem alterações. |
| Outras branches | Desenvolvimento de novas funcionalidades, experimentos, otimizações e adaptações do código. |
| `matlab` *(ou nome equivalente)* | Implementação em MATLAB baseada no modelo original em Python. |

Novas branches poderão ser adicionadas conforme o desenvolvimento do projeto.

# 🌍 Modelo Físico

A simulação baseia-se em uma solução analítica das equações que governam as **Ondas Acústico-Gravitacionais (AGWs)**.

O modelo considera:

- Propagação de ondas acústicas
- Propagação de ondas gravitacionais
- Perfil atmosférico obtido pelo modelo **NRLMSISE-00**
- Velocidade do som variável com a altitude
- Frequência de Brunt-Väisälä
- Dissipação por viscosidade
- Amplificação das ondas na termosfera

O objetivo é reproduzir a rápida propagação das AGWs e analisar a formação dos distúrbios termosféricos gerados por terremotos.

---

# 📈 Resultados

A simulação permite obter:

- Velocidade vertical das ondas
- Velocidade horizontal das ondas
- Perturbações de temperatura
- Perturbações de pressão
- Evolução espaço-temporal das AGWs
- Formação dos Distúrbios Termosféricos Co-Sísmicos (CSTDs)

Os resultados reproduzem os experimentos apresentados no artigo científico.

---

# 📚 Referência

Caso utilize este código em pesquisas ou trabalhos acadêmicos, cite:

```text
Sanchez, S. A.; Kherani, E. A.

A New Analytical Simulation Code of Acoustic-Gravity Waves of Seismic Origin
and Rapid Co-Seismic Thermospheric Disturbance Energetics.

Atmosphere, 15(5), 592, 2024.

https://doi.org/10.3390/atmos15050592
```

---

# 📄 Licença

Este repositório contém o código original associado ao artigo científico publicado na revista **Atmosphere**.

O artigo está licenciado sob a licença **Creative Commons Attribution (CC BY 4.0)**.

Caso utilize este material em pesquisas ou publicações, recomenda-se citar o artigo original.
