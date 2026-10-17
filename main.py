"""
Atividade Aula 6 - Algoritmo Genético para o problema do Caixeiro Viajante (Barbie / TSP com terrenos)
Este script resolve o problema de encontrar a menor rota visitando N pontos (amigos) e retornando ao ponto inicial (CASA).

O código está estruturado nas seguintes etapas principais:
  4.1 Geração da Instância
  4.2 Matriz de Custos (A*)
  4.3 Representação do Cromossomo
  4.4 Função de Fitness (Aptidão)
  4.5 População Inicial
  4.6 Seleção por Roleta
  4.7 Cruzamento Order Crossover (OX)
  4.8 Mutação Swap
  4.9 Elitismo
  4.10 Critério de Parada e Execução Principal
"""

import random
import time
import heapq
import itertools
import matplotlib.pyplot as plt

# Matrícula utilizada como semente aleatória para garantir a reprodutibilidade da instância
MATRICULA = 2024105231940030

# Custos de deslocamento de acordo com o tipo de terreno na matriz/mapa
# A: Asfalto (1), T: Terra (3), G: Grama (5), P: Pedregulho (10), E: Edificação/Obstáculo (Infinito)
COSTS = {'A': 1, 'T': 3, 'G': 5, 'P': 10, 'E': float('inf')}

# Representação textual do mapa 42x42 com os tipos de terreno
MAPA_TXT = [
    "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",
    "GGPPPPPPPPPPPPPPGGGGPPPPPAGGGGGGGGGGGGGGGG",
    "GGPEEEEEPEEEEEEPGGGGPEEEGAGGGPPPPPPPPPPPGG",
    "GGPEEEEEPEEEEEEPPPPPPEEAAAGGGPGGGGGGGGGPGG",
    "GGPEEPEEPEEEPEEPEEEEGEEAAAGGGPGGGTTTGGGPGG",
    "GGPPPPPPPPPPPPPPPEEEGEEEGAGGGPGGGTTTGGGPGG",
    "GGPTTTTTTTTTTTTPPEEEGGGGGAGGGPGGGTTTGGGPGG",
    "GGPTTTTTTTTTTTTPEEEEGEEEGAGGGPGGGTTTGGGPGG",
    "GGPTTTTGGGGTTTTPPPPPPEEEGAGGGPGGGGGGGGGPGG",
    "GGPTTTTGGGGTTTTPGGGGPEEAAAGGGPPPPPPPPPPPGG",
    "GGPTTTTGGGGTTTTPGGGGPEEEGAGGGGGGGGPGGGGGGG",
    "GGPTTTTTTTTTTTTPGGGGPEEEGAGGGGGGGGPGGGGGGG",
    "GGPTTTTTTTTTTTTPGGGGPGGGGAGGGGGGGGPGGGGGGG",
    "GGPPPPPPPPPPPPPPAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "GGAGGGGGGGGGGGGAGGGGGGGGGAGGGGGGGGGGGGGGGG",
    "GGAGEEEGGEEEEGGAGEEEEEEEGAGEEEEEEEGEEEEEGG",
    "GGAPPEEGGEEEAAAAGEEEEEEEGAGEEEEEEEGEEEEEGG",
    "GGAGEEEGGEEEEGGAGEAEEEAEGAGEEEEAEEGEEAEEGG",
    "GGAGGGGGGGGGGGGAGGAGGGAGGAGGGGGAGGGGGAGGGG",
    "GGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGG",
    "GGAPPPPPPAGGGGGGGGGGAGGGAGGGGGAGGAGPGGGAGG",
    "GGAPEEEEPAGEEEGGEEEGAGEEAEGEEGAGEAEPEEGAGG",
    "GGAPEEEEPAGEEEGGEEAAAGEEEEGEEGAGEEEPEEGAGG",
    "GGAPEEEPPAAAEEGGEEEGAGGGGGGEAAAGGGGPEAAAGG",
    "GGAPEEEEPAGEEEGGEEEGAGEEEEGEEGAGEEEPEEGAGG",
    "GGAPEEEEPAGEEEGGEEEGAGEEAEGEEGAGEAEPEEGAGG",
    "GGAPPPPPPAGGGGGGGGGGAGGGAGGGGGAGGAGPGGGAGG",
    "GGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGG",
    "GGAGGGGGGGGAGGGGPGAGPGGGPGAGGGAGGGGPGGGAGG",
    "GGAGEEEEEEGAGEEEPEAEPEEEPEAEGGAGEEEPEEGAGG",
    "GGAGEEAAEEGAGEAEPEEEPEAEPEEEGGAGEEEEEEGAGG",
    "GGAGGGAAGGGAGGAGPGGGPGAGPGGGGGAGGGGGGGGAGG",
    "GGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGG",
    "GGAGGGGGGGGGGGGGGGGGAGGGGGGPGGGGGGAPPPPAGG",
    "GGAGEEEEEEEEEEEEEEEGAGGGGGGPGGGGGGAPEEPAGG",
    "GGAGEAAAAAAAEEAAAEEGAGGGGGGPGGGGGGAPEEPAGG",
    "GGAGEAEEEEEAEEEEAEEGAPPPPPPPPPPPPPAPPEPAGG",
    "GGAGEAAAAEEAAAAAAEEGAGGGGGGPGGGGGGAPEEPAGG",
    "GGAGEEEEAEEEEEEEEEEGAGGGGGGPGGGGGGAPEEPAGG",
    "GGAGGGGGAGGGGGGGGGGGAGGGGGGPGGGGGGAPPPPAGG",
    "GGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGG",
    "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"
]

# Conversão da lista de strings em uma matriz de caracteres
grid = [list(l) for l in MAPA_TXT]

# Coordenada fixada do ponto inicial (CASA)
CASA = (23, 19)


# ==============================================================================
# 4.1 GERAÇÃO DA INSTÂNCIA
# ==============================================================================
def gerar_amigos(n, seed):
    """
    Sorteia N coordenadas válidas no mapa para representar as casas dos amigos.
    Uma coordenada é válida se não for um obstáculo ('E') nem a posição inicial (CASA).
    """
    rnd = random.Random(seed)
    validas = [(r, c) for r in range(len(grid)) for c in range(len(grid[0]))
               if grid[r][c] != 'E' and (r, c) != CASA]
    return rnd.sample(validas, n)


# ==============================================================================
# 4.2 MATRIZ DE CUSTOS (ALGORITMO A*)
# ==============================================================================
def heuristica(a, b):
    """Calcula a distância Manhattan entre dois pontos como heurística do A*."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_custo(inicio, destino):
    """
    Algoritmo A* para encontrar o custo do menor caminho entre dois pontos no mapa.
    Considera o custo de travessia do terreno acumulado a cada passo.
    """
    if inicio == destino:
        return 0
    g = {inicio: 0}
    heap = [(heuristica(inicio, destino), inicio)]
    while heap:
        f, atual = heapq.heappop(heap)
        if atual == destino:
            return g[atual]
        if f > g[atual] + heuristica(atual, destino):
            continue
        # Avalia os 4 vizinhos adjacentes (Cima, Baixo, Esquerda, Direita)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = atual[0] + dr, atual[1] + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                custo = COSTS[grid[nr][nc]]
                if custo == float('inf'):  # Ignora obstáculos
                    continue
                ng = g[atual] + custo
                viz = (nr, nc)
                if ng < g.get(viz, float('inf')):
                    g[viz] = ng
                    heapq.heappush(heap, (ng + heuristica(viz, destino), viz))
    return float('inf')

def matriz_custos(pontos):
    """
    Precalcula a matriz de distâncias/custos entre todos os pontos de interesse 
    (CASA e todos os amigos), otimizando a avaliação das rotas no Algoritmo Genético.
    O ponto 0 sempre representa a CASA, e os pontos de 1 a N representam os amigos.
    """
    n = len(pontos)
    M = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                M[i][j] = a_star_custo(pontos[i], pontos[j])
    return M


# ==============================================================================
# 4.3 E 4.4 REPRESETANÇÃO DO CROMOSSOMO E FUNÇÃO DE FITNESS
# ==============================================================================
def custo_rota(cromo, M):
    """
    Calcula a soma dos custos de uma rota completa:
    CASA (0) -> amigo_1 -> amigo_2 -> ... -> amigo_N -> CASA (0)
    """
    total = M[0][cromo[0]]
    for a, b in zip(cromo, cromo[1:]):
        total += M[a][b]
    total += M[cromo[-1]][0]
    return total

def fitness(cromo, M):
    """
    4.4 Função de Fitness (Aptidão)
    Como o problema busca a MINIMIZAÇÃO do custo da rota, invertemos o valor:
    fitness = 1 / (1 + custo).
    Quanto menor o custo da rota, maior a aptidão do cromossomo.
    """
    return 1.0 / (1.0 + custo_rota(cromo, M))


# ==============================================================================
# 4.5 POPULAÇÃO INICIAL
# ==============================================================================
def pop_inicial(P, n):
    """
    4.3 Cromossomo: Representado por uma permutação simples dos índices dos amigos (de 1 a n).
    4.5 População Inicial: Gera P indivíduos aleatórios contendo permutações sem repetição.
    """
    base = list(range(1, n + 1))
    return [random.sample(base, n) for _ in range(P)]


# ==============================================================================
# 4.6 SELEÇÃO POR ROLETA
# ==============================================================================
def roleta(pop, fits):
    """
    Seleciona indivíduos da população com probabilidade proporcional à sua aptidão (Fitness).
    Indivíduos com melhor fitness têm maior fatia da roleta.
    """
    soma = sum(fits)
    acumulada, acc = [], 0.0
    for f in fits:
        acc += f / soma
        acumulada.append(acc)
    nova = []
    for _ in range(len(pop)):
        r = random.random()
        for ind, lim in zip(pop, acumulada):
            if r <= lim:
                nova.append(ind[:])
                break
        else:
            nova.append(pop[-1][:])
    return nova


# ==============================================================================
# 4.7 CRUZAMENTO (ORDER CROSSOVER - OX)
# ==============================================================================
def order_crossover(p1, p2):
    """
    Operador de Cruzamento Específico para Permutações (OX):
    1. Mantém uma sub-sequência aleatória do Pai 1.
    2. Preenche o restante dos genes na ordem em que aparecem no Pai 2, 
       evitando duplicações de cidades.
    """
    n = len(p1)
    i, j = sorted(random.sample(range(n), 2))
    filho = [None] * n
    # Copia o trecho sorteado do Pai 1
    filho[i:j + 1] = p1[i:j + 1]
    segmento = set(filho[i:j + 1])
    # Pega os genes restantes do Pai 2 mantendo a ordem
    resto = [g for g in p2 if g not in segmento]
    it = iter(resto)
    # Preenche as posições vazias no filho
    for k in range(n):
        if filho[k] is None:
            filho[k] = next(it)
    return filho


# ==============================================================================
# 4.8 MUTAÇÃO SWAP
# ==============================================================================
def mutacao_swap(cromo, taxa):
    """
    Operador de Mutação Swap:
    Dado a taxa de mutação, seleciona duas posições aleatórias e troca seus valores de lugar,
    garantindo que o cromossomo continue sendo uma permutação válida.
    """
    if random.random() < taxa:
        i, j = random.sample(range(len(cromo)), 2)
        cromo[i], cromo[j] = cromo[j], cromo[i]


# ==============================================================================
# 4.9 E 4.10 ELITISMO E CRITÉRIO DE PARADA (ALGORITMO GENÉTICO)
# ==============================================================================
def algoritmo_genetico(M, n, P=100, geracoes=300, taxa_cruz=0.85,
                       taxa_mut=0.03, elite=2, paciencia=None):
    """
    Executa o ciclo evolutivo completo do Algoritmo Genético.
    - elite: Preserva os 'elite' melhores indivíduos para a próxima geração sem alterações.
    - paciencia: Critério de parada antecipada caso não haja melhora por 'paciencia' gerações.
    """
    pop = pop_inicial(P, n)
    melhor, melhor_custo = None, float('inf')
    historico = []
    ger_melhor = 0

    for g in range(geracoes):
        # Avaliação da população
        fits = [fitness(c, M) for c in pop]

        # Atualiza o melhor indivíduo global encontrado
        idx = max(range(P), key=lambda k: fits[k])
        custo_g = custo_rota(pop[idx], M)
        if custo_g < melhor_custo:
            melhor_custo, melhor, ger_melhor = custo_g, pop[idx][:], g
        historico.append(melhor_custo)

        # 4.10 Critério de parada por convergência (paciência)
        if paciencia and g - ger_melhor >= paciencia:
            break

        # 4.9 Elitismo: Separa os melhores para passagem direta
        ordem = sorted(range(P), key=lambda k: fits[k], reverse=True)
        elites = [pop[k][:] for k in ordem[:elite]]

        # 4.6 Seleção por Roleta
        selecionados = roleta(pop, fits)
        random.shuffle(selecionados)

        # 4.7 Cruzamento
        nova = []
        for k in range(0, P - 1, 2):
            p1, p2 = selecionados[k], selecionados[k + 1]
            if random.random() < taxa_cruz:
                f1, f2 = order_crossover(p1, p2), order_crossover(p2, p1)
            else:
                f1, f2 = p1[:], p2[:]
            nova += [f1, f2]
        if len(nova) < P:
            nova.append(selecionados[-1][:])

        # 4.8 Mutação
        for c in nova:
            mutacao_swap(c, taxa_mut)

        # Reinsere a elite mantendo os melhores indivíduos
        nova[:elite] = elites
        pop = nova

    return melhor, melhor_custo, historico, ger_melhor


# ==============================================================================
# ALGORITMO DE FORÇA BRUTA (PARA COMPARAÇÃO)
# ==============================================================================
def forca_bruta(M, n):
    """Calcula todas as permutações possíveis para obter o resultado ótimo perfeito em N pequeno."""
    melhor, melhor_custo = None, float('inf')
    for perm in itertools.permutations(range(1, n + 1)):
        c = custo_rota(perm, M)
        if c < melhor_custo:
            melhor, melhor_custo = list(perm), c
    return melhor, melhor_custo


# ==============================================================================
# EXPERIMENTOS E PLOTAGEM
# ==============================================================================
def experimento_A(n=8):
    """Executa o Experimento A (N=8) comparando o Algoritmo Genético com a Força Bruta."""
    print(f"===== EXPERIMENTO A (N={n}) =====")
    amigos = gerar_amigos(n, MATRICULA)
    M = matriz_custos([CASA] + amigos)
    random.seed(MATRICULA)

    t = time.time()
    rota_bf, custo_bf = forca_bruta(M, n)
    t_bf = time.time() - t

    t = time.time()
    rota_ag, custo_ag, hist, ger = algoritmo_genetico(M, n, P=200, geracoes=500, taxa_mut=0.05)
    t_ag = time.time() - t

    print(f"Força bruta : custo={custo_bf}  rota={rota_bf}  ({t_bf:.2f}s)")
    print(f"AG          : custo={custo_ag}  rota={rota_ag}  ({t_ag:.2f}s)")
    print(f"Geração em que o AG achou o melhor: {ger}")
    print(f"Diferença para o ótimo: {100*(custo_ag-custo_bf)/custo_bf:.2f}%")
    return hist, custo_bf

def experimento_B(n=15):
    """Executa o Experimento B (N=15) demonstrando o desempenho do AG em uma busca maior."""
    print(f"\n===== EXPERIMENTO B (N={n}) =====")
    amigos = gerar_amigos(n, MATRICULA)
    M = matriz_custos([CASA] + amigos)
    random.seed(MATRICULA)

    t = time.time()
    rota, custo, hist, ger = algoritmo_genetico(M, n, P=150, geracoes=500)
    t_ag = time.time() - t
    print(f"AG: melhor custo={custo}  rota={rota}")
    print(f"Estabilizou na geração {ger} | tempo={t_ag:.2f}s")
    return hist

def plotar(histA, otimo, histB):
    """Gera e salva o gráfico com as curvas de convergência do Algoritmo Genético."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(histA, label="AG")
    ax[0].axhline(otimo, color="r", ls="--", label="Ótimo (força bruta)")
    ax[0].set(title="Experimento A (N=8)", xlabel="Geração", ylabel="Melhor custo")
    ax[0].legend()
    ax[1].plot(histB)
    ax[1].set(title="Experimento B (N=15)", xlabel="Geração", ylabel="Melhor custo")
    plt.tight_layout()
    plt.savefig("convergencia.png", dpi=120)

if __name__ == "__main__":
    hA, otimo = experimento_A()
    hB = experimento_B()
    plotar(hA, otimo, hB)