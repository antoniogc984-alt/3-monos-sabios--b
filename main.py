from typing import List, Optional
import math
import random

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, confloat
from fastapi.middleware.cors import CORSMiddleware


# =========================
#  MODELOS Pydantic
# =========================

# ---- N-Queens ----
class SolveRequest(BaseModel):
    n: conint(ge=4, le=12)
    mode: str = "visual"          # "visual" | "final"
    max_steps: conint(ge=100, le=50000) = 5000


class Snapshot(BaseModel):
    row: int
    col: int
    action: str  # "place" | "remove"


class SolveResponse(BaseModel):
    n: int
    solution: List[int]
    snapshots: List[Snapshot]
    truncated: bool


# ---- Montecarlo ----
class SimulateRequest(BaseModel):
    initial_price: confloat(gt=0)
    volatility: confloat(ge=0)        # volatilidad anual
    days: conint(ge=1, le=365)
    simulations: conint(ge=10, le=5000)


class SimulateResponse(BaseModel):
    paths: List[List[float]]
    mean_path: List[float]
    p5_path: List[float]
    p95_path: List[float]
    var_95: float
    best_final: float
    worst_final: float
    mean_final: float


# ---- TSP Genético ----
class City(BaseModel):
    x: float
    y: float


class EvolveRequest(BaseModel):
    cities: List[City]
    population_size: conint(ge=10, le=1000) = 80
    mutation_rate: confloat(ge=0, le=1) = 0.1
    generations: conint(ge=1, le=500) = 80
    best_route_indices: Optional[List[int]] = None  # opcional, para continuar


class EvolveResponse(BaseModel):
    best_route_indices: List[int]
    best_distance: float
    generation: int


# =========================
#  FASTAPI APP
# =========================

app = FastAPI(title="Los 3 Monos Sabios API")

# CORS: para desarrollo local (puedes ajustar origenes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en prod pon tu dominio específico
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


# =========================
#  N-QUEENS /api/solve
# =========================

def solve_n_queens(n: int, max_steps: int, visual: bool):
    cols = [-1] * n
    snapshots = []
    step_count = 0
    truncated = False

    def is_safe(row: int, col: int) -> bool:
        for r in range(row):
            c = cols[r]
            if c == col or abs(c - col) == abs(r - row):
                return False
        return True

    def add_snapshot(row: int, col: int, action: str):
        nonlocal step_count, truncated
        if not visual or truncated:
            return
        if step_count >= max_steps:
            truncated = True
            return
        snapshots.append({"row": row, "col": col, "action": action})
        step_count += 1

    def backtrack(row: int) -> bool:
        if row == n:
            return True
        for col in range(n):
            if is_safe(row, col):
                cols[row] = col
                add_snapshot(row, col, "place")
                if backtrack(row + 1):
                    return True
                # retroceso
                add_snapshot(row, col, "remove")
                cols[row] = -1
        return False

    has_solution = backtrack(0)
    if not has_solution:
        return [], snapshots, truncated

    return cols, snapshots, truncated


@app.post("/api/solve", response_model=SolveResponse)
def solve_endpoint(req: SolveRequest):
    visual = req.mode == "visual"
    solution, snapshots, truncated = solve_n_queens(
        req.n, req.max_steps, visual
    )
    if not solution:
        raise HTTPException(status_code=400, detail="No se encontró solución para N dado.")
    return SolveResponse(
        n=req.n,
        solution=solution,
        snapshots=[Snapshot(**s) for s in snapshots],
        truncated=truncated,
    )


# =========================
#  MONTECARLO /api/simulate
# =========================

@app.post("/api/simulate", response_model=SimulateResponse)
def simulate_endpoint(req: SimulateRequest):
    # Limitamos simulaciones para no matar el tiempo de Vercel
    sims = min(req.simulations, 2000)
    days = req.days
    s0 = req.initial_price
    vol_annual = req.volatility

    # Convertimos a volatilidad diaria (modelo muy simplificado)
    trading_days = 252
    vol_daily = vol_annual / math.sqrt(trading_days)

    # paths: (sims, days+1)
    rng = np.random.default_rng()
    # Rendimientos diarios ~ N(0, vol_daily)
    daily_returns = 1 + vol_daily * rng.normal(size=(sims, days))
    paths = np.empty((sims, days + 1), dtype=float)
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.cumprod(daily_returns, axis=1)

    # Estadísticas
    mean_path = paths.mean(axis=0)
    p5_path = np.percentile(paths, 5, axis=0)
    p95_path = np.percentile(paths, 95, axis=0)

    final_prices = paths[:, -1]
    best_final = float(final_prices.max())
    worst_final = float(final_prices.min())
    mean_final = float(final_prices.mean())

    # VaR al 95%: pérdida máxima esperada con 95% de confianza
    p5_price = float(np.percentile(final_prices, 5))
    var_95 = max(0.0, s0 - p5_price)

    return SimulateResponse(
        paths=paths.tolist(),
        mean_path=mean_path.tolist(),
        p5_path=p5_path.tolist(),
        p95_path=p95_path.tolist(),
        var_95=var_95,
        best_final=best_final,
        worst_final=worst_final,
        mean_final=mean_final,
    )


# =========================
#  TSP GENETICO /api/evolve
# =========================

def euclidean_distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def route_distance(route: List[int], cities: List[City]) -> float:
    dist = 0.0
    n = len(route)
    for i in range(n):
        c1 = cities[route[i]]
        c2 = cities[route[(i + 1) % n]]
        dist += euclidean_distance((c1.x, c1.y), (c2.x, c2.y))
    return dist


def tournament_selection(population: List[List[int]], cities: List[City], k: int = 3) -> List[int]:
    selected = random.sample(population, k)
    best = min(selected, key=lambda r: route_distance(r, cities))
    return best.copy()


def ordered_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b + 1] = parent1[a:b + 1]

    pos = (b + 1) % size
    for gene in parent2:
        if gene not in child:
            child[pos] = gene
            pos = (pos + 1) % size
    return child


def evolve_tsp(
    cities: List[City],
    population_size: int,
    mutation_rate: float,
    generations: int,
    seed_route: Optional[List[int]] = None,
):
    n = len(cities)
    if n < 3:
        raise ValueError("Se requieren al menos 3 ciudades.")

    base = list(range(n))

    def random_route():
        r = base[:]
        random.shuffle(r)
        return r

    # Población inicial
    population: List[List[int]] = []
    if seed_route and len(seed_route) == n:
        population.append(seed_route[:])

    while len(population) < population_size:
        population.append(random_route())

    def fitness(route: List[int]) -> float:
        return route_distance(route, cities)

    for _ in range(generations):
        population.sort(key=fitness)
        new_pop: List[List[int]] = population[:2]  # elitismo: mejores 2

        while len(new_pop) < population_size:
            p1 = tournament_selection(population, cities)
            p2 = tournament_selection(population, cities)
            child = ordered_crossover(p1, p2)

            if random.random() < mutation_rate:
                i, j = random.sample(range(n), 2)
                child[i], child[j] = child[j], child[i]

            new_pop.append(child)

        population = new_pop

    population.sort(key=fitness)
    best_route = population[0]
    best_distance = fitness(best_route)
    return best_route, best_distance


@app.post("/api/evolve", response_model=EvolveResponse)
def evolve_endpoint(req: EvolveRequest):
    if len(req.cities) < 3:
        raise HTTPException(status_code=400, detail="Se requieren al menos 3 ciudades.")

    best_route, best_distance = evolve_tsp(
        cities=req.cities,
        population_size=req.population_size,
        mutation_rate=req.mutation_rate,
        generations=req.generations,
        seed_route=req.best_route_indices,
    )

    # Como manejamos generaciones “por llamada”, devolvemos solo el número recibido.
    return EvolveResponse(
        best_route_indices=best_route,
        best_distance=best_distance,
        generation=req.generations,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
