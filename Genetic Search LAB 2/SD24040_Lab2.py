import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ---- Fixed Lab 2 Problem Parameters ----
POP_SIZE = 300          # 1) Population setup is 300
CHROM_LEN = 80          # 3) Length of all individuals is 80
TARGET_ONES = 50        # 2) Maximum value when number of ones equals 50
MAX_FITNESS = 80        # 4) Return value is 80 when ones == 50
N_GENERATIONS = 50      # 5) Number of generations is 50

# ---- Fitness Function ----
def fitness(individual: np.ndarray) -> float:
    """
    Fitness peaks when the number of 1s equals TARGET_ONES (50).
    Max fitness is MAX_FITNESS (=80) at ones == 50.
    """
    ones = int(individual.sum())
    return float(MAX_FITNESS - abs(ones - TARGET_ONES))

# ---- GA Operators ----
def init_population(pop_size: int, chrom_len: int, start_biased: bool = True) -> np.ndarray:
    """
    Initializes population. 
    If start_biased=True, initializes bits with a 90% chance of 0 and 10% chance of 1.
    This prevents initial random 50/50 generation from hitting max fitness on Gen 1.
    """
    if start_biased:
        return np.random.choice([0, 1], size=(pop_size, chrom_len), p=[0.9, 0.1]).astype(np.int8)
    else:
        return np.random.randint(0, 2, size=(pop_size, chrom_len), dtype=np.int8)

# Selection Operators
def tournament_selection(pop: np.ndarray, fits: np.ndarray, k: int) -> np.ndarray:
    idxs = np.random.randint(0, len(pop), size=k)
    best_idx = idxs[np.argmax(fits[idxs])]
    return pop[best_idx].copy()

def roulette_wheel_selection(pop: np.ndarray, fits: np.ndarray) -> np.ndarray:
    total_fit = fits.sum()
    if total_fit > 0:
        probs = fits / total_fit
    else:
        probs = np.ones_like(fits) / len(fits)
    selected_idx = np.random.choice(len(pop), p=probs)
    return pop[selected_idx].copy()

# Crossover Operators
def single_point_crossover(p1: np.ndarray, p2: np.ndarray, crossover_rate: float):
    if np.random.rand() > crossover_rate:
        return p1.copy(), p2.copy()
    point = np.random.randint(1, CHROM_LEN)
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2

def two_point_crossover(p1: np.ndarray, p2: np.ndarray, crossover_rate: float):
    if np.random.rand() > crossover_rate:
        return p1.copy(), p2.copy()
    pts = sorted(np.random.choice(range(1, CHROM_LEN), size=2, replace=False))
    c1 = np.concatenate([p1[:pts[0]], p2[pts[0]:pts[1]], p1[pts[1]:]])
    c2 = np.concatenate([p2[:pts[0]], p1[pts[0]:pts[1]], p2[pts[1]:]])
    return c1, c2

def uniform_crossover(p1: np.ndarray, p2: np.ndarray, crossover_rate: float, p_gene: float = 0.5):
    if np.random.rand() > crossover_rate:
        return p1.copy(), p2.copy()
    mask = np.random.rand(CHROM_LEN) < p_gene
    c1 = np.where(mask, p1, p2)
    c2 = np.where(mask, p2, p1)
    return c1.copy(), c2.copy()

# Mutation Operator
def mutate(individual: np.ndarray, mutation_rate: float) -> np.ndarray:
    mask = np.random.rand(CHROM_LEN) < mutation_rate
    individual[mask] = 1 - individual[mask]
    return individual

# ---- Main Evolution Loop ----
def evolve(
    pop: np.ndarray, 
    generations: int, 
    selection_type: str, 
    crossover_type: str, 
    crossover_rate: float, 
    mutation_rate: float, 
    tournament_k: int, 
    early_stop: bool
):
    best_fitness_per_gen = []
    avg_fitness_per_gen = []
    best_individual = None
    best_f = -np.inf

    for gen in range(generations):
        fits = np.array([fitness(ind) for ind in pop])

        # Track best and average fitness per generation
        gen_best_idx = np.argmax(fits)
        gen_best = pop[gen_best_idx]
        gen_best_f = fits[gen_best_idx]
        gen_avg_f = fits.mean()

        best_fitness_per_gen.append(float(gen_best_f))
        avg_fitness_per_gen.append(float(gen_avg_f))

        if gen_best_f > best_f:
            best_f = float(gen_best_f)
            best_individual = gen_best.copy()

        # Early stopping condition
        if early_stop and best_f >= MAX_FITNESS:
            break

        # Selection & Reproduction
        new_pop = []
        while len(new_pop) < len(pop):
            # 1. Selection
            if selection_type == "Tournament":
                p1 = tournament_selection(pop, fits, tournament_k)
                p2 = tournament_selection(pop, fits, tournament_k)
            else:  # Roulette Wheel
                p1 = roulette_wheel_selection(pop, fits)
                p2 = roulette_wheel_selection(pop, fits)

            # 2. Crossover
            if crossover_type == "Single-Point":
                c1, c2 = single_point_crossover(p1, p2, crossover_rate)
            elif crossover_type == "Two-Point":
                c1, c2 = two_point_crossover(p1, p2, crossover_rate)
            else:  # Uniform
                c1, c2 = uniform_crossover(p1, p2, crossover_rate)

            # 3. Mutation
            c1 = mutate(c1, mutation_rate)
            c2 = mutate(c2, mutation_rate)

            new_pop.extend([c1, c2])

        pop = np.array(new_pop[:len(pop)], dtype=np.int8)

    return best_individual, best_f, best_fitness_per_gen, avg_fitness_per_gen

# ---- Streamlit UI ----
st.set_page_config(page_title="Bit Pattern GA (80 bits, target ones = 50)", page_icon="🧬", layout="wide")

st.title("🧬 Genetic Algorithm: Evolve an 80-bit Pattern")
st.caption(
    "Fixed requirements: Population = 300, Chromosome Length = 80, Generations = 50.\n"
    "Fitness peaks at ones = 50, with max fitness = 80."
)

# Sidebar - GA Control Options
with st.sidebar:
    st.header("⚙️ GA Control Options")
    
    st.subheader("Operator Selection")
    selection_type = st.selectbox("Selection Method", ["Tournament", "Roulette Wheel"])
    if selection_type == "Tournament":
        tournament_k = st.slider("Tournament Size (k)", min_value=2, max_value=10, value=3)
    else:
        tournament_k = 3

    crossover_type = st.selectbox("Crossover Method", ["Single-Point", "Two-Point", "Uniform"])

    st.subheader("Hyperparameters")
    crossover_rate = st.slider("Crossover Rate ($P_c$)", 0.0, 1.0, 0.9, 0.05)
    mutation_rate = st.slider("Mutation Rate ($P_m$ per bit)", 0.0, 0.1, 1.0 / CHROM_LEN, 0.005, format="%.4f")

    st.subheader("Execution Settings")
    bias_init = st.checkbox("Bias Initial Population (90% Zeros to show curve)", value=True)
    early_stop = st.checkbox("Enable Early Stopping (Stop at max fitness 80)", value=False)
    seed = st.number_input("Random seed (reproducibility)", min_value=0, value=42, step=1)

with st.expander("ℹ️ Problem setup (fixed by requirement)", expanded=True):
    st.write(
        f"""
- **Population size**: `{POP_SIZE}`
- **Chromosome length**: `{CHROM_LEN}`
- **Target number of ones**: `{TARGET_ONES}`
- **Max fitness at optimum**: `{MAX_FITNESS}` (when ones = {TARGET_ONES})
- **Generations**: `{N_GENERATIONS}`
- **Active Operators**: Selection = `{selection_type}`, Crossover = `{crossover_type}`
"""
    )

run_btn = st.button("Run Genetic Algorithm", type="primary")

if run_btn:
    random.seed(seed)
    np.random.seed(seed)

    population = init_population(POP_SIZE, CHROM_LEN, start_biased=bias_init)
    best_ind, best_fit, best_curve, avg_curve = evolve(
        pop=population,
        generations=N_GENERATIONS,
        selection_type=selection_type,
        crossover_type=crossover_type,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        tournament_k=tournament_k,
        early_stop=early_stop
    )

    ones_count = int(best_ind.sum())
    zeros_count = CHROM_LEN - ones_count
    bitstring = "".join(map(str, best_ind.tolist()))

    st.subheader("🏁 Best Individual Found")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(label="Best Fitness Score", value=f"{best_fit:.0f} / {MAX_FITNESS}")
    with col_b:
        st.metric(label="Generations Run", value=f"{len(best_curve)} / {N_GENERATIONS}")

    st.write(f"**Ones**: {ones_count} | **Zeros**: {zeros_count} | **Length**: {CHROM_LEN}")
    st.code(bitstring, language="text")

    # ---- Enhanced Convergence Plot ----
    st.subheader("📉 Convergence (Best vs. Average Fitness)")
    fig, ax = plt.subplots(figsize=(9, 4.8))

    gens = range(1, len(best_curve) + 1)

    # 1. Plot Best and Average Population Fitness Curves
    ax.plot(gens, best_curve, label="Best Fitness", linewidth=2.5, color="#1f77b4")
    ax.plot(gens, avg_curve, label="Average Fitness", linewidth=2.0, linestyle="--", color="#ff7f0e")

    # 2. Horizontal Target Reference Line (y = 80)
    ax.axhline(y=MAX_FITNESS, color="red", linestyle=":", linewidth=1.5, label=f"Target Max Fitness ({MAX_FITNESS})")

    # 3. First-Convergence Callout Annotation
    if MAX_FITNESS in best_curve:
        conv_gen = best_curve.index(MAX_FITNESS) + 1
        ax.plot(conv_gen, MAX_FITNESS, marker="o", markersize=8, color="red")
        
        # Adjust annotation offset dynamically based on generation
        x_text_offset = conv_gen - 12 if conv_gen > 20 else conv_gen + 2
        ax.annotate(
            f"Reached Target (Gen {conv_gen})",
            xy=(conv_gen, MAX_FITNESS),
            xytext=(x_text_offset, MAX_FITNESS - 7),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            fontweight="bold",
            color="red",
            fontsize=10
        )

    # 4. Plot Legend & Dynamic Title
    ax.set_title(
        f"GA Evolutionary Performance ({selection_type} + {crossover_type})\n"
        f"Configuration: $P_{{pop}}={POP_SIZE}, L={CHROM_LEN}, P_c={crossover_rate}, P_m={mutation_rate:.4f}$",
        fontsize=11,
        pad=10
    )
    ax.set_xlabel("Generation", fontsize=10)
    ax.set_ylabel("Fitness Score", fontsize=10)
    
    # Set y limits nicely
    min_y = min(min(avg_curve), min(best_curve)) - 5
    ax.set_ylim(min_y, MAX_FITNESS + 8)
    
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)

    st.pyplot(fig)

    if best_fit == MAX_FITNESS and ones_count == TARGET_ONES:
        st.success("Perfect match achieved: ones = 50 and fitness = 80 ✅")
    else:
        st.info("GA reached a near-optimal solution; adjust mutation rate or try another seed.")

st.caption("© 2026 Interactive GA demo (80 bits, optimum at 50 ones).")
