# D-LAA enhanced observer pipeline demo on an enriched U.S. power grid topology
# Author: ChatGPT (code shared per user's request)
# -----------------------------------------------------------------------------
# What this script does:
# 1) Loads the uploaded Opsahl US power grid topology (Western US HV grid) and builds an enriched dynamic model.
# 2) Selects a manageable connected subgraph, assigns synthetic generator/load roles and dynamic parameters.
# 3) Forms the DC linearized swing model matrices (A, B, E) consistent with the paper's formulation.
# 4) Instantiates three attack scenarios:
#      (i) D-LAA: load injection proportional to system frequency (feedback attack)
#     (ii) Conventional LAA: exogenous oscillatory/step load injection
#    (iii) AI data center event: synchronized large load drop and reconnection at multiple buses
# 5) Runs time-domain simulation with RK4 integration.
# 6) Implements a practical area-based residual detector (energy of local frequency errors) for localization.
# 7) Performs a simple attack identification (least-squares inversion on ydot) as a stand-in for the MIMO equivalent-control readout.
# 8) Produces plots and saves artifacts.
#
# Requirements:
#   python >= 3.9, numpy, pandas, networkx, matplotlib
#   pip install numpy pandas networkx matplotlib
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------- Config ----------------------------
BASE = Path("/mnt/data") if Path("/mnt/data").exists() else Path(".")
EDGES_CSV = BASE / "us_powergrid_edges.csv"               # created earlier or auto-generated below
RAW_EDGES  = BASE / "out.opsahl-powergrid"                # original Opsahl edge list
OUTDIR = BASE / "dlaa_demo_artifacts"
OUTDIR.mkdir(exist_ok=True, parents=True)

# Target backbone size (the script will pick a largest connected subgraph close to this; it may be smaller after filtering)
N_TARGET = 250

# Random seeds for reproducibility
SEED_CORE = 42
SEED_SYN  = 123

# ---------------------------- Load edge list ----------------------------
def load_edges():
    if EDGES_CSV.exists():
        edge_df = pd.read_csv(EDGES_CSV)
    elif RAW_EDGES.exists():
        edges = []
        with open(RAW_EDGES, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("%") or not line.strip():
                    continue
                a = line.strip().split()
                if len(a) >= 2 and all(tok.isdigit() for tok in a[:2]):
                    u, v = map(int, a[:2])
                    if u != v:
                        edges.append((u, v))
        edge_df = pd.DataFrame(edges, columns=["u","v"])
        edge_df.to_csv(EDGES_CSV, index=False)
        print(f"[info] Created {EDGES_CSV.name} from raw file.")
    else:
        raise FileNotFoundError("Neither us_powergrid_edges.csv nor out.opsahl-powergrid was found.")
    return edge_df

edge_df = load_edges()
Gfull = nx.from_pandas_edgelist(edge_df, "u", "v")

# Largest connected component
largest_cc = max(nx.connected_components(Gfull), key=len)
Gcc = Gfull.subgraph(largest_cc).copy()

# ---------------------------- Select a core backbone ----------------------------
N_target = N_TARGET
deg_sorted = sorted(Gcc.degree, key=lambda x: x[1], reverse=True)
core_nodes = [n for n, d in deg_sorted[:N_target]]
G = Gcc.subgraph(core_nodes).copy()
largest_cc2 = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc2).copy()

# Reindex nodes to 0..N-1
node_index = {n:i for i,n in enumerate(G.nodes())}
G = nx.relabel_nodes(G, node_index, copy=True)
nodes = list(G.nodes())
N = len(nodes)
print(f"[info] Selected backbone size: N={N}, E={G.number_of_edges()}")

# ---------------------------- Assign generators and loads ----------------------------
rng = np.random.default_rng(SEED_CORE)
deg = np.array([G.degree(n) for n in nodes])
ng = max(10, int(0.15 * N))  # ~15% generators
gen_candidates = np.argsort(-deg)[:ng]
gen_set = set(gen_candidates.tolist())
load_set = set(nodes) - gen_set
nl = len(load_set)

gen_idx = np.array(sorted(list(gen_set)))
load_idx = np.array(sorted(list(load_set)))

# ---------------------------- Build DC matrices (B', partitions, A,B,E) ----------------------------
edge_list = list(G.edges())
m = len(edge_list)
A_inc = np.zeros((m, N))
for ell, (i, j) in enumerate(edge_list):
    A_inc[ell, i] =  1.0
    A_inc[ell, j] = -1.0

b_line = rng.uniform(0.5, 2.0, size=m)  # synthetic susceptances
Bprime = A_inc.T @ (b_line[:, None] * A_inc)

Bgg = Bprime[np.ix_(gen_idx, gen_idx)]
Bgl = Bprime[np.ix_(gen_idx, load_idx)]
Blg = Bprime[np.ix_(load_idx, gen_idx)]
Bll = Bprime[np.ix_(load_idx, load_idx)]

eps = 1e-6
Bll_inv = np.linalg.inv(Bll + eps*np.eye(Bll.shape[0]))

Btilde = (Bgg - Bgl @ Bll_inv @ Blg)
Etilde = (Bgl @ Bll_inv)

# Inertia and damping (synthetic)
M = np.diag(rng.uniform(3.0, 7.0, size=ng))   # seconds
D = np.diag(rng.uniform(0.5, 1.5, size=ng))   # pu damping

Z = np.zeros_like(Btilde)
Ing = np.eye(ng)
A_top = np.hstack([Z, Ing])
A_bot = np.hstack([-np.linalg.inv(M) @ Btilde, -np.linalg.inv(M) @ D])
A = np.vstack([A_top, A_bot])

Bmat = np.vstack([np.zeros((ng, ng)), np.linalg.inv(M)])
Emat = np.vstack([np.zeros((ng, nl)),  np.linalg.inv(M) @ Etilde])

# Measurements: all generator frequencies
C = np.hstack([np.zeros((ng, ng)), np.eye(ng)])

# ---------------------------- Choose data center loads ----------------------------
load_degs = [(n, G.degree(n)) for n in load_idx]
dc_nodes_load_index = [n for n, d in sorted(load_degs, key=lambda x: x[1], reverse=True)[:3]]
dc_pos = [np.where(load_idx == n)[0][0] for n in dc_nodes_load_index]
attack_bus_pos = dc_pos[0]

# ---------------------------- Time grid ----------------------------
T = 40.0
dt = 0.02
t = np.arange(0.0, T+dt, dt)
nt = len(t)

Mdiag = np.diag(M)
Msum = np.sum(Mdiag)
def omega_coi(omega_vec):
    return float(np.dot(Mdiag, omega_vec)/Msum)

# ---------------------------- Attack scenarios ----------------------------
current_time = 0.0  # used by scenario_alpha()

def scenario_alpha(kind, y_hist, k_feedback=200.0, A_step=150.0, f_osc=0.25, t0=5.0, t1=15.0):
    global current_time
    alpha = np.zeros(nl)
    if kind == "dlaa":
        if current_time >= t0:
            w_coi = omega_coi(y_hist)
            alpha[attack_bus_pos] = -k_feedback * w_coi
    elif kind == "conventional":
        if t0 <= current_time <= t1:
            if f_osc > 0.0:
                alpha[attack_bus_pos] = A_step * np.sin(2*np.pi*f_osc*(current_time - t0))
            else:
                alpha[attack_bus_pos] = A_step
    elif kind == "datacenter":
        if t0 <= current_time <= t1:
            for p in dc_pos:
                alpha[p] = -A_step
    return alpha

# ---------------------------- Simulation (RK4) ----------------------------
def simulate(kind, noise_std=0.0):
    x = np.zeros(2*ng)
    Y = np.zeros((nt, ng))
    R_area = np.zeros((nt, 4))
    Alpha_true = np.zeros((nt, nl))

    # areas by modularity communities (up to 4)
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G))
    except Exception:
        comms = [set(range(N))]
    area_id = np.zeros(N, dtype=int)
    for k, cset in enumerate(comms[:4]):
        for n in cset:
            area_id[n] = k
    area_gen_lists = []
    for k in range(4):
        g_in_area = [i for i, n in enumerate(gen_idx) if area_id[n] == k]
        if len(g_in_area) == 0 and k>0:
            g_in_area = area_gen_lists[-1] if len(area_gen_lists)>0 else []
        area_gen_lists.append(g_in_area)

    for k in range(nt):
        global current_time
        current_time = t[k]
        omega = x[ng:]
        alpha = scenario_alpha(kind, omega)
        Alpha_true[k,:] = alpha

        Pm = np.zeros(ng)

        def f_state(xloc, alpha_loc):
            return A @ xloc + Bmat @ Pm + Emat @ alpha_loc

        k1 = f_state(x, alpha)
        xh = x + 0.5*dt*k1
        k2 = f_state(xh, alpha)
        xh2 = x + 0.5*dt*k2
        k3 = f_state(xh2, alpha)
        xh3 = x + dt*k3
        k4 = f_state(xh3, alpha)

        x = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        y = (C @ x) + (noise_std * np.random.standard_normal(ng))
        Y[k,:] = y

        for a in range(4):
            g_idx_list = area_gen_lists[a]
            if len(g_idx_list) == 0:
                R_area[k,a] = 0.0
            else:
                block = y[g_idx_list]
                R_area[k,a] = float(np.mean(block*block))

    return t, Y, R_area, Alpha_true, area_gen_lists

# Run scenarios
t, Y_dlaa, R_dlaa, Atrue_dlaa, area_gen_lists = simulate("dlaa", noise_std=0.0005)
t, Y_conv, R_conv, Atrue_conv, _ = simulate("conventional", noise_std=0.0005)
t, Y_dc, R_dc, Atrue_dc, _ = simulate("datacenter", noise_std=0.0005)

# ---------------------------- Identification via LS on ydot ----------------------------
def estimate_alpha_from_data(Y, Atrue):
    theta_hat = np.zeros((nt, ng))
    for k in range(1, nt):
        theta_hat[k,:] = theta_hat[k-1,:] + dt*Y[k-1,:]

    Minv = np.linalg.inv(M)
    Hy = Minv @ Etilde

    Alpha_est = np.zeros_like(Atrue)
    reg = 1e-6
    HyT = Hy.T
    Hreg_inv = np.linalg.inv(HyT @ Hy + reg*np.eye(Hy.shape[1])) @ HyT

    ydot = np.zeros_like(Y)
    ydot[1:,:] = (Y[1:,:] - Y[:-1,:]) / dt
    ydot[0,:] = ydot[1,:]

    for k in range(nt):
        rhs = -( ydot[k,:] + (Minv @ D) @ Y[k,:] + (Minv @ Btilde) @ theta_hat[k,:] )
        x_ls = Hreg_inv @ rhs
        Alpha_est[k,:] = x_ls
    return Alpha_est

Alpha_est_dlaa = estimate_alpha_from_data(Y_dlaa, Atrue_dlaa)
Alpha_est_conv = estimate_alpha_from_data(Y_conv, Atrue_conv)
Alpha_est_dc   = estimate_alpha_from_data(Y_dc,   Atrue_dc)

# ---------------------------- Visualization ----------------------------
def plot_detection_and_identification(Y, R_area, Atrue, Aest, title_suffix):
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(t, Y[:,0])
    axs[0].set_title(f"Generator frequency (bus 0) — {title_suffix}")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("ω [pu]")

    axs[1].plot(t, R_area[:,0], label="Area 1")
    axs[1].plot(t, R_area[:,1], label="Area 2")
    axs[1].plot(t, R_area[:,2], label="Area 3")
    axs[1].plot(t, R_area[:,3], label="Area 4")
    axs[1].set_title(f"Area residual energy (localization) — {title_suffix}")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Residual energy")
    axs[1].legend()

    axs[2].plot(t, Atrue[:, 0 if Atrue.shape[1]==1 else attack_bus_pos], label="True α (attacked bus)")
    axs[2].plot(t, Aest[:,  0 if Aest.shape[1]==1  else attack_bus_pos], label="Estimated α", linestyle="--")
    axs[2].set_title(f"Attack identification at attacked bus — {title_suffix}")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("α [pu]")
    axs[2].legend()

    plt.tight_layout()
    plt.show()

plot_detection_and_identification(Y_dlaa, R_dlaa, Atrue_dlaa, Alpha_est_dlaa, "D-LAA (feedback attack)")
plot_detection_and_identification(Y_conv, R_conv, Atrue_conv, Alpha_est_conv, "Conventional LAA (oscillatory)")
plot_detection_and_identification(Y_dc,   R_dc,   Atrue_dc,   Alpha_est_dc,   "AI Data Center event (drop & reconnect)")

# ---------------------------- Save artifacts ----------------------------
pd.DataFrame(Y_dlaa).to_csv(OUTDIR/"omega_dlaa.csv", index=False)
pd.DataFrame(Y_conv).to_csv(OUTDIR/"omega_conventional.csv", index=False)
pd.DataFrame(Y_dc).to_csv(OUTDIR/"omega_datacenter.csv", index=False)
pd.DataFrame(Atrue_dlaa).to_csv(OUTDIR/"alpha_true_dlaa.csv", index=False)
pd.DataFrame(Alpha_est_dlaa).to_csv(OUTDIR/"alpha_est_dlaa.csv", index=False)
pd.DataFrame(Atrue_conv).to_csv(OUTDIR/"alpha_true_conventional.csv", index=False)
pd.DataFrame(Alpha_est_conv).to_csv(OUTDIR/"alpha_est_conventional.csv", index=False)
pd.DataFrame(Atrue_dc).to_csv(OUTDIR/"alpha_true_datacenter.csv", index=False)
pd.DataFrame(Alpha_est_dc).to_csv(OUTDIR/"alpha_est_datacenter.csv", index=False)

# ---------------------------- Basic metrics ----------------------------
def metrics_for(Y, R_area, Atrue, Aest, label):
    e = Atrue[:, attack_bus_pos] - Aest[:, attack_bus_pos]
    rmse = float(np.sqrt(np.mean(e**2)))
    mask = (t >= 5.0) & (t <= 15.0)
    peak_by_area = [float(np.max(R_area[mask,a])) for a in range(R_area.shape[1])]
    peak_area = int(np.argmax(peak_by_area)) + 1
    return {"scenario": label, "alpha_RMSE_at_attacked_bus": rmse, "peak_area_id_5to15s": peak_area}

rows = []
rows.append(metrics_for(Y_dlaa, R_dlaa, Atrue_dlaa, Alpha_est_dlaa, "D-LAA (feedback)"))
rows.append(metrics_for(Y_conv, R_conv, Atrue_conv, Alpha_est_conv, "Conventional LAA (oscillatory)"))
rows.append(metrics_for(Y_dc,   R_dc,   Atrue_dc,   Alpha_est_dc,   "AI DC event (drop/reconnect)"))
df_metrics = pd.DataFrame(rows)
print("\n== Basic Metrics ==")
print(df_metrics.to_string(index=False))

# ---------------------------- Synthetic grid (degree-preserving rewiring) ----------------------------
G_syn = G.copy()
num_swaps = int(0.2 * G_syn.number_of_edges())
G_syn = nx.double_edge_swap(G_syn, nswap=num_swaps, max_tries=num_swaps*10)

nodes_syn = list(G_syn.nodes())
assert nodes_syn == list(range(N)), "Node relabel mismatch; expected same labeling"

edge_list_syn = list(G_syn.edges())
m_syn = len(edge_list_syn)
A_inc_syn = np.zeros((m_syn, N))
for ell, (i, j) in enumerate(edge_list_syn):
    A_inc_syn[ell, i] =  1.0
    A_inc_syn[ell, j] = -1.0

rng_syn = np.random.default_rng(SEED_SYN)
b_line_syn = rng_syn.uniform(0.5, 2.0, size=m_syn)
Bprime_syn = A_inc_syn.T @ (b_line_syn[:, None] * A_inc_syn)

Bgg_syn = Bprime_syn[np.ix_(gen_idx, gen_idx)]
Bgl_syn = Bprime_syn[np.ix_(gen_idx, load_idx)]
Blg_syn = Bprime_syn[np.ix_(load_idx, gen_idx)]
Bll_syn = Bprime_syn[np.ix_(load_idx, load_idx)]
Bll_inv_syn = np.linalg.inv(Bll_syn + 1e-6*np.eye(Bll_syn.shape[0]))

Btilde_syn = (Bgg_syn - Bgl_syn @ Bll_inv_syn @ Blg_syn)
Etilde_syn = (Bgl_syn @ Bll_inv_syn)

Minv = np.linalg.inv(M)
Z = np.zeros_like(Btilde_syn)
Ing = np.eye(ng)
A_top_syn = np.hstack([Z, Ing])
A_bot_syn = np.hstack([-Minv @ Btilde_syn, -Minv @ D])
A_syn = np.vstack([A_top_syn, A_bot_syn])
Bmat_syn = np.vstack([np.zeros((ng, ng)), Minv])
Emat_syn = np.vstack([np.zeros((ng, len(load_idx))), Minv @ Etilde_syn])
C_syn = C.copy()

# Areas on synthetic graph
try:
    from networkx.algorithms.community import greedy_modularity_communities
    comms_syn = list(greedy_modularity_communities(G_syn))
except Exception:
    comms_syn = [set(range(N))]
area_id_syn = np.zeros(N, dtype=int)
for k, cset in enumerate(comms_syn[:4]):
    for n in cset:
        area_id_syn[n] = k
area_gen_lists_syn = [[i for i, n in enumerate(gen_idx) if area_id_syn[n] == a] for a in range(4)]
if any(len(lst)==0 for lst in area_gen_lists_syn):
    splits = np.array_split(np.arange(ng), 4)
    area_gen_lists_syn = [list(s) for s in splits]

def simulate_on_syn():
    x = np.zeros(2*ng)
    Y = np.zeros((nt, ng))
    R_area = np.zeros((nt, 4))
    Alpha_true = np.zeros((nt, nl))
    for k in range(nt):
        global current_time
        current_time = t[k]
        alpha = np.zeros(nl)
        if 5.0 <= current_time <= 15.0:
            for p in dc_pos:
                alpha[p] = -150.0
        Alpha_true[k,:] = alpha

        k1 = A_syn @ x + Bmat_syn @ np.zeros(ng) + Emat_syn @ alpha
        k2 = A_syn @ (x + 0.5*dt*k1) + Emat_syn @ alpha
        k3 = A_syn @ (x + 0.5*dt*k2) + Emat_syn @ alpha
        k4 = A_syn @ (x + dt*k3) + Emat_syn @ alpha
        x = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

        y = (C_syn @ x)
        Y[k,:] = y

        for a in range(4):
            g_idx_list = area_gen_lists_syn[a]
            block = y[g_idx_list]
            R_area[k,a] = float(np.mean(block*block))
    return Y, R_area, Alpha_true

Y_syn, R_syn, Atrue_syn = simulate_on_syn()

# Compare real vs synthetic grid response (AI DC event)
fig, axs = plt.subplots(2,1, figsize=(10,7))
axs[0].plot(t, Y_dc[:,0], label="Real-topology (Opsahl)")
axs[0].plot(t, Y_syn[:,0], label="Synthetic rewired")
axs[0].set_title("Generator frequency (bus 0): real vs synthetic grid (AI DC event)")
axs[0].set_xlabel("Time [s]"); axs[0].set_ylabel("ω [pu]"); axs[0].legend()

axs[1].plot(t, R_dc[:,0], label="Area1 — real")
axs[1].plot(t, R_syn[:,0], label="Area1 — synthetic", linestyle="--")
axs[1].set_title("Area-1 residual energy: real vs synthetic grid (AI DC event)")
axs[1].set_xlabel("Time [s]"); axs[1].set_ylabel("Residual energy"); axs[1].legend()

plt.tight_layout()
plt.show()

# Save synthetic outputs
pd.DataFrame(Y_syn).to_csv(OUTDIR/"omega_datacenter_synthetic.csv", index=False)
pd.DataFrame(R_syn).to_csv(OUTDIR/"residuals_datacenter_synthetic.csv", index=False)
pd.DataFrame(Atrue_syn).to_csv(OUTDIR/"alpha_true_datacenter_synthetic.csv", index=False)

# ---------------------------- Mapping table (role/area/DC tags) ----------------------------
try:
    from networkx.algorithms.community import greedy_modularity_communities
    comms_real = list(greedy_modularity_communities(G))
except Exception:
    comms_real = [set(range(N))]
area_id_real = np.zeros(N, dtype=int)
for k, cset in enumerate(comms_real[:4]):
    for n in cset:
        area_id_real[n] = k
flag = np.array(["GEN" if i in gen_set else "LOAD" for i in range(N)], dtype=object)

df_map = pd.DataFrame({
    "bus_internal": np.arange(N, dtype=int),
    "role": flag,
    "area_id": area_id_real
})
dc_bus_internal = load_idx[dc_pos]
df_map["is_data_center"] = df_map["bus_internal"].isin(dc_bus_internal)
df_map.to_csv(OUTDIR/"bus_mapping_real_topology.csv", index=False)

print(f"\nArtifacts saved in: {OUTDIR.as_posix()}")
