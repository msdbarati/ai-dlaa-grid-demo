[README.md](https://github.com/user-attachments/files/21910249/README.md)

# AI D-LAA Grid Demo (Western US Topology)

Reproducible demo for **Dynamic Load‑Altering Attacks (D‑LAA)** and **AI data‑center events** on an enriched Western‑US HV grid topology (Opsahl dataset).

## Contents
- `src/dlaa_opsahl_demo.py` — End‑to‑end script (model build → simulation → logging → plots)
- `data/` — Opsahl files + derived edge list `us_powergrid_edges.csv`
- `results/` — CSV outputs for ω(t) and true/estimated attack inputs
- `figures/` — `western_us_backbone_topology.png`

## Quick start
```bash
pip install numpy pandas networkx matplotlib
python src/dlaa_opsahl_demo.py
```

## Scenarios
1) D‑LAA (feedback)  2) Conventional LAA (oscillatory)  3) AI data‑center drop/reconnect
