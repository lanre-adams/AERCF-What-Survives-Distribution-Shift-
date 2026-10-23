# AERCF — Paper-Faithful Simulation

> **Agunloye O.A., Anana A., Mathias S., Bakare E.**  
> *"AERCF: What Survives Distribution Shift? An Empirical Evaluation of an Adaptive, Explainable and Resilient Intrusion Detection and Response Architecture"*  
> IJMSRT Vol. 4 Issue 10, October 2026. DOI: [10.67317/IJMSRT26OCT017](https://doi.org/10.67317/IJMSRT26OCT017)

This repository provides a runnable implementation of the full AERCF pipeline described in the paper, with an interactive dashboard that reproduces Experiments E1–E7.

---

## Architecture (Figure 1)

```
L1 Sense    NSL-KDD 41 raw features → 121-dim (log-transform + one-hot)
L2 Detect   Isolation Forest (200)  |  VAE (121-64-32-8)
            XGBoost (300 trees d=6) |  MLP (128-64) [std + PGD-adv]
            Fusion: stacked logistic on standardised detector scores
L3 Cal+Exp  EVT/GPD threshold at α ∈ {1 %, 0.5 %, 0.1 %} + TreeSHAP
L4 Decide   Tiered lattice: monitor < step-up < isolate
            Corroboration rule (Eq. 3) + 4 adaptation policies
```

---

## Quick Start

### Prerequisites
- Docker ≥ 24 and Docker Compose V2
- 4 GB RAM, 4 GB free disk (NSL-KDD + model weights)

### Run

```bash
git clone <this-repo>
cd aercf
docker compose up --build
```

Open **http://localhost:3000** in your browser.

Click **▶ Train & Run All Experiments** on the Overview tab.  
Training downloads NSL-KDD automatically (≈ 3–5 min on 2 vCPUs).  
When done, all tabs (E1–E7) populate with paper-faithful results.

### Development (no Docker)

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
REACT_APP_API_URL=http://localhost:8000 npm start
```

---

## Experiments reproduced

| Tab | Experiment | Paper section |
|-----|-----------|---------------|
| E1  | Detection under distribution shift (Table 5) | §5.1 |
| E2  | EVT calibration ratio ρ (Figure 4) | §5.2 |
| E3  | SHAP faithfulness: additivity + deletion (Figure 5) | §5.3 |
| E4  | Adversarial robustness PGD-20 (Figure 6) | §5.4 |
| E5  | Threshold adaptation over 60 windows (Table 6 + Figure 7) | §5.5 |
| E6  | Containment tier operating points (Figure 8) | §5.6 |
| E7  | Inference cost / latency (Table 7) | §5.7 |

---

## Paper key findings reproduced

- **C1** – XGBoost achieves near-perfect in-distribution detection (AUROC ≈ 1.000) but degrades 13.3 pp under KDDTest-21 novel attacks; VAE loses only 3.5 pp.
- **C2** – EVT calibration is tight in-distribution (ρ ≈ 1) but exceeds targets 29× on KDDTest+ at α = 0.1%.
- **C3** – Naive adaptive threshold controllers (PI, alert-budget) underperform a fixed threshold (mean DR 77.2%) under rising attack prevalence.
- **C4** – PGD adversarial training raises MLP DR 49.8% → 63.1% at ε = 0.1 with no FPR cost.
- **C5** – Full pipeline processes ≈ 33 000 records/s on 2 vCPUs (batch 1024).

---

## File structure

```
aercf/
├── backend/
│   ├── aercf/
│   │   ├── core.py          # AERCF model, detectors, EVT, policies
│   │   └── experiments.py   # E1–E7 runner functions
│   ├── api/
│   │   └── main.py          # FastAPI application
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Dashboard UI (8 tabs, E1–E7)
│   │   ├── api.js           # Fetch wrappers
│   │   └── index.js
│   ├── public/index.html
│   ├── package.json
│   └── Dockerfile
├── nginx/
│   └── nginx.conf           # /api/* → backend, /* → React SPA
└── docker-compose.yml
```

---

## Appendix B note

The companion browser visualisation (Appendix B of the paper) uses fixed random distributions conditioned on true labels. It is **not** a source of experimental results and must not be cited as performance evidence. This repository runs actual ML training on real NSL-KDD data.

---

## Citation

```bibtex
@article{agunloye2026aercf,
  title   = {{AERCF}: What Survives Distribution Shift? An Empirical Evaluation of an Adaptive, Explainable and Resilient Intrusion Detection and Response Architecture},
  author  = {Agunloye, Oluwaseun A. and Anana, Ayomide and Mathias, Sunday and Bakare, Emmanuel},
  journal = {International Journal of Multidisciplinary Sciences and Research Trends},
  volume  = {4},
  number  = {10},
  year    = {2026},
  doi     = {10.67317/IJMSRT26OCT017}
}
```
