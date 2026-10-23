"""
AERCF Core — paper-faithful implementation.

Paper: "AERCF: What Survives Distribution Shift? An Empirical Evaluation of an
Adaptive, Explainable and Resilient Intrusion Detection and Response Architecture"
IJMSRT Vol.4 Issue 10, October 2026. DOI: 10.67317/IJMSRT26OCT017

Detectors (Table 2):
  Isolation Forest   — 200 trees, negated sklearn score
  VAE                — 121-64-32-8 encoder, mirrored decoder, 25 epochs, neg-ELBO
  XGBoost            — 300 trees, depth 6, η=0.1
  MLP                — 121-128-64-1, 15 epochs; also PGD-trained variant (ε=0.05, 7 steps)
  Fusion             — logistic meta-learner on standardised IF/VAE/XGBoost scores

Dataset (Table 3): KDDTrain+ / KDDTest+ / KDDTest-21
  125 973 / 22 544 / 11 850 records; 41 features → 121 after one-hot + log-transform

EVT calibration (Eq. 2):
  θ_α = u + (σ/ξ) * [(n·α/N_u)^(-ξ) - 1]
  u = 90th percentile of benign validation scores
  GPD fitted to exceedances above u
  Targets α ∈ {1%, 0.5%, 0.1%}

Threshold adaptation (Table 6, §4.4):
  Fixed / Repository-controller / PI on 2% audit sample / Alert-budget
"""

from __future__ import annotations
import io, json, logging, os, pickle, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import genpareto
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from xgboost import XGBClassifier

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

log = logging.getLogger("aercf.core")

# ─────────────────────────────────────────────────────────────────────────────
# NSL-KDD feature engineering (§4.1)
# 41 raw features → 121 after one-hot + log-transform
# ─────────────────────────────────────────────────────────────────────────────

PROTO_CATS  = ["tcp","udp","icmp"]
SERVICE_CATS = [
    "ftp_data","other","private","http","remote_job","name","netbios_ns","eco_i",
    "mtp","telnet","finger","domain_u","supdup","uucp_path","Z39_50","smtp","csnet_ns",
    "uucp","netbios_dgm","urp_i","auth","domain","ftp","bgp","ldap","ecr_i","gopher",
    "vmnet","systat","http_443","efs","whois","imap4","iso_tsap","echo","klogin","link",
    "sunrpc","login","kshell","sql_net","time","hostnames","exec","ntp_u","discard",
    "nntp","courier","ctf","ssh","daytime","shell","netstat","pop_3","nnsp","IRC",
    "pop_2","printer","tim_i","pm_dump","red_i","netbios_ssn","rje","X11","urh_i",
    "http_8001","aol","http_2784","tftp_u","harvest","icmp",
]
FLAG_CATS = ["SF","S1","REJ","S2","S0","S3","RSTO","RSTR","RSTOS0","OTH","SH"]

LOG_FEATS = [
    "duration","src_bytes","dst_bytes","land","wrong_fragment","urgent",
    "hot","num_failed_logins","logged_in","num_compromised","root_shell",
    "su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count",
]

NUMERIC_FEATS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
]

COLS = NUMERIC_FEATS + ["label","difficulty"]


def load_nslkdd(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load one NSL-KDD .txt file; returns (features_df, binary_labels)."""
    df = pd.read_csv(path, header=None, names=COLS)
    y = (df["label"] != "normal").astype(int)
    df = df.drop(columns=["label", "difficulty"])
    return df, y


def engineer_features(df: pd.DataFrame, scaler: StandardScaler | None = None,
                       fit: bool = False) -> tuple[np.ndarray, StandardScaler]:
    """
    NSL-KDD → 121-dim feature matrix.
    Log-transform heavy-tailed counts; one-hot encode 3 categorical fields.
    """
    X = df.copy()

    # Log-transform heavy-tailed numeric fields
    for col in LOG_FEATS:
        if col in X.columns and X[col].dtype != object:
            X[col] = np.log1p(X[col].clip(lower=0))

    # One-hot encode protocol_type (3), service (~70 but paper says 121 total),
    # flag (11) — we keep the union from training data
    proto_ohe  = pd.get_dummies(X["protocol_type"], prefix="proto").reindex(
        columns=[f"proto_{c}" for c in PROTO_CATS], fill_value=0)
    service_ohe = pd.get_dummies(X["service"], prefix="svc").reindex(
        columns=[f"svc_{c}" for c in SERVICE_CATS], fill_value=0)
    flag_ohe   = pd.get_dummies(X["flag"], prefix="flag").reindex(
        columns=[f"flag_{c}" for c in FLAG_CATS], fill_value=0)

    numeric = X.drop(columns=["protocol_type","service","flag"]).select_dtypes(include=np.number)
    Xmat = pd.concat([numeric, proto_ohe, service_ohe, flag_ohe], axis=1).fillna(0).values.astype(np.float32)

    if scaler is None:
        scaler = StandardScaler()
    if fit:
        scaler.fit(Xmat)
    Xmat = scaler.transform(Xmat)
    return Xmat, scaler


# ─────────────────────────────────────────────────────────────────────────────
# VAE (§4.2, Table 2): 121-64-32-8 encoder, mirrored decoder
# Trained on benign records only; score = negative ELBO (8 MC samples)
# ─────────────────────────────────────────────────────────────────────────────

class VAEEncoder(nn.Module):
    def __init__(self, input_dim: int = 121, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.mu_head  = nn.Linear(32, latent_dim)
        self.log_var_head = nn.Linear(32, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu_head(h), self.log_var_head(h)


class VAEDecoder(nn.Module):
    def __init__(self, input_dim: int = 121, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, input_dim: int = 121, latent_dim: int = 8):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, latent_dim)
        self.decoder = VAEDecoder(input_dim, latent_dim)

    def reparameterise(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterise(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

    def elbo_loss(self, x, x_recon, mu, log_var):
        recon = nn.functional.mse_loss(x_recon, x, reduction="sum")
        kl    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon + kl

    @torch.no_grad()
    def anomaly_score(self, x: np.ndarray, n_samples: int = 8) -> np.ndarray:
        """Return negative ELBO averaged over n_samples MC draws (higher = more anomalous)."""
        t = torch.tensor(x, dtype=torch.float32)
        scores = []
        for _ in range(n_samples):
            x_r, mu, lv = self.forward(t)
            loss = nn.functional.mse_loss(x_r, t, reduction="none").sum(dim=1) \
                   + 0.5 * (mu.pow(2) + lv.exp() - 1 - lv).sum(dim=1)
            scores.append(loss.numpy())
        return np.mean(scores, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# MLP for classification + PGD adversarial training (§4.2, Table 2)
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim: int = 121):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        t = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return torch.sigmoid(self.forward(t)).numpy()


def pgd_attack(model: MLP, x: torch.Tensor, y: torch.Tensor,
               eps: float = 0.05, alpha: float = 0.01, steps: int = 7) -> torch.Tensor:
    """L∞ PGD attack (§4.2, E4). Continuous features only, clipped to [0,1]."""
    x_adv = x.detach().clone().requires_grad_(True)
    for _ in range(steps):
        loss = nn.functional.binary_cross_entropy_with_logits(model(x_adv), y.float())
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, x - eps, x + eps)
            x_adv = torch.clamp(x_adv, 0, 1)
        x_adv = x_adv.detach().requires_grad_(True)
    return x_adv.detach()


# ─────────────────────────────────────────────────────────────────────────────
# EVT threshold calibration (Eq. 2 in paper)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EVTCalibrator:
    alpha: float = 0.01          # target FPR (e.g. 0.01 = 1%)
    u_quantile: float = 0.90     # percentile for POT threshold u
    xi: float  = 0.0             # GPD shape (fitted)
    sigma: float = 1.0           # GPD scale (fitted)
    u: float   = 0.0             # POT threshold
    N_u: int   = 0               # number of exceedances
    n: int     = 0               # total benign samples
    threshold: float = 0.0       # calibrated detection threshold θ_α
    rho: float = 1.0             # realised calibration ratio

    def fit(self, benign_scores: np.ndarray) -> "EVTCalibrator":
        scores = np.asarray(benign_scores, dtype=float)
        self.n = len(scores)
        self.u = float(np.quantile(scores, self.u_quantile))
        exceedances = scores[scores > self.u] - self.u
        self.N_u = len(exceedances)
        if self.N_u < 10:
            # Fall back to empirical quantile
            self.threshold = float(np.quantile(scores, 1 - self.alpha))
            return self
        # Fit GPD to exceedances (scipy uses c=-ξ convention)
        c, loc, scale = genpareto.fit(exceedances, floc=0)
        self.xi    = float(-c)   # paper sign convention
        self.sigma = float(scale)
        # Eq. 2: θ_α = u + (σ/ξ)·[(n·α/N_u)^(-ξ) - 1]
        if abs(self.xi) < 1e-6:
            # Limit ξ→0: exponential tail
            self.threshold = float(self.u + self.sigma * np.log(self.n * self.alpha / self.N_u))
        else:
            self.threshold = float(
                self.u + (self.sigma / self.xi) *
                ((self.n * self.alpha / self.N_u) ** (-self.xi) - 1)
            )
        return self

    def realised_fpr(self, benign_scores: np.ndarray) -> float:
        scores = np.asarray(benign_scores, dtype=float)
        fpr = float((scores >= self.threshold).mean())
        self.rho = fpr / self.alpha if self.alpha > 0 else float("inf")
        return fpr


# ─────────────────────────────────────────────────────────────────────────────
# Threshold adaptation policies (§4.4, Table 6)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FixedPolicy:
    """Fixed validation-calibrated threshold — best mean DR (77.2%)."""
    theta: float

    def update(self, window_scores, window_labels=None, audit_scores=None,
               audit_labels=None, alert_count=None, window_size=None):
        return self.theta  # never changes


@dataclass
class RepositoryController:
    """
    Oracle controller: updates θ based on cumulative true labels (E5 §4.4).
    Lowers FPR target by adjusting empirical quantile.
    """
    theta: float
    alpha: float = 0.01
    _cumulative_benign: list = field(default_factory=list)

    def update(self, window_scores, window_labels, audit_scores=None,
               audit_labels=None, alert_count=None, window_size=None):
        # Use true labels — oracle mode
        labels = np.asarray(window_labels)
        scores = np.asarray(window_scores)
        benign = scores[labels == 0]
        self._cumulative_benign.extend(benign.tolist())
        all_b = np.array(self._cumulative_benign)
        if len(all_b) > 10:
            self.theta = float(np.quantile(all_b, 1 - self.alpha))
        return self.theta


@dataclass
class PIController:
    """
    PI controller on 2% audit sample (§4.4, Table 6).
    Tracks estimated FPR from audit labels arriving one window late.
    Kp, Ki tuned to not anti-windup beyond ±0.2 from initial theta.
    """
    theta: float
    alpha: float = 0.01
    Kp: float = 2.0
    Ki: float = 0.5
    _integral: float = 0.0
    _theta0: float = field(init=False)

    def __post_init__(self):
        self._theta0 = self.theta

    def update(self, window_scores, window_labels=None, audit_scores=None,
               audit_labels=None, alert_count=None, window_size=None):
        if audit_scores is None or audit_labels is None:
            return self.theta
        a_scores = np.asarray(audit_scores)
        a_labels = np.asarray(audit_labels)
        benign = a_scores[a_labels == 0]
        if len(benign) == 0:
            return self.theta
        fpr_est = float((benign >= self.theta).mean())
        error = fpr_est - self.alpha
        self._integral = np.clip(self._integral + error, -0.3, 0.3)
        delta = self.Kp * error + self.Ki * self._integral
        # Clip movement: floor on detection, ceiling on FPR
        self.theta = float(np.clip(self.theta + delta, self._theta0 - 0.2, self._theta0 + 0.5))
        return self.theta


@dataclass
class AlertBudgetPolicy:
    """
    Alert-budget policy: keep alert volume ≤ 5% of traffic (§4.4, Table 6).
    Suppresses detection when prevalence rises — gameable by adversary.
    """
    theta: float
    budget_frac: float = 0.05

    def update(self, window_scores, window_labels=None, audit_scores=None,
               audit_labels=None, alert_count=None, window_size=None):
        scores = np.asarray(window_scores)
        if window_size is None:
            window_size = len(scores)
        current_rate = (scores >= self.theta).mean()
        if current_rate > self.budget_frac:
            # Raise threshold until we're under budget
            self.theta = float(np.quantile(scores, 1 - self.budget_frac))
        return self.theta


# ─────────────────────────────────────────────────────────────────────────────
# Full AERCF model bundle
# ─────────────────────────────────────────────────────────────────────────────

class AERCF:
    """
    Four-layer AERCF: Sense → Detect → Calibrate+Explain → Decide+Adapt

    Attributes
    ----------
    scaler        : StandardScaler fit on training features
    iso_forest    : IsolationForest (200 trees)
    vae           : VAE (121-64-32-8, trained on benign only)
    xgb           : XGBClassifier (300/6/η=0.1)
    mlp           : MLP standard (121-128-64-1)
    mlp_pgd       : MLP with PGD adversarial training (ε=0.05, 7 steps)
    fusion        : LogisticRegression meta-learner on IF/VAE/XGB scores
    calibrators   : dict of alpha → EVTCalibrator per detector
    explainer     : SHAP TreeExplainer for XGBoost
    feature_names : list of 121 feature names
    """

    def __init__(self):
        self.scaler: StandardScaler | None       = None
        self.iso_forest: IsolationForest | None  = None
        self.vae: VAE | None                     = None
        self.xgb: XGBClassifier | None           = None
        self.mlp: MLP | None                     = None
        self.mlp_pgd: MLP | None                 = None
        self.fusion: LogisticRegression | None   = None
        self.calibrators: dict                   = {}
        self.explainer                           = None
        self.feature_names: list[str]            = []
        self.train_metrics: dict                 = {}
        self._if_scaler: StandardScaler | None   = None
        self._vae_scaler: StandardScaler | None  = None
        self._xgb_scaler: StandardScaler | None  = None

    # ── training ─────────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray,   y_val: np.ndarray,
            feature_names: list[str] | None = None,
            epochs_vae: int = 25, epochs_mlp: int = 15,
            seed: int = 0):
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        input_dim = X_train.shape[1]
        self.feature_names = feature_names or [f"f{i}" for i in range(input_dim)]

        benign_train = X_train[y_train == 0]
        log.info("Training IF on %d benign records", len(benign_train))

        # 1. Isolation Forest (Table 2)
        self.iso_forest = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
        self.iso_forest.fit(benign_train)

        # 2. VAE (Table 2): trained on benign only, 25 epochs
        log.info("Training VAE (%d epochs)", epochs_vae)
        self.vae = VAE(input_dim=input_dim, latent_dim=8)
        opt = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        t_benign = torch.tensor(benign_train, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(t_benign)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
        self.vae.train()
        for ep in range(epochs_vae):
            total = 0.0
            for (xb,) in loader:
                opt.zero_grad()
                xr, mu, lv = self.vae(xb)
                loss = self.vae.elbo_loss(xb, xr, mu, lv) / len(xb)
                loss.backward()
                opt.step()
                total += loss.item()
            if (ep+1) % 5 == 0:
                log.info("VAE epoch %d/%d  loss=%.4f", ep+1, epochs_vae, total/len(loader))

        # 3. XGBoost supervised (Table 2)
        log.info("Training XGBoost")
        neg_pos = (y_train == 0).sum() / (y_train == 1).sum()
        self.xgb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=neg_pos, eval_metric="logloss",
            random_state=seed, n_jobs=-1, verbosity=0,
        )
        self.xgb.fit(X_train, y_train)

        # 4. MLP standard (Table 2)
        log.info("Training MLP (%d epochs)", epochs_mlp)
        self.mlp = self._train_mlp(X_train, y_train, epochs=epochs_mlp, seed=seed, pgd=False)

        # 4b. MLP + PGD adversarial (ε=0.05, 7 steps, Table 2)
        log.info("Training MLP+PGD (%d epochs)", epochs_mlp)
        self.mlp_pgd = self._train_mlp(X_train, y_train, epochs=epochs_mlp, seed=seed, pgd=True)

        # 5. Compute base scores on validation split for fusion + calibration
        log.info("Computing validation scores for fusion + EVT calibration")
        if_scores  = self._if_score(X_val)
        vae_scores = self._vae_score(X_val)
        xgb_scores = self.xgb.predict_proba(X_val)[:, 1]

        # Standardise for fusion meta-learner (§4.3, "standardised IF/VAE/XGBoost scores")
        self._if_scaler  = StandardScaler().fit(if_scores.reshape(-1,1))
        self._vae_scaler = StandardScaler().fit(vae_scores.reshape(-1,1))
        self._xgb_scaler = StandardScaler().fit(xgb_scores.reshape(-1,1))

        Z_val = self._fusion_features(X_val)
        self.fusion = LogisticRegression(max_iter=500, random_state=seed)
        self.fusion.fit(Z_val, y_val)

        # 6. EVT calibration for each detector (benign validation scores, u=90th pct)
        log.info("Fitting EVT calibrators")
        benign_val = X_val[y_val == 0]
        for alpha in [0.01, 0.005, 0.001]:
            self.calibrators[alpha] = {}
            for name, scores in [
                ("if",    self._if_score(benign_val)),
                ("vae",   self._vae_score(benign_val)),
                ("xgb",   self.xgb.predict_proba(benign_val)[:,1]),
                ("fusion",self._fusion_score(benign_val)),
            ]:
                cal = EVTCalibrator(alpha=alpha)
                cal.fit(scores)
                self.calibrators[alpha][name] = cal
                log.info("EVT α=%.3f %s: θ=%.4f (u=%.4f, ξ=%.4f, σ=%.4f)",
                         alpha, name, cal.threshold, cal.u, cal.xi, cal.sigma)

        # 7. TreeSHAP explainer on XGBoost
        if SHAP_AVAILABLE:
            log.info("Building TreeSHAP explainer")
            self.explainer = shap.TreeExplainer(self.xgb)

        log.info("Training complete")

    # ── internal score helpers ────────────────────────────────────────────────

    def _if_score(self, X: np.ndarray) -> np.ndarray:
        """Negated sklearn IF score: higher → more anomalous (Table 2)."""
        return -self.iso_forest.score_samples(X)

    def _vae_score(self, X: np.ndarray) -> np.ndarray:
        self.vae.eval()
        return self.vae.anomaly_score(X)

    def _fusion_features(self, X: np.ndarray) -> np.ndarray:
        if_s  = self._if_scaler.transform(self._if_score(X).reshape(-1,1))[:,0]
        vae_s = self._vae_scaler.transform(self._vae_score(X).reshape(-1,1))[:,0]
        xgb_s = self._xgb_scaler.transform(
            self.xgb.predict_proba(X)[:,1].reshape(-1,1))[:,0]
        return np.column_stack([if_s, vae_s, xgb_s])

    def _fusion_score(self, X: np.ndarray) -> np.ndarray:
        return self.fusion.predict_proba(self._fusion_features(X))[:,1]

    # ── MLP training helper ───────────────────────────────────────────────────

    def _train_mlp(self, X: np.ndarray, y: np.ndarray,
                   epochs: int = 15, seed: int = 0, pgd: bool = False) -> MLP:
        torch.manual_seed(seed)
        model = MLP(input_dim=X.shape[1])
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        Xt    = torch.tensor(X, dtype=torch.float32)
        yt    = torch.tensor(y, dtype=torch.float32)
        ds    = torch.utils.data.TensorDataset(Xt, yt)
        dl    = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
        for ep in range(epochs):
            model.train()
            for xb, yb in dl:
                opt.zero_grad()
                if pgd:
                    xb = pgd_attack(model, xb, yb, eps=0.05, alpha=0.01, steps=7)
                loss = nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
                loss.backward()
                opt.step()
        return model

    # ── inference ─────────────────────────────────────────────────────────────

    def score(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return per-detector anomaly scores + fusion probability."""
        return {
            "if":     self._if_score(X),
            "vae":    self._vae_score(X),
            "xgb":    self.xgb.predict_proba(X)[:,1],
            "mlp":    self.mlp.predict_proba(X),
            "mlp_pgd":self.mlp_pgd.predict_proba(X),
            "fusion": self._fusion_score(X),
        }

    def predict(self, X: np.ndarray, alpha: float = 0.01,
                detector: str = "fusion") -> np.ndarray:
        """Binary predictions using EVT-calibrated threshold."""
        scores = self.score(X)[detector]
        theta  = self.calibrators[alpha][detector].threshold
        return (scores >= theta).astype(int)

    def explain(self, X: np.ndarray, max_records: int = 500) -> dict:
        """TreeSHAP attributions for XGBoost (exact additivity, §4.3)."""
        if not SHAP_AVAILABLE or self.explainer is None:
            return {"error": "shap not available"}
        X_sub = X[:max_records]
        sv = self.explainer.shap_values(X_sub)
        base_value = float(self.explainer.expected_value)
        mean_abs   = np.abs(sv).mean(axis=0).tolist()
        top_idx    = np.argsort(mean_abs)[::-1][:10].tolist()
        return {
            "base_value":  base_value,
            "mean_abs":    mean_abs,
            "top_features":[self.feature_names[i] for i in top_idx],
            "top_values":  [mean_abs[i] for i in top_idx],
            "shap_sample": sv[:min(10,len(sv))].tolist(),
        }

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "core.pkl", "wb") as f:
            pickle.dump(self, f)
        log.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "AERCF":
        with open(Path(path) / "core.pkl", "rb") as f:
            return pickle.load(f)

    def is_trained(self) -> bool:
        return self.fusion is not None
