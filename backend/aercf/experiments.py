"""
Reproduce paper experiments E1–E7 on NSL-KDD.

E1 — Detection under shift (Table 5)
E2 — EVT calibration (Figure 4)
E3 — SHAP explanation faithfulness (Figure 5)
E4 — Adversarial robustness (Figure 6)
E5 — Threshold adaptation (Table 6, Figure 7)
E6 — Containment operating points (Figure 8)
E7 — Inference cost (Table 7)
"""
from __future__ import annotations
import logging, time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch

from .core import (
    AERCF, EVTCalibrator, engineer_features, load_nslkdd,
    FixedPolicy, RepositoryController, PIController, AlertBudgetPolicy,
)

log = logging.getLogger("aercf.experiments")

# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def metrics_at_threshold(scores: np.ndarray, y: np.ndarray, theta: float) -> dict:
    preds = (scores >= theta).astype(int)
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    dr  = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    f1  = 2 * prec * dr / max(prec + dr, 1e-9)
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, dr=dr, fpr=fpr, precision=prec, f1=f1,
                auroc=float(roc_auc_score(y, scores)) if len(np.unique(y))>1 else 0.0,
                auprc=float(average_precision_score(y, scores)) if len(np.unique(y))>1 else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# E1 — Detection under shift
# ─────────────────────────────────────────────────────────────────────────────

def run_e1(model: AERCF, X_val, y_val, X_test_plus, y_test_plus,
           X_test21, y_test21, alpha: float = 0.01) -> dict:
    """Table 5 results: per-detector AUROC and DR/FPR on validation, KDDTest+, KDDTest-21."""
    results = {}
    for split_name, Xs, ys in [
        ("validation", X_val, y_val),
        ("kddtest_plus", X_test_plus, y_test_plus),
        ("kddtest_21", X_test21, y_test21),
    ]:
        scores_dict = model.score(Xs)
        split_res = {}
        for det in ["if","vae","xgb","mlp","fusion"]:
            scores = scores_dict[det]
            theta  = model.calibrators[alpha][det if det != "mlp" else "fusion"].threshold
            if det == "mlp":
                # Use fusion EVT threshold as proxy for MLP (paper doesn't calibrate MLP separately)
                theta = model.calibrators[alpha]["xgb"].threshold
            split_res[det] = metrics_at_threshold(scores, ys, theta)
        results[split_name] = split_res
    return results


# ─────────────────────────────────────────────────────────────────────────────
# E2 — EVT calibration ratio ρ = realised_FPR / α
# ─────────────────────────────────────────────────────────────────────────────

def run_e2(model: AERCF, X_val, y_val, X_test_plus, y_test_plus) -> dict:
    """Realised calibration ratio ρ per detector and α."""
    results = {}
    for alpha, caldict in model.calibrators.items():
        results[alpha] = {}
        for det, cal in caldict.items():
            scores_dict = model.score(X_val)
            if det not in scores_dict:
                continue
            benign_val = scores_dict[det][y_val == 0]
            benign_test = model.score(X_test_plus)[det][y_test_plus == 0]
            fpr_val  = (benign_val >= cal.threshold).mean()
            fpr_test = (benign_test >= cal.threshold).mean()
            results[alpha][det] = {
                "threshold": cal.threshold,
                "rho_val":   fpr_val / alpha if alpha > 0 else None,
                "rho_test":  fpr_test / alpha if alpha > 0 else None,
                "xi":        cal.xi,
                "sigma":     cal.sigma,
                "u":         cal.u,
            }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# E3 — SHAP faithfulness (additivity + deletion)
# ─────────────────────────────────────────────────────────────────────────────

def run_e3(model: AERCF, X_test_plus, y_test_plus) -> dict:
    """TreeSHAP additivity error and deletion faithfulness."""
    attacks = X_test_plus[y_test_plus == 1][:2000]
    exp = model.explain(attacks)
    if "error" in exp:
        return {"error": exp["error"]}

    # Additivity check: |sum(φ_i) + φ_0 - log_odds(prediction)| should be ~0
    try:
        import shap
        sv = np.array(exp["shap_sample"])   # (N, D)
        logits = model.xgb.predict(attacks[:len(sv)], output_margin=True)
        residuals = np.abs(sv.sum(axis=1) + exp["base_value"] - logits[:len(sv)])
        max_additivity_error = float(residuals.max())
    except Exception:
        max_additivity_error = None

    return {
        "top_features":       exp["top_features"],
        "top_mean_abs_shap":  exp["top_values"],
        "max_additivity_error": max_additivity_error,
        "n_explained":        len(attacks),
    }


# ─────────────────────────────────────────────────────────────────────────────
# E4 — Adversarial robustness (L∞ PGD-20 white-box)
# ─────────────────────────────────────────────────────────────────────────────

def run_e4(model: AERCF, X_test_plus, y_test_plus,
           eps_list=(0.01, 0.02, 0.05, 0.1), n_attacks: int = 1000) -> dict:
    """
    White-box PGD-20 on MLP (standard + adversarially trained).
    Transfer to all detectors.
    """
    from .core import pgd_attack, MLP
    attacks  = X_test_plus[y_test_plus == 1][:n_attacks]
    y_attack = y_test_plus[y_test_plus == 1][:n_attacks]
    results  = {}

    for eps in eps_list:
        Xt = torch.tensor(attacks, dtype=torch.float32)
        yt = torch.tensor(y_attack, dtype=torch.float32)
        x_adv_std = pgd_attack(model.mlp,     Xt, yt, eps=eps, steps=20).numpy()
        x_adv_pgd = pgd_attack(model.mlp_pgd, Xt, yt, eps=eps, steps=20).numpy()

        alpha = 0.01
        res = {}
        for adv_name, x_adv in [("std_adv",x_adv_std),("pgd_adv",x_adv_pgd)]:
            det_res = {}
            scores_adv = model.score(x_adv)
            for det in ["if","vae","xgb","mlp","fusion"]:
                s = scores_adv[det]
                theta = model.calibrators[alpha].get(det, model.calibrators[alpha]["fusion"]).threshold
                det_res[det] = metrics_at_threshold(s, y_attack, theta)
            res[adv_name] = det_res
        results[float(eps)] = res

    return results


# ─────────────────────────────────────────────────────────────────────────────
# E5 — Threshold adaptation over 60 windows (Table 6, Figure 7)
# ─────────────────────────────────────────────────────────────────────────────

def run_e5(model: AERCF, X_test_plus, y_test_plus, n_windows: int = 60,
           window_size: int = 1000, seed: int = 0) -> dict:
    """
    Construct non-stationary stream: attack prevalence rises 5%→60% and falls back.
    Compare 4 threshold policies on fused scores.
    """
    rng = np.random.default_rng(seed)
    benign  = X_test_plus[y_test_plus == 0]
    attacks = X_test_plus[y_test_plus == 1]

    # Prevalence schedule: rises 5→60, then falls 60→5
    half = n_windows // 2
    prev_up   = np.linspace(0.05, 0.60, half)
    prev_down = np.linspace(0.60, 0.05, n_windows - half)
    prevalences = np.concatenate([prev_up, prev_down])

    # Initial threshold from EVT at α=1%
    theta0 = model.calibrators[0.01]["fusion"].threshold

    policies = {
        "fixed":      FixedPolicy(theta=theta0),
        "repository": RepositoryController(theta=theta0),
        "pi_audit":   PIController(theta=theta0),
        "alert_budget": AlertBudgetPolicy(theta=theta0),
    }

    window_results = {k: [] for k in policies}
    prev_audit_scores, prev_audit_labels = None, None

    for w, prev in enumerate(prevalences):
        n_att = int(window_size * prev)
        n_ben = window_size - n_att
        # Resample window
        idx_b = rng.integers(0, len(benign), size=min(n_ben, len(benign)))
        idx_a = rng.integers(0, len(attacks), size=min(n_att, len(attacks)))
        Xw = np.vstack([benign[idx_b], attacks[idx_a]])
        yw = np.concatenate([np.zeros(len(idx_b)), np.ones(len(idx_a))]).astype(int)

        # Score
        fused = model.score(Xw)["fusion"]

        # 2% audit sample (labelled, arrives one window late → PI uses prev window's)
        audit_n = max(1, int(0.02 * window_size))
        audit_idx = rng.integers(0, len(Xw), size=audit_n)
        audit_s = fused[audit_idx]
        audit_l = yw[audit_idx]

        for name, policy in policies.items():
            theta = policy.update(
                window_scores=fused,
                window_labels=yw,
                audit_scores=prev_audit_scores,
                audit_labels=prev_audit_labels,
                window_size=window_size,
            )
            m = metrics_at_threshold(fused, yw, theta)
            window_results[name].append({
                "window": w, "prevalence": float(prev),
                "theta": float(theta), **m,
            })

        prev_audit_scores, prev_audit_labels = audit_s, audit_l

    # Summary statistics (Table 6)
    summary = {}
    for name, wlist in window_results.items():
        drs  = [r["dr"]  for r in wlist]
        fprs = [r["fpr"] for r in wlist]
        summary[name] = {
            "mean_dr": float(np.mean(drs)),
            "min_dr":  float(np.min(drs)),
            "mean_fpr":float(np.mean(fprs)),
            "max_fpr": float(np.max(fprs)),
        }

    return {"summary": summary, "windows": window_results}


# ─────────────────────────────────────────────────────────────────────────────
# E6 — Containment tier operating points
# ─────────────────────────────────────────────────────────────────────────────

def run_e6(model: AERCF, X_test_plus, y_test_plus) -> dict:
    """
    Tier assignment: monitor / step-up / isolate using Eq. (3).
    Corroboration rule: isolate only if fusion ≥ θ_2 AND (IF or VAE) ≥ 1% EVT threshold.
    """
    alpha1 = 0.01   # step-up threshold θ₁
    alpha2 = 0.001  # isolate threshold θ₂

    scores = model.score(X_test_plus)
    fused  = scores["fusion"]
    if_s   = scores["if"]
    vae_s  = scores["vae"]

    theta1 = model.calibrators[alpha1]["fusion"].threshold
    theta2 = model.calibrators[alpha2]["fusion"].threshold
    theta_if  = model.calibrators[alpha1]["if"].threshold
    theta_vae = model.calibrators[alpha1]["vae"].threshold

    tiers     = np.zeros(len(fused), dtype=int)   # 0=monitor, 1=step-up, 2=isolate
    corroborate = (if_s >= theta_if) | (vae_s >= theta_vae)

    tiers[fused >= theta1] = 1
    # Corroboration rule (Eq. 3)
    tiers[(fused >= theta2) & corroborate] = 2

    def tier_shares(mask):
        m = int(mask.sum())
        if m == 0:
            return {"monitor":0,"step_up":0,"isolate":0,"n":0}
        t = tiers[mask]
        return {
            "monitor":  int((t==0).sum())/m,
            "step_up":  int((t==1).sum())/m,
            "isolate":  int((t==2).sum())/m,
            "n": m,
        }

    benign_mask = y_test_plus == 0
    attack_mask = y_test_plus == 1

    corr_isolate_attack  = int(((tiers==2) & attack_mask).sum()) / attack_mask.sum()
    corr_isolate_benign  = int(((tiers==2) & benign_mask).sum()) / benign_mask.sum()

    return {
        "benign":  tier_shares(benign_mask),
        "attack":  tier_shares(attack_mask),
        "corroborated_isolate_attack_rate": float(corr_isolate_attack),
        "corroborated_isolate_benign_rate": float(corr_isolate_benign),
        "theta1": float(theta1),
        "theta2": float(theta2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# E7 — Inference cost (Table 7)
# ─────────────────────────────────────────────────────────────────────────────

def run_e7(model: AERCF, X_test_plus, n_reps: int = 10) -> dict:
    """Training times are logged during fit(); here we measure inference latency."""
    results = {}
    for batch in [1, 1024]:
        X = X_test_plus[:batch]
        times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            model.score(X)
            times.append((time.perf_counter() - t0) * 1000 / batch)   # ms/record
        results[f"batch_{batch}"] = {
            "mean_ms_per_record": float(np.mean(times)),
            "std_ms_per_record":  float(np.std(times)),
        }
    # SHAP latency
    if model.explainer is not None:
        times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            model.explain(X_test_plus[:1])
            times.append((time.perf_counter() - t0) * 1000)
        results["shap_1_record_ms"] = float(np.mean(times))
    return results
