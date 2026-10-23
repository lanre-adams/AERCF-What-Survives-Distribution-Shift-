/**
 * AERCF Dashboard — paper-faithful companion visualisation.
 *
 * Paper: IJMSRT26OCT017 — "What Survives Distribution Shift?"
 * Authors: O.A. Agunloye, A. Anana, S. Mathias, E. Bakare
 *
 * Tabs:
 *   Overview     — system status, dataset info, quick-start
 *   E1 Shift     — Table 5: DR/FPR per detector across 3 splits
 *   E2 EVT       — Figure 4: calibration ratio ρ per detector+α
 *   E3 SHAP      — Figure 5: global attribution bar chart + deletion faithfulness
 *   E4 Adversarial — Figure 6: PGD robustness per ε per detector
 *   E5 Adapt     — Table 6 + Figure 7: 4 policies over 60 windows
 *   E6 Contain   — Figure 8: tier assignment (monitor/step-up/isolate)
 *   E7 Cost      — Table 7: latency per component
 */
import React, { useState, useEffect, useCallback, useRef } from "react";
import { api } from "./api";

// ─────────────────────────────────────────────────────────────────────────────
// Design tokens
// ─────────────────────────────────────────────────────────────────────────────
const T = {
  bg0: "#05080f", bg1: "#0d1117", bg2: "#141b24", bg3: "#1c2535",
  border: "#1e2d42", borderHi: "#2a4060",
  txt: "#c8d8e8", txtDim: "#4a6278", txtHi: "#e8f4ff",
  cyan: "#00aaff", green: "#00e87a", red: "#ff4060",
  amber: "#ffb830", purple: "#9b6dff", teal: "#00d4c8",
  pink: "#ff4db8", navy: "#162032",
};

const FONT = "'IBM Plex Sans', system-ui, sans-serif";
const MONO = "'IBM Plex Mono', 'Fira Code', monospace";

const css = {
  app: {
    fontFamily: FONT, background: T.bg0, color: T.txt,
    minHeight: "100vh", display: "flex", flexDirection: "column",
  },
  header: {
    background: T.bg1, borderBottom: `1px solid ${T.border}`,
    padding: "12px 24px", display: "flex", alignItems: "center", gap: 16,
    position: "sticky", top: 0, zIndex: 100,
  },
  logo: { color: T.cyan, fontFamily: MONO, fontWeight: 700, fontSize: 18, letterSpacing: 2 },
  badge: (color) => ({
    background: color + "22", color, border: `1px solid ${color}55`,
    borderRadius: 4, padding: "2px 8px", fontSize: 11, fontFamily: MONO,
    whiteSpace: "nowrap",
  }),
  tabs: {
    display: "flex", gap: 0, background: T.bg1,
    borderBottom: `1px solid ${T.border}`, overflowX: "auto",
    scrollbarWidth: "none",
  },
  tab: (active) => ({
    padding: "10px 18px", cursor: "pointer", fontSize: 13, fontWeight: 500,
    borderBottom: active ? `2px solid ${T.cyan}` : "2px solid transparent",
    color: active ? T.cyan : T.txtDim,
    background: "transparent", border: "none", outline: "none",
    whiteSpace: "nowrap", fontFamily: FONT,
    transition: "color 0.15s",
  }),
  main: { flex: 1, padding: "20px 24px", maxWidth: 1400, margin: "0 auto", width: "100%" },
  card: (accent) => ({
    background: T.bg1, border: `1px solid ${accent ? accent + "44" : T.border}`,
    borderRadius: 8, padding: 16, marginBottom: 16,
    boxShadow: accent ? `0 0 16px ${accent}11` : "none",
  }),
  h2: { color: T.txtHi, fontSize: 14, fontWeight: 600, margin: "0 0 12px 0", fontFamily: MONO },
  h3: { color: T.cyan, fontSize: 12, fontWeight: 600, margin: "0 0 8px 0",
         textTransform: "uppercase", letterSpacing: 1 },
  grid: (cols) => ({
    display: "grid", gridTemplateColumns: `repeat(${cols}, 1fr)`, gap: 12,
  }),
  tbl: {
    width: "100%", borderCollapse: "collapse", fontSize: 12, fontFamily: MONO,
  },
  th: {
    padding: "6px 10px", textAlign: "left", color: T.txtDim,
    borderBottom: `1px solid ${T.border}`, fontSize: 11,
  },
  td: {
    padding: "5px 10px", borderBottom: `1px solid ${T.bg2}`,
  },
  btn: (color=T.cyan) => ({
    background: color + "18", color, border: `1px solid ${color}55`,
    borderRadius: 6, padding: "8px 18px", cursor: "pointer",
    fontSize: 13, fontFamily: FONT, fontWeight: 500,
    transition: "background 0.15s",
  }),
  note: {
    fontSize: 11, color: T.txtDim, fontFamily: MONO,
    background: T.bg2, borderRadius: 4, padding: "6px 10px", marginTop: 8,
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Utility components
// ─────────────────────────────────────────────────────────────────────────────

function Badge({ label, color = T.cyan }) {
  return <span style={css.badge(color)}>{label}</span>;
}

function KV({ k, v, mono=true }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center",
                  padding: "3px 0", borderBottom: `1px solid ${T.bg3}` }}>
      <span style={{ color: T.txtDim, fontSize: 12 }}>{k}</span>
      <span style={{ color: T.txt, fontSize: 12, fontFamily: mono ? MONO : FONT }}>{v}</span>
    </div>
  );
}

function ProgressBar({ value, color=T.cyan, height=6 }) {
  return (
    <div style={{ background: T.bg3, borderRadius: 99, overflow: "hidden", height }}>
      <div style={{ width: `${Math.max(0, Math.min(100, value * 100))}%`,
                    background: color, height, borderRadius: 99,
                    transition: "width 0.4s ease" }} />
    </div>
  );
}

function Sparkline({ values, color=T.cyan, height=40, width=200 }) {
  if (!values || values.length < 2) return null;
  const mn = Math.min(...values), mx = Math.max(...values);
  const range = mx - mn || 1;
  const pts = values.map((v,i) =>
    `${(i / (values.length-1)) * width},${height - ((v - mn) / range) * (height - 4) - 2}`
  ).join(" ");
  return (
    <svg width={width} height={height} style={{ overflow: "visible" }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5}
                strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

function HBar({ value, max=1, color=T.cyan, label, sub }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
        <span style={{ color: T.txt }}>{label}</span>
        <span style={{ color, fontFamily: MONO }}>{sub ?? (value * 100).toFixed(1) + "%"}</span>
      </div>
      <ProgressBar value={value / max} color={color} height={4} />
    </div>
  );
}

function Tile({ label, value, unit="", color=T.cyan, sub }) {
  return (
    <div style={{ background: T.bg2, borderRadius: 8, padding: "12px 16px",
                  border: `1px solid ${color}22` }}>
      <div style={{ color: T.txtDim, fontSize: 11, marginBottom: 4 }}>{label}</div>
      <div style={{ color, fontSize: 22, fontFamily: MONO, fontWeight: 700,
                    lineHeight: 1 }}>
        {value}<span style={{ fontSize: 12, marginLeft: 4, color: T.txtDim }}>{unit}</span>
      </div>
      {sub && <div style={{ color: T.txtDim, fontSize: 11, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function Alert({ msg, type="info" }) {
  const colors = { info: T.cyan, warn: T.amber, error: T.red, success: T.green };
  const c = colors[type] || T.cyan;
  return (
    <div style={{ background: c + "15", border: `1px solid ${c}44`, borderRadius: 6,
                  padding: "10px 14px", color: c, fontSize: 13, marginBottom: 12 }}>
      {msg}
    </div>
  );
}

function StatusDot({ status }) {
  const map = { ready: T.green, training: T.amber, downloading: T.cyan,
                error: T.red, idle: T.txtDim };
  return <span style={{ display: "inline-block", width: 8, height: 8,
                        borderRadius: "50%", background: map[status] || T.txtDim,
                        marginRight: 6, flexShrink: 0 }} />;
}

// ─────────────────────────────────────────────────────────────────────────────
// Detector colour map
// ─────────────────────────────────────────────────────────────────────────────
const DET_COLOR = {
  if: T.cyan, vae: T.teal, xgb: T.amber, mlp: T.purple, fusion: T.green,
};
const DET_LABEL = {
  if: "Isolation Forest", vae: "VAE", xgb: "XGBoost",
  mlp: "MLP", fusion: "Fusion",
};

// ─────────────────────────────────────────────────────────────────────────────
// Tabs
// ─────────────────────────────────────────────────────────────────────────────

function TabOverview({ status, onTrain, datasets }) {
  const trained = status?.trained;
  const [training, setTraining] = useState(false);
  const [trainErr, setTrainErr] = useState(null);
  const [epochs_vae, setEpochsVae] = useState(25);
  const [epochs_mlp, setEpochsMlp] = useState(15);

  async function handleTrain() {
    setTraining(true); setTrainErr(null);
    try { await onTrain({ epochs_vae, epochs_mlp }); }
    catch (e) { setTrainErr(e.message); }
    setTraining(false);
  }

  return (
    <div>
      {/* System status */}
      <div style={css.card(trained ? T.green : T.amber)}>
        <p style={css.h2}>System Status</p>
        <div style={css.grid(4)}>
          <Tile label="Model State" value={status?.status ?? "—"}
                color={trained ? T.green : T.amber} />
          <Tile label="Progress" value={status ? Math.round(status.progress * 100) : 0}
                unit="%" color={T.cyan} />
          <Tile label="Experiments Run"
                value={status?.experiments_available?.length ?? 0}
                unit="/7" color={T.purple} />
          <Tile label="Dataset" value="NSL-KDD" color={T.teal}
                sub={datasets ? `${datasets.partitions?.["KDDTrain+"]?.records?.toLocaleString()} train records` : "—"} />
        </div>
        {status?.message && (
          <div style={{ ...css.note, marginTop: 12 }}>{status.message}</div>
        )}
        {status?.log_tail?.length > 0 && (
          <details style={{ marginTop: 8 }}>
            <summary style={{ color: T.txtDim, fontSize: 12, cursor: "pointer" }}>Training log</summary>
            <div style={{ ...css.note, maxHeight: 120, overflowY: "auto" }}>
              {status.log_tail.map((l,i) => <div key={i}>{l}</div>)}
            </div>
          </details>
        )}
      </div>

      {/* Train controls */}
      <div style={css.card()}>
        <p style={css.h2}>Train AERCF (Paper Experiment Suite)</p>
        <p style={{ color: T.txtDim, fontSize: 13, margin: "0 0 12px 0" }}>
          Trains all detectors on NSL-KDD (downloads automatically) then runs E1–E7.
          On 2 vCPUs this takes ~3–5 min at default epochs.
        </p>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 12 }}>
          <label style={{ color: T.txtDim, fontSize: 13 }}>
            VAE epochs (paper: 25)
            <input type="number" min={1} max={100} value={epochs_vae}
                   onChange={e => setEpochsVae(+e.target.value)}
                   style={{ marginLeft: 8, width: 60, background: T.bg2, color: T.txt,
                            border: `1px solid ${T.border}`, borderRadius: 4, padding: "2px 6px" }} />
          </label>
          <label style={{ color: T.txtDim, fontSize: 13 }}>
            MLP epochs (paper: 15)
            <input type="number" min={1} max={100} value={epochs_mlp}
                   onChange={e => setEpochsMlp(+e.target.value)}
                   style={{ marginLeft: 8, width: 60, background: T.bg2, color: T.txt,
                            border: `1px solid ${T.border}`, borderRadius: 4, padding: "2px 6px" }} />
          </label>
        </div>
        {trainErr && <Alert msg={trainErr} type="error" />}
        <button style={css.btn(trained ? T.amber : T.green)}
                disabled={training || ["downloading","training"].includes(status?.status)}
                onClick={handleTrain}>
          {training ? "Starting…" : trained ? "Re-train" : "▶ Train & Run All Experiments"}
        </button>
        {status && ["downloading","training"].includes(status.status) && (
          <div style={{ marginTop: 10 }}>
            <ProgressBar value={status.progress} />
          </div>
        )}
      </div>

      {/* Architecture */}
      <div style={css.card()}>
        <p style={css.h2}>AERCF Architecture (Figure 1)</p>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          <div>
            {[
              ["L1 — Sense", "NetFlow / NSL-KDD 41 features → 121 after log-transform + OHE", T.cyan],
              ["L2 — Detect", "IF (200) + VAE (121-64-32-8) + XGBoost (300/6) + MLP (128-64) → Stacked logistic fusion", T.teal],
              ["L3 — Calibrate + Explain", "EVT/GPD threshold at α∈{1%,0.5%,0.1%} + TreeSHAP attributions (exact additivity)", T.green],
              ["L4 — Decide + Contain", "Tiered lattice: monitor < step-up < isolate; 4 adaptation policies; corroboration rule", T.amber],
            ].map(([layer, desc, col]) => (
              <div key={layer} style={{ marginBottom: 10 }}>
                <div style={{ color: col, fontSize: 12, fontWeight: 700, fontFamily: MONO }}>{layer}</div>
                <div style={{ color: T.txtDim, fontSize: 12, marginTop: 2 }}>{desc}</div>
              </div>
            ))}
          </div>
          <div>
            <p style={css.h3}>Key results (Table 5)</p>
            {[
              ["XGBoost (best)", "In-distribution AUROC 1.000, DR 99.98% at 1% FPR"],
              ["XGBoost (shift)", "KDDTest+ DR 81.3%, FPR 3.85%; Test-21 DR 75.3%"],
              ["VAE (novel attacks)", "Lost only 3.5 pp under novelty vs 13.3 pp for XGBoost"],
              ["Fusion", "KDDTest+ AUROC 0.958 — worse than XGBoost alone"],
              ["EVT calibration", "ρ=1.0 in-dist; exceeded 29× on KDDTest+ at α=0.1%"],
              ["SHAP", "Exact additivity (error < 9.5×10⁻⁶); top-3 flip 53.7%"],
              ["PGD adv training", "DR raised 49.8%→63.1% under PGD ε=0.1"],
              ["Fixed threshold", "Best policy: mean DR 77.2%"],
            ].map(([k,v]) => <KV key={k} k={k} v={v} mono={false} />)}
          </div>
        </div>
      </div>

      {/* Dataset */}
      {datasets && (
        <div style={css.card()}>
          <p style={css.h2}>NSL-KDD Partitions (Table 3)</p>
          <table style={css.tbl}>
            <thead>
              <tr>{["Partition","Records","Benign","Attack","Notes"].map(h =>
                <th key={h} style={css.th}>{h}</th>)}</tr>
            </thead>
            <tbody>
              {Object.entries(datasets.partitions).map(([k, v]) => (
                <tr key={k}>
                  <td style={{ ...css.td, color: T.cyan, fontFamily: MONO }}>{k}</td>
                  <td style={css.td}>{v.records?.toLocaleString()}</td>
                  <td style={{ ...css.td, color: T.green }}>{v.benign?.toLocaleString()}</td>
                  <td style={{ ...css.td, color: T.red }}>{v.attack?.toLocaleString()}</td>
                  <td style={{ ...css.td, color: T.txtDim, fontSize: 11 }}>{v.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={css.note}>Features: {datasets.features?.raw} raw → {datasets.features?.engineered} engineered · {datasets.features?.encoding}</div>
        </div>
      )}
    </div>
  );
}

function fmt(v, pct=true) {
  if (v == null) return "—";
  return pct ? (v * 100).toFixed(1) + "%" : v.toFixed(4);
}

function TabE1({ data }) {
  if (!data) return <Alert msg="Run /train to generate E1 results (Table 5)." type="warn" />;
  const splits = ["validation","kddtest_plus","kddtest_21"];
  const splitLabels = { validation: "Validation (in-dist.)", kddtest_plus: "KDDTest+", kddtest_21: "KDDTest-21" };
  const dets = ["if","vae","xgb","mlp","fusion"];

  return (
    <div>
      <div style={css.card(T.cyan)}>
        <p style={css.h2}>E1 — Detection Under Distribution Shift (Table 5)</p>
        <p style={{ color: T.txtDim, fontSize: 13, margin: "0 0 12px 0" }}>
          DR and FPR at the 1% FPR EVT threshold, per detector, across three data splits.
          <b style={{ color: T.amber }}> Finding (C3): calibration fails under shift — FPR rises 3–16×.</b>
        </p>
        {splits.map(split => (
          <div key={split} style={{ marginBottom: 20 }}>
            <p style={css.h3}>{splitLabels[split]}</p>
            <table style={css.tbl}>
              <thead>
                <tr>
                  <th style={css.th}>Detector</th>
                  <th style={css.th}>AUROC</th>
                  <th style={css.th}>Detection Rate</th>
                  <th style={css.th}>False-Positive Rate</th>
                  <th style={css.th}>F1</th>
                  <th style={css.th}>Precision</th>
                </tr>
              </thead>
              <tbody>
                {dets.map(det => {
                  const m = data[split]?.[det];
                  if (!m) return null;
                  const drColor = m.dr >= 0.8 ? T.green : m.dr >= 0.5 ? T.amber : T.red;
                  const fprColor = m.fpr <= 0.01 ? T.green : m.fpr <= 0.05 ? T.amber : T.red;
                  return (
                    <tr key={det}>
                      <td style={{ ...css.td, color: DET_COLOR[det], fontWeight: 600 }}>{DET_LABEL[det]}</td>
                      <td style={{ ...css.td, fontFamily: MONO }}>{fmt(m.auroc, false)}</td>
                      <td style={{ ...css.td, color: drColor, fontFamily: MONO }}>{fmt(m.dr)}</td>
                      <td style={{ ...css.td, color: fprColor, fontFamily: MONO }}>{fmt(m.fpr)}</td>
                      <td style={{ ...css.td, fontFamily: MONO }}>{fmt(m.f1, false)}</td>
                      <td style={{ ...css.td, fontFamily: MONO }}>{fmt(m.precision, false)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ))}
        <div style={css.note}>
          Paper reference: XGBoost reaches DR 81.3%/FPR 3.85% on KDDTest+; VAE loses only 3.5 pp on novel attacks vs 13.3 pp for XGBoost.
          Fusion (stacked) underperforms its best member on KDDTest+ (C3 finding).
        </div>
      </div>
    </div>
  );
}

function TabE2({ data }) {
  if (!data) return <Alert msg="Run /train to generate E2 results." type="warn" />;
  const dets = ["if","vae","xgb","fusion"];
  const alphas = Object.keys(data).sort((a,b) => parseFloat(b)-parseFloat(a));

  return (
    <div style={css.card(T.teal)}>
      <p style={css.h2}>E2 — EVT Calibration Ratio ρ = Realised FPR / Target α (Figure 4)</p>
      <p style={{ color: T.txtDim, fontSize: 13, margin: "0 0 12px 0" }}>
        ρ = 1 is perfect calibration. Paper finding: EVT meets targets on validation,
        but exceeds them <b style={{ color: T.red }}>29×</b> on KDDTest+ at α=0.1%.
      </p>
      <table style={css.tbl}>
        <thead>
          <tr>
            <th style={css.th}>Detector</th>
            {alphas.map(a => (
              <React.Fragment key={a}>
                <th style={css.th}>α={a} θ</th>
                <th style={css.th}>ρ (val)</th>
                <th style={css.th}>ρ (test+)</th>
              </React.Fragment>
            ))}
          </tr>
        </thead>
        <tbody>
          {dets.map(det => (
            <tr key={det}>
              <td style={{ ...css.td, color: DET_COLOR[det], fontWeight: 600 }}>{DET_LABEL[det]}</td>
              {alphas.map(a => {
                const d = data[a]?.[det];
                if (!d) return <React.Fragment key={a}><td style={css.td}>—</td><td style={css.td}>—</td><td style={css.td}>—</td></React.Fragment>;
                const rhoVal  = d.rho_val != null ? d.rho_val.toFixed(2) : "—";
                const rhoTest = d.rho_test != null ? d.rho_test.toFixed(2) : "—";
                const rhoColor = d.rho_test > 2 ? T.red : d.rho_test > 1.5 ? T.amber : T.green;
                return (
                  <React.Fragment key={a}>
                    <td style={{ ...css.td, fontFamily: MONO, color: T.txtDim }}>{d.threshold?.toFixed(4)}</td>
                    <td style={{ ...css.td, fontFamily: MONO }}>{rhoVal}</td>
                    <td style={{ ...css.td, fontFamily: MONO, color: rhoColor }}>{rhoTest}</td>
                  </React.Fragment>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div style={css.note}>
        GPD shape ξ: paper reports ξ≈−0.20 for IF (bounded tail), +0.15 for VAE, +0.36 for Fusion (heavy logit tail).
        Detectors with lightest tails were also best calibrated under shift.
      </div>
    </div>
  );
}

function TabE3({ data }) {
  if (!data) return <Alert msg="Run /train to generate E3 results." type="warn" />;
  if (data.error) return <Alert msg={data.error} type="error" />;
  const top = data.top_features || [];
  const vals = data.top_mean_abs_shap || [];
  const maxV = Math.max(...vals, 0.01);

  return (
    <div>
      <div style={css.card(T.green)}>
        <p style={css.h2}>E3 — SHAP Explanation Faithfulness (Figure 5)</p>
        <div style={css.grid(3)}>
          <Tile label="Records explained" value={data.n_explained?.toLocaleString() ?? "—"} color={T.cyan} />
          <Tile label="Additivity error (max)" value={data.max_additivity_error != null ? data.max_additivity_error.toExponential(2) : "—"} color={T.green}
                sub="paper: < 9.5×10⁻⁶" />
          <Tile label="Top-3 deletion flip" value="53.7%" color={T.amber} sub="paper result (faithfulness)" />
        </div>
      </div>
      <div style={css.card()}>
        <p style={css.h2}>Global Attribution — Top 10 XGBoost Features (Figure 5a)</p>
        {top.map((feat, i) => (
          <HBar key={feat} value={vals[i]} max={maxV}
                label={feat} color={T.green}
                sub={vals[i]?.toFixed(3) + " log-odds"} />
        ))}
        <div style={css.note}>
          TreeSHAP satisfies exact additivity: Σφᵢ + φ₀ = model log-odds output.
          AERCF decisions rest on a handful of volume and connection-rate features (src_bytes, count, dst_bytes dominant).
        </div>
      </div>
    </div>
  );
}

function TabE4({ data }) {
  if (!data) return <Alert msg="Run /train to generate E4 results." type="warn" />;
  const epsList = Object.keys(data).map(parseFloat).sort((a,b) => a-b);
  const [eps, setEps] = useState(epsList[2] ?? 0.05);
  const dets = ["if","vae","xgb","mlp","fusion"];
  const d = data[eps];

  return (
    <div>
      <div style={css.card(T.red)}>
        <p style={css.h2}>E4 — Adversarial Robustness: L∞ PGD White-Box (Figure 6)</p>
        <p style={{ color: T.txtDim, fontSize: 13, margin: "0 0 12px 0" }}>
          PGD-20 crafted on standard MLP, transferred to all detectors.
          Paper: PGD training raises MLP DR 49.8%→63.1% at ε=0.1; OR-ensemble XGBoost+VAE kept 79.0%.
        </p>
        <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap" }}>
          {epsList.map(e => (
            <button key={e} style={css.btn(e === eps ? T.red : T.txtDim)}
                    onClick={() => setEps(e)}>ε={e}</button>
          ))}
        </div>
        {d && (
          <table style={css.tbl}>
            <thead>
              <tr>
                <th style={css.th}>Detector</th>
                <th style={css.th}>Clean DR</th>
                <th style={css.th}>Adv (std) DR</th>
                <th style={css.th}>Adv (PGD-trained) DR</th>
                <th style={css.th}>Δ PGD-trained</th>
              </tr>
            </thead>
            <tbody>
              {dets.map(det => {
                const c = d[det]?.clean?.dr;
                const s = d[det]?.adv_std?.dr;
                const p = d[det]?.adv_pgd?.dr;
                const delta = p != null && s != null ? p - s : null;
                return (
                  <tr key={det}>
                    <td style={{ ...css.td, color: DET_COLOR[det], fontWeight: 600 }}>{DET_LABEL[det]}</td>
                    <td style={{ ...css.td, color: T.green, fontFamily: MONO }}>{fmt(c)}</td>
                    <td style={{ ...css.td, color: T.red, fontFamily: MONO }}>{fmt(s)}</td>
                    <td style={{ ...css.td, color: T.amber, fontFamily: MONO }}>{fmt(p)}</td>
                    <td style={{ ...css.td, color: delta >= 0 ? T.green : T.red, fontFamily: MONO }}>
                      {delta != null ? (delta >= 0 ? "+" : "") + (delta * 100).toFixed(1) + "%" : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
        <div style={css.note}>
          Transferred attacks cut XGBoost from 81.3%→61.0% but raised VAE from 59.9%→63.7%.
          OR-combination XGBoost+VAE at jointly-calibrated 1% FPR thresholds: 78.9%→79.0% (−2.2 clean cost).
        </div>
      </div>
    </div>
  );
}

function TabE5({ data }) {
  if (!data) return <Alert msg="Run /train to generate E5 results." type="warn" />;
  const summary = data.summary || {};
  const windows = data.windows || {};
  const policies = Object.keys(summary);
  const PCOL = { fixed: T.cyan, repository: T.green, pi_audit: T.amber, alert_budget: T.red };
  const [metric, setMetric] = useState("dr");
  const w = windows[policies[0]] || [];
  const xLabels = w.map(d => d.window);

  return (
    <div>
      <div style={css.card(T.amber)}>
        <p style={css.h2}>E5 — Threshold Adaptation Over 60 Windows (Table 6 + Figure 7)</p>
        <p style={{ color: T.txtDim, fontSize: 13, margin: "0 0 12px 0" }}>
          Attack prevalence rises 5%→60% then falls. Windows of 1,000 fused-score records.
          Finding (C3): <b style={{ color: T.red }}>naive adaptation underperforms a fixed threshold.</b>
        </p>
        <div style={css.grid(4)}>
          {policies.map(p => {
            const s = summary[p];
            return (
              <div key={p} style={{ background: T.bg2, borderRadius: 6, padding: 10,
                                    border: `1px solid ${PCOL[p]}33` }}>
                <div style={{ color: PCOL[p], fontSize: 12, fontWeight: 700, marginBottom: 6 }}>
                  {p.replace("_"," ")}
                </div>
                <KV k="Mean DR" v={fmt(s.mean_dr)} />
                <KV k="Min DR"  v={fmt(s.min_dr)} />
                <KV k="Mean FPR" v={fmt(s.mean_fpr)} />
                <KV k="Max FPR" v={fmt(s.max_fpr)} />
              </div>
            );
          })}
        </div>
      </div>
      <div style={css.card()}>
        <p style={css.h2}>Per-Window {metric.toUpperCase()} Trajectory</p>
        <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
          {["dr","fpr","theta"].map(m => (
            <button key={m} style={css.btn(m === metric ? T.amber : T.txtDim)}
                    onClick={() => setMetric(m)}>{m.toUpperCase()}</button>
          ))}
        </div>
        <div style={{ overflowX: "auto" }}>
          <svg width={Math.max(600, xLabels.length * 9)} height={160}>
            {policies.map(p => {
              const vals = (windows[p] || []).map(d => d[metric] ?? d.theta ?? 0);
              if (!vals.length) return null;
              const mn = Math.min(...vals), mx = Math.max(...vals);
              const range = mx - mn || 1;
              const pts = vals.map((v, i) =>
                `${20 + i * 9},${140 - ((v - mn) / range) * 120}`
              ).join(" ");
              return <polyline key={p} points={pts} fill="none" stroke={PCOL[p]}
                               strokeWidth={1.5} strokeLinejoin="round" opacity={0.85} />;
            })}
          </svg>
        </div>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginTop: 8 }}>
          {policies.map(p => (
            <span key={p} style={{ color: PCOL[p], fontSize: 12 }}>
              ■ {p.replace("_"," ")}
            </span>
          ))}
        </div>
        <div style={css.note}>
          PI controller + alert-budget both drive detection toward zero when prevalence rises (adversarially exploitable).
          Fixed threshold achieves best mean DR (77.2%, Table 6).
        </div>
      </div>
    </div>
  );
}

function TabE6({ data }) {
  if (!data) return <Alert msg="Run /train to generate E6 results." type="warn" />;
  const tiers = ["monitor","step_up","isolate"];
  const tierColors = { monitor: T.cyan, step_up: T.amber, isolate: T.red };

  return (
    <div>
      <div style={css.card(T.purple)}>
        <p style={css.h2}>E6 — Containment Tier Operating Points (Figure 8)</p>
        <p style={{ color: T.txtDim, fontSize: 13, margin: "0 0 12px 0" }}>
          Tier thresholds at α₁=1% (step-up) and α₂=0.1% (isolate).
          Corroboration rule: isolate only if fusion ≥ θ₂ AND (IF or VAE ≥ their 1% threshold).
        </p>
        <div style={css.grid(2)}>
          {["attack","benign"].map(cls => (
            <div key={cls} style={{ background: T.bg2, borderRadius: 6, padding: 12 }}>
              <p style={css.h3}>{cls} records (n={data[cls]?.n?.toLocaleString()})</p>
              {tiers.map(t => (
                <HBar key={t} label={t.replace("_","-")}
                      value={data[cls]?.[t] ?? 0}
                      color={tierColors[t]} />
              ))}
            </div>
          ))}
        </div>
        <div style={{ ...css.grid(3), marginTop: 12 }}>
          <Tile label="θ₁ (step-up)" value={data.theta1?.toFixed(4) ?? "—"} color={T.amber} />
          <Tile label="θ₂ (isolate)" value={data.theta2?.toFixed(4) ?? "—"} color={T.red} />
          <Tile label="Corroborated isolate (attack)" value={fmt(data.corroborated_isolate_attack_rate)} color={T.green} />
        </div>
        <div style={css.note}>
          Paper: 66.7% of attacks reach isolate; 2.78% of benign also reach it — 28× the 0.1% target.
          Corroboration halves benign isolation to 1.35% while retaining 54.1% of attacks.
        </div>
      </div>
    </div>
  );
}

function TabE7({ data }) {
  if (!data) return <Alert msg="Run /train to generate E7 results." type="warn" />;
  return (
    <div>
      <div style={css.card(T.teal)}>
        <p style={css.h2}>E7 — Inference Cost (Table 7)</p>
        <p style={{ color: T.txtDim, fontSize: 13, margin: "0 0 12px 0" }}>
          Measured on this machine. Paper: 2 vCPUs, batch-1024 full pipeline ≈0.030 ms/record (≈33 000 rec/s).
        </p>
        <table style={css.tbl}>
          <thead>
            <tr>
              <th style={css.th}>Batch size</th>
              <th style={css.th}>Mean latency (ms/record)</th>
              <th style={css.th}>Std</th>
              <th style={css.th}>Records/sec</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(data)
              .filter(([k]) => k.startsWith("batch"))
              .map(([k, v]) => {
                const batch = parseInt(k.split("_")[1]);
                const rps = v.mean_ms_per_record > 0 ? Math.round(1000 / v.mean_ms_per_record) : "—";
                return (
                  <tr key={k}>
                    <td style={{ ...css.td, color: T.cyan, fontFamily: MONO }}>{batch.toLocaleString()}</td>
                    <td style={{ ...css.td, fontFamily: MONO }}>{v.mean_ms_per_record.toFixed(3)}</td>
                    <td style={{ ...css.td, fontFamily: MONO }}>{v.std_ms_per_record.toFixed(3)}</td>
                    <td style={{ ...css.td, color: T.green, fontFamily: MONO }}>{typeof rps === "number" ? rps.toLocaleString() : rps}</td>
                  </tr>
                );
              })}
          </tbody>
        </table>
        {data.shap_1_record_ms != null && (
          <div style={{ marginTop: 10 }}>
            <KV k="TreeSHAP (1 record)" v={data.shap_1_record_ms.toFixed(2) + " ms"} />
          </div>
        )}
        <div style={css.note}>
          Paper Table 7: IF batch-1 = 11.37 ms (fixed call overhead); batch-1024 = 0.018 ms.
          VAE: 0.58 / 0.009 ms. XGBoost: 0.32 / 0.002 ms. TreeSHAP: 0.90 ms.
          Full pipeline batch-1024 ≈ 33 000 records/s on 2 vCPUs.
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────────────────────────────────────

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "e1",       label: "E1 Shift" },
  { id: "e2",       label: "E2 EVT" },
  { id: "e3",       label: "E3 SHAP" },
  { id: "e4",       label: "E4 Adversarial" },
  { id: "e5",       label: "E5 Adaptation" },
  { id: "e6",       label: "E6 Containment" },
  { id: "e7",       label: "E7 Cost" },
];

export default function App() {
  const [tab, setTab]       = useState("overview");
  const [status, setStatus] = useState(null);
  const [exp, setExp]       = useState({});
  const [datasets, setDatasets] = useState(null);
  const pollRef = useRef(null);

  const fetchStatus = useCallback(async () => {
    try {
      const s = await api.status();
      setStatus(s);
      if (s.experiments_available?.length && !Object.keys(exp).length) {
        const e = await api.experiments().catch(() => null);
        if (e) setExp(e);
      }
    } catch (_) {}
  }, [exp]);

  useEffect(() => {
    api.datasets().then(setDatasets).catch(() => {});
    fetchStatus();
    pollRef.current = setInterval(fetchStatus, 3000);
    return () => clearInterval(pollRef.current);
  }, [fetchStatus]);

  async function handleTrain(params) {
    await api.train(params);
  }

  const tabContent = {
    overview:  <TabOverview status={status} onTrain={handleTrain} datasets={datasets} />,
    e1:        <TabE1 data={exp.e1} />,
    e2:        <TabE2 data={exp.e2} />,
    e3:        <TabE3 data={exp.e3} />,
    e4:        <TabE4 data={exp.e4} />,
    e5:        <TabE5 data={exp.e5} />,
    e6:        <TabE6 data={exp.e6} />,
    e7:        <TabE7 data={exp.e7} />,
  };

  return (
    <div style={css.app}>
      {/* Header */}
      <div style={css.header}>
        <span style={css.logo}>AERCF</span>
        <Badge label="IJMSRT26OCT017" color={T.cyan} />
        <Badge label="NSL-KDD" color={T.teal} />
        <Badge label="E1–E7" color={T.purple} />
        <div style={{ flex: 1 }} />
        <StatusDot status={status?.status ?? "idle"} />
        <span style={{ color: T.txtDim, fontSize: 12 }}>
          {status?.status ?? "idle"}
          {status?.status === "training" && ` ${Math.round(status.progress * 100)}%`}
        </span>
      </div>

      {/* Tabs */}
      <div style={css.tabs}>
        {TABS.map(t => (
          <button key={t.id} style={css.tab(tab === t.id)} onClick={() => setTab(t.id)}>
            {t.label}
            {t.id !== "overview" && exp[t.id] &&
              <span style={{ marginLeft: 4, color: T.green, fontSize: 9 }}>●</span>}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={css.main}>
        {tabContent[tab]}
      </div>

      {/* Footer */}
      <div style={{ background: T.bg1, borderTop: `1px solid ${T.border}`,
                    padding: "8px 24px", fontSize: 11, color: T.txtDim,
                    display: "flex", gap: 16, flexWrap: "wrap" }}>
        <span>Agunloye et al. · IJMSRT26OCT017 · October 2026</span>
        <span>DOI: 10.67317/IJMSRT26OCT017</span>
        <span style={{ marginLeft: "auto" }}>
          <span style={{ color: T.green }}>●</span> API: {status?.status ?? "connecting"}
        </span>
      </div>
    </div>
  );
}
