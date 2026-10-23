/**
 * AERCF API client — thin fetch wrappers for all backend endpoints.
 */
const BASE = process.env.REACT_APP_API_URL || "/api";

async function req(method, path, body) {
  const opts = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(`${BASE}${path}`, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export const api = {
  status:       ()           => req("GET",  "/status"),
  train:        (body={})    => req("POST", "/train", body),
  datasets:     ()           => req("GET",  "/datasets/info"),
  experiments:  ()           => req("GET",  "/experiments"),
  experiment:   (id)         => req("GET",  `/experiments/${id}`),
  calibration:  ()           => req("GET",  "/calibration"),
  predict:      (body)       => req("POST", "/predict", body),
  explain:      (body)       => req("POST", "/explain", body),
  attack:       (body)       => req("POST", "/attack",  body),
  adaptWindow:  (body)       => req("POST", "/adaptation/window", body),
};
