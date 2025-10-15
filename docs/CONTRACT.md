# Data Contract, KPIs, and SLOs

defines what success looks like (KPIs/SLOs) and the data contract for my MNIST E2E pipeline project.

## Key Performance Indicators (KPIs)

| Metric | Target | Why |
|---|---|---|
| **Test Accuracy** | **≥ 99.0%** | MNIST is easy; this keeps us honest. |
| **Robustness Drop** | **≤ 0.2% absolute** | Ensures the model doesn’t collapse with minor disruptions to the data. |

## Service Level Objectives (SLOs)

| Metric | Target | Definition |
|---|---|---|
| **p95 Latency** | **≤ 50 ms** | 95% of prediction requests complete at or under this time. |
| **5xx Error Rate** | **< 0.5% per 24h** | Percentage of requests returning HTTP 5xx (server errors). |
| **/health Uptime** | **≥ 99.5% monthly** | Percentage of time the health endpoint is reachable. |

## Data Contract (API Schemas)

### Predict — Request
Content-Type: `application/json`  
Endpoint: `POST /predict`

```json
{
  "image": "<base64 of 28x28 grayscale image>",
  "normalize": true
}
```

**Rules (validation):**
- `image` is **required** and must decode to shape **28×28×1** (grayscale).
- Pixel range must be **[0, 255]** before normalization.
- If `normalize` is `true` (default), server scales to **[0, 1]** internally.
- Missing/invalid fields → **HTTP 400**.

### Predict — Response
```json
{
  "prediction": 7,
  "model_version": "model_v12",
  "latency_ms": 12.4
}
```

Fields:
- `prediction`: integer in **{0..9}**
- `model_version`: version folder name of the live artifact
- `latency_ms`: measured wall time for the inference call

### Health — Response
Endpoint: `GET /health`
```json
{ "status": "ok", "model_version": "model_v12" }
```

### Metrics — Response
Endpoint: `GET /metrics`
```json
{
  "p95_latency": 41.8,
  "error_rate_5xx": 0.1,
  "req_count": 12345,
  "last_retrain": "2025-10-14T12:00:00Z",
  "model_version": "model_v12"
}
```

### Info — Response
Endpoint: `GET /info`
```json
{
  "prod_metrics": { "accuracy": 99.2, "robustness_drop": 0.15 },
  "data_snapshot_date": "2025-10-14"
}
```