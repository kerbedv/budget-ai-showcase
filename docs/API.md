# 📑 API Reference — Budget-AI Showcase

This backend exposes a small set of clean endpoints to demonstrate **ML models in production**.

---

## 🔍 Health & Schema

### `GET /health`
Quick health check.

**Response**
```json
{ "status": "ok" }
GET /schema
High-level input/output schema for v2 endpoints.

Response

{
  "version": "v2",
  "predict_request": { "data": "{feature_name: value, ...}" },
  "forecast_request": {
    "start": "YYYY-MM-DD",
    "end": "YYYY-MM-DD",
    "history": "[{ds,y}] (optional)"
  }
}
🤖 Predictions
POST /v2/predict
Run a prediction with XGBoost + KMeans models.

Request

{
  "data": {
    "Income": 3000,
    "Savings_Rate": 0.22,
    "Weekend_Percentage": 0.3,
    "Food_Percentage": 0.12,
    "Credit_Rate": 0.1,
    "Housing_Rate": 0.28
    // Behavior_Cluster omitted — server computes via KMeans
  }
}
Response

  "version": "v2",
  "model": "xgb+kmeans",
  "outputs": {
    "Predicted_Spending_Rate": 0.3175,
    "Behavior_Cluster": 1
  }
}
📈 Forecasting
POST /v2/forecast
Generate a time-series forecast with Prophet (or demo fallback).

Request

{
  "start": "2025-08-01",
  "end":   "2025-08-07",
  "history": [
    {"ds": "2025-07-25", "y": 88.4},
    {"ds": "2025-07-26", "y": 93.0},
    {"ds": "2025-07-27", "y": 97.1}
  ]
}
Response

{
  "version": "v2",
  "model": "prophet",
  "mode": "client_history",
  "history_points": 3,
  "requested": {
    "start": "2025-08-01",
    "end": "2025-08-07",
    "horizon_days": 7
  },
  "forecast": [
    {"ds": "2025-08-01", "yhat": 101.2, "yhat_lower": 94.8, "yhat_upper": 107.6},
    {"ds": "2025-08-02", "yhat": 102.9, "yhat_lower": 96.1, "yhat_upper": 109.7},
    {"ds": "2025-08-03", "yhat": 103.7, "yhat_lower": 96.9, "yhat_upper": 110.5},
    ...
  ]
}
⚠️ Common Errors
400 Bad Request

invalid_request → payload doesn’t match schema

bad_history → not enough history points for Prophet (need ≥30 for full forecast)

404 Not Found

Wrong path (try /v2/predict not /predict)

401 Unauthorized

If auth enabled, include X-Firebase-Token: <idToken> header

🧪 Curl Examples
bash
Copy code
# Health
curl http://localhost:8000/health

# Schema
curl http://localhost:8000/schema

# Predict
curl -X POST http://localhost:8000/v2/predict \
  -H "Content-Type: application/json" \
  -d '{"data":{"Income":3000,"Savings_Rate":0.2,"Weekend_Percentage":0.3}}'

# Forecast (with short history)
curl -X POST http://localhost:8000/v2/forecast \
  -H "Content-Type: application/json" \
  -d '{"start":"2025-08-01","end":"2025-08-07","history":[{"ds":"2025-07-25","y":88.4},{"ds":"2025-07-26","y":93.0}]}' 