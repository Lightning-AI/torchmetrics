# AGENTS.md — Functional Metrics (torchmetrics/functional/)

## 🏃 Functional Metrics Usage

- Each metric available as a function: `torchmetrics.functional.<domain>.<metric>(preds, target, ...)`
- Functional metrics MUST NOT store internal state; each call is independent

## 🔢 Input Shape

- Accept shapes: `[N]`, `[N, C]` or per-metric docstring (always documented)
- Agents MUST normalize input to match function signature

## ✅ Aggregation & Arithmetic

- For batch-wise or accumulation over multiple batches, aggregate manually, or use module-based version instead
- Some metrics support composition via arithmetic, e.g., `Combined = accuracy + precision`

## 🚦 Distributed Data Parallel (DDP)

- Functional metrics DO NOT synchronize across devices or threads
- Agents needing DDP support MUST use the module-based version (`torchmetrics.<domain>.<Metric>`)

## 📍 Location

- Place new functional metrics in: `torchmetrics/functional/<domain>/<new_metric>.py`
- DO NOT cross-import module-based logic; retain separation

