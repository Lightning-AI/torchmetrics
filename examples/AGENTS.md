# 🤖 AGENTS.md for examples/

## 🧪 Purpose

This folder contains runnable demos of TorchMetrics in action. Agents can use these to:

- Validate metric implementations
- Benchmark performance
- Generate plots for documentation

## 🚀 How to Run

- Install all extras (includes plotting dependencies):
  pip install torchmetrics[all]

- Run any example script:
  python examples/<script>.py

## 📊 Plotting

Examples that call `.plot()` require matplotlib. Ensure it’s installed.

## 🧼 Formatting

Use Black and Ruff on the examples:
black examples/
ruff check --fix examples/

## 🧠 Agent Tips

- Extract common metric usage patterns for docs.
- Auto-generate demo plots for READMEs.
- Compare outputs against synthetic test cases.
