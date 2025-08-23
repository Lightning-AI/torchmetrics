# AGENTS.md — Plotting Utilities (`src/torchmetrics/utilities/`)

## 🎨 Summary

- Provides matplotlib-based plotting helpers for all major TorchMetrics classes
- Adds .plot() methods for metric modules, MetricCollection, and per-domain metrics

## 📦 Dependencies

- MUST install: `pip install matplotlib scienceplots`
- SciencePlots styles are used for publication-quality output
- Optional: LaTeX is recommended for advanced plots (see docs for installer script)

## 🧰 Key Functions

| Function                    | Description                                                   |
|-----------------------------|---------------------------------------------------------------|
| `plot_single_or_multi_val`  | Plots scalar or multi-value metric output w/ bounds/optimum   |
| `plot_confusion_matrix`     | Draws task-specific confusion matrices                        |
| `plot_curve`                | Renders ROC, precision-recall, and similar curves             |

## 🧑‍💻 Usage & Extension

- Each plotting function uses `@style_change(_style)` context for consistent style
- Always check `if not _MATPLOTLIB_AVAILABLE` —> raise informative ModuleNotFoundError
- Add new functions as needed for new metric types; follow docstring conventions

## 🧪 Testing

- Test plotting with mock data and popular metric classes
- Use pytest where possible; avoid hard assertions on figures (visual output), prefer checking for error-free run

## 🖌️ Style

- Docstrings: Google or NumPy preferred
- Follow lint: `black`, `isort`, `flake8`
- Name all functions with action-oriented verbs (`plot_*`)

