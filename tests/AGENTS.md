# AGENTS.md — Tests Directory

## 🦾 Testing Principles

- Each metric MUST have tests for both functional and module APIs
- File naming: `tests/<domain>/test_<metric>.py`

## 🚥 Test Class Patterns

- Use shared class: `from tests.helpers import MetricTester`
- Parametrize tests for all relevant cases: dtypes, device, batch size, input shape
- Use `@pytest.mark.parametrize("ddp", [True, False])` for DDP/single-device

## 📦 Test Inputs and Coverage

- Tests MUST cover:
  - Normal, edge, and error-throwing cases
  - State reset, accumulation, compute-after-multiple-updates
  - Integration with DDP (if supported by metric)
- Prefer synthetic test data but MAY use fixed random seeds for reproducibility

## 🚨 Error Assertion

- Always handle intentional assertion errors: `with pytest.raises(ExceptionType): ...`
- Include tests for device/dtype mismatch, shape mismatch, and missing dependencies

## 📄 Naming and Lint Rules

- Test naming: `test_<function>_<aspect>`
- Follow lint: `black`, `isort`, `flake8`
