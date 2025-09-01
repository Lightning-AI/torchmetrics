# AGENTS.md — Requirements and Extras

## Files-to-Extras Mapping

- Each *.txt here (without leading underscore) becomes an installable extra via setup.py.
  - Example: requirements/image.txt -> pip install torchmetrics[image]
- Files starting with "_" are internal (not exposed as extras).

## Base Requirements

- requirements/base.txt sets core constraints (e.g., torch >=2.0.0). Do not violate these in docs or AGENTS files.

## Adding a New Extra

- Create <name>.txt, add pinned/upper-bounded deps consistent with project policy.
- Avoid URL-based requirements; see setup.py parser behavior.
- Run: python setup.py --name (implicit during packaging) to verify extras resolve.

## Dev and All

- "all" extra is auto-created from available extras (excluding internal).
- "dev" = "all" + tests, useful for contributors: pip install -e .[dev]
