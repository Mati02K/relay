# Python Style Guide

## Formatter and Linter

Use Ruff.

```bash
.venv/bin/ruff check src
.venv/bin/ruff format --check src
```

Project config is in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I"]
```

## Naming

The current codebase is mixed because older modules use camelCase and newer
runtime/telemetry modules use snake_case. Match the file you are editing.

Guidelines:

- New modules should use standard Python `snake_case` for functions, methods, and variables.
- Existing camelCase FastAPI handlers or legacy membership/network methods may stay camelCase.
- Classes use `PascalCase`.
- Module constants use `UPPER_SNAKE_CASE`.
- Do not rename public functions only for style unless the task asks for a migration.

## Type Hints

Every `def` and `async def` should have parameter and return annotations.

Use focused mypy checks for the edited area:

```bash
.venv/bin/mypy src/relay src/coordinator src/telemetry src/worker/daemon.py src/worker/main.py src/worker/inference
```

Avoid `Any` unless the boundary really is untyped JSON, FastAPI request data, or external metadata.

## Async and I/O

Use async APIs in async paths:

- HTTP: `httpx.AsyncClient`
- FastAPI handlers: `async def`
- gRPC client calls: `grpc.aio`

Do not add blocking network or subprocess waits inside an async request path unless wrapped appropriately.

## Data Models

Use Pydantic models for persistent config and cross-layer schemas:

- `src/relay/config.py`
- `src/telemetry/schemas.py`
- FastAPI response models where practical

OpenAI-compatible request bodies and membership metadata are partly dynamic JSON,
so carefully validate raw `dict[str, object]` at those boundaries.

## Docstrings and Comments

Docstrings should explain purpose and important contracts. Short one-line
docstrings are fine for simple helpers. Multi-line docstrings are acceptable
when the function encodes scheduling math, telemetry meaning, lifecycle rules,
or non-obvious behavior.

Use comments sparingly. Add comments for:

- scheduling formula terms
- process lifecycle ordering
- telemetry source boundaries
- compatibility or workaround details

Do not add comments that merely restate the code.

## Source Hygiene

- No commented-out code.
- No emojis in source files or docs unless explicitly requested.
- Do not commit generated `__pycache__`, `.mypy_cache`, `.ruff_cache`, or `.pytest_cache`.
- Do not silently replace the scheduler formula with a different heuristic.
- Do not reintroduce the removed local Python membership service as the default path.
