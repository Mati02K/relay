# Python Style Guide

## Formatter and Linter

Ruff. Run before every commit:
```bash
ruff check . && ruff format .
```

Config in `pyproject.toml`:
```toml
[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I"]   # N (pep8-naming) disabled — project uses camelCase
```

## Naming

| Thing | Convention | Example |
|---|---|---|
| Variables | camelCase | `nodeId`, `myAddress`, `memberList` |
| Functions / methods | camelCase | `getMyAddress()`, `electLeader()` |
| Classes | PascalCase | `EtcdMembership`, `TailscaleNetwork` |
| Module-level constants | UPPER_SNAKE_CASE | `SERVICE_TYPE`, `BROWSE_TIMEOUT_SECONDS` |

Standard Python `snake_case` is **not** used in this project.

## Type Hints

Every `def` and `async def` must be fully annotated — parameters and return type. Run `mypy src/` to verify:
```bash
mypy src/
```

Config:
```toml
[tool.mypy]
python_version = "3.11"
strict = true
```

No `Any` types unless there is no alternative.

## Async

All I/O-bound code uses `async/await`:
- gRPC calls → `grpc.aio` channel and stubs
- HTTP calls → `httpx.AsyncClient`
- FastAPI handlers → always `async def`

Never call blocking I/O directly inside an async function. If a library is sync-only, use `asyncio.run_in_executor`.

## Data Models

Use `pydantic.BaseModel` for all request/response schemas and config objects passed across layer or function boundaries. Do not pass raw `dict` between layers.

```python
class ChatRequest(BaseModel):
    messages: list[Message]
```

## Docstrings

Every `def` and `async def` gets **exactly one short docstring line** explaining what the function does — not how. This line describes the function's purpose, not a step-by-step of the implementation.

```python
async def register(self, nodeId: str, metadata: dict) -> None:
    """Register this node in the cluster with the given metadata."""
```

No multi-line docstrings. No `Args:` / `Returns:` blocks.

## What Not to Do

- No `print()` statements (use `logging` if needed)
- No commented-out code
- No emojis anywhere in source files
- No inline comments unless the logic is genuinely non-obvious to a reader (e.g. a tricky invariant or a workaround for a specific library bug)
