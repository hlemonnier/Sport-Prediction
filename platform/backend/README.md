# Sport Prediction Backend

Local Rust API for orchestrating sport research pipelines.

## Run

```bash
cd platform/backend
cargo run
```

Env options (optional):
- `PORT` (default 4000)
- `REPO_ROOT` (defaults to current directory)

## Notes
- Runs are stored in `platform/backend/data/` (SQLite + artefacts).
- The backend expects Python pipelines to live in their project `Python/` folders.
