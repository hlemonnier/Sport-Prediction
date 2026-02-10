# Sport Prediction Backend

Local Rust API for orchestrating sport research pipelines.

## Run

```bash
cd platform/backend
cargo run
```

Env options:
- `PORT` (default 4000)
- `REPO_ROOT` (defaults to current directory)
- `DATABASE_URL` (required, Supabase Postgres URI)
  - Direct connection example: `postgresql://postgres:<password>@db.<project-ref>.supabase.co:5432/postgres?sslmode=require`

## Notes
- Runs metadata and sweep history are stored in Supabase Postgres.
- Run artefacts (`config.json`, `result.json`, logs) are still stored in `platform/backend/data/`.
- The backend expects Python pipelines to live in their project `Python/` folders.
- On startup, the backend enforces RLS and revokes `anon`/`authenticated` table privileges on `runs`, `sweeps`, and `sweep_runs`.
