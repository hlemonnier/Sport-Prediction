# Sport Prediction Backend (Bun)

Local Bun API for orchestrating sport research pipelines.

## Install & Run

```bash
cd apps/api
bun install
bun run dev
```

Production style run:

```bash
cd apps/api
bun run start
```

Env options:
- `PORT` (default `4000`)
- `REPO_ROOT` (defaults to detected repository root)
- `DATABASE_URL` (required, Supabase Postgres URI)
  - Example: `postgresql://postgres:<password>@db.<project-ref>.supabase.co:5432/postgres?sslmode=require`

## Notes
- Runs metadata and sweep history are stored in Supabase Postgres.
- Run metadata cache files (`config.json`, `result.json`, logs) are stored in
  `apps/api/data/` by default. Override with `SPORT_PREDICTION_API_DATA_DIR`.
- The backend expects Python pipelines in each project `Python/` folder.
- On startup, the backend enables RLS and revokes `anon`/`authenticated` table privileges on `runs`, `sweeps`, `sweep_runs`, and `user_preferences`.
