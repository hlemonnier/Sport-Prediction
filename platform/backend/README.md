# Sport Prediction Backend (Bun)

Local Bun API for orchestrating sport research pipelines.

## Install & Run

```bash
cd platform/backend
bun install
bun run dev
```

Production style run:

```bash
cd platform/backend
bun run start
```

Env options:
- `PORT` (default `4000`)
- `REPO_ROOT` (defaults to detected repository root)
- `DATABASE_URL` (required, Supabase Postgres URI)
  - Example: `postgresql://postgres:<password>@db.<project-ref>.supabase.co:5432/postgres?sslmode=require`

## Notes
- Runs metadata and sweep history are stored in Supabase Postgres.
- Run artifacts (`config.json`, `result.json`, logs) are stored in `platform/backend/data/`.
- The backend expects Python pipelines in each project `Python/` folder.
- On startup, the backend enables RLS and revokes `anon`/`authenticated` table privileges on `runs`, `sweeps`, `sweep_runs`, and `user_preferences`.
