# F1 Platform Service

Python/FastAPI service for the F1 live and replay platform.

This service owns the near-live contract described in the implementation plan:

- OpenF1-style raw events are ingested as appendable/upsertable events.
- Reducers build compact current session state from sessions, drivers, laps,
  position, intervals, stints, pit, race-control, weather, car-data, location,
  overtake, and session-result topics.
- Sector-only lap messages update current lap and sector timing but do not
  create completed lap-chart rows or prediction refreshes until a full lap
  duration arrives.
- Replays run the same reducer path as live ingestion so historical sessions can
  be deterministic test fixtures.
- Prediction snapshots are versioned and kept as a timeline, not overwritten.
- Race prediction snapshots include expected finishing position plus a 10th to
  90th percentile finishing-position range.
- FastAPI exposes snapshots, event ingress, replay reset, and WebSocket streams.
- OpenF1 historical REST imports can be persisted as JSONL replay fixtures.
- Parsed API live events are written to the JSONL replay store before reduction,
  including duplicate or stale source messages.
- Stored replays can be started at controlled speed through the same live
  `ingest()` path used by OpenF1 events.
- Parsed live/replay events are appended to an ordered event stream. Local
  runs use an in-memory stream; `F1_PLATFORM_REDIS_URL` enables Redis Streams.
- JSONL replays, Redis stream reads, and dead-letter recovery rehydrate the same
  raw event record contract, preserving `received_at` rather than restamping it.
- Reduced sessions are projected into SQL tables for drivers, laps, stints,
  session metadata, pit stops, race control, weather samples, overtakes, final results,
  custom micro-sectors, predictions, and derived analytics. Local runs use SQLite;
  `F1_PLATFORM_DATABASE_URL` enables PostgreSQL.
- OpenF1 live credentials stay server-side behind `OpenF1TokenManager`.
- Prediction snapshots can come from a separate model service through
  `F1_PLATFORM_PREDICTION_URL`, with deterministic local fallback.

The service starts with an in-memory event stream plus a local SQLite projection
store so development works without PostgreSQL or Redis. `infra/docker-compose.yml`
adds PostgreSQL and Redis for the production storage boundary.

## Local Run

```bash
cd services/f1-platform
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn f1_platform.app:create_app --factory --reload --port 8001
```

Open:

```text
http://localhost:8001/health
http://localhost:8001/api/f1/sessions/sample-race/snapshot
ws://localhost:8001/api/f1/sessions/sample-race/stream
```

The WebSocket stream sends compact deltas plus periodic full snapshots so a
missed delta cannot permanently corrupt the browser state. Tune the snapshot
cadence with:

```text
F1_PLATFORM_WS_SNAPSHOT_INTERVAL_SECONDS=15
```

Import a historical OpenF1 session into the replay store:

```bash
curl -X POST http://localhost:8001/api/f1/openf1/import \
  -H "Content-Type: application/json" \
  -d '{"session_key":9165,"session_name":"Race","include_telemetry":false,"limit_per_topic":3000}'
```

Then open:

```text
http://localhost:8001/api/f1/sessions/9165/snapshot
```

Replay that stored session through the live reducer:

```bash
curl -X POST http://localhost:8001/api/f1/sessions/9165/replay/start \
  -H "Content-Type: application/json" \
  -d '{"speed":20,"max_delay_seconds":2}'
```

Inspect replay progress and recent raw stream records:

```text
http://localhost:8001/api/f1/sessions/9165/replay/status
http://localhost:8001/api/f1/sessions/9165/events?count=25
http://localhost:8001/api/f1/sessions/9165/projection
```

Generate post-session FastF1 engineering artifacts:

```bash
curl -X POST http://localhost:8001/api/f1/fastf1/import \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2026,
    "event": "Austria",
    "session_name": "R",
    "drivers": ["VER", "RUS"],
    "include_telemetry": true,
    "telemetry_laps_per_driver": 1,
    "distance_step_meters": 5,
    "output_format": "parquet"
  }'
```

The FastF1 import is a batch/post-session job, not the live transport. It writes:

```text
fastf1/year=.../event=.../session=.../laps.parquet
fastf1/year=.../event=.../session=.../weather.parquet
fastf1/year=.../event=.../session=.../race_control.parquet
telemetry/year=.../event=.../session=.../driver=.../lap=.../part-000.parquet
corner_metrics/year=.../event=.../session=.../driver=.../lap=.../part-000.parquet
centerline/year=.../event=.../session=.../canonical.parquet
telemetry_comparison/year=.../event=.../session=.../*.parquet
```

Telemetry artifacts are distance-aligned on the requested metre grid. The
corner metrics artifact uses local speed-minimum windows to estimate entry,
minimum and exit speed, braking duration, throttle reapplication, full-throttle
share, and corner time. The centreline is derived from a clean FastF1 telemetry
lap with X/Y coordinates, and live OpenF1 locations are projected onto that
centreline when the API can match the live session to a generated artifact.
Race micro-sectors are generated from track-progress crossings as custom
equal-distance segments. They are intentionally labelled custom micro-sectors
and should not be presented as official FIA/FOM mini-sector data.
Weather updates are retained as a bounded sample history, persisted to SQL, and
summarized as `weather_evolution_v1` for track temperature, rainfall, and wind
trend views.
Reduced-state lap timing also feeds `pace_analysis_v1` for rolling pace,
consistency and field-median views. Adjacent order gaps, DRS state, tyre age
and recent lap pace feed `battle_dashboard_v1` so the first dashboard can show
DRS trains and likely overtake windows without claiming exact car placement.

Inspect generated FastF1 artifacts through the API:

```text
http://localhost:8001/api/f1/fastf1/artifacts?session_key=fastf1:2026:austria:r
http://localhost:8001/api/f1/fastf1/artifacts/{artifact_id}/rows?limit=100
http://localhost:8001/api/f1/fastf1/engineering-summary?session_key=fastf1:2026:austria:r
```

Map an OpenF1 live/replay session to a FastF1 centreline explicitly:

```text
F1_PLATFORM_CENTERLINE_SESSION_MAP=9165=fastf1:2026:austria:r
```

Or pass the mapping when importing:

```bash
curl -X POST http://localhost:8001/api/f1/openf1/import \
  -H "Content-Type: application/json" \
  -d '{"session_key":9165,"session_name":"Race","fastf1_session_key":"fastf1:2026:austria:r"}'
```

`F1_PLATFORM_CENTERLINE_FALLBACK_TO_LATEST=1` allows the latest available
centerline artifact to be used when no exact session mapping exists. Keep it
off unless you are sure the current live session and latest centerline are the
same circuit.

For authenticated live OpenF1 work, set these on the backend process only:

```text
OPENF1_USERNAME=...
OPENF1_PASSWORD=...
```

Run the live MQTT ingestor as a separate backend process. It obtains the
OAuth2 token server-side, subscribes to OpenF1 MQTT topics, normalizes each
message into the platform event contract, and posts the event into the FastAPI
ingress endpoint:

```bash
cd services/f1-platform
OPENF1_USERNAME=... \
OPENF1_PASSWORD=... \
F1_PLATFORM_API_URL=http://127.0.0.1:8001 \
python -m f1_platform.live_ingestor
```

Optional live-ingestor environment variables:

```text
OPENF1_MQTT_BROKER=mqtt.openf1.org
OPENF1_MQTT_PORT=8883
OPENF1_MQTT_TOPICS=v1/sessions,v1/drivers,v1/laps,v1/position,v1/intervals,v1/stints,v1/pit,v1/race_control,v1/weather,v1/car_data,v1/location,v1/overtakes
OPENF1_MQTT_USERNAME=...
OPENF1_FALLBACK_SESSION_KEY=...
OPENF1_TOKEN_RECONNECT_SECONDS=3300
OPENF1_RECONNECT_DELAY_SECONDS=5
OPENF1_SINK_RETRY_ATTEMPTS=3
OPENF1_SINK_RETRY_BACKOFF_SECONDS=0.5
OPENF1_DEAD_LETTER_PATH=data/raw/f1/openf1-dead-letter.jsonl
```

After all submit retries are exhausted, failed live events are appended to the
dead-letter JSONL file. Set `OPENF1_DEAD_LETTER_PATH=off` only if you are
comfortable losing events while the API is unavailable.

Replay dead-lettered events after the API is healthy again:

```bash
cd services/f1-platform
F1_PLATFORM_API_URL=http://127.0.0.1:8001 \
python -m f1_platform.dead_letter replay \
  --path data/raw/f1/openf1-dead-letter.jsonl
```

Successful rows are compacted out of the spool. Rows that still fail submit stay
in the file with `lastReplayError`, `lastReplayAt`, and `replayAttempts` fields.
Use `--dry-run` to validate the file without posting or rewriting anything.

For Redis Streams-backed event logging, set:

```text
F1_PLATFORM_REDIS_URL=redis://localhost:6379/0
```

For PostgreSQL-backed analytical projections, set:

```text
F1_PLATFORM_DATABASE_URL=postgresql://sport_prediction:sport_prediction@localhost:5432/sport_prediction
```

For model-service-backed prediction snapshots, run `services/f1-prediction-service`
and set:

```text
F1_PLATFORM_PREDICTION_URL=http://localhost:8002
F1_PLATFORM_PREDICTION_TIMEOUT_SECONDS=8
F1_PLATFORM_PREDICTION_FALLBACK=1
```

Without `F1_PLATFORM_DATABASE_URL`, projections are written to:

```text
data/raw/f1/platform-projections.sqlite
```

Without explicit FastF1 paths, artifacts and cache are written to:

```text
data/raw/f1/fastf1-artifacts
data/raw/f1/fastf1-cache
```

## Tests

```bash
cd services/f1-platform
PYTHONPATH=. pytest -q
```

Suggested commit name: `build-f1-platform-live-infra`
