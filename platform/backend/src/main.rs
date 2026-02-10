use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use dotenvy::dotenv;
use mime_guess::from_path;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::{postgres::PgPoolOptions, PgPool, Row};
use std::{
    collections::HashMap,
    fs,
    io::Write,
    path::{Path as FsPath, PathBuf},
    process::Stdio,
    time::Instant,
};
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;
use walkdir::{DirEntry, WalkDir};

#[derive(Clone)]
struct AppState {
    repo_root: PathBuf,
    data_dir: PathBuf,
    db: PgPool,
}

#[derive(Debug)]
struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
        }
    }

    fn forbidden(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = Json(json!({"error": self.message}));
        (self.status, body).into_response()
    }
}

type AppResult<T> = Result<T, AppError>;

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}

#[derive(Debug, Clone)]
struct ProjectInfo {
    sport: String,
    name: String,
    python_dir: PathBuf,
    notebook: Option<PathBuf>,
    kind: ProjectKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ProjectKind {
    F1,
    Football,
    Unknown,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ParamDef {
    name: String,
    label: String,
    kind: String,
    required: bool,
    default: Option<Value>,
    options: Option<Vec<String>>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct CatalogProject {
    sport: String,
    name: String,
    python_dir: String,
    notebook: Option<String>,
    params: Vec<ParamDef>,
}

#[derive(Serialize)]
struct CatalogResponse {
    projects: Vec<CatalogProject>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DataFileStatus {
    path: String,
    format: String,
    exists: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FootballDataStatus {
    teams: DataFileStatus,
    matches: DataFileStatus,
    fixtures: DataFileStatus,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DataStatus {
    football: FootballDataStatus,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PaperEntry {
    sport: String,
    title: String,
    file: String,
    source: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NotebookEntry {
    sport: String,
    project: String,
    file: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FootballFixture {
    match_id: String,
    date: String,
    season: String,
    league: String,
    home_team_id: String,
    away_team_id: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FootballFixturesResponse {
    fixtures: Vec<FootballFixture>,
    warning: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct FixturesQuery {
    limit: Option<usize>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct OpenRequest {
    path: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RunRequest {
    sport: String,
    project: String,
    params: HashMap<String, Value>,
    cache_dir: Option<String>,
    tags: Option<Vec<String>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RunResponse {
    run_id: String,
    status: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SweepRequest {
    sport: String,
    project: String,
    base_params: HashMap<String, Value>,
    sweep: SweepSpec,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SweepSpec {
    param: String,
    values: Vec<Value>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SweepResponse {
    sweep_id: String,
    run_ids: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RunSummary {
    id: String,
    created_at: String,
    sport: String,
    project: String,
    status: String,
    duration_ms: Option<i64>,
    sweep_id: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RunDetail {
    id: String,
    created_at: String,
    sport: String,
    project: String,
    status: String,
    duration_ms: Option<i64>,
    sweep_id: Option<String>,
    config: Value,
    result: Option<Value>,
    stdout: Option<String>,
    stderr: Option<String>,
    result_path: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SweepRow {
    id: String,
    created_at: String,
    sport: String,
    project: String,
    param: String,
    values_json: String,
    base_config_json: String,
    status: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SweepSummaryPoint {
    param_value: String,
    score: Option<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SweepRunDetail {
    id: String,
    created_at: String,
    status: String,
    param_value: String,
    result: Option<Value>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SweepDetail {
    id: String,
    created_at: String,
    sport: String,
    project: String,
    param: String,
    values: Value,
    status: String,
    runs: Vec<SweepRunDetail>,
    summary: Vec<SweepSummaryPoint>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();

    let current_dir = std::env::current_dir()?;
    let repo_root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| find_repo_root(&current_dir));

    let data_dir = repo_root.join("platform/backend/data");
    fs::create_dir_all(&data_dir)?;

    let database_url = std::env::var("DATABASE_URL").map_err(|_| {
        anyhow::anyhow!("DATABASE_URL is required (use your Supabase Postgres URI)")
    })?;
    let db = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await?;

    init_db(&db).await?;

    let state = AppState {
        repo_root,
        data_dir,
        db,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/health", get(health))
        .route("/api/catalog", get(catalog))
        .route("/api/data-status", get(data_status))
        .route("/api/papers", get(papers))
        .route("/api/notebooks", get(notebooks))
        .route("/api/football/fixtures", get(football_fixtures))
        .route("/api/files/*path", get(serve_file))
        .route("/api/open", post(open_path))
        .route("/api/runs", post(create_run).get(list_runs))
        .route("/api/runs/:id", get(get_run))
        .route("/api/sweeps", post(create_sweep).get(list_sweeps))
        .route("/api/sweeps/:id", get(get_sweep))
        .layer(cors)
        .with_state(state);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4000);

    let addr = format!("0.0.0.0:{}", port);
    println!("Backend running on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app.into_make_service()).await?;

    Ok(())
}

async fn init_db(db: &PgPool) -> anyhow::Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            sport TEXT NOT NULL,
            project TEXT NOT NULL,
            status TEXT NOT NULL,
            config_json TEXT NOT NULL,
            result_path TEXT,
            stdout_path TEXT,
            stderr_path TEXT,
            duration_ms INTEGER,
            sweep_id TEXT
        );
        "#,
    )
    .execute(db)
    .await?;

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS sweeps (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            sport TEXT NOT NULL,
            project TEXT NOT NULL,
            param TEXT NOT NULL,
            values_json TEXT NOT NULL,
            base_config_json TEXT NOT NULL,
            status TEXT NOT NULL
        );
        "#,
    )
    .execute(db)
    .await?;

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS sweep_runs (
            sweep_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            param_value TEXT NOT NULL,
            PRIMARY KEY (sweep_id, run_id)
        );
        "#,
    )
    .execute(db)
    .await?;

    harden_db_security(db).await?;

    Ok(())
}

async fn harden_db_security(db: &PgPool) -> anyhow::Result<()> {
    sqlx::query("ALTER TABLE public.runs ENABLE ROW LEVEL SECURITY")
        .execute(db)
        .await?;
    sqlx::query("ALTER TABLE public.sweeps ENABLE ROW LEVEL SECURITY")
        .execute(db)
        .await?;
    sqlx::query("ALTER TABLE public.sweep_runs ENABLE ROW LEVEL SECURITY")
        .execute(db)
        .await?;

    sqlx::query(
        "REVOKE ALL PRIVILEGES ON TABLE public.runs, public.sweeps, public.sweep_runs FROM PUBLIC",
    )
    .execute(db)
    .await?;

    sqlx::query(
        r#"
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
                REVOKE ALL PRIVILEGES ON TABLE public.runs, public.sweeps, public.sweep_runs FROM anon;
            END IF;

            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
                REVOKE ALL PRIVILEGES ON TABLE public.runs, public.sweeps, public.sweep_runs FROM authenticated;
            END IF;

            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
                GRANT ALL PRIVILEGES ON TABLE public.runs, public.sweeps, public.sweep_runs TO service_role;
            END IF;
        END
        $$;
        "#,
    )
    .execute(db)
    .await?;

    Ok(())
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

async fn catalog(State(state): State<AppState>) -> AppResult<Json<CatalogResponse>> {
    let projects = build_catalog(&state.repo_root)?;
    let response_projects = projects
        .into_iter()
        .map(|project| CatalogProject {
            sport: project.sport.clone(),
            name: project.name.clone(),
            python_dir: to_relative(&state.repo_root, &project.python_dir)
                .unwrap_or_else(|| project.python_dir.to_string_lossy().to_string()),
            notebook: project
                .notebook
                .as_ref()
                .and_then(|p| to_relative(&state.repo_root, p)),
            params: params_for_project(&project.kind),
        })
        .collect();

    Ok(Json(CatalogResponse {
        projects: response_projects,
    }))
}

async fn papers(State(state): State<AppState>) -> AppResult<Json<Vec<PaperEntry>>> {
    let papers = load_papers(&state.repo_root)?;
    Ok(Json(papers))
}

async fn notebooks(State(state): State<AppState>) -> AppResult<Json<Vec<NotebookEntry>>> {
    let notebooks = load_notebooks(&state.repo_root)?;
    Ok(Json(notebooks))
}

async fn data_status(State(state): State<AppState>) -> AppResult<Json<DataStatus>> {
    let data_root = state.repo_root.join("data/football");
    let teams = data_file_status(&data_root, "teams", &["parquet", "csv"]);
    let matches = data_file_status(&data_root, "matches", &["parquet", "csv"]);
    let fixtures = data_file_status(&data_root, "fixtures", &["parquet", "csv"]);

    Ok(Json(DataStatus {
        football: FootballDataStatus {
            teams,
            matches,
            fixtures,
        },
    }))
}

async fn football_fixtures(
    State(state): State<AppState>,
    Query(query): Query<FixturesQuery>,
) -> AppResult<Json<FootballFixturesResponse>> {
    let data_root = state.repo_root.join("data/football");
    let (path, format, exists) = data_file_status_raw(&data_root, "fixtures", &["parquet", "csv"]);
    if !exists {
        return Ok(Json(FootballFixturesResponse {
            fixtures: Vec::new(),
            warning: Some("Fixtures file not found.".to_string()),
        }));
    }

    if format != "csv" {
        return Ok(Json(FootballFixturesResponse {
            fixtures: Vec::new(),
            warning: Some("Parquet fixtures detected. CSV parsing only in MVP.".to_string()),
        }));
    }

    let limit = query.limit.unwrap_or(5);
    let fixtures = read_fixtures_csv(&path, limit).unwrap_or_default();

    Ok(Json(FootballFixturesResponse {
        fixtures,
        warning: None,
    }))
}

async fn serve_file(
    State(state): State<AppState>,
    Path(path): Path<String>,
) -> AppResult<Response> {
    let sanitized = sanitize_path(&state.repo_root, &path)?;
    if !sanitized.exists() {
        return Err(AppError::not_found("File not found"));
    }
    let mime = from_path(&sanitized).first_or_octet_stream();
    let bytes = tokio::fs::read(&sanitized)
        .await
        .map_err(|_| AppError::not_found("File not found"))?;

    let mut headers = HeaderMap::new();
    headers.insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_str(mime.as_ref())
            .unwrap_or_else(|_| HeaderValue::from_static("application/octet-stream")),
    );

    Ok((headers, bytes).into_response())
}

async fn open_path(
    State(state): State<AppState>,
    Json(payload): Json<OpenRequest>,
) -> AppResult<Json<Value>> {
    let sanitized = sanitize_path(&state.repo_root, &payload.path)?;
    if !sanitized.exists() {
        return Err(AppError::not_found("File not found"));
    }
    let status = std::process::Command::new("open")
        .arg(&sanitized)
        .status()
        .map_err(|_| AppError::internal("Failed to open file"))?;

    if !status.success() {
        return Err(AppError::internal("Open command failed"));
    }

    Ok(Json(json!({"status": "ok"})))
}

async fn list_runs(State(state): State<AppState>) -> AppResult<Json<Vec<RunSummary>>> {
    let rows = sqlx::query(
        r#"SELECT id, created_at, sport, project, status, duration_ms, sweep_id
        FROM runs
        ORDER BY created_at DESC"#,
    )
    .fetch_all(&state.db)
    .await
    .map_err(|_| AppError::internal("Failed to load runs"))?;

    let summaries = rows
        .into_iter()
        .map(|row| RunSummary {
            id: row.try_get("id").unwrap_or_default(),
            created_at: row.try_get("created_at").unwrap_or_default(),
            sport: row.try_get("sport").unwrap_or_default(),
            project: row.try_get("project").unwrap_or_default(),
            status: row.try_get("status").unwrap_or_default(),
            duration_ms: row.try_get("duration_ms").ok(),
            sweep_id: row.try_get("sweep_id").ok(),
        })
        .collect();

    Ok(Json(summaries))
}

async fn get_run(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> AppResult<Json<RunDetail>> {
    let row = sqlx::query(
        r#"SELECT id, created_at, sport, project, status, config_json, result_path, stdout_path, stderr_path, duration_ms, sweep_id
        FROM runs
        WHERE id = $1"#,
    )
        .bind(&id)
        .fetch_optional(&state.db)
        .await
        .map_err(|_| AppError::internal("Failed to load run"))?
        .ok_or_else(|| AppError::not_found("Run not found"))?;

    let config_json: String = row
        .try_get("config_json")
        .unwrap_or_else(|_| "{}".to_string());
    let result_path: Option<String> = row.try_get("result_path").ok();
    let stdout_path: Option<String> = row.try_get("stdout_path").ok();
    let stderr_path: Option<String> = row.try_get("stderr_path").ok();

    let config: Value = serde_json::from_str(&config_json).unwrap_or(json!({}));
    let result = match result_path.clone() {
        Some(path) => read_json_path(&state.repo_root, &path),
        None => Ok(None),
    }?;
    let stdout = match stdout_path {
        Some(path) => read_text_path(&state.repo_root, &path),
        None => Ok(None),
    }?;
    let stderr = match stderr_path {
        Some(path) => read_text_path(&state.repo_root, &path),
        None => Ok(None),
    }?;

    let detail = RunDetail {
        id: row.try_get("id").unwrap_or_default(),
        created_at: row.try_get("created_at").unwrap_or_default(),
        sport: row.try_get("sport").unwrap_or_default(),
        project: row.try_get("project").unwrap_or_default(),
        status: row.try_get("status").unwrap_or_default(),
        duration_ms: row.try_get("duration_ms").ok(),
        sweep_id: row.try_get("sweep_id").ok(),
        config,
        result,
        stdout,
        stderr,
        result_path: result_path.and_then(|p| to_relative(&state.repo_root, &PathBuf::from(p))),
    };

    Ok(Json(detail))
}

async fn create_run(
    State(state): State<AppState>,
    Json(payload): Json<RunRequest>,
) -> AppResult<Json<RunResponse>> {
    let project = resolve_project(&state.repo_root, &payload.sport, &payload.project)?;
    let run_id = Uuid::new_v4().to_string();
    let run_dir = state.data_dir.join("runs").join(&run_id);
    fs::create_dir_all(&run_dir).map_err(|_| AppError::internal("Failed to create run dir"))?;

    let mut params = payload.params.clone();
    if let Some(cache_dir) = payload.cache_dir.clone() {
        params
            .entry("cache_dir".to_string())
            .or_insert(Value::String(cache_dir));
    }

    let config_json = json!({
        "sport": payload.sport,
        "project": payload.project,
        "params": params,
        "tags": payload.tags,
    });

    let config_path = run_dir.join("config.json");
    write_json(&config_path, &config_json)?;

    let output_path = run_dir.join("result.json");
    let stdout_path = run_dir.join("stdout.log");
    let stderr_path = run_dir.join("stderr.log");

    let start = Instant::now();
    let status = execute_python_run(
        &project,
        &config_json,
        &output_path,
        &stdout_path,
        &stderr_path,
    )?;
    let duration_ms = start.elapsed().as_millis() as i64;

    let status_text = if status { "success" } else { "failed" };

    sqlx::query(
        r#"INSERT INTO runs (id, created_at, sport, project, status, config_json, result_path, stdout_path, stderr_path, duration_ms, sweep_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        "#,
    )
    .bind(&run_id)
    .bind(Utc::now().to_rfc3339())
    .bind(&project.sport)
    .bind(&project.name)
    .bind(status_text)
    .bind(config_json.to_string())
    .bind(output_path.to_string_lossy().to_string())
    .bind(stdout_path.to_string_lossy().to_string())
    .bind(stderr_path.to_string_lossy().to_string())
    .bind(duration_ms)
    .bind(Option::<String>::None)
    .execute(&state.db)
    .await
    .map_err(|_| AppError::internal("Failed to persist run"))?;

    Ok(Json(RunResponse {
        run_id,
        status: status_text.to_string(),
    }))
}

async fn list_sweeps(State(state): State<AppState>) -> AppResult<Json<Vec<SweepRow>>> {
    let rows = sqlx::query(
        r#"SELECT id, created_at, sport, project, param, values_json, base_config_json, status
        FROM sweeps
        ORDER BY created_at DESC"#,
    )
    .fetch_all(&state.db)
    .await
    .map_err(|_| AppError::internal("Failed to load sweeps"))?;

    let sweeps = rows
        .into_iter()
        .map(|row| SweepRow {
            id: row.try_get("id").unwrap_or_default(),
            created_at: row.try_get("created_at").unwrap_or_default(),
            sport: row.try_get("sport").unwrap_or_default(),
            project: row.try_get("project").unwrap_or_default(),
            param: row.try_get("param").unwrap_or_default(),
            values_json: row
                .try_get("values_json")
                .unwrap_or_else(|_| "[]".to_string()),
            base_config_json: row
                .try_get("base_config_json")
                .unwrap_or_else(|_| "{}".to_string()),
            status: row.try_get("status").unwrap_or_default(),
        })
        .collect();
    Ok(Json(sweeps))
}

async fn create_sweep(
    State(state): State<AppState>,
    Json(payload): Json<SweepRequest>,
) -> AppResult<Json<SweepResponse>> {
    let project = resolve_project(&state.repo_root, &payload.sport, &payload.project)?;
    let sweep_id = Uuid::new_v4().to_string();

    let base_config = json!({
        "sport": payload.sport,
        "project": payload.project,
        "params": payload.base_params,
    });

    sqlx::query(
        r#"INSERT INTO sweeps (id, created_at, sport, project, param, values_json, base_config_json, status)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        "#,
    )
    .bind(&sweep_id)
    .bind(Utc::now().to_rfc3339())
    .bind(&project.sport)
    .bind(&project.name)
    .bind(&payload.sweep.param)
    .bind(serde_json::to_string(&payload.sweep.values).unwrap_or("[]".to_string()))
    .bind(base_config.to_string())
    .bind("running")
    .execute(&state.db)
    .await
    .map_err(|_| AppError::internal("Failed to create sweep"))?;

    let mut run_ids = Vec::new();
    let mut any_failed = false;

    for value in payload.sweep.values.iter() {
        let run_id = Uuid::new_v4().to_string();
        run_ids.push(run_id.clone());
        let run_dir = state.data_dir.join("runs").join(&run_id);
        fs::create_dir_all(&run_dir).map_err(|_| AppError::internal("Failed to create run dir"))?;

        let mut params = payload.base_params.clone();
        params.insert(payload.sweep.param.clone(), value.clone());

        let config_json = json!({
            "sport": payload.sport,
            "project": payload.project,
            "params": params,
            "sweep_id": sweep_id,
        });

        let config_path = run_dir.join("config.json");
        write_json(&config_path, &config_json)?;

        let output_path = run_dir.join("result.json");
        let stdout_path = run_dir.join("stdout.log");
        let stderr_path = run_dir.join("stderr.log");

        let start = Instant::now();
        let status = execute_python_run(
            &project,
            &config_json,
            &output_path,
            &stdout_path,
            &stderr_path,
        )?;
        let duration_ms = start.elapsed().as_millis() as i64;

        let status_text = if status { "success" } else { "failed" };
        if !status {
            any_failed = true;
        }

        sqlx::query(
            r#"INSERT INTO runs (id, created_at, sport, project, status, config_json, result_path, stdout_path, stderr_path, duration_ms, sweep_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            "#,
        )
        .bind(&run_id)
        .bind(Utc::now().to_rfc3339())
        .bind(&project.sport)
        .bind(&project.name)
        .bind(status_text)
        .bind(config_json.to_string())
        .bind(output_path.to_string_lossy().to_string())
        .bind(stdout_path.to_string_lossy().to_string())
        .bind(stderr_path.to_string_lossy().to_string())
        .bind(duration_ms)
        .bind(&sweep_id)
        .execute(&state.db)
        .await
        .map_err(|_| AppError::internal("Failed to persist run"))?;

        let param_value = match value {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            _ => serde_json::to_string(value).unwrap_or_else(|_| value.to_string()),
        };
        sqlx::query(
            r#"INSERT INTO sweep_runs (sweep_id, run_id, param_value) VALUES ($1, $2, $3)"#,
        )
        .bind(&sweep_id)
        .bind(&run_id)
        .bind(param_value)
        .execute(&state.db)
        .await
        .map_err(|_| AppError::internal("Failed to persist sweep run"))?;
    }

    let sweep_status = if any_failed { "partial" } else { "success" };
    sqlx::query("UPDATE sweeps SET status = $1 WHERE id = $2")
        .bind(sweep_status)
        .bind(&sweep_id)
        .execute(&state.db)
        .await
        .map_err(|_| AppError::internal("Failed to finalize sweep"))?;

    Ok(Json(SweepResponse { sweep_id, run_ids }))
}

async fn get_sweep(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> AppResult<Json<SweepDetail>> {
    let sweep = sqlx::query(
        r#"SELECT id, created_at, sport, project, param, values_json, base_config_json, status
        FROM sweeps
        WHERE id = $1"#,
    )
    .bind(&id)
    .fetch_optional(&state.db)
    .await
    .map_err(|_| AppError::internal("Failed to load sweep"))?
    .ok_or_else(|| AppError::not_found("Sweep not found"))?;

    let run_rows = sqlx::query(
        r#"SELECT r.id, r.created_at, r.status, r.result_path, sr.param_value
            FROM runs r
            JOIN sweep_runs sr ON r.id = sr.run_id
            WHERE sr.sweep_id = $1
            ORDER BY r.created_at ASC"#,
    )
    .bind(&id)
    .fetch_all(&state.db)
    .await
    .map_err(|_| AppError::internal("Failed to load sweep runs"))?;

    let mut runs = Vec::new();
    let mut summary = Vec::new();

    for row in run_rows {
        let run_id: String = row.try_get("id").unwrap_or_default();
        let created_at: String = row.try_get("created_at").unwrap_or_default();
        let status: String = row.try_get("status").unwrap_or_default();
        let result_path: Option<String> = row.try_get("result_path").ok();
        let param_value: String = row.try_get("param_value").unwrap_or_default();
        let result = match result_path.clone() {
            Some(path) => read_json_path(&state.repo_root, &path),
            None => Ok(None),
        }?;

        let score = result.as_ref().and_then(score_from_result);
        summary.push(SweepSummaryPoint {
            param_value: param_value.clone(),
            score,
        });

        runs.push(SweepRunDetail {
            id: run_id,
            created_at,
            status,
            param_value,
            result,
        });
    }

    let values_json: String = sweep
        .try_get("values_json")
        .unwrap_or_else(|_| "[]".to_string());
    let values: Value = serde_json::from_str(&values_json).unwrap_or(json!([]));

    Ok(Json(SweepDetail {
        id: sweep.try_get("id").unwrap_or_default(),
        created_at: sweep.try_get("created_at").unwrap_or_default(),
        sport: sweep.try_get("sport").unwrap_or_default(),
        project: sweep.try_get("project").unwrap_or_default(),
        param: sweep.try_get("param").unwrap_or_default(),
        values,
        status: sweep.try_get("status").unwrap_or_default(),
        runs,
        summary,
    }))
}

fn build_catalog(repo_root: &PathBuf) -> AppResult<Vec<ProjectInfo>> {
    let mut projects = Vec::new();
    let projects_root = resolve_projects_root(repo_root);
    for entry in
        fs::read_dir(&projects_root).map_err(|_| AppError::internal("Failed to scan repo"))?
    {
        let entry = entry.map_err(|_| AppError::internal("Failed to scan repo"))?;
        let path = entry.path();
        if !is_sport_dir(&path) {
            continue;
        }
        let sport_name = entry.file_name().to_string_lossy().to_string();
        for project_entry in
            fs::read_dir(&path).map_err(|_| AppError::internal("Failed to scan sport"))?
        {
            let project_entry =
                project_entry.map_err(|_| AppError::internal("Failed to scan sport"))?;
            let project_path = project_entry.path();
            if !project_path.is_dir() {
                continue;
            }
            let project_name = project_entry.file_name().to_string_lossy().to_string();
            let python_dir = project_path.join("Python");
            if !python_dir.exists() {
                continue;
            }
            let notebook = project_path.join("Jupyter").join("model-research.ipynb");
            let kind = match (sport_name.as_str(), project_name.as_str()) {
                ("F1", "Rising Qualification Prediction") => ProjectKind::F1,
                ("Football", "Match Result Prediction") => ProjectKind::Football,
                _ => ProjectKind::Unknown,
            };
            projects.push(ProjectInfo {
                sport: sport_name.clone(),
                name: project_name,
                python_dir,
                notebook: notebook.exists().then_some(notebook),
                kind,
            });
        }
    }
    Ok(projects)
}

fn is_sport_dir(path: &FsPath) -> bool {
    if !path.is_dir() {
        return false;
    }
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if name.starts_with('.') {
        return false;
    }
    if matches!(
        name,
        "Research" | "research" | "platform" | "papers" | "projects" | ".git"
    ) {
        return false;
    }
    let mut has_project = false;
    if let Ok(read_dir) = fs::read_dir(path) {
        for entry in read_dir.flatten() {
            let child = entry.path();
            if child.is_dir() && !entry.file_name().to_string_lossy().starts_with('.') {
                if child.join("Python").exists() || child.join("Jupyter").exists() {
                    has_project = true;
                    break;
                }
            }
        }
    }
    has_project
}

fn params_for_project(kind: &ProjectKind) -> Vec<ParamDef> {
    match kind {
        ProjectKind::F1 => vec![
            ParamDef {
                name: "mode".to_string(),
                label: "Mode".to_string(),
                kind: "select".to_string(),
                required: true,
                default: Some(Value::String("qualifying".to_string())),
                options: Some(vec!["qualifying".into(), "race".into()]),
            },
            ParamDef {
                name: "source".to_string(),
                label: "Source".to_string(),
                kind: "select".to_string(),
                required: true,
                default: Some(Value::String("fastf1".to_string())),
                options: Some(vec!["fastf1".into(), "openf1".into()]),
            },
            ParamDef {
                name: "year".to_string(),
                label: "Year".to_string(),
                kind: "int".to_string(),
                required: true,
                default: Some(Value::Number(2026.into())),
                options: None,
            },
            ParamDef {
                name: "round_number".to_string(),
                label: "Round".to_string(),
                kind: "int".to_string(),
                required: true,
                default: Some(Value::Number(1.into())),
                options: None,
            },
            ParamDef {
                name: "train_seasons".to_string(),
                label: "Train Seasons".to_string(),
                kind: "string".to_string(),
                required: false,
                default: Some(Value::String("auto".to_string())),
                options: None,
            },
            ParamDef {
                name: "include_standings".to_string(),
                label: "Include Standings".to_string(),
                kind: "bool".to_string(),
                required: false,
                default: Some(Value::Bool(false)),
                options: None,
            },
            ParamDef {
                name: "cache_dir".to_string(),
                label: "Cache Dir".to_string(),
                kind: "string".to_string(),
                required: false,
                default: Some(Value::String(".cache/fastf1".to_string())),
                options: None,
            },
            ParamDef {
                name: "meeting_name".to_string(),
                label: "Meeting Name".to_string(),
                kind: "string".to_string(),
                required: false,
                default: None,
                options: None,
            },
            ParamDef {
                name: "country_name".to_string(),
                label: "Country Name".to_string(),
                kind: "string".to_string(),
                required: false,
                default: None,
                options: None,
            },
        ],
        ProjectKind::Football => vec![
            ParamDef {
                name: "mode".to_string(),
                label: "Mode".to_string(),
                kind: "select".to_string(),
                required: true,
                default: Some(Value::String("match_result".to_string())),
                options: Some(vec!["match_result".into(), "scoreline".into()]),
            },
            ParamDef {
                name: "league".to_string(),
                label: "League".to_string(),
                kind: "string".to_string(),
                required: true,
                default: Some(Value::String("epl".to_string())),
                options: None,
            },
            ParamDef {
                name: "season".to_string(),
                label: "Season".to_string(),
                kind: "int".to_string(),
                required: true,
                default: Some(Value::Number(2026.into())),
                options: None,
            },
            ParamDef {
                name: "round_number".to_string(),
                label: "Round".to_string(),
                kind: "int".to_string(),
                required: true,
                default: Some(Value::Number(1.into())),
                options: None,
            },
            ParamDef {
                name: "data_source".to_string(),
                label: "Data Source".to_string(),
                kind: "string".to_string(),
                required: false,
                default: Some(Value::String("placeholder".to_string())),
                options: None,
            },
            ParamDef {
                name: "train_seasons".to_string(),
                label: "Train Seasons".to_string(),
                kind: "string".to_string(),
                required: false,
                default: Some(Value::String("auto".to_string())),
                options: None,
            },
            ParamDef {
                name: "cache_dir".to_string(),
                label: "Cache Dir".to_string(),
                kind: "string".to_string(),
                required: false,
                default: None,
                options: None,
            },
        ],
        ProjectKind::Unknown => vec![],
    }
}

fn load_papers(repo_root: &PathBuf) -> AppResult<Vec<PaperEntry>> {
    let index = parse_research_index(repo_root);
    let research_dir = resolve_papers_root(repo_root);
    let mut papers = Vec::new();
    if !research_dir.exists() {
        return Ok(papers);
    }
    for entry in
        fs::read_dir(&research_dir).map_err(|_| AppError::internal("Failed to read research"))?
    {
        let entry = entry.map_err(|_| AppError::internal("Failed to read research"))?;
        let sport_dir = entry.path();
        if !sport_dir.is_dir() {
            continue;
        }
        let sport = entry.file_name().to_string_lossy().to_string();
        for pdf_entry in
            fs::read_dir(&sport_dir).map_err(|_| AppError::internal("Failed to read papers"))?
        {
            let pdf_entry = pdf_entry.map_err(|_| AppError::internal("Failed to read papers"))?;
            let pdf_path = pdf_entry.path();
            if pdf_path.extension().and_then(|s| s.to_str()) != Some("pdf") {
                continue;
            }
            let file_name = pdf_path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            let meta = index.get(file_name);
            let title = meta
                .and_then(|m| m.get("title"))
                .cloned()
                .unwrap_or_else(|| {
                    pdf_path
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .replace('-', " ")
                });
            let source = meta.and_then(|m| m.get("source")).cloned();
            let file = to_relative(repo_root, &pdf_path)
                .unwrap_or_else(|| pdf_path.to_string_lossy().to_string());
            papers.push(PaperEntry {
                sport: meta
                    .and_then(|m| m.get("sport"))
                    .cloned()
                    .unwrap_or_else(|| sport.clone()),
                title,
                file,
                source,
            });
        }
    }
    Ok(papers)
}

fn parse_research_index(repo_root: &PathBuf) -> HashMap<String, HashMap<String, String>> {
    let mut index: HashMap<String, HashMap<String, String>> = HashMap::new();
    let readme = resolve_papers_root(repo_root).join("README.md");
    if !readme.exists() {
        return index;
    }
    let content = fs::read_to_string(readme).unwrap_or_default();
    let mut current_sport: Option<String> = None;
    let mut current_title: Option<String> = None;
    let mut current_file: Option<String> = None;

    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("**") && line.ends_with("**") && line.matches("**").count() >= 2 {
            let label = line.trim_matches('*').trim().to_string();
            if label.to_lowercase() != "research papers" {
                current_sport = Some(label);
            }
            continue;
        }
        if line.starts_with("- ") && !line.starts_with("- File:") && !line.starts_with("- Source:")
        {
            current_title = Some(line[2..].trim().to_string());
            current_file = None;
            continue;
        }
        if line.starts_with("- File:") {
            let file = line.split('`').nth(1).unwrap_or("").trim().to_string();
            let file_name = FsPath::new(&file)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            if file_name.is_empty() {
                continue;
            }
            current_file = Some(file_name.clone());
            let meta = index.entry(file_name).or_default();
            if let Some(title) = current_title.clone() {
                meta.insert("title".to_string(), title);
            }
            if let Some(sport) = current_sport.clone() {
                meta.insert("sport".to_string(), sport);
            }
            continue;
        }
        if line.starts_with("- Source:") {
            if let Some(file) = current_file.clone() {
                let meta = index.entry(file).or_default();
                let source = line.splitn(2, ':').nth(1).unwrap_or("").trim().to_string();
                meta.insert("source".to_string(), source);
            }
        }
    }
    index
}

fn load_notebooks(repo_root: &PathBuf) -> AppResult<Vec<NotebookEntry>> {
    let mut notebooks = Vec::new();
    let projects_root = resolve_projects_root(repo_root);
    for entry in WalkDir::new(&projects_root)
        .into_iter()
        .filter_entry(|entry| !should_skip(entry))
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                if name.ends_with(".ipynb") {
                    let file = to_relative(repo_root, path)
                        .unwrap_or_else(|| path.to_string_lossy().to_string());
                    let rel = path.strip_prefix(&projects_root).unwrap_or(path);
                    let mut components = rel.components();
                    let sport = components
                        .next()
                        .and_then(|c| c.as_os_str().to_str())
                        .unwrap_or("Unknown");
                    let project = components
                        .next()
                        .and_then(|c| c.as_os_str().to_str())
                        .unwrap_or("Unknown");
                    notebooks.push(NotebookEntry {
                        sport: sport.to_string(),
                        project: project.to_string(),
                        file,
                    });
                }
            }
        }
    }
    Ok(notebooks)
}

fn should_skip(entry: &DirEntry) -> bool {
    if entry.file_type().is_file() {
        return false;
    }
    let name = entry.file_name().to_string_lossy();
    matches!(
        name.as_ref(),
        ".git" | "node_modules" | ".next" | "target" | "data" | ".turbo" | "platform" | "papers"
    )
}

fn resolve_project(repo_root: &PathBuf, sport: &str, project: &str) -> AppResult<ProjectInfo> {
    let projects = build_catalog(repo_root)?;
    let mut matches: Vec<ProjectInfo> = projects
        .into_iter()
        .filter(|p| {
            normalize(&p.sport) == normalize(sport) && normalize(&p.name) == normalize(project)
        })
        .collect();
    if matches.is_empty() {
        return Err(AppError::not_found("Project not found"));
    }
    Ok(matches.remove(0))
}

fn normalize(input: &str) -> String {
    input.to_lowercase().replace('_', " ").trim().to_string()
}

fn resolve_projects_root(repo_root: &PathBuf) -> PathBuf {
    let candidate = repo_root.join("research/projects");
    if candidate.exists() {
        return candidate;
    }
    let legacy = repo_root.join("projects");
    if legacy.exists() {
        return legacy;
    }
    repo_root.clone()
}

fn resolve_papers_root(repo_root: &PathBuf) -> PathBuf {
    let candidate = repo_root.join("research/papers");
    if candidate.exists() {
        return candidate;
    }
    let legacy = repo_root.join("Research");
    if legacy.exists() {
        return legacy;
    }
    repo_root.join("papers")
}

fn find_repo_root(start: &PathBuf) -> PathBuf {
    for ancestor in start.ancestors() {
        if ancestor.join("research").exists()
            || ancestor.join("Research").exists()
            || ancestor.join(".git").exists()
        {
            return ancestor.to_path_buf();
        }
    }
    start.clone()
}

fn sanitize_path(repo_root: &PathBuf, input: &str) -> AppResult<PathBuf> {
    let rel = input.trim_start_matches('/');
    let candidate = repo_root.join(rel);
    let canonical = candidate
        .canonicalize()
        .map_err(|_| AppError::not_found("File not found"))?;
    let root = repo_root
        .canonicalize()
        .map_err(|_| AppError::internal("Failed to resolve repo"))?;
    if !canonical.starts_with(&root) {
        return Err(AppError::forbidden("Path not allowed"));
    }
    Ok(canonical)
}

fn to_relative(repo_root: &FsPath, path: &FsPath) -> Option<String> {
    path.strip_prefix(repo_root)
        .ok()
        .map(|p| p.to_string_lossy().to_string())
}

fn execute_python_run(
    project: &ProjectInfo,
    config: &Value,
    output_path: &PathBuf,
    stdout_path: &PathBuf,
    stderr_path: &PathBuf,
) -> AppResult<bool> {
    let params = config
        .get("params")
        .and_then(|v| v.as_object())
        .ok_or_else(|| AppError::bad_request("Params missing"))?;

    let (command, args) = build_command(project, params, output_path)?;

    let mut cmd = std::process::Command::new(command);
    cmd.current_dir(&project.python_dir)
        .args(args)
        .stdout(Stdio::from(
            fs::File::create(stdout_path)
                .map_err(|_| AppError::internal("Failed to open stdout"))?,
        ))
        .stderr(Stdio::from(
            fs::File::create(stderr_path)
                .map_err(|_| AppError::internal("Failed to open stderr"))?,
        ));

    let status = cmd
        .status()
        .map_err(|_| AppError::internal("Failed to run python"))?;

    Ok(status.success() && output_path.exists())
}

fn build_command(
    project: &ProjectInfo,
    params: &serde_json::Map<String, Value>,
    output_path: &PathBuf,
) -> AppResult<(String, Vec<String>)> {
    let script = project.python_dir.join("run_prediction.py");
    if !script.exists() {
        return Err(AppError::internal("run_prediction.py not found"));
    }

    let mut args = Vec::new();
    args.push(script.to_string_lossy().to_string());

    match project.kind {
        ProjectKind::F1 => {
            let mode = get_str(params, "mode", None, true)?;
            let source = get_str(params, "source", None, true)?;
            let year = get_i64(params, "year", Some(2026), true)?;
            let round = get_i64(
                params,
                "round_number",
                params.get("round").and_then(|v| v.as_i64()),
                true,
            )?;
            let train_seasons = get_str(params, "train_seasons", Some("auto"), false)?;
            let include_standings = get_bool(params, "include_standings", false);
            let cache_dir = params
                .get("cache_dir")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let meeting_name = params
                .get("meeting_name")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let country_name = params
                .get("country_name")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());

            args.extend(["--mode".to_string(), mode]);
            args.extend(["--source".to_string(), source]);
            args.extend(["--year".to_string(), year.to_string()]);
            args.extend(["--round".to_string(), round.to_string()]);
            args.extend(["--train-seasons".to_string(), train_seasons]);
            if include_standings {
                args.push("--include-standings".to_string());
            }
            if let Some(cache) = cache_dir {
                args.extend(["--cache-dir".to_string(), cache.to_string()]);
            }
            if let Some(meeting) = meeting_name {
                args.extend(["--meeting-name".to_string(), meeting.to_string()]);
            }
            if let Some(country) = country_name {
                args.extend(["--country-name".to_string(), country.to_string()]);
            }
        }
        ProjectKind::Football => {
            let mode = get_str(params, "mode", None, true)?;
            let league = get_str(params, "league", None, true)?;
            let season = get_i64(params, "season", Some(2026), true)?;
            let round = get_i64(
                params,
                "round_number",
                params.get("round").and_then(|v| v.as_i64()),
                true,
            )?;
            let data_source = get_str(params, "data_source", Some("placeholder"), false)?;
            let train_seasons = get_str(params, "train_seasons", Some("auto"), false)?;
            let cache_dir = params
                .get("cache_dir")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());

            args.extend(["--mode".to_string(), mode]);
            args.extend(["--league".to_string(), league]);
            args.extend(["--season".to_string(), season.to_string()]);
            args.extend(["--round".to_string(), round.to_string()]);
            args.extend(["--data-source".to_string(), data_source]);
            args.extend(["--train-seasons".to_string(), train_seasons]);
            if let Some(cache) = cache_dir {
                args.extend(["--cache-dir".to_string(), cache.to_string()]);
            }
        }
        ProjectKind::Unknown => return Err(AppError::bad_request("Unknown project kind")),
    }

    args.extend([
        "--output-format".to_string(),
        "json".to_string(),
        "--output-path".to_string(),
        output_path.to_string_lossy().to_string(),
        "--quiet".to_string(),
    ]);

    Ok(("python".to_string(), args))
}

fn get_str(
    params: &serde_json::Map<String, Value>,
    key: &str,
    default: Option<&str>,
    required: bool,
) -> AppResult<String> {
    if let Some(value) = params.get(key) {
        if let Some(s) = value.as_str() {
            return Ok(s.to_string());
        }
        return Ok(value.to_string());
    }
    if let Some(default) = default {
        return Ok(default.to_string());
    }
    if required {
        return Err(AppError::bad_request(format!("Missing param: {}", key)));
    }
    Ok(String::new())
}

fn get_i64(
    params: &serde_json::Map<String, Value>,
    key: &str,
    fallback: Option<i64>,
    required: bool,
) -> AppResult<i64> {
    if let Some(value) = params.get(key) {
        if let Some(n) = value.as_i64() {
            return Ok(n);
        }
        if let Some(s) = value.as_str() {
            return s
                .parse::<i64>()
                .map_err(|_| AppError::bad_request(format!("Invalid param: {}", key)));
        }
    }
    if let Some(default) = fallback {
        return Ok(default);
    }
    if required {
        return Err(AppError::bad_request(format!("Missing param: {}", key)));
    }
    Ok(0)
}

fn get_bool(params: &serde_json::Map<String, Value>, key: &str, default: bool) -> bool {
    params.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
}

fn read_json_path(repo_root: &PathBuf, path: &str) -> AppResult<Option<Value>> {
    let path_buf = PathBuf::from(path);
    let abs = if path_buf.is_absolute() {
        path_buf
    } else {
        repo_root.join(path_buf)
    };
    if !abs.exists() {
        return Ok(None);
    }
    let content =
        fs::read_to_string(abs).map_err(|_| AppError::internal("Failed to read result"))?;
    let parsed: Value =
        serde_json::from_str(&content).map_err(|_| AppError::internal("Invalid JSON"))?;
    Ok(Some(parsed))
}

fn read_text_path(repo_root: &PathBuf, path: &str) -> AppResult<Option<String>> {
    let path_buf = PathBuf::from(path);
    let abs = if path_buf.is_absolute() {
        path_buf
    } else {
        repo_root.join(path_buf)
    };
    if !abs.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(abs).map_err(|_| AppError::internal("Failed to read log"))?;
    Ok(Some(content))
}

fn score_from_result(result: &Value) -> Option<f64> {
    let rows = result.get("rows")?.as_array()?;
    let mut values = Vec::new();
    for row in rows {
        if let Some(pred) = row.get("pred").and_then(|v| v.as_f64()) {
            values.push(pred);
        }
    }
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn write_json(path: &PathBuf, value: &Value) -> AppResult<()> {
    let mut file =
        fs::File::create(path).map_err(|_| AppError::internal("Failed to write JSON"))?;
    let content = serde_json::to_string_pretty(value)
        .map_err(|_| AppError::internal("Failed to serialize"))?;
    file.write_all(content.as_bytes())
        .map_err(|_| AppError::internal("Failed to write JSON"))?;
    Ok(())
}

fn data_file_status(root: &PathBuf, name: &str, formats: &[&str]) -> DataFileStatus {
    let (path, format, exists) = data_file_status_raw(root, name, formats);
    DataFileStatus {
        path: path.to_string_lossy().to_string(),
        format,
        exists,
    }
}

fn data_file_status_raw(root: &PathBuf, name: &str, formats: &[&str]) -> (PathBuf, String, bool) {
    for format in formats {
        let candidate = root.join(format!("{}.{}", name, format));
        if candidate.exists() {
            return (candidate, format.to_string(), true);
        }
    }
    let default_format = formats.first().copied().unwrap_or("csv");
    (
        root.join(format!("{}.{}", name, default_format)),
        default_format.to_string(),
        false,
    )
}

fn read_fixtures_csv(path: &PathBuf, limit: usize) -> Result<Vec<FootballFixture>, AppError> {
    let mut fixtures = Vec::new();
    let mut reader = csv::Reader::from_path(path)
        .map_err(|_| AppError::internal("Failed to read fixtures CSV"))?;
    for result in reader.deserialize::<HashMap<String, String>>() {
        let record = result.map_err(|_| AppError::internal("Invalid fixtures CSV"))?;
        let fixture = FootballFixture {
            match_id: record.get("match_id").cloned().unwrap_or_default(),
            date: record.get("date").cloned().unwrap_or_default(),
            season: record.get("season").cloned().unwrap_or_default(),
            league: record.get("league").cloned().unwrap_or_default(),
            home_team_id: record.get("home_team_id").cloned().unwrap_or_default(),
            away_team_id: record.get("away_team_id").cloned().unwrap_or_default(),
        };
        fixtures.push(fixture);
        if fixtures.len() >= limit {
            break;
        }
    }
    Ok(fixtures)
}
