export const F1_TRACK_LAYOUT_API_BASE =
  process.env.NEXT_PUBLIC_F1_TRACK_LAYOUT_API_URL ||
  "https://raw.githubusercontent.com/bacinger/f1-circuits/master";

export type F1TrackCornerKind = "slow" | "medium" | "fast";

export type F1TrackCornerMarker = {
  number: number;
  x: number;
  y: number;
  progress: number;
  severity: number;
  kind: F1TrackCornerKind;
};

export type F1TrackLayoutProfile = {
  source: string;
  sourceUrl: string;
  datasetId: string;
  circuitId?: string | null;
  circuitName: string;
  location?: string | null;
  lengthKm?: string | null;
  pathD: string;
  points: Array<{ x: number; y: number }>;
  slow: number;
  medium: number;
  fast: number;
  totalCorners: number;
  corners: F1TrackCornerMarker[];
};

type GeoJsonFeatureCollection = {
  type: "FeatureCollection";
  features?: GeoJsonFeature[];
};

type GeoJsonFeature = {
  type: "Feature";
  properties?: Record<string, unknown>;
  geometry?: {
    type?: string;
    coordinates?: unknown;
  };
};

type RawPoint = {
  x: number;
  y: number;
};

type SampledPoint = RawPoint & {
  distance: number;
};

const SVG_WIDTH = 640;
const SVG_HEIGHT = 360;
const SVG_PADDING = 36;

const ERGAST_CIRCUIT_LAYOUT_IDS: Record<string, string> = {
  albert_park: "au-1953",
  americas: "us-2012",
  baku: "az-2016",
  bahrain: "bh-2002",
  catalunya: "es-1991",
  hungaroring: "hu-1986",
  imola: "it-1953",
  interlagos: "br-1940",
  jeddah: "sa-2021",
  losail: "qa-2004",
  marina_bay: "sg-2008",
  madring: "es-2026",
  miami: "us-2022",
  monaco: "mc-1929",
  monza: "it-1922",
  red_bull_ring: "at-1969",
  rodriguez: "mx-1962",
  shanghai: "cn-2004",
  silverstone: "gb-1948",
  spa: "be-1925",
  suzuka: "jp-1962",
  vegas: "us-2023",
  villeneuve: "ca-1978",
  yas_marina: "ae-2009",
  zandvoort: "nl-1948",
};

const EVENT_LAYOUT_IDS: Record<string, string> = {
  abu_dhabi: "ae-2009",
  azerbaijan: "az-2016",
  australian: "au-1953",
  austrian: "at-1969",
  bahrain: "bh-2002",
  belgian: "be-1925",
  brazilian: "br-1940",
  british: "gb-1948",
  canadian: "ca-1978",
  chinese: "cn-2004",
  dutch: "nl-1948",
  emilia_romagna: "it-1953",
  hungarian: "hu-1986",
  italian: "it-1922",
  japanese: "jp-1962",
  las_vegas: "us-2023",
  madrid: "es-2026",
  madring: "es-2026",
  miami: "us-2022",
  mexico_city: "mx-1962",
  monaco: "mc-1929",
  qatar: "qa-2004",
  saudi_arabian: "sa-2021",
  singapore: "sg-2008",
  spanish: "es-1991",
  united_states: "us-2012",
};

const KNOWN_CORNER_COUNTS: Record<string, number> = {
  "ae-2009": 16,
  "at-1969": 10,
  "au-1953": 14,
  "az-2016": 20,
  "be-1925": 19,
  "bh-2002": 15,
  "br-1940": 15,
  "ca-1978": 14,
  "cn-2004": 16,
  "es-1991": 14,
  "es-2026": 22,
  "gb-1948": 18,
  "hu-1986": 14,
  "it-1922": 11,
  "it-1953": 19,
  "jp-1962": 18,
  "mc-1929": 19,
  "mx-1962": 17,
  "nl-1948": 14,
  "qa-2004": 16,
  "sa-2021": 27,
  "sg-2008": 19,
  "us-2012": 20,
  "us-2022": 19,
  "us-2023": 17,
};

export async function getF1TrackLayoutProfile(params: {
  circuitId?: string | null;
  circuitName?: string | null;
  eventName?: string | null;
}): Promise<F1TrackLayoutProfile> {
  const datasetId = resolveLayoutDatasetId(params);
  if (!datasetId) {
    throw new Error(`No F1 track layout mapping found for ${params.circuitId ?? params.circuitName ?? params.eventName ?? "circuit"}`);
  }

  const baseUrl = F1_TRACK_LAYOUT_API_BASE.replace(/\/+$/, "");
  const sourceUrl = `${baseUrl}/circuits/${datasetId}.geojson`;
  const response = await fetch(sourceUrl, { cache: "force-cache" });
  if (!response.ok) {
    throw new Error(`F1 track layout failed: ${response.status}`);
  }

  const payload = (await response.json()) as unknown;
  const feature = geoJsonFeature(payload);
  const coordinates = lineStringCoordinates(feature);
  if (coordinates.length < 3) {
    throw new Error(`F1 track layout ${datasetId} has no usable line geometry`);
  }

  return buildTrackLayoutProfile({
    circuitId: params.circuitId,
    datasetId,
    feature,
    coordinates,
    sourceUrl,
  });
}

function resolveLayoutDatasetId(params: {
  circuitId?: string | null;
  circuitName?: string | null;
  eventName?: string | null;
}): string | null {
  const circuitKey = normalizeLayoutKey(params.circuitId);
  const eventKey = normalizeLayoutKey(params.eventName);
  const nameKey = normalizeLayoutKey(params.circuitName);
  return (
    ERGAST_CIRCUIT_LAYOUT_IDS[circuitKey] ??
    EVENT_LAYOUT_IDS[eventKey] ??
    EVENT_LAYOUT_IDS[nameKey] ??
    null
  );
}

function buildTrackLayoutProfile(params: {
  circuitId?: string | null;
  datasetId: string;
  feature: GeoJsonFeature;
  coordinates: Array<[number, number]>;
  sourceUrl: string;
}): F1TrackLayoutProfile {
  const properties = params.feature.properties ?? {};
  const rawPoints = projectLongitudeLatitude(params.coordinates);
  const totalDistance = cumulativeDistance(rawPoints);
  const normalizedPoints = normalizeTrackPoints(rawPoints);
  const pathD = svgPathFromPoints(normalizedPoints);
  const desiredCorners = KNOWN_CORNER_COUNTS[params.datasetId];
  const rawCorners = detectCornerMarkers(rawPoints, totalDistance, desiredCorners);
  const corners = classifyCorners(
    rawCorners.map((corner, index) => {
      const point = interpolateSvgPoint(normalizedPoints, rawPoints, corner.distance);
      return {
        number: index + 1,
        x: point.x,
        y: point.y,
        progress: totalDistance > 0 ? corner.distance / totalDistance : 0,
        severity: corner.severity,
        kind: "medium" as F1TrackCornerKind,
      };
    })
  );

  const counts = corners.reduce(
    (accumulator, corner) => {
      accumulator[corner.kind] += 1;
      return accumulator;
    },
    { slow: 0, medium: 0, fast: 0 }
  );

  return {
    source: "bacinger/f1-circuits",
    sourceUrl: params.sourceUrl,
    datasetId: params.datasetId,
    circuitId: params.circuitId,
    circuitName: optionalString(properties.Name) ?? params.datasetId,
    location: optionalString(properties.Location),
    lengthKm: formatLengthKm(properties.length),
    pathD,
    points: normalizedPoints,
    slow: counts.slow,
    medium: counts.medium,
    fast: counts.fast,
    totalCorners: corners.length,
    corners,
  };
}

function projectLongitudeLatitude(coordinates: Array<[number, number]>): RawPoint[] {
  const averageLatitude =
    (coordinates.reduce((sum, coordinate) => sum + coordinate[1], 0) / coordinates.length) *
    (Math.PI / 180);
  return coordinates.map(([longitude, latitude]) => ({
    x: longitude * Math.cos(averageLatitude) * 111_320,
    y: latitude * 111_320,
  }));
}

function normalizeTrackPoints(points: RawPoint[]): RawPoint[] {
  const minX = Math.min(...points.map((point) => point.x));
  const maxX = Math.max(...points.map((point) => point.x));
  const minY = Math.min(...points.map((point) => point.y));
  const maxY = Math.max(...points.map((point) => point.y));
  const width = Math.max(1, maxX - minX);
  const height = Math.max(1, maxY - minY);
  const scale = Math.min((SVG_WIDTH - SVG_PADDING * 2) / width, (SVG_HEIGHT - SVG_PADDING * 2) / height);
  const offsetX = (SVG_WIDTH - width * scale) / 2;
  const offsetY = (SVG_HEIGHT - height * scale) / 2;

  return points.map((point) => ({
    x: roundSvg(offsetX + (point.x - minX) * scale),
    y: roundSvg(SVG_HEIGHT - (offsetY + (point.y - minY) * scale)),
  }));
}

function svgPathFromPoints(points: RawPoint[]): string {
  if (!points.length) return "";
  const commands = [`M ${points[0].x} ${points[0].y}`];
  for (const point of points.slice(1)) {
    commands.push(`L ${point.x} ${point.y}`);
  }
  commands.push("Z");
  return commands.join(" ");
}

function detectCornerMarkers(points: RawPoint[], totalDistance: number, desiredCorners?: number): Array<{
  distance: number;
  severity: number;
}> {
  const targetCorners = desiredCorners ?? 14;
  const sampled = resampleTrack(points, Math.max(18, totalDistance / 420));
  const windowSize = 5;
  const candidates: Array<{ distance: number; severity: number }> = [];

  for (let index = windowSize; index < sampled.length - windowSize; index += 1) {
    const left = sampled[index - windowSize];
    const center = sampled[index];
    const right = sampled[index + windowSize];
    const headingIn = Math.atan2(center.y - left.y, center.x - left.x);
    const headingOut = Math.atan2(right.y - center.y, right.x - center.x);
    const severity = Math.abs(angleDelta(headingIn, headingOut));
    if (severity > 0.08) {
      candidates.push({ distance: center.distance, severity });
    }
  }

  candidates.sort((left, right) => right.severity - left.severity);
  let minSeparation = Math.max(55, totalDistance / (targetCorners * 1.55));
  let selected: Array<{ distance: number; severity: number }> = [];

  while (minSeparation >= 45) {
    selected = [];
    for (const candidate of candidates) {
      if (selected.every((corner) => Math.abs(corner.distance - candidate.distance) > minSeparation)) {
        selected.push(candidate);
      }
      if (selected.length >= targetCorners) break;
    }
    if (selected.length >= Math.min(targetCorners, candidates.length)) break;
    minSeparation *= 0.85;
  }

  return selected.sort((left, right) => left.distance - right.distance);
}

function classifyCorners(corners: F1TrackCornerMarker[]): F1TrackCornerMarker[] {
  if (!corners.length) return corners;
  const severities = corners.map((corner) => corner.severity).sort((left, right) => left - right);
  const mediumThreshold = severities[Math.max(0, Math.floor(severities.length * 0.38))];
  const slowThreshold = severities[Math.max(0, Math.floor(severities.length * 0.72))];

  return corners.map((corner) => ({
    ...corner,
    kind: corner.severity >= slowThreshold ? "slow" : corner.severity >= mediumThreshold ? "medium" : "fast",
  }));
}

function resampleTrack(points: RawPoint[], step: number): SampledPoint[] {
  if (!points.length) return [];
  const sampled: SampledPoint[] = [{ ...points[0], distance: 0 }];
  let accumulated = 0;
  let nextDistance = step;

  for (let index = 1; index < points.length; index += 1) {
    const start = points[index - 1];
    const end = points[index];
    const segment = pointDistance(start, end);
    while (segment > 0 && accumulated + segment >= nextDistance) {
      const ratio = (nextDistance - accumulated) / segment;
      sampled.push({
        x: start.x + (end.x - start.x) * ratio,
        y: start.y + (end.y - start.y) * ratio,
        distance: nextDistance,
      });
      nextDistance += step;
    }
    accumulated += segment;
  }

  return sampled;
}

function interpolateSvgPoint(svgPoints: RawPoint[], rawPoints: RawPoint[], targetDistance: number): RawPoint {
  let accumulated = 0;
  for (let index = 1; index < rawPoints.length; index += 1) {
    const segment = pointDistance(rawPoints[index - 1], rawPoints[index]);
    if (accumulated + segment >= targetDistance && segment > 0) {
      const ratio = (targetDistance - accumulated) / segment;
      return {
        x: roundSvg(svgPoints[index - 1].x + (svgPoints[index].x - svgPoints[index - 1].x) * ratio),
        y: roundSvg(svgPoints[index - 1].y + (svgPoints[index].y - svgPoints[index - 1].y) * ratio),
      };
    }
    accumulated += segment;
  }
  return svgPoints[svgPoints.length - 1] ?? { x: SVG_WIDTH / 2, y: SVG_HEIGHT / 2 };
}

function cumulativeDistance(points: RawPoint[]): number {
  let total = 0;
  for (let index = 1; index < points.length; index += 1) {
    total += pointDistance(points[index - 1], points[index]);
  }
  return total;
}

function pointDistance(left: RawPoint, right: RawPoint): number {
  return Math.hypot(right.x - left.x, right.y - left.y);
}

function angleDelta(left: number, right: number): number {
  let delta = right - left;
  while (delta > Math.PI) delta -= Math.PI * 2;
  while (delta < -Math.PI) delta += Math.PI * 2;
  return delta;
}

function geoJsonFeature(payload: unknown): GeoJsonFeature {
  if (isRecord(payload) && payload.type === "Feature") return payload as GeoJsonFeature;
  if (isRecord(payload) && payload.type === "FeatureCollection") {
    const collection = payload as GeoJsonFeatureCollection;
    const feature = collection.features?.[0];
    if (feature) return feature;
  }
  throw new Error("F1 track layout returned malformed GeoJSON");
}

function lineStringCoordinates(feature: GeoJsonFeature): Array<[number, number]> {
  const geometry = feature.geometry;
  if (!geometry || geometry.type !== "LineString" || !Array.isArray(geometry.coordinates)) return [];
  return geometry.coordinates.flatMap((coordinate) => {
    if (!Array.isArray(coordinate) || coordinate.length < 2) return [];
    const longitude = Number(coordinate[0]);
    const latitude = Number(coordinate[1]);
    return Number.isFinite(longitude) && Number.isFinite(latitude) ? [[longitude, latitude] as [number, number]] : [];
  });
}

function normalizeLayoutKey(value?: string | null): string {
  return String(value ?? "")
    .toLowerCase()
    .replace(/grand prix/g, "")
    .replace(/circuit/g, "")
    .replace(/autodrome/g, "")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function optionalString(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  const text = String(value).trim();
  return text || null;
}

function formatLengthKm(value: unknown): string | null {
  const length = Number(value);
  if (!Number.isFinite(length) || length <= 0) return null;
  return (length / 1000).toFixed(3);
}

function roundSvg(value: number): number {
  return Math.round(value * 100) / 100;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
