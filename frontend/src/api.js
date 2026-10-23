export const API_BASE =
  import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

async function readJson(response) {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(
      body.detail || `Request failed with status ${response.status}`
    );
  }

  return response.json();
}

export async function getDatasetStatus() {
  return readJson(
    await fetch(`${API_BASE}/api/v1/dataset/status`)
  );
}

export async function getRecords({
  offset = 0,
  limit = 25,
  search = "",
  generator = "",
} = {}) {
  const params = new URLSearchParams({
    offset: String(offset),
    limit: String(limit),
  });

  if (search) params.set("search", search);
  if (generator) params.set("generator", generator);

  return readJson(
    await fetch(
      `${API_BASE}/api/v1/dataset/records?${params.toString()}`
    )
  );
}

export async function runModule1(imageName) {
  return readJson(
    await fetch(
      `${API_BASE}/api/v1/dataset/analyze/${encodeURIComponent(imageName)}`,
      { method: "POST" }
    )
  );
}

export function absoluteUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_BASE}${path}`;
}

export function handoffUrl(imageId) {
  return `${API_BASE}/api/v1/images/${imageId}/handoff`;
}
