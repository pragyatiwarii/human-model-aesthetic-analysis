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

export async function getPerturbations() {
  return readJson(
    await fetch(`${API_BASE}/api/v1/perturbations`)
  );
}

export async function generateCounterfactuals({
  imageId,
  regionId,
  perturbationType,
}) {
  return readJson(
    await fetch(`${API_BASE}/api/v1/counterfactuals/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: imageId,
        region_id: regionId,
        perturbation_type: perturbationType,
        severities: ["subtle", "moderate", "strong"],
      }),
    })
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

export function module3HandoffUrl(batchId) {
  return `${API_BASE}/api/v1/counterfactuals/batches/${batchId}/handoff`;
}
