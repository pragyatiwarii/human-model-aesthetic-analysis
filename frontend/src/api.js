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

export async function getAestheticModelStatus() {
  return readJson(
    await fetch(`${API_BASE}/api/v1/aesthetic-model/status`)
  );
}

export async function trainAestheticModel({ force = false } = {}) {
  return readJson(
    await fetch(`${API_BASE}/api/v1/aesthetic-model/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force }),
    })
  );
}

export async function scoreAestheticBatch(batchId) {
  return readJson(
    await fetch(`${API_BASE}/api/v1/aesthetic/batches/${batchId}/score`, {
      method: "POST",
    })
  );
}

export async function submitHumanPreference({
  batchId,
  participantCode,
  ratings,
  preferredCandidateId,
  notes = "",
}) {
  return readJson(
    await fetch(`${API_BASE}/api/v1/human-preferences`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        batch_id: batchId,
        participant_code: participantCode,
        ratings,
        preferred_candidate_id: preferredCandidateId,
        notes: notes || null,
      }),
    })
  );
}

export function module4HandoffUrl(batchId) {
  return `${API_BASE}/api/v1/module3/batches/${batchId}/handoff`;
}
