import React, { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Aperture,
  ArrowLeft,
  ArrowRight,
  Beaker,
  CheckCircle2,
  BarChart3,
  Database,
  Download,
  Dice5,
  FlaskConical,
  LoaderCircle,
  Search,
  ShieldCheck,
  Sparkles,
  WandSparkles,
} from "lucide-react";

import {
  absoluteUrl,
  analyzeUploadedImage,
  generateCounterfactuals,
  getAestheticModelStatus,
  getDatasetStatus,
  getModule4Analysis,
  getModule4Dashboard,
  getPerturbations,
  getRecords,
  handoffUrl,
  module3HandoffUrl,
  module4AnalysisUrl,
  module4CsvUrl,
  module4ProjectCsvUrl,
  module4HandoffUrl,
  runModule1,
  scoreAestheticBatch,
  submitHumanPreference,
  submitPairwisePreference,
  trainAestheticModel,
} from "./api";

const PAGE_SIZE = 25;

function ScoreCard({ label, value, std }) {
  const available = value !== null && value !== undefined && Number.isFinite(Number(value));
  return (
    <div className="scoreCard">
      <span>{label}</span>
      <strong>{available ? Number(value).toFixed(3) : "N/A"}</strong>
      <small>{available && std !== null && std !== undefined ? `STD ${Number(std).toFixed(3)}` : "Custom image"}</small>
    </div>
  );
}

function RegionImage({ result, selectedRegion, onSelectRegion }) {
  return (
    <div className="imageStage">
      <div className="mediaFrame">
        <img
          src={absoluteUrl(result.original_image_url)}
          alt={result.dataset_image_name}
        />

        <div className="overlayLayer">
          {result.regions.map((region) => {
            const [x1, y1, x2, y2] = region.bbox;

            return (
              <button
                type="button"
                key={region.region_id}
                className={[
                  "regionBox",
                  selectedRegion?.region_id === region.region_id
                    ? "selected"
                    : "",
                  region.is_fallback ? "fallback" : "",
                ].join(" ")}
                style={{
                  left: `${x1 * 100}%`,
                  top: `${y1 * 100}%`,
                  width: `${(x2 - x1) * 100}%`,
                  height: `${(y2 - y1) * 100}%`,
                }}
                onClick={() => onSelectRegion(region)}
                title={
                  region.is_fallback
                    ? "Full image fallback"
                    : `${region.label} ${(region.confidence * 100).toFixed(1)}%`
                }
              >
                <span>
                  {region.is_fallback
                    ? "full image"
                    : `${region.label} ${(region.confidence * 100).toFixed(0)}%`}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function VariantCard({ variant }) {
  return (
    <article className="variantCard">
      <div className="variantHeader">
        <div>
          <span>Severity</span>
          <strong>{variant.severity}</strong>
        </div>
        <CheckCircle2 size={18} />
      </div>

      <img
        src={absoluteUrl(variant.counterfactual_url)}
        alt={`${variant.perturbation_type} ${variant.severity}`}
      />

      <div className="variantMeta">
        <div>
          <span>Parameters</span>
          <code>{JSON.stringify(variant.parameters)}</code>
        </div>
        <div>
          <span>Outside region</span>
          <strong>
            {variant.validation?.outside_region_unchanged
              ? "unchanged ✓"
              : "validation failed"}
          </strong>
        </div>
        <div>
          <span>Changed inside</span>
          <strong>
            {((variant.validation?.inside_region_changed_fraction || 0) * 100).toFixed(1)}%
          </strong>
        </div>
      </div>
    </article>
  );
}


function HumanCandidateCard({ candidate, label, rating, preferred, onRating, onPreferred, revealed, modelScore }) {
  const imageUrl = candidate.source_type === "original"
    ? candidate.image_url
    : candidate.counterfactual_url;
  return (
    <article className={["humanCandidate", preferred ? "preferred" : ""].join(" ")}>
      <div className="candidateTop">
        <div>
          <span>Human study candidate</span>
          <strong>{label}</strong>
        </div>
        <label className="bestChoice">
          <input
            type="radio"
            name="preferredCandidate"
            checked={preferred}
            onChange={onPreferred}
          />
          Best
        </label>
      </div>
      <img src={absoluteUrl(imageUrl)} alt={`Candidate ${label}`} />
      <div className="ratingBlock">
        <span>Aesthetic rating</span>
        <div className="ratingButtons">
          {[1, 2, 3, 4, 5].map((value) => (
            <button
              type="button"
              className={rating === value ? "active" : ""}
              onClick={() => onRating(value)}
              key={value}
            >
              {value}
            </button>
          ))}
        </div>
      </div>
      {revealed && (
        <div className="revealedMeta">
          <span>
            {candidate.source_type === "original"
              ? "Original"
              : `Counterfactual · ${candidate.severity}`}
          </span>
          <strong>
            Model score {modelScore == null ? "not scored" : Number(modelScore).toFixed(3)}
          </strong>
        </div>
      )}
    </article>
  );
}

function metricText(value, digits = 3) {
  if (value == null || Number.isNaN(Number(value))) return "Unavailable";
  return Number(value).toFixed(digits);
}

function AgreementMetric({ label, value, note }) {
  return (
    <div className="agreementMetric">
      <span>{label}</span>
      <strong>{value}</strong>
      {note && <small>{note}</small>}
    </div>
  );
}

function IssueBadge({ issue }) {
  return (
    <div className={`issueBadge ${issue.level}`}>
      <AlertTriangle size={16} />
      <div>
        <strong>{issue.code.replaceAll("_", " ")}</strong>
        <span>{issue.message}</span>
      </div>
    </div>
  );
}


function SeverityResponseChart({ rows = [] }) {
  if (!rows.length) return null;
  const width = 720;
  const height = 270;
  const pad = 46;
  const x = (index) => pad + (index / 3) * (width - pad * 2);
  const y = (value) => height - pad - ((Number(value) - 1) / 4) * (height - pad * 2);
  const modelPoints = rows.map((row) => `${x(row.severity_index)},${y(row.model_score)}`).join(" ");
  const humanPoints = rows.map((row) => `${x(row.severity_index)},${y(row.human_mean_rating)}`).join(" ");
  return (
    <div className="researchChart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Severity response curves">
        {[1, 2, 3, 4, 5].map((tick) => (
          <g key={tick}>
            <line x1={pad} y1={y(tick)} x2={width - pad} y2={y(tick)} className="chartGrid" />
            <text x={pad - 12} y={y(tick) + 4} textAnchor="end" className="chartLabel">{tick}</text>
          </g>
        ))}
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} className="chartAxis" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} className="chartAxis" />
        <polyline points={modelPoints} className="modelLine" />
        <polyline points={humanPoints} className="humanLine" />
        {rows.map((row) => (
          <g key={row.severity}>
            <circle cx={x(row.severity_index)} cy={y(row.model_score)} r="5" className="modelPoint" />
            <circle cx={x(row.severity_index)} cy={y(row.human_mean_rating)} r="5" className="humanPoint" />
            <text x={x(row.severity_index)} y={height - 18} textAnchor="middle" className="chartLabel">
              {row.severity}
            </text>
          </g>
        ))}
      </svg>
      <div className="chartLegend"><span><i className="modelSwatch" />Model score</span><span><i className="humanSwatch" />Human mean</span></div>
    </div>
  );
}

function CorrelationScatter({ points = [] }) {
  if (!points.length) return <div className="emptyChart">No correlation points yet.</div>;
  const width = 600;
  const height = 330;
  const pad = 48;
  const x = (value) => pad + ((Number(value) - 1) / 4) * (width - pad * 2);
  const y = (value) => height - pad - ((Number(value) - 1) / 4) * (height - pad * 2);
  return (
    <div className="researchChart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Human model correlation scatter plot">
        {[1, 2, 3, 4, 5].map((tick) => (
          <g key={tick}>
            <line x1={x(tick)} y1={pad} x2={x(tick)} y2={height - pad} className="chartGrid" />
            <line x1={pad} y1={y(tick)} x2={width - pad} y2={y(tick)} className="chartGrid" />
            <text x={x(tick)} y={height - 20} textAnchor="middle" className="chartLabel">{tick}</text>
            <text x={pad - 12} y={y(tick) + 4} textAnchor="end" className="chartLabel">{tick}</text>
          </g>
        ))}
        <line x1={x(1)} y1={y(1)} x2={x(5)} y2={y(5)} className="idealLine" />
        {points.map((point, index) => (
          <circle key={`${point.batch_id}-${point.candidate_id}-${index}`} cx={x(point.model_score)} cy={y(point.human_mean_rating)} r="6" className="scatterPoint">
            <title>{`${point.candidate}: model ${Number(point.model_score).toFixed(2)}, human ${Number(point.human_mean_rating).toFixed(2)}`}</title>
          </circle>
        ))}
        <text x={width / 2} y={height - 2} textAnchor="middle" className="chartAxisTitle">Model aesthetic score</text>
        <text x="15" y={height / 2} textAnchor="middle" transform={`rotate(-90 15 ${height / 2})`} className="chartAxisTitle">Human mean rating</text>
      </svg>
      <small className="chartNote">Dashed diagonal = perfect score agreement.</small>
    </div>
  );
}

function DisagreementBars({ rows = [], label = "Category" }) {
  if (!rows.length) return <div className="emptyChart">No grouped data yet.</div>;
  return (
    <div className="barChartList">
      {rows.map((row) => (
        <div className="barChartRow" key={row.name}>
          <div><strong>{row.name}</strong><small>{row.batch_count} batch{row.batch_count === 1 ? "" : "es"}</small></div>
          <div className="barTrack"><span style={{ width: `${Math.max(0, Math.min(100, (row.disagreement_rate || 0) * 100))}%` }} /></div>
          <b>{((row.disagreement_rate || 0) * 100).toFixed(1)}%</b>
        </div>
      ))}
      <small className="chartNote">{label} disagreement rate uses the configured substantial score-gap threshold.</small>
    </div>
  );
}

function DisagreementHeatmap({ cells = [] }) {
  if (!cells.length) return <div className="emptyChart">Run more experiments to populate the heatmap.</div>;
  const regions = [...new Set(cells.map((cell) => cell.region))];
  const perturbations = [...new Set(cells.map((cell) => cell.perturbation))];
  const lookup = new Map(cells.map((cell) => [`${cell.region}|||${cell.perturbation}`, cell]));
  return (
    <div className="heatmapWrap">
      <table className="heatmapTable">
        <thead><tr><th>Region</th>{perturbations.map((name) => <th key={name}>{name}</th>)}</tr></thead>
        <tbody>
          {regions.map((region) => (
            <tr key={region}>
              <th>{region}</th>
              {perturbations.map((perturbation) => {
                const cell = lookup.get(`${region}|||${perturbation}`);
                const rate = cell?.disagreement_rate;
                const alpha = rate == null ? 0.03 : 0.12 + rate * 0.68;
                return <td key={perturbation} style={{ background: `rgba(126, 96, 255, ${alpha})` }}>{rate == null ? "—" : `${(rate * 100).toFixed(0)}%`}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function App() {
  const [status, setStatus] = useState(null);
  const [records, setRecords] = useState([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);

  const [search, setSearch] = useState("");
  const [generator, setGenerator] = useState("");

  const [selected, setSelected] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [selectedRegion, setSelectedRegion] = useState(null);

  const [perturbations, setPerturbations] = useState([]);
  const [perturbationType, setPerturbationType] = useState("blur");
  const [batch, setBatch] = useState(null);

  const [modelStatus, setModelStatus] = useState(null);
  const [modelScores, setModelScores] = useState(null);
  const [participantCode, setParticipantCode] = useState("P001");
  const [humanRatings, setHumanRatings] = useState({});
  const [preferredCandidateId, setPreferredCandidateId] = useState("");
  const [preferenceSubmitted, setPreferenceSubmitted] = useState(false);
  const [pairwiseResponses, setPairwiseResponses] = useState({});
  const [pairwiseSubmitted, setPairwiseSubmitted] = useState(false);
  const [submittingPairwise, setSubmittingPairwise] = useState(false);
  const [blindStudy, setBlindStudy] = useState(false);
  const [trainingModel, setTrainingModel] = useState(false);
  const [scoringBatch, setScoringBatch] = useState(false);
  const [submittingPreference, setSubmittingPreference] = useState(false);
  const [agreementAnalysis, setAgreementAnalysis] = useState(null);
  const [projectDashboard, setProjectDashboard] = useState(null);
  const [analysingAgreement, setAnalysingAgreement] = useState(false);
  const [loadingDashboard, setLoadingDashboard] = useState(false);

  const [loadingList, setLoadingList] = useState(false);
  const [analysing, setAnalysing] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState("");
  const [customFile, setCustomFile] = useState(null);
  const [customPrompt, setCustomPrompt] = useState("");
  const [customGenerator, setCustomGenerator] = useState("custom-upload");
  const [customCategory, setCustomCategory] = useState("");
  const [uploadingCustom, setUploadingCustom] = useState(false);

  const generators = status?.generators || [];

  const page = Math.floor(offset / PAGE_SIZE) + 1;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const first = total ? offset + 1 : 0;
  const last = Math.min(offset + PAGE_SIZE, total);

  async function refresh({
    nextOffset = 0,
    nextSearch = search,
    nextGenerator = generator,
  } = {}) {
    setLoadingList(true);
    setError("");

    try {
      const data = await getRecords({
        offset: nextOffset,
        limit: PAGE_SIZE,
        search: nextSearch,
        generator: nextGenerator,
      });

      setRecords(data.records);
      setTotal(data.total);
      setOffset(nextOffset);

      const stillVisible =
        selected &&
        data.records.some((record) => record.name === selected.name);

      if (!stillVisible) {
        setSelected(data.records[0] || null);
        setAnalysis(null);
        setSelectedRegion(null);
        setBatch(null);
        setModelScores(null);
        setHumanRatings({});
        setPreferredCandidateId("");
        setPreferenceSubmitted(false);
        setPairwiseResponses({});
        setPairwiseSubmitted(false);
        setBlindStudy(false);
        setAgreementAnalysis(null);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingList(false);
    }
  }

  useEffect(() => {
    getDatasetStatus()
      .then(setStatus)
      .catch((err) => setError(err.message));

    getAestheticModelStatus()
      .then(setModelStatus)
      .catch((err) => setError(err.message));

    getPerturbations()
      .then((items) => {
        setPerturbations(items);
        if (items.length && !items.some((item) => item.key === "blur")) {
          setPerturbationType(items[0].key);
        }
      })
      .catch((err) => setError(err.message));

    refresh({
      nextOffset: 0,
      nextSearch: "",
      nextGenerator: "",
    });
  }, []);

  async function analyseCustomImage(event) {
    event?.preventDefault();
    if (!customFile) {
      setError("Choose an image file first.");
      return;
    }
    if (!customPrompt.trim()) {
      setError("Enter the original generation prompt for the custom image.");
      return;
    }
    setUploadingCustom(true);
    setError("");
    try {
      const data = await analyzeUploadedImage({
        file: customFile,
        prompt: customPrompt.trim(),
        generator: customGenerator.trim() || "custom-upload",
        category: customCategory.trim(),
      });
      setSelected({
        name: data.dataset_image_name,
        prompt: data.prompt,
        generator: data.generator || "custom-upload",
        mos_quality: null,
        std_quality: null,
        mos_align: null,
        std_align: null,
        image_exists: true,
        image_url: data.original_image_url,
        is_custom: true,
      });
      setAnalysis(data);
      setSelectedRegion(data.regions?.[0] || null);
      setBatch(null);
      setModelScores(null);
      setHumanRatings({});
      setPreferredCandidateId("");
      setPreferenceSubmitted(false);
      setPairwiseResponses({});
      setPairwiseSubmitted(false);
      setAgreementAnalysis(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploadingCustom(false);
    }
  }

  async function analyse() {
    if (!selected) return;

    setAnalysing(true);
    setAnalysis(null);
    setSelectedRegion(null);
    setBatch(null);
    setModelScores(null);
    setHumanRatings({});
    setPreferredCandidateId("");
    setPreferenceSubmitted(false);
    setPairwiseResponses({});
    setPairwiseSubmitted(false);
    setBlindStudy(false);
    setAgreementAnalysis(null);
    setError("");

    try {
      const data = await runModule1(selected.name);
      setAnalysis(data);
      setSelectedRegion(data.regions?.[0] || null);
    } catch (err) {
      setError(err.message);
    } finally {
      setAnalysing(false);
    }
  }

  async function generate() {
    if (!analysis || !selectedRegion || !perturbationType) return;

    setGenerating(true);
    setBatch(null);
    setError("");

    try {
      const data = await generateCounterfactuals({
        imageId: analysis.image_id,
        regionId: selectedRegion.region_id,
        perturbationType,
      });
      setBatch(data);
      setModelScores(null);
      setHumanRatings({});
      setPreferredCandidateId("");
      setPreferenceSubmitted(false);
      setPairwiseResponses({});
      setPairwiseSubmitted(false);
      setBlindStudy(false);
      setAgreementAnalysis(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setGenerating(false);
    }
  }


  async function trainModel() {
    setTrainingModel(true);
    setError("");
    try {
      const data = await trainAestheticModel();
      setModelStatus(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setTrainingModel(false);
    }
  }

  async function scoreBatch() {
    if (!batch) return;
    setScoringBatch(true);
    setError("");
    try {
      const data = await scoreAestheticBatch(batch.batch_id);
      setModelScores(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setScoringBatch(false);
    }
  }

  function nextParticipantCode(current) {
    const value = (current || "").trim();
    const match = value.match(/^(.*?)(\d+)$/);
    if (!match) return "P002";

    const [, prefix, digits] = match;
    const nextNumber = String(Number(digits) + 1).padStart(digits.length, "0");
    return `${prefix}${nextNumber}`;
  }

  function startNextParticipant() {
    setParticipantCode((current) => nextParticipantCode(current));
    setHumanRatings({});
    setPreferredCandidateId("");
    setPreferenceSubmitted(false);
    setPairwiseResponses({});
    setPairwiseSubmitted(false);
    setBlindStudy(true);
    setAgreementAnalysis(null);
    setError("");
  }

  async function submitPreference() {
    if (!batch) return;
    const candidates = module3Candidates;
    if (!participantCode.trim()) {
      setError("Enter a participant code before submitting.");
      return;
    }
    if (candidates.some((candidate) => !humanRatings[candidate.candidate_id])) {
      setError("Rate every candidate from 1 to 5 before submitting.");
      return;
    }
    if (!preferredCandidateId) {
      setError("Choose the single best candidate before submitting.");
      return;
    }
    setSubmittingPreference(true);
    setError("");
    try {
      await submitHumanPreference({
        batchId: batch.batch_id,
        participantCode: participantCode.trim(),
        ratings: humanRatings,
        preferredCandidateId,
      });
      setPreferenceSubmitted(true);
      setBlindStudy(false);
      setAgreementAnalysis(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmittingPreference(false);
    }
  }

  async function submitPairwise() {
    if (!batch) return;
    const counterfactuals = batch.variants || [];
    if (counterfactuals.some((variant) => !pairwiseResponses[variant.counterfactual_id])) {
      setError("Choose Original, Modified, or About the same for every severity.");
      return;
    }
    setSubmittingPairwise(true);
    setError("");
    try {
      await submitPairwisePreference({
        batchId: batch.batch_id,
        participantCode: participantCode.trim(),
        responses: pairwiseResponses,
      });
      setPairwiseSubmitted(true);
      setAgreementAnalysis(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmittingPairwise(false);
    }
  }

  async function analyseHumanModelAgreement() {
    if (!batch) return;
    setAnalysingAgreement(true);
    setError("");
    try {
      const data = await getModule4Analysis(batch.batch_id);
      setAgreementAnalysis(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setAnalysingAgreement(false);
    }
  }

  async function refreshProjectDashboard() {
    setLoadingDashboard(true);
    setError("");
    try {
      const data = await getModule4Dashboard();
      setProjectDashboard(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingDashboard(false);
    }
  }

  function randomPage() {
    if (!total) return;

    const randomIndex = Math.floor(Math.random() * total);
    const nextOffset = Math.floor(randomIndex / PAGE_SIZE) * PAGE_SIZE;

    refresh({ nextOffset });
  }

  const objectSummary = useMemo(() => {
    if (!analysis) return "";

    if (analysis.context?.detector_objects?.length) {
      return analysis.context.detector_objects
        .map((item) => `${item.label} ${(item.confidence * 100).toFixed(1)}% (${item.reliability || "uncertain"})`)
        .join(", ");
    }

    if (analysis.regions.some((region) => region.is_fallback)) {
      return "No detector object passed the reliability threshold. Full image is available as the editable region.";
    }

    return "No detector-confirmed object detected.";
  }, [analysis]);

  const contextEntitySummary = (items) =>
    items?.length ? items.map((item) => item.label).join(" · ") : "None extracted";


  const module3Candidates = useMemo(() => {
    if (!batch) return [];
    const original = {
      candidate_id: `original:${batch.original_image_id}`,
      source_type: "original",
      severity: null,
      image_url: batch.original_image_url,
    };
    const variants = batch.variants.map((variant) => ({
      ...variant,
      candidate_id: variant.counterfactual_id,
      source_type: "counterfactual",
    }));
    const all = [original, ...variants];
    // Stable anti-bias display order derived from the batch ID.
    return all
      .map((item) => ({
        item,
        key: `${batch.batch_id}:${item.candidate_id}`.split("").reduce((a, c) => ((a * 33) ^ c.charCodeAt(0)) >>> 0, 5381),
      }))
      .sort((a, b) => a.key - b.key)
      .map(({ item }) => item);
  }, [batch]);

  const scoreByCandidate = useMemo(() => {
    const map = {};
    for (const item of modelScores?.candidates || []) map[item.candidate_id] = item.score;
    return map;
  }, [modelScores]);

  const selectedPerturbation = perturbations.find(
    (item) => item.key === perturbationType
  );

  return (
    <main className={blindStudy ? "appShell blindStudy" : "appShell"}>
      <header className="topbar">
        <div className="brand">
          <div className="logo">
            <Aperture size={20} />
          </div>

          <div>
            <h1>Aesthetic Lens</h1>
            <p>AGIQA-3K · Modules 1 → 4 · context-quality enhanced</p>
          </div>
        </div>

        <div className="modulePills">
          <div className="ownerPill">01 · Vennela</div>
          <div className="ownerPill">02 · Sanyojita</div>
          <div className="ownerPill">03 · Muskan</div>
          <div className="ownerPill activeModule">04 · Pragya</div>
        </div>
      </header>

      <section className="hero">
        <div>
          <span className="eyebrow">
            CONTEXT → REGION → COUNTERFACTUAL → HUMAN + MODEL → AGREEMENT · 04/04
          </span>

          <h2>
            Measure <em>where humans and the model agree</em> — and where they do not.
          </h2>

          <p>
            Module 1 extracts structured visual/context information and editable regions. Module 2 creates controlled counterfactuals; Module 3 collects model scores plus both blind ratings and the PDF pairwise preference protocol for Module 4.
          </p>
        </div>

        <div className="statusCard">
          <Database size={20} />
          <div>
            <strong>AGIQA-3K</strong>
            <span>
              {status
                ? `${status.record_count} rows · ${status.available_image_count} images found`
                : "Checking dataset…"}
            </span>
          </div>
          <i className={status?.ready ? "ready" : ""} />
        </div>
      </section>

      {status && !status.ready && (
        <div className="warning">
          Copy `data.csv` and the AGIQA-3K images into
          `backend/data/agiqa3k/` before testing.
        </div>
      )}

      <section className="panel customInputPanel">
        <div className="panelHeading">
          <Sparkles size={18} />
          <div>
            <h3>Analyze a new AI-generated image</h3>
            <p>Optional acceptance-criteria path; AGIQA-3K remains the main experiment dataset</p>
          </div>
        </div>
        <form className="customInputForm" onSubmit={analyseCustomImage}>
          <label>
            <span>Image</span>
            <input type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => setCustomFile(event.target.files?.[0] || null)} />
          </label>
          <label className="customPromptField">
            <span>Original generation prompt</span>
            <input value={customPrompt} onChange={(event) => setCustomPrompt(event.target.value)} placeholder="e.g. a woman holding a red umbrella beside a white car" />
          </label>
          <label>
            <span>Generator/source</span>
            <input value={customGenerator} onChange={(event) => setCustomGenerator(event.target.value)} placeholder="custom-upload" />
          </label>
          <label>
            <span>Category/style</span>
            <input value={customCategory} onChange={(event) => setCustomCategory(event.target.value)} placeholder="optional" />
          </label>
          <button type="submit" className="primary" disabled={uploadingCustom}>
            {uploadingCustom ? <LoaderCircle className="spin" size={18} /> : <Sparkles size={18} />}
            {uploadingCustom ? "Analyzing image…" : "Run Module 1 on custom image"}
          </button>
        </form>
      </section>

      <section className="workspace">
        <aside className="panel sidebar">
          <div className="panelHeading">
            <Database size={18} />
            <div>
              <h3>AGIQA-3K samples</h3>
              <p>{first}–{last} of {total}</p>
            </div>
          </div>

          <form
            className="searchBox"
            onSubmit={(event) => {
              event.preventDefault();
              refresh({
                nextOffset: 0,
                nextSearch: search,
                nextGenerator: generator,
              });
            }}
          >
            <Search size={15} />
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search prompt or filename"
            />
          </form>

          <div className="tools">
            <select
              value={generator}
              onChange={(event) => {
                const value = event.target.value;
                setGenerator(value);
                refresh({
                  nextOffset: 0,
                  nextSearch: search,
                  nextGenerator: value,
                });
              }}
            >
              <option value="">All generators</option>
              {generators.map((item) => (
                <option value={item} key={item}>{item}</option>
              ))}
            </select>

            <button type="button" onClick={randomPage}>
              <Dice5 size={14} />
              Random
            </button>
          </div>

          <div className="recordList">
            {loadingList ? (
              <div className="loading">
                <LoaderCircle className="spin" size={18} />
                Loading records…
              </div>
            ) : records.length ? (
              records.map((record) => (
                <button
                  type="button"
                  key={record.name}
                  className={[
                    "record",
                    selected?.name === record.name ? "active" : "",
                  ].join(" ")}
                  onClick={() => {
                    setSelected(record);
                    setAnalysis(null);
                    setSelectedRegion(null);
                    setBatch(null);
                  }}
                >
                  <div>
                    <strong>{record.name}</strong>
                    <span className={record.image_exists ? "available" : "missing"}>
                      {record.image_exists ? "image" : "missing"}
                    </span>
                  </div>
                  <p>{record.prompt}</p>
                  <small>{record.generator}</small>
                </button>
              ))
            ) : (
              <div className="emptyList">No matching records.</div>
            )}
          </div>

          <div className="pagination">
            <button
              type="button"
              disabled={offset === 0}
              onClick={() => refresh({ nextOffset: Math.max(0, offset - PAGE_SIZE) })}
            >
              <ArrowLeft size={15} />
            </button>
            <span>Page {page} of {pages}</span>
            <button
              type="button"
              disabled={offset + PAGE_SIZE >= total}
              onClick={() => refresh({ nextOffset: offset + PAGE_SIZE })}
            >
              <ArrowRight size={15} />
            </button>
          </div>
        </aside>

        <section className="panel content">
          {!selected ? (
            <div className="emptyState">
              <Database size={42} />
              <strong>Select an AGIQA-3K record</strong>
            </div>
          ) : (
            <>
              <div className="sampleHeader">
                <div>
                  <span className="eyebrow">{selected.generator}</span>
                  <h3>{selected.name}</h3>
                  <p>{selected.prompt}</p>
                </div>

                <button
                  type="button"
                  className="primary"
                  disabled={analysing || !selected.image_exists || selected.is_custom}
                  onClick={analyse}
                >
                  {analysing ? (
                    <LoaderCircle className="spin" size={18} />
                  ) : (
                    <Sparkles size={18} />
                  )}
                  {selected.is_custom ? "Custom image analyzed" : analysing ? "Running Module 1…" : "Run Module 1"}
                </button>
              </div>

              <div className="scoreGrid">
                <ScoreCard
                  label="Human perceptual MOS"
                  value={selected.mos_quality}
                  std={selected.std_quality}
                />
                <ScoreCard
                  label="Human alignment MOS"
                  value={selected.mos_align}
                  std={selected.std_align}
                />
              </div>

              {!analysis && selected.image_url && (
                <div className="previewOnly">
                  <img src={absoluteUrl(selected.image_url)} alt={selected.name} />
                </div>
              )}

              {analysis && (
                <>
                  <RegionImage
                    result={analysis}
                    selectedRegion={selectedRegion}
                    onSelectRegion={(region) => {
                      setSelectedRegion(region);
                      setBatch(null);
                    }}
                  />

                  <div className="regionStrip">
                    {analysis.regions.map((region) => (
                      <button
                        type="button"
                        key={region.region_id}
                        className={
                          selectedRegion?.region_id === region.region_id
                            ? "active"
                            : ""
                        }
                        onClick={() => {
                          setSelectedRegion(region);
                          setBatch(null);
                        }}
                      >
                        <span>{region.is_fallback ? "full image" : region.label}</span>
                        <small>
                          {region.is_fallback
                            ? "fallback"
                            : `${(region.confidence * 100).toFixed(1)}%`}
                        </small>
                      </button>
                    ))}
                  </div>
                </>
              )}
            </>
          )}

          {error && <div className="error">{error}</div>}
        </section>
      </section>

      {analysis && (
        <section className="resultGrid">
          <article className="panel">
            <div className="panelHeading">
              <ShieldCheck size={18} />
              <div>
                <h3>Module 1 result</h3>
                <p>v1.6 context-quality + region intelligence</p>
              </div>
            </div>

            <div className="facts">
              <div><span>Original prompt</span><strong>{analysis.prompt}</strong></div>
              <div><span>Reliable objects</span><strong>{objectSummary}</strong></div>
              <div>
                <span>Confidence threshold</span>
                <strong>{analysis.detector.confidence_threshold}</strong>
              </div>
              <div>
                <span>Selected region</span>
                <strong>
                  {selectedRegion?.is_fallback
                    ? "full_image fallback"
                    : selectedRegion?.label}
                </strong>
              </div>
            </div>
          </article>

          <article className="panel contextPanel">
            <div className="panelHeading">
              <Activity size={18} />
              <div>
                <h3>Structured context extraction</h3>
                <p>VLM-assisted with prompt + detector fallback</p>
              </div>
            </div>
            <div className="contextFacts">
              <div>
                <span>Extraction mode</span>
                <strong>{analysis.context?.extraction?.mode || "prompt+detector-fallback"}</strong>
              </div>
              <div>
                <span>Visual caption</span>
                <strong>{analysis.context?.visual_caption || "VLM unavailable; structured fallback used"}</strong>
              </div>
              <div>
                <span>Scene/context</span>
                <strong>{analysis.context?.scene?.tags?.length ? analysis.context.scene.tags.join(", ") : "No explicit scene tags"}</strong>
              </div>
              <div>
                <span>Detector objects</span>
                <strong>{analysis.context?.detector_objects?.length ? analysis.context.detector_objects.map((item) => `${item.label} ${(item.confidence * 100).toFixed(1)}% · ${item.reliability}`).join(" · ") : "No detector-confirmed objects"}</strong>
              </div>
              <div>
                <span>Prompt entities</span>
                <strong>{contextEntitySummary(analysis.context?.prompt_entities)}</strong>
              </div>
              <div>
                <span>VLM entities</span>
                <strong>{contextEntitySummary(analysis.context?.vlm_entities)}</strong>
              </div>
              <div className="contextWide">
                <span>Attributes</span>
                <strong>{analysis.context?.attributes?.length ? analysis.context.attributes.map((item) => `${item.object}: ${item.value}`).join(" · ") : "No explicit attributes extracted"}</strong>
              </div>
              <div className="contextWide">
                <span>Relationships</span>
                <strong>{analysis.context?.relationships?.length ? analysis.context.relationships.map((item) => `${item.subject} → ${item.relation.replaceAll("_", " ")} → ${item.object}`).join(" · ") : "No reliable relationship extracted"}</strong>
              </div>
              {analysis.context?.detector_objects?.some((item) => item.reliability === "uncertain") && (
                <div className="contextWide">
                  <span>Detection note</span>
                  <strong>Uncertain detector labels are retained for editing but should be visually verified before being treated as semantic ground truth.</strong>
                </div>
              )}
            </div>
          </article>

          <article className="panel">
            <div className="panelHeading">
              <CheckCircle2 size={18} />
              <div>
                <h3>Module 1 → Module 2 handoff</h3>
                <p>Stable v1.4 contract</p>
              </div>
            </div>

            <pre>
              {JSON.stringify(
                {
                  image_id: analysis.image_id,
                  dataset_image_name: analysis.dataset_image_name,
                  prompt: analysis.prompt,
                  context: analysis.context,
                  selected_region: selectedRegion,
                  ground_truth: {
                    mos_quality: analysis.mos_quality,
                    mos_align: analysis.mos_align,
                  },
                },
                null,
                2
              )}
            </pre>

            <a
              className="handoffButton"
              href={handoffUrl(analysis.image_id)}
              target="_blank"
              rel="noreferrer"
            >
              Open complete Module 1 handoff JSON
            </a>
          </article>
        </section>
      )}

      {analysis && selectedRegion && (
        <section className="module2Section">
          <div className="module2Intro">
            <div>
              <span className="eyebrow">MODULE 2 · SANYOJITA</span>
              <h2>Controlled counterfactual generation</h2>
              <p>
                The selected region is the only area permitted to change.
                Outputs are saved as lossless PNG and automatically checked
                to confirm that pixels outside the region are unchanged.
              </p>
            </div>
            <FlaskConical size={38} />
          </div>

          <div className="module2Grid">
            <article className="panel controlPanel">
              <div className="panelHeading">
                <Beaker size={18} />
                <div>
                  <h3>Experiment controls</h3>
                  <p>One perturbation · three severities</p>
                </div>
              </div>

              <div className="controlFacts">
                <div>
                  <span>Target region</span>
                  <strong>
                    {selectedRegion.is_fallback ? "full_image" : selectedRegion.label}
                  </strong>
                </div>
                <div>
                  <span>Region ID</span>
                  <code>{selectedRegion.region_id}</code>
                </div>
              </div>

              <label className="fieldLabel" htmlFor="perturbation">
                Perturbation
              </label>
              <select
                id="perturbation"
                className="perturbationSelect"
                value={perturbationType}
                onChange={(event) => {
                  setPerturbationType(event.target.value);
                  setBatch(null);
                }}
              >
                {perturbations.map((item) => (
                  <option value={item.key} key={item.key}>{item.label}</option>
                ))}
              </select>

              {selectedPerturbation && (
                <div className="specBox">
                  <strong>{selectedPerturbation.label}</strong>
                  <p>{selectedPerturbation.description}</p>
                  <div className="severitySpec">
                    {Object.entries(selectedPerturbation.severities).map(
                      ([severity, params]) => (
                        <span key={severity}>
                          <b>{severity}</b> {JSON.stringify(params)}
                        </span>
                      )
                    )}
                  </div>
                </div>
              )}

              <button
                type="button"
                className="primary generateButton"
                disabled={generating || !perturbationType}
                onClick={generate}
              >
                {generating ? (
                  <LoaderCircle className="spin" size={18} />
                ) : (
                  <WandSparkles size={18} />
                )}
                {generating
                  ? "Generating counterfactuals…"
                  : "Generate subtle · moderate · strong"}
              </button>
            </article>

            <article className="panel methodologyPanel">
              <div className="panelHeading">
                <ShieldCheck size={18} />
                <div>
                  <h3>Isolation guarantee</h3>
                  <p>Designed for scientific comparison</p>
                </div>
              </div>

              <div className="methodFlow">
                <span>original image</span>
                <b>→</b>
                <span>selected mask</span>
                <b>→</b>
                <span>controlled change</span>
                <b>→</b>
                <span>PNG + validation</span>
              </div>

              <p className="methodNote">
                Module 2 composites the modified candidate only inside the
                Module 1 mask/polygon. Outside-region maximum pixel difference
                must be exactly 0 before the batch is accepted.
              </p>
            </article>
          </div>

          {batch && (
            <>
              <div className="variantGrid">
                {batch.variants.map((variant) => (
                  <VariantCard
                    key={variant.counterfactual_id}
                    variant={variant}
                  />
                ))}
              </div>

              <article className="panel module3Handoff">
                <div className="panelHeading">
                  <CheckCircle2 size={18} />
                  <div>
                    <h3>Module 3 handoff ready</h3>
                    <p>Original + counterfactuals + exact perturbation metadata</p>
                  </div>
                </div>

                <pre>
                  {JSON.stringify(
                    {
                      batch_id: batch.batch_id,
                      original_image_id: batch.original_image_id,
                      region: batch.region,
                      perturbation_type: batch.perturbation_type,
                      variants: batch.variants.map((variant) => ({
                        severity: variant.severity,
                        parameters: variant.parameters,
                        validation: variant.validation,
                        counterfactual_url: variant.counterfactual_url,
                      })),
                    },
                    null,
                    2
                  )}
                </pre>

                <a
                  className="handoffButton"
                  href={module3HandoffUrl(batch.batch_id)}
                  target="_blank"
                  rel="noreferrer"
                >
                  Open complete Module 3 handoff JSON
                </a>
              </article>
            </>
          )}
        </section>
      )}


      {batch && (
        <section className="module3Section">
          <div className="module2Intro module3Intro">
            <div>
              <span className="eyebrow">MODULE 3 · MUSKAN</span>
              <h2>Aesthetic scoring + human preference collection</h2>
              <p>
                Score the original and all three counterfactuals with the same
                cached AGIQA-trained model, then collect blind 1–5 human ratings
                and one preferred candidate for Module 4 agreement analysis.
              </p>
            </div>
            <Sparkles size={40} />
          </div>

          <div className="module3Controls">
            <article className="panel modelPanel">
              <div className="panelHeading">
                <Database size={18} />
                <div>
                  <h3>Local aesthetic model</h3>
                  <p>AGIQA-3K MOS-trained ridge regressor</p>
                </div>
              </div>
              <div className="modelFacts">
                <div><span>Status</span><strong>{modelStatus?.trained ? "trained ✓" : "not trained"}</strong></div>
                <div><span>Version</span><strong>{modelStatus?.version || "agiqa-ridge-v1"}</strong></div>
                {modelStatus?.validation_metrics && (
                  <>
                    <div><span>Validation MAE</span><strong>{Number(modelStatus.validation_metrics.mae).toFixed(3)}</strong></div>
                    <div><span>Validation R²</span><strong>{Number(modelStatus.validation_metrics.r2).toFixed(3)}</strong></div>
                  </>
                )}
              </div>
              {!modelStatus?.trained ? (
                <button type="button" className="primary" disabled={trainingModel} onClick={trainModel}>
                  {trainingModel ? <LoaderCircle className="spin" size={18} /> : <Sparkles size={18} />}
                  {trainingModel ? "Training from AGIQA-3K…" : "Train aesthetic model"}
                </button>
              ) : (
                <button type="button" className="primary" disabled={scoringBatch} onClick={scoreBatch}>
                  {scoringBatch ? <LoaderCircle className="spin" size={18} /> : <CheckCircle2 size={18} />}
                  {scoringBatch ? "Scoring four candidates…" : "Score original + 3 counterfactuals"}
                </button>
              )}
              <p className="methodNote">
                The model is project-specific and interpretable. It learns AGIQA-3K
                quality MOS from deterministic visual features and is cached after training.
              </p>
            </article>

            <article className="panel humanPanel">
              <div className="panelHeading">
                <ShieldCheck size={18} />
                <div>
                  <h3>Human preference study</h3>
                  <p>Blind scoring before identities/model scores are revealed</p>
                </div>
              </div>
              <label className="fieldLabel" htmlFor="participantCode">Participant code</label>
              <input
                id="participantCode"
                className="participantInput"
                value={participantCode}
                onChange={(event) => setParticipantCode(event.target.value)}
                disabled={preferenceSubmitted}
                placeholder="P001"
              />
              {!preferenceSubmitted ? (
                <button type="button" className="secondaryStudyButton" onClick={() => setBlindStudy(true)}>
                  Open blind participant view
                </button>
              ) : pairwiseSubmitted ? (
                <button type="button" className="secondaryStudyButton" onClick={startNextParticipant}>
                  Start next participant
                </button>
              ) : (
                <button type="button" className="secondaryStudyButton" disabled>
                  Complete PDF pairwise protocol below
                </button>
              )}
              <p className="methodNote">
                Ask the participant to rate every candidate from 1 (poor) to 5
                (excellent) and mark exactly one as Best. Candidate identities are
                hidden until submission to reduce bias.
              </p>
            </article>
          </div>

          <div className="humanCandidateGrid">
            {module3Candidates.map((candidate, index) => (
              <HumanCandidateCard
                key={candidate.candidate_id}
                candidate={candidate}
                label={String.fromCharCode(65 + index)}
                rating={humanRatings[candidate.candidate_id] || 0}
                preferred={preferredCandidateId === candidate.candidate_id}
                onRating={(value) => !preferenceSubmitted && setHumanRatings((current) => ({ ...current, [candidate.candidate_id]: value }))}
                onPreferred={() => !preferenceSubmitted && setPreferredCandidateId(candidate.candidate_id)}
                revealed={preferenceSubmitted}
                modelScore={scoreByCandidate[candidate.candidate_id]}
              />
            ))}
          </div>

          <div className="module3SubmitRow">
            <button
              type="button"
              className="primary"
              disabled={submittingPreference || preferenceSubmitted}
              onClick={submitPreference}
            >
              {submittingPreference ? <LoaderCircle className="spin" size={18} /> : <ShieldCheck size={18} />}
              {preferenceSubmitted ? "Human preference saved ✓" : "Submit human preference"}
            </button>
            {!modelScores && modelStatus?.trained && (
              <span>Score the batch to prepare the Module 4 model side.</span>
            )}
          </div>

          {preferenceSubmitted && (
            <article className="panel pairwisePanel">
              <div className="panelHeading">
                <ShieldCheck size={18} />
                <div>
                  <h3>PDF pairwise preference protocol</h3>
                  <p>For each severity: Original / Modified / About the same</p>
                </div>
              </div>
              <p className="methodNote">This exact pairwise response is stored in addition to the blind 1–5 ratings and Best vote.</p>
              <div className="pairwiseRows">
                {batch.variants.map((variant) => (
                  <div className="pairwiseRow" key={variant.counterfactual_id}>
                    <div className="pairwiseImages">
                      <div><span>Original</span><img src={absoluteUrl(batch.original_image_url)} alt="Original" /></div>
                      <div><span>{variant.severity}</span><img src={absoluteUrl(variant.counterfactual_url)} alt={variant.severity} /></div>
                    </div>
                    <div className="pairwiseChoices">
                      {[
                        ["original", "Original"],
                        ["modified", "Modified"],
                        ["same", "About the same"],
                      ].map(([value, label]) => (
                        <button
                          type="button"
                          key={value}
                          className={pairwiseResponses[variant.counterfactual_id] === value ? "active" : ""}
                          disabled={pairwiseSubmitted}
                          onClick={() => setPairwiseResponses((current) => ({ ...current, [variant.counterfactual_id]: value }))}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
              <button type="button" className="primary" disabled={pairwiseSubmitted || submittingPairwise} onClick={submitPairwise}>
                {submittingPairwise ? <LoaderCircle className="spin" size={18} /> : <CheckCircle2 size={18} />}
                {pairwiseSubmitted ? "Pairwise responses saved ✓" : "Submit pairwise preferences"}
              </button>
            </article>
          )}

          {preferenceSubmitted && pairwiseSubmitted && modelScores && (
            <article className="panel module3Handoff">
              <div className="panelHeading">
                <CheckCircle2 size={18} />
                <div>
                  <h3>Module 4 handoff ready</h3>
                  <p>Model scores + human ratings/preferences + experiment metadata</p>
                </div>
              </div>
              <div className="modelPreference">
                <span>Model-preferred candidate</span>
                <strong>{modelScores.model_preferred_candidate_id}</strong>
              </div>
              <a
                className="handoffButton"
                href={module4HandoffUrl(batch.batch_id)}
                target="_blank"
                rel="noreferrer"
              >
                Open complete Module 4 handoff JSON
              </a>
            </article>
          )}
        </section>
      )}

      {batch && preferenceSubmitted && pairwiseSubmitted && modelScores && (
        <section className="module4Section">
          <div className="module2Intro module4Intro">
            <div>
              <span className="eyebrow">MODULE 4 · PRAGYA · PDF-ALIGNED v4.3</span>
              <h2>Human–model quantitative analysis + research dashboard</h2>
              <p>
                Compute the quantitative measures required by the project specification, visualize
                severity and disagreement patterns, and keep every result reproducible from stored
                model scores, human responses and perturbation metadata.
              </p>
            </div>
            <BarChart3 size={42} />
          </div>

          <div className="module4Actions">
            <button type="button" className="primary" disabled={analysingAgreement} onClick={analyseHumanModelAgreement}>
              {analysingAgreement ? <LoaderCircle className="spin" size={18} /> : <Activity size={18} />}
              {analysingAgreement ? "Analyzing agreement…" : "Analyze this human–model batch"}
            </button>
            <button type="button" className="secondaryStudyButton" disabled={loadingDashboard} onClick={refreshProjectDashboard}>
              {loadingDashboard ? "Refreshing dashboard…" : "Refresh project dashboard"}
            </button>
          </div>

          {agreementAnalysis && (
            <>
              <div className="agreementMetricGrid wideMetrics">
                <AgreementMetric label="Participants" value={agreementAnalysis.participant_count} />
                <AgreementMetric label="Top-choice agreement" value={agreementAnalysis.top_choice_agreement ? "YES ✓" : "NO"} note={`${(agreementAnalysis.human_consensus_strength * 100).toFixed(0)}% human consensus`} />
                <AgreementMetric label="Directional agreement" value={agreementAnalysis.directional_agreement_rate == null ? "Unavailable" : `${(agreementAnalysis.directional_agreement_rate * 100).toFixed(1)}%`} note="human/model Δ direction" />
                <AgreementMetric label="Disagreement rate" value={`${(agreementAnalysis.disagreement_rate * 100).toFixed(1)}%`} note={`gap ≥ ${agreementAnalysis.disagreement_score_gap_threshold}`} />
                <AgreementMetric label="Pearson r" value={metricText(agreementAnalysis.pearson_r)} note="linear association" />
                <AgreementMetric label="Spearman ρ" value={metricText(agreementAnalysis.spearman_rho)} note="rank consistency" />
                <AgreementMetric label="MAE" value={metricText(agreementAnalysis.mae)} note="human–model error" />
                <AgreementMetric label="RMSE" value={metricText(agreementAnalysis.rmse)} note="penalizes larger errors" />
                <AgreementMetric label="Model severity slope" value={metricText(agreementAnalysis.model_severity_response_slope)} />
                <AgreementMetric label="Human severity slope" value={metricText(agreementAnalysis.human_severity_response_slope)} />
              </div>

              <div className="module4TwoCol chartsTwoCol">
                <article className="panel chartPanel">
                  <div className="panelHeading"><Activity size={18} /><div><h3>Severity–response curve</h3><p>Original → subtle → moderate → strong</p></div></div>
                  <SeverityResponseChart rows={agreementAnalysis.severity_response} />
                </article>
                <article className="panel chartPanel">
                  <div className="panelHeading"><BarChart3 size={18} /><div><h3>Batch quantitative summary</h3><p>Specification-aligned measures</p></div></div>
                  <div className="specMetricList">
                    <div><span>Original preference rate</span><strong>{(agreementAnalysis.human_preference_rate.original_preferred_rate * 100).toFixed(1)}%</strong></div>
                    <div><span>Counterfactual preference rate</span><strong>{(agreementAnalysis.human_preference_rate.counterfactual_preferred_rate * 100).toFixed(1)}%</strong></div>
                    <div><span>About the same rate</span><strong>{agreementAnalysis.human_preference_rate.same_rate == null ? "Unavailable" : `${(agreementAnalysis.human_preference_rate.same_rate * 100).toFixed(1)}%`}</strong></div>
                    <div><span>Mean absolute score gap</span><strong>{metricText(agreementAnalysis.mean_absolute_score_gap)}</strong></div>
                    <div><span>Mean absolute rank gap</span><strong>{metricText(agreementAnalysis.mean_absolute_rank_gap)}</strong></div>
                  </div>
                  <small className="chartNote">{agreementAnalysis.human_preference_rate.protocol_note}</small>
                </article>
              </div>

              {agreementAnalysis.human_preference_rate_by_severity?.length > 0 && (
                <article className="panel agreementTablePanel">
                  <div className="panelHeading"><ShieldCheck size={18} /><div><h3>Pairwise human preference by severity</h3><p>Exact PDF protocol: Original / Modified / About the same</p></div></div>
                  <div className="agreementTableWrap">
                    <table className="agreementTable">
                      <thead><tr><th>Severity</th><th>Responses</th><th>Original</th><th>Modified</th><th>About the same</th></tr></thead>
                      <tbody>{agreementAnalysis.human_preference_rate_by_severity.map((row) => (
                        <tr key={row.severity}><td><strong>{row.severity}</strong></td><td>{row.response_count}</td><td>{(row.original_rate * 100).toFixed(1)}%</td><td>{(row.modified_rate * 100).toFixed(1)}%</td><td>{(row.same_rate * 100).toFixed(1)}%</td></tr>
                      ))}</tbody>
                    </table>
                  </div>
                </article>
              )}

              <article className="panel agreementTablePanel">
                <div className="panelHeading"><BarChart3 size={18} /><div><h3>Human vs model candidate comparison</h3><p>Scores, confidence intervals, ranks and preference rates</p></div></div>
                <div className="agreementTableWrap">
                  <table className="agreementTable">
                    <thead><tr><th>Candidate</th><th>Model</th><th>Human mean</th><th>Human 95% CI</th><th>Gap</th><th>Model Δ</th><th>Human Δ</th><th>Ranks M/H</th><th>Best votes</th></tr></thead>
                    <tbody>
                      {agreementAnalysis.candidates.map((row) => (
                        <tr key={row.candidate_id} className={row.substantial_disagreement ? "disagreementRow" : ""}>
                          <td><strong>{row.label}</strong></td>
                          <td>{Number(row.model_score).toFixed(3)}</td>
                          <td>{Number(row.human_mean_rating).toFixed(3)}</td>
                          <td>{row.human_mean_ci_95 ? `${Number(row.human_mean_ci_95.lower).toFixed(2)}–${Number(row.human_mean_ci_95.upper).toFixed(2)}` : "n<2"}</td>
                          <td>{Number(row.absolute_score_gap).toFixed(3)}</td>
                          <td>{Number(row.model_delta_from_original).toFixed(3)}</td>
                          <td>{Number(row.human_delta_from_original).toFixed(3)}</td>
                          <td>{row.model_rank} / {row.human_rank}</td>
                          <td>{row.preference_count} ({(row.preference_share * 100).toFixed(0)}%)</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </article>

              <article className="panel agreementTablePanel">
                <div className="panelHeading"><Activity size={18} /><div><h3>Human sensitivity vs model sensitivity</h3><p>Δ response, disagreement magnitude, effect size and 95% confidence interval</p></div></div>
                <div className="agreementTableWrap">
                  <table className="agreementTable">
                    <thead><tr><th>Severity</th><th>Human Δ</th><th>Model Δ</th><th>|Δ difference|</th><th>Direction agrees</th><th>Cohen's dz</th><th>Human Δ 95% CI</th></tr></thead>
                    <tbody>
                      {agreementAnalysis.perturbation_sensitivity.map((row) => (
                        <tr key={row.candidate_id}>
                          <td><strong>{row.severity}</strong></td>
                          <td>{Number(row.human_delta_from_original).toFixed(3)}</td>
                          <td>{Number(row.model_delta_from_original).toFixed(3)}</td>
                          <td>{Number(row.absolute_sensitivity_disagreement).toFixed(3)}</td>
                          <td>{row.directional_agreement ? "YES ✓" : "NO"}</td>
                          <td>{metricText(row.human_effect_size_cohens_dz)}</td>
                          <td>{row.human_delta_ci_95 ? `${Number(row.human_delta_ci_95.lower).toFixed(2)}–${Number(row.human_delta_ci_95.upper).toFixed(2)}` : "n<2"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </article>

              <article className="panel">
                <div className="panelHeading"><AlertTriangle size={18} /><div><h3>Visual / contextual disagreement report</h3><p>Automatically detected research issues for this batch</p></div></div>
                <div className="contextReportSummary">
                  <span>Context</span>
                  <strong>{agreementAnalysis.original?.context?.visual_caption || agreementAnalysis.original?.prompt || "Context unavailable"}</strong>
                  {agreementAnalysis.original?.context?.relationships?.length ? (
                    <small>{agreementAnalysis.original.context.relationships.map((item) => `${item.subject} → ${item.relation.replaceAll("_", " ")} → ${item.object}`).join(" · ")}</small>
                  ) : null}
                </div>
                <div className="issueList">
                  {agreementAnalysis.issues.length ? agreementAnalysis.issues.map((issue, index) => (
                    <IssueBadge key={`${issue.code}-${issue.candidate_id || index}`} issue={issue} />
                  )) : <div className="cleanAnalysis">No disagreement flags for this batch ✓</div>}
                </div>
              </article>

              <div className="analysisLinks">
                <a className="handoffButton" href={module4AnalysisUrl(batch.batch_id)} target="_blank" rel="noreferrer">Open final analysis JSON</a>
                <a className="handoffButton secondaryLink" href={module4CsvUrl(batch.batch_id)}><Download size={17} /> Export candidate report CSV</a>
              </div>
            </>
          )}

          {projectDashboard && (
            <div className="projectDashboardStack">
              <article className="panel projectDashboard">
                <div className="panelHeading"><Database size={18} /><div><h3>Project-level quantitative dashboard</h3><p>All analyzable Module 3 batches stored locally</p></div></div>
                <div className="agreementMetricGrid wideMetrics compact">
                  <AgreementMetric label="Analyzed batches" value={projectDashboard.batch_count} />
                  <AgreementMetric label="Human responses" value={projectDashboard.participant_response_count} />
                  <AgreementMetric label="Top-choice agreement" value={projectDashboard.top_choice_agreement_rate == null ? "Unavailable" : `${(projectDashboard.top_choice_agreement_rate * 100).toFixed(1)}%`} note={projectDashboard.top_choice_agreement_ci_95 ? `95% CI ${(projectDashboard.top_choice_agreement_ci_95.lower * 100).toFixed(1)}–${(projectDashboard.top_choice_agreement_ci_95.upper * 100).toFixed(1)}%` : ""} />
                  <AgreementMetric label="Directional agreement" value={projectDashboard.directional_agreement_rate == null ? "Unavailable" : `${(projectDashboard.directional_agreement_rate * 100).toFixed(1)}%`} />
                  <AgreementMetric label="Disagreement rate" value={projectDashboard.disagreement_rate == null ? "Unavailable" : `${(projectDashboard.disagreement_rate * 100).toFixed(1)}%`} />
                  <AgreementMetric label="Original preferred" value={projectDashboard.original_preference_rate == null ? "Unavailable" : `${(projectDashboard.original_preference_rate * 100).toFixed(1)}%`} />
                  <AgreementMetric label="Counterfactual preferred" value={projectDashboard.counterfactual_preference_rate == null ? "Unavailable" : `${(projectDashboard.counterfactual_preference_rate * 100).toFixed(1)}%`} />
                  <AgreementMetric label="Mean Pearson r" value={metricText(projectDashboard.mean_pearson_r)} />
                  <AgreementMetric label="Mean Spearman ρ" value={metricText(projectDashboard.mean_spearman_rho)} />
                  <AgreementMetric label="MAE" value={metricText(projectDashboard.mae)} />
                  <AgreementMetric label="RMSE" value={metricText(projectDashboard.rmse)} />
                  <AgreementMetric label="Mean score gap" value={metricText(projectDashboard.mean_absolute_score_gap)} note={projectDashboard.mean_absolute_score_gap_ci_95 ? `95% CI ${Number(projectDashboard.mean_absolute_score_gap_ci_95.lower).toFixed(2)}–${Number(projectDashboard.mean_absolute_score_gap_ci_95.upper).toFixed(2)}` : ""} />
                </div>
              </article>

              <div className="module4TwoCol chartsTwoCol">
                <article className="panel chartPanel">
                  <div className="panelHeading"><BarChart3 size={18} /><div><h3>Human–model response correlation plot</h3><p>All candidate observations across analyzed batches</p></div></div>
                  <CorrelationScatter points={projectDashboard.correlation_points} />
                </article>
                <article className="panel chartPanel">
                  <div className="panelHeading"><Activity size={18} /><div><h3>Disagreement rate by perturbation</h3><p>Required project figure</p></div></div>
                  <DisagreementBars rows={projectDashboard.by_perturbation} label="Perturbation" />
                </article>
              </div>

              <article className="panel agreementTablePanel">
                <div className="panelHeading"><Activity size={18} /><div><h3>Human vs model sensitivity by perturbation</h3><p>Required comparison table aggregated across experiments</p></div></div>
                <div className="agreementTableWrap">
                  <table className="agreementTable">
                    <thead><tr><th>Perturbation</th><th>Batches</th><th>Human sensitivity</th><th>Model sensitivity</th><th>Sensitivity gap</th><th>Directional agreement</th><th>Disagreement rate</th></tr></thead>
                    <tbody>{projectDashboard.by_perturbation.map((row) => <tr key={row.name}><td><strong>{row.name}</strong></td><td>{row.batch_count}</td><td>{metricText(row.mean_human_sensitivity)}</td><td>{metricText(row.mean_model_sensitivity)}</td><td>{metricText(row.mean_sensitivity_disagreement)}</td><td>{row.directional_agreement_rate == null ? "—" : `${(row.directional_agreement_rate * 100).toFixed(1)}%`}</td><td>{row.disagreement_rate == null ? "—" : `${(row.disagreement_rate * 100).toFixed(1)}%`}</td></tr>)}</tbody>
                  </table>
                </div>
              </article>

              <div className="module4TwoCol chartsTwoCol">
                <article className="panel chartPanel">
                  <div className="panelHeading"><Activity size={18} /><div><h3>Disagreement rate by image region</h3><p>Detected region labels including full_image fallback</p></div></div>
                  <DisagreementBars rows={projectDashboard.by_region_label} label="Region" />
                </article>
                <article className="panel chartPanel">
                  <div className="panelHeading"><Activity size={18} /><div><h3>Disagreement rate by image category</h3><p>AGIQA style/generator category aggregation</p></div></div>
                  <DisagreementBars rows={projectDashboard.by_category || []} label="Category" />
                </article>
                <article className="panel chartPanel">
                  <div className="panelHeading"><BarChart3 size={18} /><div><h3>Severity sensitivity summary</h3><p>Mean response changes across all batches</p></div></div>
                  <div className="agreementTableWrap">
                    <table className="agreementTable compactTable">
                      <thead><tr><th>Severity</th><th>Human Δ</th><th>Model Δ</th><th>|Δ diff|</th><th>Direction agree</th><th>Mean dz</th></tr></thead>
                      <tbody>{projectDashboard.by_severity.map((row) => <tr key={row.severity}><td><strong>{row.severity}</strong></td><td>{Number(row.mean_human_delta).toFixed(3)}</td><td>{Number(row.mean_model_delta).toFixed(3)}</td><td>{Number(row.mean_absolute_sensitivity_disagreement).toFixed(3)}</td><td>{(row.directional_agreement_rate * 100).toFixed(0)}%</td><td>{metricText(row.mean_human_effect_size_cohens_dz)}</td></tr>)}</tbody>
                    </table>
                  </div>
                </article>
              </div>

              <article className="panel chartPanel">
                <div className="panelHeading"><Database size={18} /><div><h3>Human–model disagreement heatmap</h3><p>Region × perturbation substantial-disagreement rate</p></div></div>
                <DisagreementHeatmap cells={projectDashboard.heatmap} />
              </article>

              <article className="panel findingsPanel">
                <div className="panelHeading"><CheckCircle2 size={18} /><div><h3>Summary of important findings</h3><p>Automatically generated from the stored experimental results</p></div></div>
                {projectDashboard.findings?.length ? <div className="findingsList">{projectDashboard.findings.map((item, index) => <div key={`${item.type}-${index}`}><strong>{item.type}</strong><span>{item.text}</span></div>)}</div> : <div className="cleanAnalysis">Add more analyzed batches to generate cross-experiment findings.</div>}
              </article>

              {projectDashboard.top_disagreement_batches?.length > 0 && (
                <article className="panel">
                  <div className="dashboardBatchList">
                    <span>Highest-priority disagreement batches</span>
                    {projectDashboard.top_disagreement_batches.slice(0, 10).map((row) => (
                      <div key={row.batch_id}><strong>{row.dataset_image_name || row.batch_id}</strong><span>{row.perturbation} · {row.region_label} · disagreement {((row.disagreement_rate || 0) * 100).toFixed(0)}% · gap {Number(row.mean_absolute_score_gap).toFixed(3)}</span></div>
                    ))}
                  </div>
                </article>
              )}

              <div className="analysisLinks">
                <a className="handoffButton secondaryLink" href={module4ProjectCsvUrl()}><Download size={17} /> Export project quantitative CSV</a>
              </div>
            </div>
          )}
        </section>
      )}

      <footer>
        <span>Module 1 · Vennela ✓</span>
        <span>Module 2 · Sanyojita ✓</span>
        <span>Module 3 · Muskan ✓</span>
        <strong>Module 4 · Pragya · agreement + dashboard</strong>
      </footer>
    </main>
  );
}
