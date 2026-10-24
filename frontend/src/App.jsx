import React, { useEffect, useMemo, useState } from "react";
import {
  Aperture,
  ArrowLeft,
  ArrowRight,
  Beaker,
  CheckCircle2,
  Database,
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
  generateCounterfactuals,
  getAestheticModelStatus,
  getDatasetStatus,
  getPerturbations,
  getRecords,
  handoffUrl,
  module3HandoffUrl,
  module4HandoffUrl,
  runModule1,
  scoreAestheticBatch,
  submitHumanPreference,
  trainAestheticModel,
} from "./api";

const PAGE_SIZE = 25;

function ScoreCard({ label, value, std }) {
  return (
    <div className="scoreCard">
      <span>{label}</span>
      <strong>{Number(value).toFixed(3)}</strong>
      <small>STD {Number(std).toFixed(3)}</small>
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
  const [blindStudy, setBlindStudy] = useState(false);
  const [trainingModel, setTrainingModel] = useState(false);
  const [scoringBatch, setScoringBatch] = useState(false);
  const [submittingPreference, setSubmittingPreference] = useState(false);

  const [loadingList, setLoadingList] = useState(false);
  const [analysing, setAnalysing] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState("");

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
        setBlindStudy(false);
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
    setBlindStudy(false);
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
      setBlindStudy(false);
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
    setBlindStudy(true);
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
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmittingPreference(false);
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

    if (analysis.context.objects?.length) {
      return analysis.context.objects
        .map((item) => `${item.label} ×${item.count}`)
        .join(", ");
    }

    if (analysis.regions.some((region) => region.is_fallback)) {
      return "No reliable semantic object detected. Full image is available as the editable region.";
    }

    return "No reliable semantic object detected.";
  }, [analysis]);


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
            <p>AGIQA-3K · Modules 1 → 3</p>
          </div>
        </div>

        <div className="modulePills">
          <div className="ownerPill">01 · Vennela</div>
          <div className="ownerPill">02 · Sanyojita</div>
          <div className="ownerPill activeModule">03 · Muskan</div>
        </div>
      </header>

      <section className="hero">
        <div>
          <span className="eyebrow">
            COUNTERFACTUALS → SCORE → HUMAN PREFERENCE · 03/04
          </span>

          <h2>
            Score <em>what changed</em> and record what people actually prefer.
          </h2>

          <p>
            Modules 1 and 2 prepare controlled original/counterfactual sets.
            Module 3 gives every candidate a model aesthetic score and collects
            blind human ratings for the final agreement analysis.
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
                  disabled={analysing || !selected.image_exists}
                  onClick={analyse}
                >
                  {analysing ? (
                    <LoaderCircle className="spin" size={18} />
                  ) : (
                    <Sparkles size={18} />
                  )}
                  {analysing ? "Running Module 1…" : "Run Module 1"}
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
                <p>Frozen v1.4 region intelligence</p>
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
              ) : (
                <button type="button" className="secondaryStudyButton" onClick={startNextParticipant}>
                  Start next participant
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

          {preferenceSubmitted && modelScores && (
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

      <footer>
        <span>Module 1 · Vennela ✓</span>
        <span>Module 2 · Sanyojita ✓</span>
        <strong>Module 3 · Muskan · scoring + human preference</strong>
      </footer>
    </main>
  );
}
