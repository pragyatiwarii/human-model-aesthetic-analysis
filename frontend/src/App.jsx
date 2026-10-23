import React, { useEffect, useMemo, useState } from "react";
import {
  Aperture,
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  Database,
  Dice5,
  LoaderCircle,
  Search,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

import {
  absoluteUrl,
  getDatasetStatus,
  getRecords,
  handoffUrl,
  runModule1,
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

function RegionImage({
  result,
  selectedRegion,
  onSelectRegion,
}) {
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

  const [loadingList, setLoadingList] = useState(false);
  const [analysing, setAnalysing] = useState(false);
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

  function randomPage() {
    if (!total) return;

    const randomIndex = Math.floor(Math.random() * total);
    const nextOffset =
      Math.floor(randomIndex / PAGE_SIZE) * PAGE_SIZE;

    refresh({ nextOffset });
  }

  const objectSummary = useMemo(() => {
    if (!analysis) return "";

    if (analysis.context.objects?.length) {
      return analysis.context.objects
        .map((item) => `${item.label} ×${item.count}`)
        .join(", ");
    }

    if (analysis.regions.some((r) => r.is_fallback)) {
      return "No reliable semantic object detected. Full image is available as the editable region.";
    }

    return "No reliable semantic object detected.";
  }, [analysis]);

  return (
    <main className="appShell">
      <header className="topbar">
        <div className="brand">
          <div className="logo">
            <Aperture size={20} />
          </div>

          <div>
            <h1>Aesthetic Lens</h1>
            <p>AGIQA-3K · Module 1 core handoff</p>
          </div>
        </div>

        <div className="ownerPill">
          Module 1 · Vennela
        </div>
      </header>

      <section className="hero">
        <div>
          <span className="eyebrow">
            DATASET → REGION → HANDOFF · 01/04
          </span>

          <h2>
            Prepare an <em>AGIQA-3K image</em> for controlled modification.
          </h2>

          <p>
            This version keeps only what later modules need:
            original dataset metadata, reliable editable regions,
            segmentation information and the Module 2 handoff.
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
              <p>
                {first}–{last} of {total}
              </p>
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
                <option value={item} key={item}>
                  {item}
                </option>
              ))}
            </select>

            <button
              type="button"
              onClick={randomPage}
              title="Jump to a random page"
            >
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
                    selected?.name === record.name
                      ? "active"
                      : "",
                  ].join(" ")}
                  onClick={() => {
                    setSelected(record);
                    setAnalysis(null);
                    setSelectedRegion(null);
                  }}
                >
                  <div>
                    <strong>{record.name}</strong>
                    <span
                      className={
                        record.image_exists
                          ? "available"
                          : "missing"
                      }
                    >
                      {record.image_exists
                        ? "image"
                        : "missing"}
                    </span>
                  </div>

                  <p>{record.prompt}</p>
                  <small>{record.generator}</small>
                </button>
              ))
            ) : (
              <div className="emptyList">
                No matching records.
              </div>
            )}
          </div>

          <div className="pagination">
            <button
              type="button"
              disabled={offset === 0}
              onClick={() =>
                refresh({
                  nextOffset: Math.max(
                    0,
                    offset - PAGE_SIZE
                  ),
                })
              }
            >
              <ArrowLeft size={15} />
            </button>

            <span>
              Page {page} of {pages}
            </span>

            <button
              type="button"
              disabled={offset + PAGE_SIZE >= total}
              onClick={() =>
                refresh({
                  nextOffset: offset + PAGE_SIZE,
                })
              }
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
                  <span className="eyebrow">
                    {selected.generator}
                  </span>
                  <h3>{selected.name}</h3>
                  <p>{selected.prompt}</p>
                </div>

                <button
                  type="button"
                  className="primary"
                  disabled={
                    analysing || !selected.image_exists
                  }
                  onClick={analyse}
                >
                  {analysing ? (
                    <LoaderCircle
                      className="spin"
                      size={18}
                    />
                  ) : (
                    <Sparkles size={18} />
                  )}

                  {analysing
                    ? "Running region analysis…"
                    : "Run Module 1"}
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
                  <img
                    src={absoluteUrl(selected.image_url)}
                    alt={selected.name}
                  />
                </div>
              )}

              {analysis && (
                <>
                  <RegionImage
                    result={analysis}
                    selectedRegion={selectedRegion}
                    onSelectRegion={setSelectedRegion}
                  />

                  <div className="regionStrip">
                    {analysis.regions.map((region) => (
                      <button
                        type="button"
                        key={region.region_id}
                        className={
                          selectedRegion?.region_id ===
                          region.region_id
                            ? "active"
                            : ""
                        }
                        onClick={() =>
                          setSelectedRegion(region)
                        }
                      >
                        <span>
                          {region.is_fallback
                            ? "full image"
                            : region.label}
                        </span>

                        <small>
                          {region.is_fallback
                            ? "fallback"
                            : `${(
                                region.confidence * 100
                              ).toFixed(1)}%`}
                        </small>
                      </button>
                    ))}
                  </div>
                </>
              )}
            </>
          )}

          {error && (
            <div className="error">{error}</div>
          )}
        </section>
      </section>

      {analysis && (
        <section className="resultGrid">
          <article className="panel">
            <div className="panelHeading">
              <ShieldCheck size={18} />
              <div>
                <h3>Module 1 core result</h3>
                <p>
                  Only information needed downstream
                </p>
              </div>
            </div>

            <div className="facts">
              <div>
                <span>Original prompt</span>
                <strong>{analysis.prompt}</strong>
              </div>

              <div>
                <span>Reliable objects</span>
                <strong>{objectSummary}</strong>
              </div>

              <div>
                <span>Confidence threshold</span>
                <strong>
                  {analysis.detector.confidence_threshold}
                </strong>
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
                <h3>Module 2 handoff</h3>
                <p>
                  Stable contract for Sanyojita's module
                </p>
              </div>
            </div>

            <pre>
              {JSON.stringify(
                {
                  image_id: analysis.image_id,
                  dataset_image_name:
                    analysis.dataset_image_name,
                  prompt: analysis.prompt,
                  selected_region: selectedRegion,
                  ground_truth: {
                    mos_quality:
                      analysis.mos_quality,
                    mos_align:
                      analysis.mos_align,
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
              Open complete handoff JSON
            </a>
          </article>
        </section>
      )}

      <footer>
        <span>
          Module 1 · AGIQA-3K region intelligence
        </span>
        <strong>
          Next → Module 2 controlled counterfactuals
        </strong>
      </footer>
    </main>
  );
}
