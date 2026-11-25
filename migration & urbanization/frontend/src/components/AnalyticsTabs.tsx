import { useState } from "react";
import { Cpu, Download, Layers, Orbit, ShieldCheck, TrendingUp } from "lucide-react";
import type { AnalysisResponse } from "../types";

interface AnalyticsTabsProps {
  data: AnalysisResponse;
  hubName: string;
}

const ZONE_COLOR: Record<number, string> = {
  0: "#06B6D4",
  1: "#22C55E",
  2: "#EF4444",
  3: "#84CC16",
};

export default function AnalyticsTabs({ data, hubName }: AnalyticsTabsProps) {
  const [activeTab, setActiveTab] = useState<"proof" | "blueprint" | "arch">("proof");

  const roi = data.roi;
  const maxNdbi = Math.max(0.25, ...data.cells.map((c) => Math.abs(c.delta_ndbi)));
  const maxViirs = Math.max(20, ...data.cells.map((c) => c.viirs));

  const downloadCsv = () => {
    const headers = [
      "Rank",
      "Source Lat",
      "Source Lon",
      "Target Lat",
      "Target Lon",
      "Length (km)",
      "HH Served",
      "Cost Early (Rs Cr)",
      "Cost Deferred (Rs Cr)",
      "Savings (Rs Cr)",
      "Risk Score",
    ];
    const rows = data.corridors.map((c) => [
      `#${c.rank}`,
      c.source_lat.toFixed(5),
      c.source_lon.toFixed(5),
      c.target_lat.toFixed(5),
      c.target_lon.toFixed(5),
      c.corridor_km.toFixed(1),
      c.served_hh,
      c.cost_cr.toFixed(2),
      (c.cost_cr * 3.4).toFixed(2),
      (c.cost_cr * 2.4).toFixed(2),
      c.risk_score.toFixed(3),
    ]);
    const csv = [headers, ...rows].map((r) => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `urbanpulse_blueprint_${hubName.replace(/[^a-z0-9]+/gi, "_").toLowerCase()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const maxCostBar = Math.max(1, roi.deferred_cost_cr, roi.total_funded_cost_cr);

  return (
    <div className="mt-6">
      {/* Tab Navigation */}
      <div className="flex flex-wrap gap-2 rounded-xl border border-cyan-500/20 bg-[#080D1A]/90 p-1.5 backdrop-blur-md">
        <button
          onClick={() => setActiveTab("proof")}
          className={`flex items-center gap-2 rounded-lg px-4 py-2.5 text-xs font-display font-semibold tracking-wide transition-all ${
            activeTab === "proof"
              ? "bg-cyan-500/15 text-cyan-300 border border-cyan-400/40 shadow-[0_0_20px_rgba(6,182,212,0.18)]"
              : "text-slate-400 hover:text-slate-200 border border-transparent"
          }`}
        >
          <Cpu className="h-4 w-4" />
          Multi-Sensor Proof &amp; AI Accuracy
        </button>
        <button
          onClick={() => setActiveTab("blueprint")}
          className={`flex items-center gap-2 rounded-lg px-4 py-2.5 text-xs font-display font-semibold tracking-wide transition-all ${
            activeTab === "blueprint"
              ? "bg-cyan-500/15 text-cyan-300 border border-cyan-400/40 shadow-[0_0_20px_rgba(6,182,212,0.18)]"
              : "text-slate-400 hover:text-slate-200 border border-transparent"
          }`}
        >
          <TrendingUp className="h-4 w-4" />
          Planner Blueprint &amp; Export
        </button>
        <button
          onClick={() => setActiveTab("arch")}
          className={`flex items-center gap-2 rounded-lg px-4 py-2.5 text-xs font-display font-semibold tracking-wide transition-all ${
            activeTab === "arch"
              ? "bg-cyan-500/15 text-cyan-300 border border-cyan-400/40 shadow-[0_0_20px_rgba(6,182,212,0.18)]"
              : "text-slate-400 hover:text-slate-200 border border-transparent"
          }`}
        >
          <Orbit className="h-4 w-4" />
          Orbit-to-Economy Architecture
        </button>
      </div>

      {/* TAB 1: Multi-Sensor Proof & AI Accuracy */}
      {activeTab === "proof" && (
        <div className="mt-4 space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Scatter Plot: ΔNDBI vs Night-Light Proxy */}
            <div className="rounded-2xl border border-slate-800 bg-[#090F1E]/90 p-5 shadow-xl">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-widest text-cyan-400">
                    Optical vs. Illumination Fusion
                  </div>
                  <h3 className="font-display text-sm font-bold text-slate-100">
                    Daytime Concrete Surge (ΔNDBI) vs. Night-Light Intensity
                  </h3>
                </div>
                <span className="rounded-full border border-red-500/30 bg-red-500/10 px-2.5 py-0.5 font-mono text-[10px] text-red-300">
                  Red = Night-Light Gap
                </span>
              </div>
              <svg viewBox="0 0 460 240" className="w-full h-56 overflow-visible">
                {/* Grid lines */}
                {[40, 85, 130, 175].map((y) => (
                  <line key={y} x1="42" y1={y} x2="445" y2={y} stroke="rgba(148,163,184,0.12)" />
                ))}
                <line x1="42" y1="20" x2="42" y2="205" stroke="rgba(148,163,184,0.3)" />
                <line x1="42" y1="205" x2="445" y2="205" stroke="rgba(148,163,184,0.3)" />
                {data.cells.slice(0, 320).map((cell, idx) => {
                  const x = 48 + ((cell.delta_ndbi + maxNdbi) / (2 * maxNdbi)) * 390;
                  const y = 200 - (Math.min(cell.viirs, maxViirs) / maxViirs) * 175;
                  const color = ZONE_COLOR[cell.zone] ?? "#64748B";
                  return (
                    <circle
                      key={idx}
                      cx={Math.max(46, Math.min(442, x))}
                      cy={Math.max(22, Math.min(202, y))}
                      r={cell.zone === 2 ? 3.2 : 2.3}
                      fill={color}
                      fillOpacity={cell.zone === 2 ? 0.85 : 0.55}
                    />
                  );
                })}
                <text x="240" y="230" textAnchor="middle" fill="#64748B" fontSize="10" fontFamily="JetBrains Mono">
                  ΔNDBI (2021 → 2026 Concrete Expansion)
                </text>
                <text
                  x="14"
                  y="115"
                  textAnchor="middle"
                  fill="#64748B"
                  fontSize="10"
                  fontFamily="JetBrains Mono"
                  transform="rotate(-90 14 115)"
                >
                  Night-Light
                </text>
              </svg>
            </div>

            {/* Feature Importances */}
            <div className="rounded-2xl border border-slate-800 bg-[#090F1E]/90 p-5 shadow-xl">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-widest text-cyan-400">
                    Gradient Boosting Explainability
                  </div>
                  <h3 className="font-display text-sm font-bold text-slate-100">
                    AI Model Feature Importances (CV Accuracy: {(data.cv_accuracy * 100).toFixed(1)}%)
                  </h3>
                </div>
                <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-0.5 font-mono text-[10px] text-emerald-300">
                  5-Fold StratifiedKFold
                </span>
              </div>
              <div className="space-y-2.5 mt-3">
                {data.feat_importances.map((f) => {
                  const pct = Math.min(100, f.importance * 100);
                  return (
                    <div key={f.feature}>
                      <div className="flex justify-between font-mono text-[11px] mb-1">
                        <span className="text-slate-300">{f.feature}</span>
                        <span className="text-cyan-300 font-semibold">{pct.toFixed(1)}%</span>
                      </div>
                      <div className="h-2 w-full rounded-full bg-slate-800/90 overflow-hidden">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-sky-400 shadow-[0_0_10px_rgba(6,182,212,0.5)]"
                          style={{ width: `${Math.max(3, pct * 1.4)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Radar VV Distribution + Telemetry Summary */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-2 rounded-2xl border border-slate-800 bg-[#090F1E]/90 p-5">
              <div className="font-mono text-[10px] uppercase tracking-widest text-cyan-400">
                Sentinel-1 SAR Backscatter (VV dB)
              </div>
              <h3 className="font-display text-sm font-bold text-slate-100 mb-4">
                Structural Height &amp; Density Separation Across Classified Zones
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {data.radar_stats.map((r) => (
                  <div
                    key={r.zone}
                    className="rounded-xl border border-slate-800 bg-[#050811]/80 p-3.5"
                    style={{ borderTopColor: r.color, borderTopWidth: 2 }}
                  >
                    <div className="font-mono text-[10px] uppercase tracking-wider text-slate-400">
                      {r.label}
                    </div>
                    <div className="font-display text-xl font-bold text-slate-100 mt-1">
                      {r.mean.toFixed(1)} <span className="text-xs text-slate-400">dB</span>
                    </div>
                    <div className="font-mono text-[10px] text-slate-500 mt-1">
                      P10–P90: {r.p10} to {r.p90} dB
                    </div>
                    <div className="font-mono text-[10px] text-cyan-400 mt-0.5">
                      {r.count.toLocaleString()} cells
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-2xl border border-emerald-500/25 bg-[#090F1E]/90 p-5 flex flex-col justify-between">
              <div>
                <div className="font-mono text-[10px] uppercase tracking-widest text-emerald-400">
                  Validation Telemetry
                </div>
                <div className="font-display text-3xl font-bold text-emerald-400 mt-1">
                  {(data.cv_accuracy * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-slate-400 mt-1">
                  Gradient Boosting Classifier · 5-Fold CV
                </div>
              </div>
              <div className="mt-4 pt-3 border-t border-slate-800 space-y-1.5 font-mono text-[11px] text-slate-300">
                <div className="flex justify-between">
                  <span className="text-slate-500">Grid Cell Size</span>
                  <span>250m × 250m (2,304 cells)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Sprawl Mean / Max</span>
                  <span className="text-amber-300">
                    {data.sprawl_prob_mean.toFixed(2)} / {data.sprawl_prob_max.toFixed(2)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">STAC Link</span>
                  <span className="text-cyan-300">Planetary Computer</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: Planner Blueprint & Export */}
      {activeTab === "blueprint" && (
        <div className="mt-4 space-y-4">
          {/* ROI Visual Comparison */}
          <div className="rounded-2xl border border-slate-800 bg-[#090F1E]/90 p-5">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <div>
                <div className="font-mono text-[10px] uppercase tracking-widest text-cyan-400">
                  Act Now vs. Wait 3 Years (3.4× Demolition &amp; Retrofit Multiplier)
                </div>
                <h3 className="font-display text-base font-bold text-slate-100">
                  Municipal Infrastructure Cost Comparison
                </h3>
              </div>
              <div className="font-mono text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/30 rounded-lg px-3 py-1.5">
                NET SAVINGS: ₹{roi.net_savings_cr.toFixed(1)} Cr (${roi.net_savings_usd_m.toFixed(1)}M)
              </div>
            </div>

            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-semibold text-cyan-300">Early Corridor Reservation (Act Now)</span>
                  <span className="font-mono text-cyan-300">₹{roi.total_funded_cost_cr.toFixed(1)} Cr</span>
                </div>
                <div className="h-4 w-full rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-sky-400 transition-all duration-300"
                    style={{ width: `${Math.max(4, (roi.total_funded_cost_cr / maxCostBar) * 100)}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-semibold text-red-400">Deferred 3-Year Retrofit &amp; Demolition (3.4×)</span>
                  <span className="font-mono text-red-400">₹{roi.deferred_cost_cr.toFixed(1)} Cr</span>
                </div>
                <div className="h-4 w-full rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-red-500 to-amber-500 transition-all duration-300"
                    style={{ width: `${Math.max(8, (roi.deferred_cost_cr / maxCostBar) * 100)}%` }}
                  />
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-5">
              {[
                { label: "Early Cost", val: `₹${roi.total_funded_cost_cr.toFixed(1)} Cr`, color: "text-cyan-300" },
                { label: "Deferred Cost", val: `₹${roi.deferred_cost_cr.toFixed(1)} Cr`, color: "text-red-400" },
                { label: "HH Connected", val: roi.hh_connected.toLocaleString(), color: "text-emerald-400" },
                { label: "Sprawl Saved", val: `${roi.sprawl_prevented_sqkm.toFixed(2)} km²`, color: "text-amber-300" },
              ].map((item) => (
                <div key={item.label} className="rounded-xl border border-slate-800 bg-[#050811]/85 p-3 text-center">
                  <div className="font-mono text-[10px] uppercase tracking-wider text-slate-500">{item.label}</div>
                  <div className={`font-display text-xl font-bold mt-1 ${item.color}`}>{item.val}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Priority Table + CSV Download */}
          <div className="rounded-2xl border border-slate-800 bg-[#090F1E]/90 p-5">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <div>
                <div className="font-mono text-[10px] uppercase tracking-widest text-cyan-400">
                  NetworkX Dijkstra Auto-Routed Corridors
                </div>
                <h3 className="font-display text-base font-bold text-slate-100">
                  Top-5 Priority Intervention Corridors
                </h3>
              </div>
              <button
                onClick={downloadCsv}
                className="flex items-center gap-2 rounded-xl border border-cyan-400/50 bg-gradient-to-r from-cyan-600 to-sky-600 px-4 py-2 text-xs font-display font-bold text-white shadow-[0_0_20px_rgba(6,182,212,0.3)] hover:brightness-110 transition-all"
              >
                <Download className="h-4 w-4" />
                Download Planner Blueprint (CSV)
              </button>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-slate-800 font-mono text-[10px] uppercase tracking-wider text-cyan-400">
                    <th className="py-2.5 px-3">Rank</th>
                    <th className="py-2.5 px-3">Source Coords</th>
                    <th className="py-2.5 px-3">Target Arterial</th>
                    <th className="py-2.5 px-3">Length</th>
                    <th className="py-2.5 px-3">HH Served</th>
                    <th className="py-2.5 px-3">Early Cost</th>
                    <th className="py-2.5 px-3">Deferred (3.4×)</th>
                    <th className="py-2.5 px-3">Savings</th>
                    <th className="py-2.5 px-3">Risk</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70 font-mono">
                  {data.corridors.map((c) => (
                    <tr key={c.rank} className="hover:bg-cyan-500/5 transition-colors">
                      <td className="py-2.5 px-3 font-bold text-cyan-300">#{c.rank}</td>
                      <td className="py-2.5 px-3 text-slate-300">
                        {c.source_lat.toFixed(4)}, {c.source_lon.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-3 text-slate-400">
                        {c.target_lat.toFixed(4)}, {c.target_lon.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-3 text-slate-200">{c.corridor_km.toFixed(1)} km</td>
                      <td className="py-2.5 px-3 text-slate-200">{c.served_hh.toLocaleString()}</td>
                      <td className="py-2.5 px-3 text-cyan-300">₹{c.cost_cr.toFixed(2)} Cr</td>
                      <td className="py-2.5 px-3 text-red-400">₹{(c.cost_cr * 3.4).toFixed(2)} Cr</td>
                      <td className="py-2.5 px-3 text-emerald-400 font-semibold">
                        ₹{(c.cost_cr * 2.4).toFixed(2)} Cr
                      </td>
                      <td className="py-2.5 px-3 text-amber-300">{c.risk_score.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: Orbit-to-Economy Architecture */}
      {activeTab === "arch" && (
        <div className="mt-4 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[
              {
                stage: "STAGE 01 // ORBITAL INGEST",
                title: "Planetary Computer STAC",
                desc: "Zero-API-key windowed COG reads of Sentinel-2 L2A, Sentinel-1 SAR, and ESA WorldCover over a 12 km × 12 km swath.",
                accent: "border-cyan-500/35",
              },
              {
                stage: "STAGE 02 // SENSOR FUSION",
                title: "data_pipeline.py",
                desc: "Aligns 2021→2026 ΔNDBI concrete expansion, NDVI green cover, SAR VV/VH backscatter, and OpenStreetMap road/water layers onto 250m cells.",
                accent: "border-sky-500/35",
              },
              {
                stage: "STAGE 03 // AI CLASSIFICATION",
                title: "ml_engine.py (Classifier)",
                desc: "Gradient Boosting 4-Zone Classifier + percentile Night-Light Gap detection flagging the top underserved growth clusters.",
                accent: "border-red-500/35",
              },
              {
                stage: "STAGE 04 // GRAVITY FORECAST",
                title: "ml_engine.py (Forecaster)",
                desc: "Spatial Random Forest Regressor models 2027–28 migration sprawl probability using road pull, neighbor pressure, and darkness gaps.",
                accent: "border-amber-500/35",
              },
              {
                stage: "STAGE 05 // AUTO-ROUTER & ROI",
                title: "planner_engine.py",
                desc: "NetworkX Dijkstra routes shortest bus & power corridors around lakes/green buffers and simulates Act Now vs. 3.4× retrofit savings.",
                accent: "border-emerald-500/35",
              },
              {
                stage: "STAGE 06 // MISSION CONTROL",
                title: "BDA / TUDA / MUDA Integration",
                desc: "Delivers interactive GeoJSON corridor paths and KTCP Act reservation blueprints directly to Karnataka Urban Development Authorities.",
                accent: "border-cyan-400/40",
              },
            ].map((card) => (
              <div
                key={card.stage}
                className={`rounded-2xl border ${card.accent} bg-[#090F1E]/90 p-5 shadow-lg`}
              >
                <div className="font-mono text-[10px] uppercase tracking-widest text-cyan-400 mb-1">
                  {card.stage}
                </div>
                <div className="font-display text-base font-bold text-slate-100 mb-2 flex items-center gap-2">
                  <Layers className="h-4 w-4 text-cyan-400" />
                  {card.title}
                </div>
                <p className="text-xs text-slate-400 leading-relaxed">{card.desc}</p>
              </div>
            ))}
          </div>

          <div className="rounded-2xl border border-cyan-500/25 bg-[#080E1C]/90 p-4 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-300">
            <div className="flex items-center gap-2 font-display font-semibold text-cyan-300">
              <ShieldCheck className="h-4 w-4" />
              Team Track A — Unified Orbit-to-Economy Ecosystem:
            </div>
            <div className="font-mono text-[11px] text-slate-400">
              🛰️ UrbanPulse AI (Migration) · 🌊 FloodSense (DisasterRisk) · 🌾 AgroSat · 🏗️ InfraScope · 📡 TrafficOrbit
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
