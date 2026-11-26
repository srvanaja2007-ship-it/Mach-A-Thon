import { useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Compass,
  IndianRupee,
  Layers,
  MapPin,
  Radar,
  Satellite,
  Search,
  Shield,
  Sparkles,
  Users,
} from "lucide-react";
import AnalyticsTabs from "./components/AnalyticsTabs";
import MissionMap from "./components/MissionMap";
import type { AnalysisResponse, PresetHub } from "./types";

type AppMode = "urban_pulse" | "border_security";

function getInitialMode(): AppMode {
  const params = new URLSearchParams(window.location.search);
  return params.get("mode") === "border_security" ? "border_security" : "urban_pulse";
}

export default function App() {
  const [mode, setMode] = useState<AppMode>(getInitialMode);
  const [urbanPresets, setUrbanPresets] = useState<Record<string, PresetHub>>({});
  const [borderSectors, setBorderSectors] = useState<Record<string, PresetHub>>({});
  const [selectedHub, setSelectedHub] = useState<string>("");
  const [customQuery, setCustomQuery] = useState<string>("");
  const [activeCoords, setActiveCoords] = useState<{ lat: number; lon: number; name: string } | null>(null);

  const [budgetCr, setBudgetCr] = useState<number>(50);
  const [data, setData] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [showZones, setShowZones] = useState<boolean>(true);
  const [showNlg, setShowNlg] = useState<boolean>(true);
  const [showForecast, setShowForecast] = useState<boolean>(true);
  const [showCorridors, setShowCorridors] = useState<boolean>(true);

  const roiDebounceRef = useRef<number | null>(null);

  const activePresetMap = mode === "border_security" ? borderSectors : urbanPresets;

  // 1. Load urban hubs + border sectors on mount
  useEffect(() => {
    fetch("/api/presets")
      .then((r) => r.json())
      .then((res) => {
        const hubs: Record<string, PresetHub> = res.presets || {};
        const sectors: Record<string, PresetHub> = res.border_sectors || {};
        setUrbanPresets(hubs);
        setBorderSectors(sectors);

        const initialMap = mode === "border_security" ? sectors : hubs;
        const firstKey = Object.keys(initialMap)[0];
        if (firstKey) {
          setSelectedHub(firstKey);
          setActiveCoords({
            lat: initialMap[firstKey].lat,
            lon: initialMap[firstKey].lon,
            name: firstKey,
          });
        }
      })
      .catch((e) => {
        setError(`Failed to connect to UrbanPulse API: ${e.message}`);
        setLoading(false);
      });
  }, []);

  // Switch mode cleanly & sync ?mode= URL query param
  const handleToggleMode = (e: React.MouseEvent<HTMLAnchorElement>) => {
    e.preventDefault();
    const nextMode: AppMode = mode === "urban_pulse" ? "border_security" : "urban_pulse";
    const url = new URL(window.location.href);
    url.searchParams.set("mode", nextMode);
    window.history.pushState({}, "", url.toString());

    setMode(nextMode);
    const nextMap = nextMode === "border_security" ? borderSectors : urbanPresets;
    const firstKey = Object.keys(nextMap)[0];
    if (firstKey) {
      setSelectedHub(firstKey);
      setActiveCoords({
        lat: nextMap[firstKey].lat,
        lon: nextMap[firstKey].lon,
        name: firstKey,
      });
    }
  };

  // 2. Run full analysis when activeCoords changes
  useEffect(() => {
    if (!activeCoords) return;
    setLoading(true);
    setError(null);
    fetch(`/api/analyze?lat=${activeCoords.lat}&lon=${activeCoords.lon}&budget_cr=${budgetCr}`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((res: AnalysisResponse) => {
        setData(res);
        setLoading(false);
      })
      .catch((e) => {
        setError(`Analysis failed: ${e.message}`);
        setLoading(false);
      });
  }, [activeCoords]);

  // 3. Instant ROI update when budget slider changes
  const handleBudgetChange = (val: number) => {
    setBudgetCr(val);
    if (!activeCoords) return;
    if (roiDebounceRef.current) window.clearTimeout(roiDebounceRef.current);
    roiDebounceRef.current = window.setTimeout(() => {
      fetch(`/api/roi?lat=${activeCoords.lat}&lon=${activeCoords.lon}&budget_cr=${val}`)
        .then((r) => r.json())
        .then((res) => {
          if (res.roi) {
            setData((prev) => (prev ? { ...prev, roi: res.roi } : prev));
          }
        })
        .catch(() => {});
    }, 60);
  };

  const handleSelectHub = (hubKey: string) => {
    setSelectedHub(hubKey);
    if (hubKey !== "__custom__" && activePresetMap[hubKey]) {
      setActiveCoords({
        lat: activePresetMap[hubKey].lat,
        lon: activePresetMap[hubKey].lon,
        name: hubKey,
      });
    }
  };

  const handleCustomSearch = async () => {
    if (!customQuery.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/geocode?q=${encodeURIComponent(customQuery.trim())}`);
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || "Location not found");
      setSelectedHub("__custom__");
      setActiveCoords({
        lat: json.lat,
        lon: json.lon,
        name: `Custom (${json.lat.toFixed(3)}°N, ${json.lon.toFixed(3)}°E)`,
      });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Geocoding error");
      setLoading(false);
    }
  };

  const hubInfo = selectedHub && activePresetMap[selectedHub] ? activePresetMap[selectedHub] : null;
  const roi = data?.roi;
  const isBorder = mode === "border_security";

  return (
    <div className="min-h-screen flex flex-col lg:flex-row">
      {/* LEFT FLIGHT-DECK SIDEBAR */}
      <aside className="w-full lg:w-80 shrink-0 border-b lg:border-b-0 lg:border-r border-cyan-500/15 bg-[#060B17]/95 p-5 flex flex-col gap-5 backdrop-blur-xl">
        {/* Brand + Active Mode Indicator */}
        <div className="flex items-center gap-3 pb-4 border-b border-slate-800/80">
          <div
            className={`flex h-11 w-11 items-center justify-center rounded-xl border ${
              isBorder
                ? "border-amber-400/50 bg-amber-500/10 shadow-[0_0_20px_rgba(245,158,11,0.25)]"
                : "border-cyan-400/40 bg-cyan-500/10 shadow-[0_0_20px_rgba(6,182,212,0.25)]"
            }`}
          >
            {isBorder ? (
              <Shield className="h-5 w-5 text-amber-300" />
            ) : (
              <Satellite className="h-5 w-5 text-cyan-300" />
            )}
          </div>
          <div>
            <div className="font-display text-base font-bold tracking-wide text-white">
              URBANPULSE AI
            </div>
            <div
              className={`font-mono text-[10px] uppercase tracking-[0.14em] ${
                isBorder ? "text-amber-400" : "text-cyan-400"
              }`}
            >
              {isBorder ? "Border & Perimeter Mode" : "Urban Migration Mode"}
            </div>
          </div>
        </div>

        {/* Target Selection */}
        <div>
          <label className="font-mono text-[10px] uppercase tracking-widest text-cyan-400 flex items-center gap-1.5 mb-2">
            <Compass className="h-3.5 w-3.5" />{" "}
            {isBorder ? "Border & Defense Sector" : "Karnataka Preset Hub"}
          </label>
          <select
            value={selectedHub}
            onChange={(e) => handleSelectHub(e.target.value)}
            className="w-full rounded-xl border border-cyan-500/25 bg-[#0B1224] px-3 py-2.5 text-xs text-slate-100 outline-none focus:border-cyan-400"
          >
            {Object.keys(activePresetMap).map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
            <option value="__custom__">— Live Custom Satellite Search —</option>
          </select>

          {hubInfo && (
            <div className="mt-2.5 rounded-xl border border-slate-800 bg-[#0A1020] p-3 text-[11px] text-slate-400 space-y-1">
              <div className="text-slate-200 font-medium">{hubInfo.description}</div>
              <div>
                <span className="text-cyan-400 font-mono">AUTHORITY:</span> {hubInfo.authority}
              </div>
              <div className="font-mono text-[10px] text-slate-500">
                {hubInfo.lat.toFixed(4)}°N · {hubInfo.lon.toFixed(4)}°E
              </div>
            </div>
          )}
        </div>

        {/* Live Custom Search */}
        <div>
          <label className="font-mono text-[10px] uppercase tracking-widest text-cyan-400 flex items-center gap-1.5 mb-2">
            <Search className="h-3.5 w-3.5" /> Live Orbital Scan
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={customQuery}
              onChange={(e) => setCustomQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleCustomSearch()}
              placeholder={isBorder ? "Sector or lat,lon" : "City or lat,lon"}
              className="flex-1 rounded-xl border border-cyan-500/25 bg-[#0B1224] px-3 py-2 text-xs text-slate-100 placeholder-slate-500 outline-none focus:border-cyan-400"
            />
            <button
              onClick={handleCustomSearch}
              className="rounded-xl border border-cyan-400/50 bg-cyan-600/80 px-3 py-2 text-xs font-display font-bold text-white hover:bg-cyan-500 transition-colors"
            >
              Scan
            </button>
          </div>
        </div>

        {/* Budget Slider */}
        <div className="rounded-xl border border-cyan-500/20 bg-[#0A1020]/90 p-3.5">
          <div className="flex items-center justify-between mb-2">
            <span className="font-mono text-[10px] uppercase tracking-widest text-cyan-400">
              {isBorder ? "Surveillance & Fence Budget" : "Early Intervention Budget"}
            </span>
            <span className="font-mono text-sm font-bold text-cyan-300">₹{budgetCr} Cr</span>
          </div>
          <input
            type="range"
            min={10}
            max={500}
            step={5}
            value={budgetCr}
            onChange={(e) => handleBudgetChange(Number(e.target.value))}
            className="aero-slider w-full"
          />
          <div className="flex justify-between font-mono text-[10px] text-slate-500 mt-1.5">
            <span>₹10 Cr</span>
            <span>₹250 Cr</span>
            <span>₹500 Cr</span>
          </div>
        </div>

        {/* Layer Toggles */}
        <div>
          <div className="font-mono text-[10px] uppercase tracking-widest text-cyan-400 flex items-center gap-1.5 mb-2.5">
            <Layers className="h-3.5 w-3.5" /> Orbital Map Layers
          </div>
          <div className="space-y-2 text-xs">
            {[
              { label: "4-Zone AI Classification", val: showZones, set: setShowZones, dot: "bg-cyan-400" },
              {
                label: isBorder ? "Covert Encroachment Alerts" : "Night-Light Gap Alerts",
                val: showNlg,
                set: setShowNlg,
                dot: "bg-red-500",
              },
              {
                label: isBorder ? "2027–28 Infiltration Risk" : "2027–28 Migration Forecast",
                val: showForecast,
                set: setShowForecast,
                dot: "bg-amber-400",
              },
              {
                label: isBorder ? "Patrol & Smart Fence Links" : "AI Bus & Power Corridors",
                val: showCorridors,
                set: setShowCorridors,
                dot: "bg-sky-300",
              },
            ].map((tog) => (
              <label
                key={tog.label}
                className="flex items-center justify-between rounded-lg border border-slate-800/80 bg-[#0A1020]/60 px-3 py-2 cursor-pointer hover:border-cyan-500/30 transition-colors"
              >
                <span className="flex items-center gap-2 text-slate-300">
                  <span className={`h-2 w-2 rounded-full ${tog.dot}`} />
                  {tog.label}
                </span>
                <input
                  type="checkbox"
                  checked={tog.val}
                  onChange={(e) => tog.set(e.target.checked)}
                  className="accent-cyan-400 h-3.5 w-3.5"
                />
              </label>
            ))}
          </div>
        </div>

        {/* Footer telemetry */}
        <div className="mt-auto pt-3 border-t border-slate-800/80 font-mono text-[10px] text-slate-500 space-y-1">
          <div className="flex items-center gap-1.5 text-emerald-400">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
            MICROSOFT PLANETARY COMPUTER STAC
          </div>
          <div>SENTINEL-2 · SENTINEL-1 · WORLDCOVER · OSM</div>
        </div>
      </aside>

      {/* MAIN COMMAND CANVAS */}
      <main className="flex-1 p-4 lg:p-6 max-w-[1440px] mx-auto w-full">
        {/* TOP HEADER WITH HOVER-ANIMATED NAVIGATION LINK */}
        <header className="rounded-2xl border border-cyan-500/25 border-t-2 border-t-cyan-400 bg-gradient-to-r from-[#0A1122]/95 via-[#08152B]/90 to-[#0A1122]/95 p-5 shadow-[0_14px_40px_rgba(0,0,0,0.45)] backdrop-blur-xl">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-cyan-400 mb-1">
                {isBorder
                  ? "AEROMANIA '26 // DUAL-USE ORBITAL SURVEILLANCE // BORDER & PERIMETER SECURITY"
                  : "AEROMANIA '26 // ORBIT-TO-ECONOMY // URBANPULSE: URBAN MIGRATION & SPRAWL"}
              </div>
              <h1 className="font-display text-2xl sm:text-3xl font-bold tracking-tight bg-gradient-to-r from-white via-sky-300 to-cyan-400 bg-clip-text text-transparent">
                {isBorder
                  ? "UrbanPulse AI — Border & Perimeter Security"
                  : "UrbanPulse AI — Urban Migration & Sprawl"}
              </h1>
              <div className="flex flex-wrap gap-2 mt-3">
                {[
                  "▲ SENTINEL-2 L2A",
                  "◈ SENTINEL-1 SAR",
                  "★ VIIRS / NIGHT-LIGHT",
                  "⬡ OSM ARTERIAL GRAPH",
                ].map((pill) => (
                  <span
                    key={pill}
                    className="rounded-full border border-cyan-500/35 bg-cyan-500/10 px-3 py-0.5 font-mono text-[10px] font-semibold tracking-wider text-cyan-300"
                  >
                    {pill}
                  </span>
                ))}
              </div>
            </div>

            {/* Hover-animated mode switch link */}
            <a
              href={`?mode=${isBorder ? "urban_pulse" : "border_security"}`}
              onClick={handleToggleMode}
              title={
                isBorder
                  ? "Hover/Click to switch back to UrbanPulse: Urban Migration & Sprawl"
                  : "Hover/Click to switch to Border & Perimeter Encroachment Surveillance"
              }
              className={`group relative inline-flex items-center gap-2 rounded-xl border px-4 py-2.5 font-display text-xs font-bold tracking-wide transition-all duration-300 hover:-translate-y-0.5 ${
                isBorder
                  ? "border-cyan-400/50 bg-cyan-500/15 text-cyan-200 shadow-[0_0_20px_rgba(6,182,212,0.22)] hover:bg-cyan-500/25 hover:shadow-[0_0_32px_rgba(34,211,238,0.5)]"
                  : "border-amber-400/50 bg-amber-500/15 text-amber-200 shadow-[0_0_20px_rgba(245,158,11,0.22)] hover:bg-amber-500/25 hover:shadow-[0_0_32px_rgba(251,191,36,0.5)]"
              }`}
            >
              <span>
                {isBorder
                  ? "🏙️ ← UrbanPulse: Migration & Sprawl"
                  : "🛡️ Border & Perimeter Security →"}
              </span>
            </a>
          </div>
        </header>

        {/* DYNAMIC VALUE BANNER */}
        <div className="mt-3 rounded-xl border border-amber-500/30 border-l-4 border-l-amber-400 bg-gradient-to-r from-amber-500/10 via-[#0B1222]/90 to-[#0B1222]/90 px-4 py-2.5 text-xs text-slate-200 flex items-center gap-2.5">
          <Sparkles className="h-4 w-4 text-amber-400 shrink-0" />
          {isBorder ? (
            <div>
              <strong className="text-amber-300">Who Pays &amp; Strategic Value:</strong>{" "}
              <strong>Border Security Forces (BSF), Ministry of Defence (MoD), and Defense Estates</strong> pay
              for automated dual-use satellite surveillance — detecting unlit perimeter encroachment early and
              optimizing all-weather patrol &amp; smart-fence routing at{" "}
              <strong className="text-amber-300">3.4× lower lifecycle cost</strong>.
            </div>
          ) : (
            <div>
              <strong className="text-amber-300">Who Pays &amp; Economic Value:</strong> Urban Development
              Authorities (<strong>BDA · TUDA · MUDA · NMPA</strong>) &amp; Municipal Corporations pay for
              UrbanPulse AI to{" "}
              <strong className="text-white">lock in transit &amp; power corridors before encroachment</strong> —
              saving ₹ Crores instead of demolishing &amp; retrofitting 3 years later at{" "}
              <strong className="text-amber-300">3.4× the cost</strong>.
            </div>
          )}
        </div>

        {error && (
          <div className="mt-4 rounded-xl border border-red-500/40 bg-red-500/10 p-4 text-xs text-red-300">
            {error}
          </div>
        )}

        {loading && !data ? (
          <div className="mt-10 flex flex-col items-center justify-center py-20 text-cyan-300">
            <Radar className="h-10 w-10 animate-spin mb-3" />
            <div className="font-mono text-xs tracking-widest uppercase">
              Fusing Sentinel-2, Sentinel-1 SAR &amp; OSM Telemetry…
            </div>
          </div>
        ) : (
          data &&
          roi && (
            <>
              {/* 4 KPI HUD CARDS */}
              <section className="mt-4 grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3.5">
                <div className="rounded-2xl border border-slate-800 border-t-2 border-t-cyan-400 bg-[#090F1E]/90 p-4 shadow-lg">
                  <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-widest text-slate-400">
                    <span>
                      {isBorder ? "New Perimeter Structures (2021–26)" : "New Built-Up Growth"}
                    </span>
                    <Activity className="h-4 w-4 text-cyan-400" />
                  </div>
                  <div className="font-display text-3xl font-bold text-cyan-300 mt-1.5">
                    {data.new_built_sqkm.toFixed(1)} km²
                  </div>
                  <div className="text-xs text-slate-400 mt-1">
                    {isBorder ? "Protected Buffer Saved:" : "Prevented Slum Sprawl:"}{" "}
                    <span className="text-cyan-300 font-semibold">
                      {roi.sprawl_prevented_sqkm.toFixed(1)} km²
                    </span>
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800 border-t-2 border-t-red-500 bg-[#090F1E]/90 p-4 shadow-lg">
                  <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-widest text-slate-400">
                    <span>
                      {isBorder ? "Unlit / Covert Encroachment Alerts" : "Night-Light Gap Alerts"}
                    </span>
                    <AlertTriangle className="h-4 w-4 text-red-400" />
                  </div>
                  <div className="font-display text-3xl font-bold text-red-400 mt-1.5">
                    {data.n_nlg_alerts.toLocaleString()}
                  </div>
                  <div className="text-xs text-slate-400 mt-1">
                    {data.nlg_area_sqkm.toFixed(1)} km² flagged ·{" "}
                    <span className="text-red-300 font-semibold">
                      {roi.n_funded} {isBorder ? "patrol links" : "corridors"}
                    </span>
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800 border-t-2 border-t-amber-400 bg-[#090F1E]/90 p-4 shadow-lg">
                  <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-widest text-slate-400">
                    <span>
                      {isBorder
                        ? "2027–28 Infiltration / Encroachment Risk"
                        : "2027–28 Predicted Influx"}
                    </span>
                    <Users className="h-4 w-4 text-amber-400" />
                  </div>
                  <div className="font-display text-3xl font-bold text-amber-300 mt-1.5">
                    {(roi.unmitigated_influx / 1000).toFixed(1)}K
                  </div>
                  <div className="text-xs text-slate-400 mt-1">
                    {isBorder ? "Baseline" : "Total"} {(data.influx_2027_28 / 1000).toFixed(1)}K ·{" "}
                    <span className="text-amber-300 font-semibold">
                      {(roi.influx_covered / 1000).toFixed(1)}K {isBorder ? "monitored" : "sheltered"}
                    </span>
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800 border-t-2 border-t-emerald-400 bg-[#090F1E]/90 p-4 shadow-lg">
                  <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-widest text-slate-400">
                    <span>
                      {isBorder ? "Surveillance & Patrol Savings (₹ Cr)" : "Net Planner Savings"}
                    </span>
                    <IndianRupee className="h-4 w-4 text-emerald-400" />
                  </div>
                  <div className="font-display text-3xl font-bold text-emerald-400 mt-1.5">
                    ₹{roi.net_savings_cr.toFixed(0)} Cr{" "}
                    <span className="text-lg text-emerald-300/80">/ ${roi.net_savings_usd_m.toFixed(1)}M</span>
                  </div>
                  <div className="text-xs text-slate-400 mt-1">
                    {isBorder ? "Cueing vs. manual patrol" : "Act now vs. 3.4× retrofit"} ·{" "}
                    <span className="text-emerald-300 font-semibold">{roi.coverage_ratio.toFixed(0)}% cov</span>
                  </div>
                </div>
              </section>

              {/* CENTERPIECE MAP */}
              <section className="mt-5">
                <div className="flex items-center justify-between mb-2">
                  <div className="font-mono text-[11px] uppercase tracking-widest text-cyan-400 flex items-center gap-1.5">
                    <MapPin className="h-3.5 w-3.5" />
                    {isBorder ? "Perimeter Surveillance Map" : "Geospatial Command Map"} — {activeCoords?.name}
                  </div>
                  {loading && (
                    <span className="font-mono text-[10px] text-cyan-300 animate-pulse">
                      UPDATING TELEMETRY…
                    </span>
                  )}
                </div>
                <MissionMap
                  data={data}
                  mode={mode}
                  showZones={showZones}
                  showNlg={showNlg}
                  showForecast={showForecast}
                  showCorridors={showCorridors}
                />
              </section>

              {/* EXECUTIVE TABS */}
              <AnalyticsTabs data={data} hubName={activeCoords?.name || "Karnataka"} />
            </>
          )
        )}
      </main>
    </div>
  );
}
