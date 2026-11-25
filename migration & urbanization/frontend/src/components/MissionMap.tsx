import { useEffect, useRef } from "react";
import L from "leaflet";
import type { AnalysisResponse } from "../types";

interface MissionMapProps {
  data: AnalysisResponse;
  showZones: boolean;
  showNlg: boolean;
  showForecast: boolean;
  showCorridors: boolean;
}

const ZONE_COLOR: Record<number, string> = {
  0: "#06B6D4",
  1: "#22C55E",
  2: "#EF4444",
  3: "#84CC16",
};

export default function MissionMap({
  data,
  showZones,
  showNlg,
  showForecast,
  showCorridors,
}: MissionMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const layersRef = useRef<{
    zones: L.LayerGroup;
    nlg: L.LayerGroup;
    forecast: L.LayerGroup;
    corridors: L.LayerGroup;
  } | null>(null);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = L.map(containerRef.current, {
      center: [data.lat, data.lon],
      zoom: 12,
      maxZoom: 16,
      zoomControl: true,
      preferCanvas: true,
    });

    const darkTiles = L.tileLayer(
      "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      {
        attribution: "&copy; OpenStreetMap &copy; CARTO",
        maxZoom: 16,
      }
    ).addTo(map);

    const satTiles = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      {
        attribution: "&copy; Esri World Imagery",
        maxZoom: 19,
      }
    );

    L.control
      .layers(
        {
          "Orbital Dark Telemetry": darkTiles,
          "Satellite Imagery (Esri)": satTiles,
        },
        {},
        { position: "topright" }
      )
      .addTo(map);

    const zones = L.layerGroup().addTo(map);
    const forecast = L.layerGroup().addTo(map);
    const nlg = L.layerGroup().addTo(map);
    const corridors = L.layerGroup().addTo(map);

    mapRef.current = map;
    layersRef.current = { zones, nlg, forecast, corridors };

    return () => {
      map.remove();
      mapRef.current = null;
      layersRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    const layers = layersRef.current;
    if (!map || !layers) return;

    map.setView([data.lat, data.lon], 12, { animate: true });

    layers.zones.clearLayers();
    layers.forecast.clearLayers();
    layers.nlg.clearLayers();
    layers.corridors.clearLayers();

    if (showZones) {
      const half = 0.00113;
      for (const cell of data.cells) {
        const color = ZONE_COLOR[cell.zone] ?? "#64748B";
        L.rectangle(
          [
            [cell.lat - half, cell.lon - half],
            [cell.lat + half, cell.lon + half],
          ],
          {
            color,
            weight: 0.4,
            opacity: 0.5,
            fillColor: color,
            fillOpacity: cell.zone === 2 ? 0.45 : 0.22,
          }
        ).addTo(layers.zones);
      }
    }

    if (showForecast) {
      for (const cell of data.cells) {
        if (cell.sprawl < 0.32) continue;
        L.circle([cell.lat, cell.lon], {
          radius: 170 + cell.sprawl * 160,
          stroke: false,
          fillColor: cell.sprawl > 0.6 ? "#F59E0B" : "#FBBF24",
          fillOpacity: Math.min(0.42, cell.sprawl * 0.48),
        }).addTo(layers.forecast);
      }
    }

    if (showNlg) {
      const half = 0.00113;
      for (const cell of data.nlg_cells) {
        const rect = L.rectangle(
          [
            [cell.lat - half, cell.lon - half],
            [cell.lat + half, cell.lon + half],
          ],
          {
            color: "#EF4444",
            weight: 1.5,
            opacity: 0.9,
            fillColor: "#EF4444",
            fillOpacity: 0.55,
          }
        );
        rect.bindPopup(`
          <div style="font-family:Inter,sans-serif;min-width:210px;font-size:12px;">
            <div style="color:#F87171;font-weight:700;margin-bottom:6px;letter-spacing:0.5px;">
              ⚠ NIGHT-LIGHT GAP ALERT
            </div>
            <div style="color:#94A3B8;line-height:1.6;">
              <div><b style="color:#E2E8F0">Coords:</b> ${cell.lat.toFixed(4)}°N, ${cell.lon.toFixed(4)}°E</div>
              <div><b style="color:#E2E8F0">ΔNDBI Growth:</b> <span style="color:#22D3EE">${cell.delta_ndbi >= 0 ? "+" : ""}${cell.delta_ndbi.toFixed(3)}</span></div>
              <div><b style="color:#E2E8F0">Night-Light Proxy:</b> <span style="color:#FBBF24">${cell.viirs.toFixed(1)}</span></div>
              <div><b style="color:#E2E8F0">Road Distance:</b> ${Math.round(cell.road_m)} m</div>
              <div><b style="color:#E2E8F0">2027–28 Sprawl Risk:</b> <span style="color:#F87171">${(cell.sprawl * 100).toFixed(0)}%</span></div>
            </div>
          </div>
        `);
        rect.addTo(layers.nlg);
      }
    }

    if (showCorridors) {
      for (const c of data.corridors) {
        if (!c.coords || c.coords.length < 2) continue;
        // Outer glow
        L.polyline(c.coords, {
          color: "#06B6D4",
          weight: 9,
          opacity: 0.25,
        }).addTo(layers.corridors);

        // Crisp inner line
        const line = L.polyline(c.coords, {
          color: "#22D3EE",
          weight: 3.5,
          opacity: 0.95,
          dashArray: "8, 6",
        });

        const deferredCost = (c.cost_cr * 3.4).toFixed(2);
        const netSavings = (c.cost_cr * 2.4).toFixed(2);
        line.bindPopup(`
          <div style="font-family:Inter,sans-serif;min-width:225px;font-size:12px;">
            <div style="color:#22D3EE;font-weight:700;margin-bottom:6px;letter-spacing:0.5px;">
              🛰️ AI CORRIDOR #${c.rank}
            </div>
            <div style="color:#94A3B8;line-height:1.6;">
              <div><b style="color:#E2E8F0">Corridor Length:</b> ${c.corridor_km.toFixed(2)} km</div>
              <div><b style="color:#E2E8F0">Households Served:</b> ${c.served_hh.toLocaleString()}</div>
              <div><b style="color:#E2E8F0">Early Cost:</b> <span style="color:#22D3EE">₹${c.cost_cr.toFixed(2)} Cr</span></div>
              <div><b style="color:#E2E8F0">3-Yr Deferred Cost:</b> <span style="color:#F87171">₹${deferredCost} Cr</span></div>
              <div><b style="color:#E2E8F0">Net Corridor Savings:</b> <span style="color:#4ADE80">₹${netSavings} Cr</span></div>
            </div>
          </div>
        `);
        line.addTo(layers.corridors);

        L.circleMarker([c.source_lat, c.source_lon], {
          radius: 6,
          color: "#FFFFFF",
          weight: 1.5,
          fillColor: "#EF4444",
          fillOpacity: 1,
        }).addTo(layers.corridors);

        L.circleMarker([c.target_lat, c.target_lon], {
          radius: 5,
          color: "#FFFFFF",
          weight: 1.5,
          fillColor: "#22D3EE",
          fillOpacity: 1,
        }).addTo(layers.corridors);
      }
    }
  }, [data, showZones, showNlg, showForecast, showCorridors]);

  return (
    <div className="relative rounded-2xl overflow-hidden border border-cyan-500/25 shadow-[0_14px_45px_rgba(0,0,0,0.55)]">
      <div ref={containerRef} className="w-full h-[540px]" />

      {/* Top-left tactical coordinate HUD */}
      <div className="pointer-events-none absolute top-3 left-14 z-[500] hidden sm:flex items-center gap-3 rounded-lg border border-cyan-500/25 bg-[#04070F]/85 px-3 py-1.5 backdrop-blur-md">
        <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_8px_#34d399]" />
        <span className="font-mono text-[11px] tracking-wider text-slate-300">
          SWATH LOCK: <strong className="text-cyan-300">{data.lat.toFixed(4)}°N, {data.lon.toFixed(4)}°E</strong> · 12×12 KM
        </span>
      </div>

      {/* Bottom-left compact legend */}
      <div className="pointer-events-none absolute bottom-4 left-4 z-[500] rounded-xl border border-slate-700/70 bg-[#060B17]/90 px-3.5 py-2.5 backdrop-blur-md">
        <div className="font-mono text-[10px] uppercase tracking-widest text-cyan-400 mb-1.5">
          Orbital Layer Key
        </div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] text-slate-300">
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-cyan-400" /> Tech Core
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-emerald-500" /> Legal Suburb
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-red-500 shadow-[0_0_6px_#ef4444]" /> Night-Light Gap
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-lime-500" /> Green / Water
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-amber-400/80" /> 2027–28 Risk
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-0.5 w-3.5 bg-cyan-300 shadow-[0_0_6px_#22d3ee]" /> AI Corridor
          </div>
        </div>
      </div>
    </div>
  );
}
