import React, { useState } from 'react';
import {
  Activity,
  Calendar,
  ChevronDown,
  ChevronRight,
  Cpu,
  Droplets,
  Flame,
  Globe2,
  Layers,
  Play,
  Plus,
  Plug,
  Wind,
  X,
} from 'lucide-react';
import type {
  BoundingBox,
  ConnectedModule,
  DisasterType,
  EventSummary,
  LayerVisibility,
} from '../types/claimpulse';

interface LeftSidebarProps {
  events: EventSummary[];
  activeEvent: EventSummary;
  layers: LayerVisibility;
  modules: ConnectedModule[];
  isProcessing: boolean;
  processingStage: string;
  selectedDisasterType: DisasterType;
  onSelectDisasterType: (type: DisasterType) => void;
  onSelectPreset: (presetId: string) => void;
  onRunCustomEvent: (params: {
    preset_id?: string | null;
    disaster_type: DisasterType;
    country: string;
    state: string;
    district: string;
    start_date: string;
    end_date: string;
    bbox?: BoundingBox;
    intensity_multiplier: number;
  }) => void;
  onToggleLayer: (key: keyof LayerVisibility) => void;
  onRegisterModule: (mod: Omit<ConnectedModule, 'module_id' | 'status'>) => Promise<void>;
  onTriggerModuleAction: (mod: ConnectedModule) => void;
}

export const LeftSidebar: React.FC<LeftSidebarProps> = ({
  events,
  activeEvent,
  layers,
  modules,
  isProcessing,
  processingStage,
  selectedDisasterType,
  onSelectDisasterType,
  onSelectPreset,
  onRunCustomEvent,
  onToggleLayer,
  onRegisterModule,
  onTriggerModuleAction,
}) => {
  const [country, setCountry] = useState(activeEvent.country);
  const [state, setState] = useState(activeEvent.state);
  const [district, setDistrict] = useState(activeEvent.district);
  const [startDate, setStartDate] = useState(activeEvent.start_date);
  const [endDate, setEndDate] = useState(activeEvent.end_date);
  const [multiplier, setMultiplier] = useState(activeEvent.intensity_multiplier || 1.0);
  const [showAdvancedRegion, setShowAdvancedRegion] = useState(false);
  const [showModules, setShowModules] = useState(false);
  const [bbox, setBbox] = useState<BoundingBox>(activeEvent.bbox);
  const [showModuleModal, setShowModuleModal] = useState(false);

  React.useEffect(() => {
    setCountry(activeEvent.country);
    setState(activeEvent.state);
    setDistrict(activeEvent.district);
    setStartDate(activeEvent.start_date);
    setEndDate(activeEvent.end_date);
    setMultiplier(activeEvent.intensity_multiplier || 1.0);
    setBbox(activeEvent.bbox);
  }, [activeEvent]);

  const [modName, setModName] = useState('');
  const [modTeam, setModTeam] = useState('');
  const [modCategory, setModCategory] =
    useState<ConnectedModule['category']>('data_connector');
  const [modDesc, setModDesc] = useState('');
  const [modWebhook, setModWebhook] = useState('');

  const handleRun = () => {
    onRunCustomEvent({
      preset_id: showAdvancedRegion ? null : activeEvent.preset_id,
      disaster_type: selectedDisasterType,
      country,
      state,
      district,
      start_date: startDate,
      end_date: endDate,
      bbox: showAdvancedRegion ? bbox : undefined,
      intensity_multiplier: multiplier,
    });
  };

  const handleModuleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!modName.trim()) return;
    await onRegisterModule({
      name: modName.trim(),
      partner_team: modTeam.trim() || 'Partner Team',
      category: modCategory,
      description: modDesc.trim() || 'Partner telemetry & risk enrichment stream.',
      layer_id: 'hazard_extent',
      action_label: 'Sync Partner Data',
      kpis: { Integration: 'REST + Webhook', Status: 'Active' },
      webhook_url: modWebhook.trim() || null,
    });
    setModName('');
    setModTeam('');
    setModDesc('');
    setModWebhook('');
    setShowModuleModal(false);
  };

  const layerItems: Array<{ key: keyof LayerVisibility; label: string; color: string }> = [
    { key: 'hazardExtent', label: 'Damage & Flood Zones', color: '#EF4444' },
    { key: 'buildings', label: 'Building Footprints', color: '#2563EB' },
    { key: 'population', label: 'Population Grid', color: '#8B5CF6' },
    { key: 'roads', label: 'Roads & Corridors', color: '#F59E0B' },
    { key: 'farmland', label: 'Farmland & Crops', color: '#22C55E' },
    { key: 'criticalInfra', label: 'Hospitals & Power', color: '#22D3EE' },
    { key: 'insurancePortfolio', label: 'Insured Policies', color: '#FACC15' },
    { key: 'lossHeatmap', label: 'Loss Heatmap', color: '#F97316' },
    { key: 'satelliteImagery', label: 'Satellite Imagery', color: '#9FB0C7' },
  ];

  const filteredPresets = events.filter(
    (ev) =>
      ev.preset_id &&
      ev.intensity_multiplier === 1 &&
      ev.disaster_type === selectedDisasterType
  );

  return (
    <aside
      className="flex h-full w-full flex-col overflow-y-auto border-r border-[#1E2A40] bg-[#0E1524] text-[#F4F7FB]"
      aria-label="Event controls and map layer settings"
    >
      {/* Event Selection */}
      <div className="border-b border-[#1E2A40] p-3.5 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-[11px] font-semibold uppercase tracking-wider text-[#9FB0C7] flex items-center gap-1.5">
            <Globe2 className="h-3.5 w-3.5 text-[#22D3EE]" /> Event Control
          </h2>
          <span className="rounded border border-[#F59E0B]/40 bg-[#F59E0B]/10 px-1.5 py-0.5 text-[10px] font-medium text-[#F59E0B]">
            Demo scenario
          </span>
        </div>

        <div>
          <label htmlFor="preset-select" className="sr-only">
            Preset Disaster Event
          </label>
          <select
            id="preset-select"
            value={activeEvent.preset_id || ''}
            onChange={(e) => {
              if (e.target.value) onSelectPreset(e.target.value);
            }}
            className="w-full rounded-lg border border-[#1E2A40] bg-[#141C2E] px-2.5 py-2 text-xs font-medium text-[#F4F7FB] transition-colors duration-150 hover:border-[#2C3D5B]"
          >
            {filteredPresets.map((ev) => (
              <option key={ev.id} value={ev.preset_id || ev.id}>
                {ev.title}
              </option>
            ))}
          </select>
        </div>

        <div className="grid grid-cols-3 gap-1.5" role="group" aria-label="Disaster hazard type">
          {(
            [
              { id: 'flood', label: 'Flood', Icon: Droplets },
              { id: 'wildfire', label: 'Wildfire', Icon: Flame },
              { id: 'cyclone', label: 'Cyclone', Icon: Wind },
            ] as const
          ).map(({ id, label, Icon }) => {
            const active = selectedDisasterType === id;
            return (
              <button
                key={id}
                type="button"
                onClick={() => {
                  onSelectDisasterType(id);
                }}
                className={`flex items-center justify-center gap-1 rounded-lg border py-1.5 text-xs font-medium transition-colors duration-150 ${
                  active
                    ? 'border-[#2563EB] bg-[#2563EB]/20 text-[#F4F7FB]'
                    : 'border-[#1E2A40] bg-[#141C2E] text-[#9FB0C7] hover:text-[#F4F7FB]'
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                {label}
              </button>
            );
          })}
        </div>

        {/* Severity Stress Control */}
        <div className="rounded-lg border border-[#1E2A40] bg-[#141C2E] p-2.5">
          <div className="flex items-center justify-between text-[11px]">
            <label htmlFor="severity-slider" className="text-[#9FB0C7] font-medium">
              Intensity Sensitivity
            </label>
            <span className="font-mono-tabular font-semibold text-[#22D3EE]">
              {multiplier === 1
                ? '1.00x (Observed)'
                : `${multiplier.toFixed(2)}x (${Math.round((multiplier - 1) * 100) > 0 ? '+' : ''}${Math.round(
                    (multiplier - 1) * 100
                  )}%)`}
            </span>
          </div>
          <input
            id="severity-slider"
            type="range"
            min={0.7}
            max={1.5}
            step={0.05}
            value={multiplier}
            onChange={(e) => setMultiplier(Number(e.target.value))}
            className="mt-2 w-full accent-[#2563EB]"
          />
          <div className="mt-1 flex justify-between text-[10px] text-[#9FB0C7]">
            <button type="button" onClick={() => setMultiplier(1.0)} className="hover:text-[#F4F7FB]">
              Reset 1.0x
            </button>
            <button type="button" onClick={() => setMultiplier(1.2)} className="hover:text-[#22D3EE]">
              Stress +20%
            </button>
          </div>
        </div>

        {/* Custom Region & Dates */}
        <button
          type="button"
          onClick={() => setShowAdvancedRegion((v) => !v)}
          aria-expanded={showAdvancedRegion}
          className="flex w-full items-center justify-between rounded-md border border-[#1E2A40] bg-[#141C2E]/60 px-2.5 py-1.5 text-[11px] text-[#9FB0C7] hover:text-[#F4F7FB] transition-colors duration-150"
        >
          <span>Region, Dates &amp; Bounding Box</span>
          {showAdvancedRegion ? (
            <ChevronDown className="h-3.5 w-3.5" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5" />
          )}
        </button>

        {showAdvancedRegion && (
          <div className="space-y-2 rounded-lg border border-[#1E2A40] bg-[#070B14] p-2.5 text-[10px]">
            <div className="grid grid-cols-2 gap-1.5">
              <div>
                <label className="text-[#9FB0C7] block">State, Country</label>
                <input
                  value={`${state}, ${country}`}
                  onChange={(e) => {
                    const parts = e.target.value.split(',');
                    setState(parts[0]?.trim() || '');
                    if (parts[1]) setCountry(parts[1].trim());
                  }}
                  className="mt-0.5 w-full rounded border border-[#1E2A40] bg-[#141C2E] px-2 py-1 text-[11px] text-[#F4F7FB]"
                />
              </div>
              <div>
                <label className="text-[#9FB0C7] block">District / Basin</label>
                <input
                  value={district}
                  onChange={(e) => setDistrict(e.target.value)}
                  className="mt-0.5 w-full rounded border border-[#1E2A40] bg-[#141C2E] px-2 py-1 text-[11px] text-[#F4F7FB]"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-1.5">
              <div>
                <label className="flex items-center gap-1 text-[#9FB0C7]">
                  <Calendar className="h-3 w-3" /> Start
                </label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="mt-0.5 w-full rounded border border-[#1E2A40] bg-[#141C2E] px-1.5 py-1 font-mono-tabular text-[10px] text-[#F4F7FB]"
                />
              </div>
              <div>
                <label className="flex items-center gap-1 text-[#9FB0C7]">
                  <Calendar className="h-3 w-3" /> End
                </label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="mt-0.5 w-full rounded border border-[#1E2A40] bg-[#141C2E] px-1.5 py-1 font-mono-tabular text-[10px] text-[#F4F7FB]"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-1.5">
              <div>
                <span className="text-[#9FB0C7]">Min Lon / Lat</span>
                <div className="mt-0.5 flex gap-1">
                  <input
                    type="number"
                    step="0.01"
                    aria-label="Minimum longitude"
                    value={bbox.min_lon}
                    onChange={(e) => setBbox({ ...bbox, min_lon: Number(e.target.value) })}
                    className="w-1/2 rounded border border-[#1E2A40] bg-[#141C2E] px-1 py-0.5 font-mono-tabular text-[#F4F7FB]"
                  />
                  <input
                    type="number"
                    step="0.01"
                    aria-label="Minimum latitude"
                    value={bbox.min_lat}
                    onChange={(e) => setBbox({ ...bbox, min_lat: Number(e.target.value) })}
                    className="w-1/2 rounded border border-[#1E2A40] bg-[#141C2E] px-1 py-0.5 font-mono-tabular text-[#F4F7FB]"
                  />
                </div>
              </div>
              <div>
                <span className="text-[#9FB0C7]">Max Lon / Lat</span>
                <div className="mt-0.5 flex gap-1">
                  <input
                    type="number"
                    step="0.01"
                    aria-label="Maximum longitude"
                    value={bbox.max_lon}
                    onChange={(e) => setBbox({ ...bbox, max_lon: Number(e.target.value) })}
                    className="w-1/2 rounded border border-[#1E2A40] bg-[#141C2E] px-1 py-0.5 font-mono-tabular text-[#F4F7FB]"
                  />
                  <input
                    type="number"
                    step="0.01"
                    aria-label="Maximum latitude"
                    value={bbox.max_lat}
                    onChange={(e) => setBbox({ ...bbox, max_lat: Number(e.target.value) })}
                    className="w-1/2 rounded border border-[#1E2A40] bg-[#141C2E] px-1 py-0.5 font-mono-tabular text-[#F4F7FB]"
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        <button
          type="button"
          onClick={handleRun}
          disabled={isProcessing}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-[#2563EB] px-3 py-2 text-xs font-semibold text-[#F4F7FB] transition-colors duration-150 hover:bg-[#1D4ED8] disabled:opacity-60"
        >
          {isProcessing ? (
            <>
              <Activity className="h-3.5 w-3.5 animate-spin" />
              {processingStage}
            </>
          ) : (
            <>
              <Play className="h-3.5 w-3.5 fill-current" />
              Run Economic Engine
            </>
          )}
        </button>
      </div>

      {/* Layer Toggles */}
      <div className="border-b border-[#1E2A40] p-3.5">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-[11px] font-semibold uppercase tracking-wider text-[#9FB0C7] flex items-center gap-1.5">
            <Layers className="h-3.5 w-3.5 text-[#22D3EE]" /> Map Layers
          </h2>
          <span className="font-mono-tabular text-[10px] text-[#9FB0C7]">9 layers</span>
        </div>
        <div className="space-y-1">
          {layerItems.map((item) => {
            const active = layers[item.key];
            return (
              <button
                key={item.key}
                type="button"
                role="switch"
                aria-checked={active}
                onClick={() => onToggleLayer(item.key)}
                className={`flex w-full items-center justify-between rounded-md px-2 py-1.5 text-xs transition-colors duration-150 ${
                  active
                    ? 'bg-[#141C2E] text-[#F4F7FB]'
                    : 'text-[#9FB0C7] hover:bg-[#141C2E]/60 hover:text-[#F4F7FB]'
                }`}
              >
                <span className="flex items-center gap-2">
                  <span
                    className="h-2 w-2 rounded-full"
                    style={{ backgroundColor: active ? item.color : '#1E2A40' }}
                  />
                  {item.label}
                </span>
                <span
                  className={`h-3.5 w-6 rounded-full p-0.5 transition-colors duration-150 ${
                    active ? 'bg-[#2563EB]' : 'bg-[#1E2A40]'
                  }`}
                >
                  <span
                    className={`block h-2.5 w-2.5 rounded-full bg-[#F4F7FB] transition-transform duration-150 ${
                      active ? 'translate-x-2.5' : 'translate-x-0'
                    }`}
                  />
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Connected Modules */}
      <div className="p-3.5">
        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={() => setShowModules((v) => !v)}
            aria-expanded={showModules}
            className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-[#9FB0C7] hover:text-[#F4F7FB]"
          >
            <Plug className="h-3.5 w-3.5 text-[#22D3EE]" /> Connected Modules ({modules.length})
            {showModules ? (
              <ChevronDown className="h-3.5 w-3.5" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" />
            )}
          </button>
          <button
            type="button"
            onClick={() => {
              setShowModules(true);
              setShowModuleModal(true);
            }}
            aria-label="Register new partner module"
            className="flex items-center gap-1 rounded border border-[#1E2A40] bg-[#141C2E] px-1.5 py-0.5 text-[10px] font-medium text-[#22D3EE] hover:border-[#2563EB]"
          >
            <Plus className="h-3 w-3" /> Add
          </button>
        </div>

        {showModules && (
          <div className="mt-2.5 space-y-2">
            {modules.map((mod) => (
              <div
                key={mod.module_id}
                className="rounded-lg border border-[#1E2A40] bg-[#141C2E] p-2.5"
              >
                <div className="flex items-start justify-between gap-1.5">
                  <div>
                    <div className="flex items-center gap-1.5">
                      <Cpu className="h-3 w-3 text-[#22C55E] shrink-0" />
                      <span className="text-xs font-semibold text-[#F4F7FB]">{mod.name}</span>
                    </div>
                    <div className="text-[10px] text-[#9FB0C7] mt-0.5">{mod.partner_team}</div>
                  </div>
                  <span className="rounded bg-[#22C55E]/15 px-1.5 py-0.5 text-[9px] font-semibold uppercase text-[#22C55E]">
                    {mod.status}
                  </span>
                </div>
                <p className="mt-1 text-[11px] leading-snug text-[#9FB0C7]">{mod.description}</p>
                {mod.action_label && (
                  <button
                    type="button"
                    onClick={() => onTriggerModuleAction(mod)}
                    className="mt-2 flex w-full items-center justify-center gap-1 rounded border border-[#1E2A40] bg-[#0E1524] py-1 text-[10px] font-medium text-[#F4F7FB] hover:border-[#2563EB]"
                  >
                    {mod.action_label}
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Register Partner Module Modal */}
      {showModuleModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-[#070B14]/80 p-4 backdrop-blur-sm"
          role="dialog"
          aria-modal="true"
          aria-label="Register partner module"
        >
          <div className="w-full max-w-md rounded-xl border border-[#1E2A40] bg-[#0E1524] p-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-[#1E2A40] pb-2.5">
              <h3 className="text-sm font-semibold text-[#F4F7FB]">Add Partner Module</h3>
              <button
                type="button"
                onClick={() => setShowModuleModal(false)}
                className="text-[#9FB0C7] hover:text-[#F4F7FB]"
                aria-label="Close modal"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <form onSubmit={handleModuleSubmit} className="mt-3 space-y-2.5 text-xs">
              <div>
                <label className="block text-[#9FB0C7] mb-1">Module Name</label>
                <input
                  required
                  placeholder="e.g. HydroMesh River Gauges"
                  value={modName}
                  onChange={(e) => setModName(e.target.value)}
                  className="w-full rounded border border-[#1E2A40] bg-[#141C2E] px-2.5 py-1.5 text-[#F4F7FB]"
                />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-[#9FB0C7] mb-1">Partner Team</label>
                  <input
                    placeholder="e.g. Team HydroMesh"
                    value={modTeam}
                    onChange={(e) => setModTeam(e.target.value)}
                    className="w-full rounded border border-[#1E2A40] bg-[#141C2E] px-2.5 py-1.5 text-[#F4F7FB]"
                  />
                </div>
                <div>
                  <label className="block text-[#9FB0C7] mb-1">Interface Type</label>
                  <select
                    value={modCategory}
                    onChange={(e) => setModCategory(e.target.value as ConnectedModule['category'])}
                    className="w-full rounded border border-[#1E2A40] bg-[#141C2E] px-2 py-1.5 text-[#F4F7FB]"
                  >
                    <option value="data_connector">Data Connector</option>
                    <option value="hazard_module">Hazard Module</option>
                    <option value="exposure_module">Exposure Module</option>
                    <option value="economic_model">Economic Model</option>
                    <option value="decision_workflow">Decision Workflow</option>
                  </select>
                </div>
              </div>
              <div>
                <label className="block text-[#9FB0C7] mb-1">Description</label>
                <textarea
                  rows={2}
                  placeholder="How this module enriches damage or valuation..."
                  value={modDesc}
                  onChange={(e) => setModDesc(e.target.value)}
                  className="w-full rounded border border-[#1E2A40] bg-[#141C2E] px-2.5 py-1.5 text-[#F4F7FB]"
                />
              </div>
              <div>
                <label className="block text-[#9FB0C7] mb-1">Webhook URL (optional)</label>
                <input
                  placeholder="https://partner-api.example.com/webhook"
                  value={modWebhook}
                  onChange={(e) => setModWebhook(e.target.value)}
                  className="w-full rounded border border-[#1E2A40] bg-[#141C2E] px-2.5 py-1.5 text-[#F4F7FB]"
                />
              </div>
              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowModuleModal(false)}
                  className="rounded-lg border border-[#1E2A40] px-3 py-1.5 text-[#9FB0C7] hover:text-[#F4F7FB]"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="rounded-lg bg-[#2563EB] px-3.5 py-1.5 font-semibold text-[#F4F7FB] hover:bg-[#1D4ED8]"
                >
                  Connect Module
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </aside>
  );
};
