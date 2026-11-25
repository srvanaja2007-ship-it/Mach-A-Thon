import { useEffect, useState } from 'react';
import {
  Activity,
  Map as MapIcon,
  Radar,
  Sliders,
} from 'lucide-react';
import {
  getEvent,
  getModules,
  listEvents,
  registerModule,
  reviewDecision,
  runEventAnalysis,
} from './api/client';
import { BottomBar } from './components/BottomBar';
import { LeftSidebar } from './components/LeftSidebar';
import { MissionMap } from './components/MissionMap';
import { RightPanel } from './components/RightPanel';
import type {
  BoundingBox,
  ConnectedModule,
  DisasterType,
  EventFullResponse,
  EventSummary,
  LayerVisibility,
  UserPersona,
} from './types/claimpulse';

const DEFAULT_LAYERS: LayerVisibility = {
  satelliteImagery: false,
  hazardExtent: true,
  buildings: true,
  population: true,
  roads: true,
  farmland: true,
  criticalInfra: true,
  insurancePortfolio: true,
  lossHeatmap: false,
};

export function App() {
  const [events, setEvents] = useState<EventSummary[]>([]);
  const [activeData, setActiveData] = useState<EventFullResponse | null>(null);
  const [selectedDisasterType, setSelectedDisasterType] = useState<DisasterType>('flood');
  const [modules, setModules] = useState<ConnectedModule[]>([]);
  const [layers, setLayers] = useState<LayerVisibility>(DEFAULT_LAYERS);
  const [persona, setPersona] = useState<UserPersona>('insurer');
  const [timelineIndex, setTimelineIndex] = useState(4);
  const [focusZoneId, setFocusZoneId] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStage, setProcessingStage] = useState('Loading scene...');
  const [toast, setToast] = useState<string | null>(null);
  const [mobilePane, setMobilePane] = useState<'controls' | 'map' | 'intel'>('map');

  const notify = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3500);
  };

  useEffect(() => {
    const bootstrap = async () => {
      const [evList, odisha, mods] = await Promise.all([
        listEvents(),
        getEvent('odisha-flood-2026'),
        getModules(),
      ]);
      setEvents(evList);
      setActiveData(odisha);
      setModules(mods);
      setTimelineIndex(Math.max(0, odisha.damage.metrics.timeline_dates.length - 1));
    };
    void bootstrap();
  }, []);

  const handleSelectDisasterType = (type: DisasterType) => {
    setSelectedDisasterType(type);
    // Find matching base preset event for the selected disaster type
    const match = events.find(
      (ev) => ev.preset_id && ev.intensity_multiplier === 1 && ev.disaster_type === type
    );
    if (match?.preset_id && match.preset_id !== activeData?.event.preset_id) {
      void handleSelectPreset(match.preset_id);
    } else {
      notify(`Filtered to ${type.toUpperCase()} events & map zones`);
    }
  };

  const handleSelectPreset = async (presetId: string) => {
    setIsProcessing(true);
    setProcessingStage('Loading satellite scene...');
    try {
      const full = await getEvent(presetId);
      setActiveData(full);
      setSelectedDisasterType(full.event.disaster_type);
      setTimelineIndex(Math.max(0, full.damage.metrics.timeline_dates.length - 1));
      setFocusZoneId(null);
      notify(`Loaded: ${full.event.title}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleRunCustomEvent = async (params: {
    preset_id?: string | null;
    disaster_type: DisasterType;
    country: string;
    state: string;
    district: string;
    start_date: string;
    end_date: string;
    bbox?: BoundingBox;
    intensity_multiplier: number;
  }) => {
    setIsProcessing(true);
    setProcessingStage('Ingesting SAR & DEM...');
    const t1 = setTimeout(() => setProcessingStage('Overlaying assets...'), 220);
    const t2 = setTimeout(() => setProcessingStage('Pricing models in ₹...'), 440);
    try {
      const full = await runEventAnalysis(params);
      setActiveData(full);
      setEvents(await listEvents());
      setTimelineIndex(Math.max(0, full.damage.metrics.timeline_dates.length - 1));
      setFocusZoneId(null);
      notify(`Updated: ${full.event.title}`);
    } finally {
      clearTimeout(t1);
      clearTimeout(t2);
      setIsProcessing(false);
    }
  };

  const handleToggleLayer = (key: keyof LayerVisibility) => {
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handleReviewClaim = async (
    decisionId: string,
    action: 'Approve' | 'Escalate' | 'Request Field Verification',
    notes: string
  ) => {
    if (!activeData) return;
    const updated = await reviewDecision(
      activeData.event.id,
      decisionId,
      action,
      'Senior Claims Analyst',
      notes
    );
    setActiveData(updated);
    notify(`Audit recorded: ${action} on ${decisionId}`);
  };

  const handleRegisterModule = async (mod: Omit<ConnectedModule, 'module_id' | 'status'>) => {
    const created = await registerModule(mod);
    setModules(await getModules());
    notify(`Connected module "${created.name}"`);
  };

  const handleTriggerModuleAction = (mod: ConnectedModule) => {
    if (mod.layer_id === 'farmland') {
      setLayers((l) => ({ ...l, farmland: true }));
    } else if (mod.layer_id === 'critical_infrastructure') {
      setLayers((l) => ({ ...l, criticalInfra: true }));
    } else if (mod.layer_id === 'insurance_portfolio') {
      setLayers((l) => ({ ...l, insurancePortfolio: true }));
    }
    notify(`${mod.name}: ${mod.action_label || 'Synced'}`);
  };

  // Matching skeleton loader while bootstrapping
  if (!activeData) {
    return (
      <div className="flex h-screen flex-col bg-[#070B14] text-[#F4F7FB]">
        <div className="h-12 border-b border-[#1E2A40] bg-[#0E1524] px-4 flex items-center justify-between">
          <div className="h-5 w-48 rounded bg-[#141C2E] animate-pulse" />
          <div className="h-6 w-72 rounded bg-[#141C2E] animate-pulse" />
        </div>
        <div className="flex flex-1 overflow-hidden">
          <div className="hidden lg:block w-[305px] border-r border-[#1E2A40] bg-[#0E1524] p-3.5 space-y-3">
            <div className="h-44 rounded-lg bg-[#141C2E] animate-pulse" />
            <div className="h-64 rounded-lg bg-[#141C2E] animate-pulse" />
          </div>
          <div className="flex-1 bg-[#070B14] flex flex-col items-center justify-center gap-2">
            <Radar className="h-8 w-8 text-[#22D3EE] animate-spin" />
            <span className="text-xs font-medium text-[#9FB0C7]">
              Initializing ClaimPulse Decision Console...
            </span>
          </div>
          <div className="hidden lg:block w-[455px] border-l border-[#1E2A40] bg-[#070B14] p-3 space-y-3">
            <div className="grid grid-cols-3 gap-2">
              {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="h-14 rounded-lg bg-[#0E1524] border border-[#1E2A40] animate-pulse" />
              ))}
            </div>
            <div className="h-32 rounded-xl bg-[#0E1524] border border-[#1E2A40] animate-pulse" />
            <div className="h-44 rounded-xl bg-[#0E1524] border border-[#1E2A40] animate-pulse" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-[#070B14] text-[#F4F7FB]">
      {/* Fixed Top Command Bar */}
      <header className="relative flex flex-wrap items-center justify-between gap-2 border-b border-[#1E2A40] bg-[#0E1524] px-4 py-2">
        {/* Processing Progress Ribbon */}
        {isProcessing && (
          <div className="absolute top-0 left-0 right-0 h-0.5 overflow-hidden bg-[#1E2A40]">
            <div className="h-full w-1/3 bg-[#22D3EE] animate-ribbon" />
          </div>
        )}

        <div className="flex items-center gap-2.5">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-[#2563EB] text-[#F4F7FB]">
            <Radar className="h-4 w-4" aria-hidden="true" />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm font-bold tracking-tight text-[#F4F7FB]">ClaimPulse</span>
            <span className="hidden sm:inline text-[11px] text-[#9FB0C7]">
              Satellite Risk &amp; Decision Console (INR)
            </span>
            {activeData.event.is_demo && (
              <span className="rounded border border-[#F59E0B]/40 bg-[#F59E0B]/10 px-1.5 py-0.5 text-[10px] font-semibold text-[#F59E0B]">
                Demo data
              </span>
            )}
          </div>
        </div>

        {/* Disaster Type Quick Selector (Single Source of Truth) */}
        <div
          className="hidden lg:flex items-center gap-1 rounded-lg border border-[#1E2A40] bg-[#070B14] p-0.5"
          role="group"
          aria-label="Filter events by disaster type"
        >
          {(
            [
              { id: 'flood', label: 'Flood' },
              { id: 'wildfire', label: 'Wildfire' },
              { id: 'cyclone', label: 'Cyclone' },
            ] as const
          ).map((item) => {
            const active = selectedDisasterType === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => handleSelectDisasterType(item.id)}
                className={`rounded-md px-3 py-1 text-xs font-medium transition-colors duration-150 ${
                  active
                    ? 'bg-[#2563EB] text-[#F4F7FB]'
                    : 'text-[#9FB0C7] hover:text-[#F4F7FB]'
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </div>

        {/* Stakeholder Economic Model Selector */}
        <div className="hidden md:flex items-center gap-2">
          <label htmlFor="stakeholder-select" className="text-[11px] text-[#9FB0C7]">
            Decision Model:
          </label>
          <select
            id="stakeholder-select"
            value={persona}
            onChange={(e) => setPersona(e.target.value as UserPersona)}
            className="rounded-lg border border-[#1E2A40] bg-[#141C2E] px-2.5 py-1 text-xs font-medium text-[#F4F7FB] transition-colors duration-150 hover:border-[#2563EB]"
          >
            <option value="insurer">Insurer (Claims &amp; Payouts)</option>
            <option value="government">Government Funds (SDRF/NDRF Relief)</option>
            <option value="humanitarian">Humanitarian Aid (Emergency Kits &amp; Cash)</option>
            <option value="lender">Bank / Lender (Collateral &amp; Credit VaR)</option>
            <option value="business">Business (Operations &amp; Supply Chain)</option>
          </select>
        </div>

        {/* Mobile View Switcher */}
        <div className="flex lg:hidden items-center gap-1 rounded-lg border border-[#1E2A40] bg-[#141C2E] p-0.5 text-[11px]">
          <button
            type="button"
            onClick={() => setMobilePane('controls')}
            className={`flex items-center gap-1 rounded px-2 py-1 ${
              mobilePane === 'controls' ? 'bg-[#2563EB] text-[#F4F7FB]' : 'text-[#9FB0C7]'
            }`}
          >
            <Sliders className="h-3 w-3" /> Controls
          </button>
          <button
            type="button"
            onClick={() => setMobilePane('map')}
            className={`flex items-center gap-1 rounded px-2 py-1 ${
              mobilePane === 'map' ? 'bg-[#2563EB] text-[#F4F7FB]' : 'text-[#9FB0C7]'
            }`}
          >
            <MapIcon className="h-3 w-3" /> Map
          </button>
          <button
            type="button"
            onClick={() => setMobilePane('intel')}
            className={`flex items-center gap-1 rounded px-2 py-1 ${
              mobilePane === 'intel' ? 'bg-[#2563EB] text-[#F4F7FB]' : 'text-[#9FB0C7]'
            }`}
          >
            <Activity className="h-3 w-3" /> Decisions
          </button>
        </div>
      </header>

      {/* Main 3-Pane Workspace */}
      <main className="relative flex flex-1 overflow-hidden">
        <div
          className={`${
            mobilePane === 'controls' ? 'flex w-full' : 'hidden'
          } lg:flex lg:w-[295px] xl:w-[310px] shrink-0 overflow-hidden`}
        >
          <LeftSidebar
            events={events}
            activeEvent={activeData.event}
            layers={layers}
            modules={modules}
            isProcessing={isProcessing}
            processingStage={processingStage}
            selectedDisasterType={selectedDisasterType}
            onSelectDisasterType={handleSelectDisasterType}
            onSelectPreset={handleSelectPreset}
            onRunCustomEvent={handleRunCustomEvent}
            onToggleLayer={handleToggleLayer}
            onRegisterModule={handleRegisterModule}
            onTriggerModuleAction={handleTriggerModuleAction}
          />
        </div>

        <div
          className={`${
            mobilePane === 'map' ? 'flex' : 'hidden'
          } lg:flex flex-1 relative overflow-hidden`}
        >
          <MissionMap
            data={activeData}
            events={events}
            persona={persona}
            layers={layers}
            timelineIndex={timelineIndex}
            focusZoneId={focusZoneId}
            selectedDisasterType={selectedDisasterType}
            onInspectZone={(zid) => setFocusZoneId(zid)}
            onSelectPreset={(pid) => void handleSelectPreset(pid)}
          />
        </div>

        <div
          className={`${
            mobilePane === 'intel' ? 'flex w-full' : 'hidden'
          } lg:flex lg:w-[435px] xl:w-[465px] shrink-0 overflow-hidden`}
        >
          <RightPanel
            data={activeData}
            persona={persona}
            onFocusZone={(zid) => {
              setFocusZoneId(zid);
              setMobilePane('map');
            }}
            onReviewClaim={handleReviewClaim}
          />
        </div>

        {toast && (
          <div
            role="status"
            aria-live="polite"
            className="glass-float pointer-events-none absolute top-3 left-1/2 z-50 -translate-x-1/2 rounded-lg px-4 py-1.5 text-xs font-medium text-[#F4F7FB] shadow-xl"
          >
            {toast}
          </div>
        )}
      </main>

      <BottomBar
        data={activeData}
        timelineIndex={timelineIndex}
        onChangeTimeline={setTimelineIndex}
      />
    </div>
  );
}

export default App;
