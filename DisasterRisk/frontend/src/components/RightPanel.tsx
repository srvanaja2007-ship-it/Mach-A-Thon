import React, { useState } from 'react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Bot,
  Building2,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock,
  FileCheck2,
  IndianRupee,
  Landmark,
  MapPin,
  Send,
  ShieldAlert,
  ShieldCheck,
  Users,
  Webhook,
} from 'lucide-react';
import { askClaimPulse, getWebhooks } from '../api/client';
import type {
  AskResponse,
  EventFullResponse,
  InsuranceClaimDecision,
  SeverityLevel,
  UserPersona,
} from '../types/claimpulse';
import {
  SEVERITY_BADGE_CLASSES,
  SEVERITY_COLORS,
  SEVERITY_LEFT_BORDERS,
  formatInr,
  formatNumber,
} from '../utils/format';
import { CountUpValue } from './CountUpValue';

interface RightPanelProps {
  data: EventFullResponse;
  persona: UserPersona;
  onFocusZone: (zoneId: string) => void;
  onReviewClaim: (
    decisionId: string,
    action: 'Approve' | 'Escalate' | 'Request Field Verification',
    notes: string
  ) => Promise<void>;
}

type MainTab = 'impact' | 'decisions' | 'analyst';
type DecisionSubTab = 'insurance' | 'government' | 'lenders' | 'audit';

interface ChatTurn {
  id: string;
  question: string;
  response: AskResponse;
}

const STARTER_PROMPTS = [
  'Which areas have the highest expected loss?',
  'Which claims should be fast-tracked?',
  'What is the estimated agricultural loss?',
  'Which critical infrastructure is at risk?',
  'Why was this claim flagged for review?',
  'How would a 20% increase in flood depth change the payout?',
];

const PARAM_PALETTE = ['#2563EB', '#22D3EE', '#22C55E', '#F59E0B'];

export const RightPanel: React.FC<RightPanelProps> = ({
  data,
  persona,
  onFocusZone,
  onReviewClaim,
}) => {
  const [activeTab, setActiveTab] = useState<MainTab>('impact');
  const [decisionTab, setDecisionTab] = useState<DecisionSubTab>(
    persona === 'government' || persona === 'humanitarian'
      ? 'government'
      : persona === 'lender'
      ? 'lenders'
      : 'insurance'
  );
  const [showAssumptions, setShowAssumptions] = useState(false);
  const [reviewingId, setReviewingId] = useState<string | null>(null);
  const [reviewNotes, setReviewNotes] = useState('');
  const [webhookLog, setWebhookLog] = useState<{
    subscriptions: Array<{ webhook_id: string; partner_name: string; url: string; events: string[]; active: boolean }>;
    recent_deliveries: Array<{ delivery_id: string; webhook_id: string; event_type: string; timestamp: string; status: string }>;
  } | null>(null);

  const [question, setQuestion] = useState('');
  const [asking, setAsking] = useState(false);
  const [chatHistory, setChatHistory] = useState<ChatTurn[]>([]);

  React.useEffect(() => {
    if (persona === 'government' || persona === 'humanitarian') {
      setDecisionTab('government');
    } else if (persona === 'lender') {
      setDecisionTab('lenders');
    } else {
      setDecisionTab('insurance');
    }
  }, [persona]);

  React.useEffect(() => {
    setChatHistory([]);
  }, [data.event.id]);

  const loadWebhooks = async () => {
    try {
      const res = await getWebhooks();
      setWebhookLog(res);
    } catch {
      // ignore
    }
  };

  const handleAsk = async (qText: string) => {
    const trimmed = qText.trim();
    if (!trimmed || asking) return;
    setAsking(true);
    setQuestion('');
    try {
      const resp = await askClaimPulse(data.event.id, trimmed);
      setChatHistory((prev) => [
        ...prev,
        { id: `${Date.now()}`, question: trimmed, response: resp },
      ]);
    } finally {
      setAsking(false);
    }
  };

  const handleReviewAction = async (
    claim: InsuranceClaimDecision,
    action: 'Approve' | 'Escalate' | 'Request Field Verification'
  ) => {
    setReviewingId(claim.decision_id);
    try {
      await onReviewClaim(claim.decision_id, action, reviewNotes);
      setReviewNotes('');
      await loadWebhooks();
    } finally {
      setReviewingId(null);
    }
  };

  const dmg = data.damage.metrics;
  const exp = data.exposure.summary;
  const loss = data.loss;
  const dec = data.decisions;
  const stakeholderModel = loss.stakeholder_models?.[persona];

  // Chart 1: Stakeholder Parameter Breakdown (dynamically follows selected option!)
  const paramChartData = (
    stakeholderModel?.parameters ||
    loss.by_sector.map((s) => ({
      name: s.label.split(' ')[0],
      amount_inr: s.expected_loss_inr,
      share_pct: s.share_pct,
      driver_note: `${s.asset_count} assets`,
    }))
  ).map((p, idx) => ({
    name: p.name.length > 16 ? `${p.name.slice(0, 15)}…` : p.name,
    fullName: p.name,
    amount_cr: Number((p.amount_inr / 1e7).toFixed(2)),
    share_pct: p.share_pct,
    driver_note: p.driver_note,
    fill: PARAM_PALETTE[idx % PARAM_PALETTE.length],
  }));

  // Chart 2: Top Affected Zones for selected stakeholder option
  const topZonesChartData = loss.top_10_zones.slice(0, 6).map((z) => {
    const skAmt = stakeholderModel?.zone_amounts_inr?.[z.zone_id] ?? z.expected_loss_inr;
    return {
      zone_id: z.zone_id,
      name: z.zone_name.split('–')[0].trim(),
      severity: z.severity,
      amount_cr: Number((skAmt / 1e7).toFixed(2)),
    };
  });

  // Chart 3: Damage Severity Distribution
  const severityChartData = (['extreme', 'high', 'medium', 'low', 'safe'] as SeverityLevel[]).map(
    (sev) => ({
      name: sev.toUpperCase(),
      sev,
      area_km2: dmg.severity_distribution_km2[sev] || 0,
      loss_cr: Number(((loss.by_severity[sev] || 0) / 1e7).toFixed(2)),
    })
  );

  // Chart 4: Time-Based Loss Progression
  const timeChartData = loss.time_series.map((t) => ({
    date: t.date.slice(5),
    loss_cr: Number((t.cumulative_loss_inr / 1e7).toFixed(2)),
  }));

  return (
    <aside
      className="flex h-full w-full flex-col overflow-hidden border-l border-[#1E2A40] bg-[#070B14] text-[#F4F7FB]"
      aria-label="Economic impact and decision console"
    >
      {/* Quiet KPI Strip with Prominent Count-Up Numbers */}
      <div className="grid grid-cols-3 gap-2 border-b border-[#1E2A40] bg-[#0E1524] p-3">
        <div className="rounded-lg border border-[#1E2A40] bg-[#141C2E]/70 p-2.5">
          <div className="flex items-center justify-between text-[10px] font-medium text-[#9FB0C7]">
            <span>Affected Area</span>
            <MapPin className="h-3 w-3 text-[#22D3EE]" aria-hidden="true" />
          </div>
          <div className="mt-1 text-sm font-bold text-[#F4F7FB]">
            <CountUpValue
              value={dmg.affected_area_km2}
              formatter={(v) => `${v.toFixed(1)}`}
            />{' '}
            <span className="text-[10px] font-normal text-[#9FB0C7]">km²</span>
          </div>
        </div>

        <div className="rounded-lg border border-[#1E2A40] bg-[#141C2E]/70 p-2.5">
          <div className="flex items-center justify-between text-[10px] font-medium text-[#9FB0C7]">
            <span>Buildings</span>
            <Building2 className="h-3 w-3 text-[#2563EB]" aria-hidden="true" />
          </div>
          <div className="mt-1 text-sm font-bold text-[#F4F7FB]">
            <CountUpValue value={exp.buildings_affected} formatter={(v) => formatNumber(v)} />
          </div>
        </div>

        <div className="rounded-lg border border-[#1E2A40] bg-[#141C2E]/70 p-2.5">
          <div className="flex items-center justify-between text-[10px] font-medium text-[#9FB0C7]">
            <span>Exposed Pop.</span>
            <Users className="h-3 w-3 text-[#8B5CF6]" aria-hidden="true" />
          </div>
          <div className="mt-1 text-sm font-bold text-[#F4F7FB]">
            <CountUpValue value={exp.population_exposed} formatter={(v) => formatNumber(v)} />
          </div>
        </div>

        <div className="rounded-lg border border-[#1E2A40] border-l-2 border-l-[#22D3EE] bg-[#141C2E]/70 p-2.5">
          <div className="flex items-center justify-between text-[10px] font-medium text-[#9FB0C7]">
            <span className="truncate pr-1">
              {stakeholderModel ? stakeholderModel.primary_metric_title : 'Est. Economic Loss'}
            </span>
            <IndianRupee className="h-3 w-3 text-[#22D3EE] shrink-0" aria-hidden="true" />
          </div>
          <div className="mt-1 text-sm font-bold text-[#22D3EE]">
            <CountUpValue
              value={stakeholderModel?.primary_amount_inr ?? loss.total_expected_loss_inr}
              formatter={(v) => formatInr(v)}
            />
          </div>
        </div>

        <div className="rounded-lg border border-[#1E2A40] border-l-2 border-l-[#22C55E] bg-[#141C2E]/70 p-2.5">
          <div className="flex items-center justify-between text-[10px] font-medium text-[#9FB0C7]">
            <span>Fast-Tracked</span>
            <CheckCircle2 className="h-3 w-3 text-[#22C55E]" aria-hidden="true" />
          </div>
          <div className="mt-1 text-sm font-bold text-[#F4F7FB]">
            <CountUpValue
              value={dec.claims_fast_tracked_count}
              formatter={(v) => `${Math.round(v)}`}
            />{' '}
            <span className="font-mono-tabular text-[10px] font-normal text-[#9FB0C7]">
              / {dec.claims.length}
            </span>
          </div>
        </div>

        <div className="rounded-lg border border-[#1E2A40] border-l-2 border-l-[#F59E0B] bg-[#141C2E]/70 p-2.5">
          <div className="flex items-center justify-between text-[10px] font-medium text-[#9FB0C7]">
            <span>Priority Zones</span>
            <ShieldAlert className="h-3 w-3 text-[#F59E0B]" aria-hidden="true" />
          </div>
          <div className="mt-1 text-sm font-bold text-[#F4F7FB]">
            <CountUpValue
              value={dec.relief_priority_zones_count}
              formatter={(v) => `${Math.round(v)}`}
            />{' '}
            <span className="text-[10px] font-normal text-[#9FB0C7]">zones</span>
          </div>
        </div>
      </div>

      {/* Primary Tabs */}
      <div
        className="grid grid-cols-3 border-b border-[#1E2A40] bg-[#0E1524] text-xs font-medium"
        role="tablist"
        aria-label="Decision console sections"
      >
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'impact'}
          onClick={() => setActiveTab('impact')}
          className={`flex items-center justify-center gap-1.5 py-2.5 border-b-2 transition-colors duration-150 ${
            activeTab === 'impact'
              ? 'border-[#2563EB] text-[#F4F7FB] bg-[#141C2E]/60'
              : 'border-transparent text-[#9FB0C7] hover:text-[#F4F7FB]'
          }`}
        >
          <BarChart3 className="h-3.5 w-3.5 text-[#22D3EE]" /> Impact &amp; Models
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'decisions'}
          onClick={() => {
            setActiveTab('decisions');
            void loadWebhooks();
          }}
          className={`flex items-center justify-center gap-1.5 py-2.5 border-b-2 transition-colors duration-150 ${
            activeTab === 'decisions'
              ? 'border-[#2563EB] text-[#F4F7FB] bg-[#141C2E]/60'
              : 'border-transparent text-[#9FB0C7] hover:text-[#F4F7FB]'
          }`}
        >
          <FileCheck2 className="h-3.5 w-3.5 text-[#22C55E]" /> Decisions
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'analyst'}
          onClick={() => setActiveTab('analyst')}
          className={`flex items-center justify-center gap-1.5 py-2.5 border-b-2 transition-colors duration-150 ${
            activeTab === 'analyst'
              ? 'border-[#8B5CF6] text-[#F4F7FB] bg-[#141C2E]/60'
              : 'border-transparent text-[#9FB0C7] hover:text-[#F4F7FB]'
          }`}
        >
          <Bot className="h-3.5 w-3.5 text-[#8B5CF6]" /> Ask Analyst
        </button>
      </div>

      {/* Scrollable Body */}
      <div className="flex-1 overflow-y-auto p-3.5 space-y-3.5">
        {/* TAB 1: IMPACT & STAKEHOLDER PREDICTION MODELS */}
        {activeTab === 'impact' && (
          <>
            {/* Selected Option Prediction Card */}
            {stakeholderModel && (
              <div className="rounded-xl border border-[#1E2A40] border-l-4 border-l-[#2563EB] bg-[#0E1524] p-3.5">
                <div className="flex items-center justify-between gap-2 text-[10px] font-semibold uppercase tracking-wider text-[#9FB0C7]">
                  <span>{stakeholderModel.label}</span>
                  <span className="font-mono-tabular text-[#22D3EE]">
                    {stakeholderModel.secondary_metric_label}:{' '}
                    <b className="text-[#F4F7FB]">{stakeholderModel.secondary_metric_value}</b>
                  </span>
                </div>

                <div className="mt-2 flex items-baseline justify-between gap-2">
                  <div className="font-mono-tabular text-xl font-bold text-[#F4F7FB]">
                    <CountUpValue
                      value={stakeholderModel.primary_amount_inr}
                      formatter={(v) => formatInr(v)}
                    />
                  </div>
                  <div className="font-mono-tabular text-[11px] text-[#9FB0C7]">
                    Range: {formatInr(stakeholderModel.uncertainty_low_inr)} –{' '}
                    {formatInr(stakeholderModel.uncertainty_high_inr)}
                  </div>
                </div>

                <p className="mt-1.5 text-xs leading-relaxed text-[#F4F7FB]">
                  {stakeholderModel.headline}
                </p>

                <div className="mt-2 rounded border border-[#1E2A40] bg-[#070B14] px-2.5 py-1.5 font-mono-tabular text-[10px] text-[#9FB0C7]">
                  {stakeholderModel.formula}
                </div>

                <div className="mt-2.5 flex flex-wrap gap-2 text-[10px] text-[#9FB0C7]">
                  <span className="rounded bg-[#141C2E] px-2 py-0.5 border border-[#1E2A40]">
                    Total Macro Loss:{' '}
                    <b className="font-mono-tabular text-[#F4F7FB]">
                      {formatInr(loss.total_expected_loss_inr)}
                    </b>
                  </span>
                  <span className="rounded bg-[#141C2E] px-2 py-0.5 border border-[#1E2A40]">
                    Roads: <b className="font-mono-tabular text-[#F4F7FB]">{exp.roads_affected_km} km</b>
                  </span>
                  <span className="rounded bg-[#141C2E] px-2 py-0.5 border border-[#1E2A40]">
                    Farmland:{' '}
                    <b className="font-mono-tabular text-[#F4F7FB]">
                      {formatNumber(exp.farmland_affected_ha)} ha
                    </b>
                  </span>
                </div>
              </div>
            )}

            {/* CHART 1: Stakeholder Parameter Breakdown */}
            <div
              className="rounded-xl border border-[#1E2A40] bg-[#0E1524] p-3"
              role="img"
              aria-label={`Bar chart showing ${stakeholderModel?.label || 'sector'} breakdown in INR Crore`}
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-semibold text-[#F4F7FB]">
                  {stakeholderModel
                    ? `${stakeholderModel.label} Parameters (₹ Cr)`
                    : 'Loss by Sector (₹ Cr)'}
                </h3>
                <span className="font-mono-tabular text-[10px] text-[#9FB0C7]">
                  Dynamic model
                </span>
              </div>
              <div className="h-36 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={paramChartData} layout="vertical" margin={{ left: 6, right: 16 }}>
                    <XAxis type="number" tick={{ fill: '#9FB0C7', fontSize: 10 }} unit=" Cr" />
                    <YAxis
                      type="category"
                      dataKey="name"
                      width={108}
                      tick={{ fill: '#F4F7FB', fontSize: 10 }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0E1524',
                        border: '1px solid #1E2A40',
                        borderRadius: '8px',
                        fontSize: '11px',
                        color: '#F4F7FB',
                      }}
                      formatter={(val: unknown) => [`₹${val} Cr`, 'Predicted Allocation']}
                    />
                    <Bar dataKey="amount_cr" radius={[0, 4, 4, 0]}>
                      {paramChartData.map((entry) => (
                        <Cell key={entry.fullName} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-2 grid grid-cols-1 gap-1.5 text-[11px]">
                {(stakeholderModel?.parameters || []).map((p, idx) => (
                  <div
                    key={p.name}
                    className="flex items-center justify-between rounded bg-[#141C2E] px-2.5 py-1.5 border border-[#1E2A40]"
                  >
                    <div className="min-w-0 pr-2">
                      <div className="flex items-center gap-1.5 text-[#F4F7FB] font-medium truncate">
                        <span
                          className="h-2 w-2 rounded-sm shrink-0"
                          style={{ backgroundColor: PARAM_PALETTE[idx % PARAM_PALETTE.length] }}
                        />
                        {p.name}
                      </div>
                      <div className="text-[10px] text-[#9FB0C7] truncate">{p.driver_note}</div>
                    </div>
                    <div className="text-right shrink-0">
                      <div className="font-mono-tabular font-semibold text-[#F4F7FB]">
                        {formatInr(p.amount_inr)}
                      </div>
                      <div className="font-mono-tabular text-[10px] text-[#22D3EE]">
                        {p.share_pct}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* CHART 2: Top Affected Zones (Stakeholder Impact) */}
            <div
              className="rounded-xl border border-[#1E2A40] bg-[#0E1524] p-3"
              role="img"
              aria-label="Top affected zones horizontal bar chart in INR Crore"
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-semibold text-[#F4F7FB]">
                  Top Affected Zones — {stakeholderModel?.primary_metric_title || 'Loss'} (₹ Cr)
                </h3>
                <span className="text-[10px] text-[#22D3EE]">Top 6</span>
              </div>
              <div className="h-36 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={topZonesChartData} layout="vertical" margin={{ left: 4, right: 16 }}>
                    <XAxis type="number" tick={{ fill: '#9FB0C7', fontSize: 10 }} unit=" Cr" />
                    <YAxis
                      type="category"
                      dataKey="name"
                      width={104}
                      tick={{ fill: '#F4F7FB', fontSize: 10 }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0E1524',
                        border: '1px solid #1E2A40',
                        borderRadius: '8px',
                        fontSize: '11px',
                        color: '#F4F7FB',
                      }}
                      formatter={(val: unknown) => [`₹${val} Cr`, stakeholderModel?.primary_metric_title || 'Loss']}
                    />
                    <Bar dataKey="amount_cr" radius={[0, 4, 4, 0]}>
                      {topZonesChartData.map((entry) => (
                        <Cell key={entry.zone_id} fill={SEVERITY_COLORS[entry.severity]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* CHART 3 & CHART 4 */}
            <div className="grid grid-cols-1 gap-3">
              <div
                className="rounded-xl border border-[#1E2A40] bg-[#0E1524] p-3"
                role="img"
                aria-label="Damage severity distribution chart in INR Crore"
              >
                <h3 className="text-xs font-semibold text-[#F4F7FB] mb-2">
                  Damage Severity Distribution (₹ Cr)
                </h3>
                <div className="h-28 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={severityChartData}>
                      <XAxis dataKey="name" tick={{ fill: '#9FB0C7', fontSize: 10 }} />
                      <YAxis tick={{ fill: '#9FB0C7', fontSize: 10 }} unit=" Cr" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#0E1524',
                          border: '1px solid #1E2A40',
                          borderRadius: '8px',
                          fontSize: '11px',
                          color: '#F4F7FB',
                        }}
                        formatter={(val: unknown) => [`₹${val} Cr`, 'Macro Loss']}
                      />
                      <Bar dataKey="loss_cr" radius={[4, 4, 0, 0]}>
                        {severityChartData.map((entry) => (
                          <Cell key={entry.sev} fill={SEVERITY_COLORS[entry.sev]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div
                className="rounded-xl border border-[#1E2A40] bg-[#0E1524] p-3"
                role="img"
                aria-label="Time-based cumulative economic loss progression chart"
              >
                <h3 className="text-xs font-semibold text-[#F4F7FB] mb-2">
                  Time-Based Loss Progression (₹ Cr)
                </h3>
                <div className="h-28 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={timeChartData}>
                      <defs>
                        <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#22D3EE" stopOpacity={0.35} />
                          <stop offset="95%" stopColor="#22D3EE" stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="date" tick={{ fill: '#9FB0C7', fontSize: 10 }} />
                      <YAxis tick={{ fill: '#9FB0C7', fontSize: 10 }} unit=" Cr" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#0E1524',
                          border: '1px solid #1E2A40',
                          borderRadius: '8px',
                          fontSize: '11px',
                          color: '#F4F7FB',
                        }}
                        formatter={(val: unknown) => [`₹${val} Cr`, 'Cumulative Loss']}
                      />
                      <Area
                        type="monotone"
                        dataKey="loss_cr"
                        stroke="#22D3EE"
                        fill="url(#lossGrad)"
                        strokeWidth={2}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Dynamic Disaster Impact & Intensity Simulation Meter */}
              <div className="rounded-xl border border-[#1E2A40] bg-[#0E1524] p-3">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-xs font-semibold text-[#F4F7FB] flex items-center gap-1.5">
                    <Activity className="h-3.5 w-3.5 text-[#22D3EE]" /> Impact &amp; Intensity Simulation
                  </h3>
                  <span className="font-mono-tabular text-[10px] text-[#22D3EE] font-semibold">
                    {dmg.max_intensity_label}
                  </span>
                </div>
                
                {/* Visual Hazard Gauge */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-[11px] text-[#9FB0C7]">
                    <span>Mean Hazard Footprint</span>
                    <span className="font-mono-tabular font-bold text-[#F4F7FB]">
                      {dmg.mean_intensity_value} {dmg.intensity_unit}
                    </span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-[#141C2E] overflow-hidden flex">
                    <div
                      className="bg-[#22C55E]"
                      style={{
                        width: `${Math.round(((dmg.severity_distribution_km2.safe || 0) / (dmg.affected_area_km2 || 1)) * 100)}%`,
                      }}
                      title="Safe"
                    />
                    <div
                      className="bg-[#FACC15]"
                      style={{
                        width: `${Math.round(((dmg.severity_distribution_km2.low || 0) / (dmg.affected_area_km2 || 1)) * 100)}%`,
                      }}
                      title="Low Severity"
                    />
                    <div
                      className="bg-[#F97316]"
                      style={{
                        width: `${Math.round(((dmg.severity_distribution_km2.medium || 0) / (dmg.affected_area_km2 || 1)) * 100)}%`,
                      }}
                      title="Medium Severity"
                    />
                    <div
                      className="bg-[#EF4444]"
                      style={{
                        width: `${Math.round(((dmg.severity_distribution_km2.high || 0) / (dmg.affected_area_km2 || 1)) * 100)}%`,
                      }}
                      title="High Severity"
                    />
                    <div
                      className="bg-[#B91C1C]"
                      style={{
                        width: `${Math.round(((dmg.severity_distribution_km2.extreme || 0) / (dmg.affected_area_km2 || 1)) * 100)}%`,
                      }}
                      title="Extreme Severity"
                    />
                  </div>
                  <div className="flex justify-between text-[10px] text-[#9FB0C7] font-mono-tabular">
                    <span className="text-[#22C55E]">Safe: {dmg.severity_distribution_km2.safe || 0} km²</span>
                    <span className="text-[#F97316]">Med: {dmg.severity_distribution_km2.medium || 0} km²</span>
                    <span className="text-[#B91C1C]">Extr: {dmg.severity_distribution_km2.extreme || 0} km²</span>
                  </div>
                </div>

                {/* Sector Sensitivity Badges */}
                <div className="mt-3 grid grid-cols-2 gap-2 text-[10px]">
                  <div className="rounded-lg border border-[#1E2A40] bg-[#141C2E] p-2">
                    <div className="text-[#9FB0C7]">Vulnerability Index</div>
                    <div className="mt-0.5 font-mono-tabular text-sm font-bold text-[#F4F7FB]">
                      {(loss.top_10_zones.reduce((acc, z) => acc + (z.vulnerability_score || 0.8), 0) / Math.max(1, loss.top_10_zones.length)).toFixed(2)} / 1.00
                    </div>
                  </div>
                  <div className="rounded-lg border border-[#1E2A40] bg-[#141C2E] p-2">
                    <div className="text-[#9FB0C7]">Infrastructure Criticality</div>
                    <div className="mt-0.5 font-mono-tabular text-sm font-bold text-[#22D3EE]">
                      {(loss.top_10_zones.reduce((acc, z) => acc + (z.infrastructure_criticality || 0.85), 0) / Math.max(1, loss.top_10_zones.length)).toFixed(2)} / 1.00
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Priority Zones with Color-Coded Left Borders & Text Labels */}
            <div className="rounded-xl border border-[#1E2A40] bg-[#0E1524] p-3">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-semibold text-[#F4F7FB]">
                  Priority Zones (Click to Inspect on Map)
                </h3>
                <span className="font-mono-tabular text-[10px] text-[#9FB0C7]">Top 10</span>
              </div>
              <div className="space-y-1.5">
                {loss.top_10_zones.map((z) => {
                  const skAmt =
                    stakeholderModel?.zone_amounts_inr?.[z.zone_id] ?? z.expected_loss_inr;
                  return (
                    <button
                      key={z.zone_id}
                      type="button"
                      onClick={() => onFocusZone(z.zone_id)}
                      className={`w-full text-left rounded-lg border border-[#1E2A40] border-l-4 ${
                        SEVERITY_LEFT_BORDERS[z.severity]
                      } bg-[#141C2E]/70 px-2.5 py-2 hover:bg-[#141C2E] transition-colors duration-150`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2 min-w-0">
                          <span className="font-mono-tabular text-[10px] font-bold text-[#9FB0C7]">
                            #{z.rank}
                          </span>
                          <span className="text-xs font-semibold text-[#F4F7FB] truncate">
                            {z.zone_name}
                          </span>
                        </div>
                        <span
                          className={`rounded border px-1.5 py-0.5 text-[9px] font-semibold uppercase shrink-0 ${
                            SEVERITY_BADGE_CLASSES[z.severity]
                          }`}
                        >
                          {z.severity}
                        </span>
                      </div>
                      <div className="mt-1 flex items-center justify-between text-[10px] text-[#9FB0C7]">
                        <span>
                          {formatNumber(z.population_exposed)} people • {formatNumber(z.buildings_affected)} bldgs
                        </span>
                        <span className="font-mono-tabular font-semibold text-[#22D3EE]">
                          {formatInr(skAmt)}
                        </span>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Assumptions & Limitations */}
            <div className="rounded-xl border border-[#1E2A40] bg-[#0E1524] p-3">
              <button
                type="button"
                onClick={() => setShowAssumptions((v) => !v)}
                aria-expanded={showAssumptions}
                className="flex w-full items-center justify-between text-xs font-semibold text-[#F4F7FB]"
              >
                <span>Formula, Assumptions &amp; Limitations</span>
                {showAssumptions ? (
                  <ChevronDown className="h-4 w-4 text-[#9FB0C7]" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-[#9FB0C7]" />
                )}
              </button>

              <div className="mt-2 rounded bg-[#070B14] p-2 font-mono-tabular text-[10px] text-[#22D3EE] border border-[#1E2A40]">
                {loss.formula_text}
              </div>

              {showAssumptions && (
                <div className="mt-2.5 space-y-2 text-[11px]">
                  {loss.assumptions.map((a) => (
                    <div
                      key={a.parameter}
                      className="rounded-lg border border-[#1E2A40] bg-[#141C2E]/70 p-2"
                    >
                      <div className="flex items-center justify-between font-semibold text-[#F4F7FB]">
                        <span>{a.parameter}</span>
                        <span className="font-mono-tabular text-[10px] text-[#22D3EE]">{a.source}</span>
                      </div>
                      <div className="mt-0.5 font-mono-tabular text-[10px] text-[#F59E0B]">
                        {a.value}
                      </div>
                      <p className="mt-1 text-[#9FB0C7] leading-snug">{a.rationale}</p>
                    </div>
                  ))}
                  <div className="rounded-lg border border-[#F59E0B]/40 bg-[#F59E0B]/10 p-2">
                    <div className="font-semibold text-[#F59E0B] mb-1 flex items-center gap-1">
                      <AlertTriangle className="h-3.5 w-3.5" /> Model Limitations
                    </div>
                    <ul className="list-disc pl-4 space-y-1 text-[#F4F7FB]">
                      {loss.limitations.map((lim) => (
                        <li key={lim}>{lim}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          </>
        )}

        {/* TAB 2: DECISIONS & AUDIT */}
        {activeTab === 'decisions' && (
          <>
            <div className="rounded-xl border border-[#1E2A40] bg-[#0E1524] p-3">
              <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-[#9FB0C7]">
                <span>Decision Allocation Summary</span>
                <span className="font-mono-tabular text-[#22C55E]">
                  {Math.round(dec.overall_confidence * 100)}% Confidence
                </span>
              </div>
              <div className="mt-2 grid grid-cols-3 gap-2 text-center">
                <div className="rounded-lg bg-[#141C2E] p-2 border border-[#1E2A40]">
                  <div className="text-[10px] text-[#9FB0C7]">Insurer Payout</div>
                  <div className="font-mono-tabular text-xs font-bold text-[#22C55E]">
                    {formatInr(dec.total_provisional_payout_inr)}
                  </div>
                </div>
                <div className="rounded-lg bg-[#141C2E] p-2 border border-[#1E2A40]">
                  <div className="text-[10px] text-[#9FB0C7]">Gov. Relief</div>
                  <div className="font-mono-tabular text-xs font-bold text-[#22D3EE]">
                    {formatInr(
                      loss.stakeholder_models?.government?.primary_amount_inr ??
                        dec.total_recommended_relief_inr
                    )}
                  </div>
                </div>
                <div className="rounded-lg bg-[#141C2E] p-2 border border-[#1E2A40]">
                  <div className="text-[10px] text-[#9FB0C7]">Humanitarian Aid</div>
                  <div className="font-mono-tabular text-xs font-bold text-[#F59E0B]">
                    {formatInr(
                      loss.stakeholder_models?.humanitarian?.primary_amount_inr ??
                        dec.total_lender_exposure_at_risk_inr
                    )}
                  </div>
                </div>
              </div>
              <div className="mt-2 flex items-center justify-between text-[10px] text-[#9FB0C7]">
                <span className="font-mono-tabular">
                  Range: {formatInr(dec.uncertainty_range_inr[0])} –{' '}
                  {formatInr(dec.uncertainty_range_inr[1])}
                </span>
                <span className="text-[#22C55E] font-medium">Zero Auto-Denials</span>
              </div>
            </div>

            <div className="grid grid-cols-4 gap-1 rounded-lg bg-[#0E1524] border border-[#1E2A40] p-1 text-[10px] font-semibold">
              {(
                [
                  { id: 'insurance', label: 'Claims' },
                  { id: 'government', label: 'Relief & Aid' },
                  { id: 'lenders', label: 'Lenders' },
                  { id: 'audit', label: 'Audit' },
                ] as const
              ).map((t) => (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => setDecisionTab(t.id)}
                  className={`rounded-md py-1.5 transition-colors duration-150 ${
                    decisionTab === t.id
                      ? 'bg-[#2563EB] text-[#F4F7FB]'
                      : 'text-[#9FB0C7] hover:text-[#F4F7FB]'
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </div>

            {decisionTab === 'insurance' && (
              <div className="space-y-2.5">
                <div className="rounded-lg border border-[#F59E0B]/40 bg-[#F59E0B]/10 px-2.5 py-2 text-[11px] text-[#F4F7FB] flex items-center gap-2">
                  <ShieldCheck className="h-4 w-4 shrink-0 text-[#F59E0B]" />
                  <span>
                    Claims &gt; ₹8.82 Cr or confidence &lt; 86% require Human Review. Low confidence never auto-denies.
                  </span>
                </div>

                {dec.claims.map((c) => {
                  const recBadge =
                    c.recommendation === 'Auto-fast-track'
                      ? 'bg-[#22C55E]/15 text-[#22C55E] border-[#22C55E]/40'
                      : c.recommendation === 'Manual review'
                      ? 'bg-[#F59E0B]/15 text-[#F59E0B] border-[#F59E0B]/40'
                      : c.recommendation === 'Investigate'
                      ? 'bg-[#EF4444]/15 text-[#EF4444] border-[#EF4444]/40'
                      : 'bg-[#141C2E] text-[#9FB0C7] border-[#1E2A40]';

                  return (
                    <div
                      key={c.decision_id}
                      className={`rounded-xl border border-[#1E2A40] border-l-4 ${
                        SEVERITY_LEFT_BORDERS[c.severity]
                      } bg-[#0E1524] p-3 space-y-2`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <div className="flex items-center gap-1.5">
                            <span className="font-mono-tabular text-[10px] font-semibold text-[#22D3EE]">
                              {c.policy_id}
                            </span>
                            <span
                              className={`rounded border px-1.5 py-0.5 text-[9px] font-semibold uppercase ${
                                SEVERITY_BADGE_CLASSES[c.severity]
                              }`}
                            >
                              {c.severity}
                            </span>
                          </div>
                          <div className="mt-0.5 text-xs font-semibold text-[#F4F7FB]">
                            {c.asset_name}
                          </div>
                          <div className="text-[10px] text-[#9FB0C7]">
                            {c.policyholder} • {c.zone_name}
                          </div>
                        </div>
                        <span
                          className={`rounded-md border px-2 py-0.5 text-[10px] font-semibold whitespace-nowrap ${recBadge}`}
                        >
                          {c.recommendation}
                        </span>
                      </div>

                      <div className="grid grid-cols-3 gap-1.5 rounded-lg bg-[#141C2E] p-2 text-[10px] border border-[#1E2A40]">
                        <div>
                          <span className="text-[#9FB0C7] block">Est. Payout</span>
                          <span className="font-mono-tabular font-bold text-[#F4F7FB] text-xs">
                            {formatInr(c.estimated_claim_inr)}
                          </span>
                        </div>
                        <div>
                          <span className="text-[#9FB0C7] block">Damage Prob.</span>
                          <span className="font-mono-tabular font-semibold text-[#22D3EE] text-xs">
                            {Math.round(c.damage_probability * 100)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-[#9FB0C7] block">Confidence</span>
                          <span className="font-mono-tabular font-semibold text-[#22C55E] text-xs">
                            {Math.round(c.confidence * 100)}%
                          </span>
                        </div>
                      </div>

                      <div className="text-[10px] text-[#9FB0C7]">
                        <b className="text-[#F4F7FB]">Calculation:</b>{' '}
                        <span className="font-mono-tabular text-[#22D3EE]">
                          {c.calculation_breakdown}
                        </span>
                      </div>

                      <div className="text-[10px] text-[#9FB0C7]">
                        <b className="text-[#F4F7FB]">Evidence:</b> {c.evidence_links.join(' • ')}
                      </div>

                      <div className="flex flex-wrap items-center justify-between gap-1 text-[10px] text-[#9FB0C7]">
                        <span className="font-mono-tabular">
                          Range: {formatInr(c.uncertainty_low_inr)} – {formatInr(c.uncertainty_high_inr)}
                        </span>
                        <span>
                          Status: <b className="text-[#F4F7FB]">{c.review_status}</b>
                        </span>
                      </div>

                      {c.human_review_required && c.review_status === 'Pending' && (
                        <div className="border-t border-[#1E2A40] pt-2 space-y-1.5">
                          <div className="text-[10px] font-semibold text-[#F59E0B]">
                            Human Review Required
                          </div>
                          <input
                            placeholder="Optional analyst note..."
                            aria-label={`Review note for ${c.policy_id}`}
                            value={reviewNotes}
                            onChange={(e) => setReviewNotes(e.target.value)}
                            className="w-full rounded border border-[#1E2A40] bg-[#070B14] px-2 py-1 text-[10px] text-[#F4F7FB]"
                          />
                          <div className="grid grid-cols-3 gap-1.5">
                            <button
                              type="button"
                              disabled={reviewingId === c.decision_id}
                              onClick={() => handleReviewAction(c, 'Approve')}
                              className="rounded bg-[#22C55E]/90 py-1 text-[10px] font-semibold text-[#070B14] hover:bg-[#22C55E]"
                            >
                              Approve
                            </button>
                            <button
                              type="button"
                              disabled={reviewingId === c.decision_id}
                              onClick={() => handleReviewAction(c, 'Escalate')}
                              className="rounded bg-[#F59E0B]/90 py-1 text-[10px] font-semibold text-[#070B14] hover:bg-[#F59E0B]"
                            >
                              Escalate
                            </button>
                            <button
                              type="button"
                              disabled={reviewingId === c.decision_id}
                              onClick={() =>
                                handleReviewAction(c, 'Request Field Verification')
                              }
                              className="rounded border border-[#1E2A40] bg-[#141C2E] py-1 text-[10px] font-semibold text-[#F4F7FB] hover:border-[#2563EB]"
                            >
                              Field Verify
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}

            {decisionTab === 'government' && (
              <div className="space-y-2.5">
                <div className="rounded-lg border border-[#1E2A40] bg-[#141C2E] p-2.5 text-[11px] text-[#F4F7FB]">
                  Showing{' '}
                  <b className="text-[#22D3EE]">
                    {persona === 'humanitarian' ? 'Humanitarian Aid' : 'Government Relief'}
                  </b>{' '}
                  priority allocation per zone based on loss, population, and vulnerability.
                </div>
                {dec.relief_allocations.map((r) => {
                  const zoneAidOrGov =
                    persona === 'humanitarian'
                      ? loss.stakeholder_models?.humanitarian?.zone_amounts_inr?.[r.zone_id] ??
                        r.recommended_relief_inr
                      : loss.stakeholder_models?.government?.zone_amounts_inr?.[r.zone_id] ??
                        r.recommended_relief_inr;
                  return (
                    <button
                      key={r.zone_id}
                      type="button"
                      onClick={() => onFocusZone(r.zone_id)}
                      className={`w-full text-left rounded-xl border border-[#1E2A40] border-l-4 ${
                        SEVERITY_LEFT_BORDERS[r.severity]
                      } bg-[#0E1524] p-2.5 hover:bg-[#141C2E] transition-colors duration-150`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="flex h-5 w-5 items-center justify-center rounded bg-[#141C2E] border border-[#1E2A40] font-mono-tabular text-[10px] font-bold text-[#22D3EE]">
                            {r.rank}
                          </span>
                          <div>
                            <div className="text-xs font-semibold text-[#F4F7FB]">
                              {r.zone_name}
                            </div>
                            <div className="text-[10px] text-[#9FB0C7]">
                              {r.district_or_ward} • {formatNumber(r.population_exposed)} exposed
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="font-mono-tabular text-xs font-bold text-[#22D3EE]">
                            {formatInr(zoneAidOrGov)}
                          </div>
                          <div className="text-[9px] text-[#9FB0C7]">
                            {persona === 'humanitarian' ? 'Aid Est.' : 'Relief Budget'}
                          </div>
                        </div>
                      </div>
                      <div className="mt-1.5 flex items-center justify-between text-[10px] text-[#9FB0C7]">
                        <span className="font-mono-tabular">
                          Vuln: {r.vulnerability_score.toFixed(2)} • Crit: {r.infrastructure_criticality.toFixed(2)}
                        </span>
                        <span className="text-[#F4F7FB] truncate max-w-[200px]">
                          {r.priority_action}
                        </span>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}

            {decisionTab === 'lenders' && (
              <div className="space-y-2.5">
                <div className="rounded-lg border border-[#F59E0B]/40 bg-[#F59E0B]/10 p-2.5 text-[11px] text-[#F4F7FB] flex items-center gap-2">
                  <Landmark className="h-4 w-4 shrink-0 text-[#F59E0B]" />
                  <span>Collateral LTV stress &amp; bps risk-premium adjustments (INR).</span>
                </div>
                {dec.lender_alerts.map((l) => (
                  <div
                    key={l.loan_id}
                    className="rounded-xl border border-[#1E2A40] bg-[#0E1524] p-3 space-y-1.5"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono-tabular text-[10px] font-semibold text-[#22D3EE]">
                        {l.loan_id} • {l.borrower}
                      </span>
                      <span
                        className={`rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase ${
                          l.risk_flag === 'Critical'
                            ? 'bg-[#EF4444]/20 text-[#EF4444]'
                            : 'bg-[#F59E0B]/20 text-[#F59E0B]'
                        }`}
                      >
                        {l.risk_flag}
                      </span>
                    </div>
                    <div className="text-xs font-semibold text-[#F4F7FB]">{l.collateral_name}</div>
                    <div className="grid grid-cols-3 gap-1.5 text-[10px] bg-[#141C2E] rounded p-2 border border-[#1E2A40]">
                      <div>
                        <span className="text-[#9FB0C7] block">Exposure</span>
                        <span className="font-mono-tabular font-semibold text-[#F4F7FB]">
                          {formatInr(l.portfolio_exposure_inr)}
                        </span>
                      </div>
                      <div>
                        <span className="text-[#9FB0C7] block">LTV Stress</span>
                        <span className="font-mono-tabular font-semibold text-[#EF4444]">
                          +{l.ltv_stress_pct}%
                        </span>
                      </div>
                      <div>
                        <span className="text-[#9FB0C7] block">Rate Adj.</span>
                        <span className="font-mono-tabular font-semibold text-[#F59E0B]">
                          +{l.risk_premium_bps_adj} bps
                        </span>
                      </div>
                    </div>
                    <div className="text-[10px] text-[#9FB0C7]">
                      <b className="text-[#F4F7FB]">Action:</b> {l.recommended_action}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {decisionTab === 'audit' && (
              <div className="space-y-3">
                <div>
                  <h4 className="text-xs font-semibold text-[#F4F7FB] mb-2 flex items-center gap-1.5">
                    <Clock className="h-3.5 w-3.5 text-[#22D3EE]" /> Audit Trail
                  </h4>
                  <div className="space-y-2">
                    {dec.audit_trail.map((a) => (
                      <div
                        key={a.audit_id}
                        className="rounded-lg border border-[#1E2A40] bg-[#0E1524] p-2.5 text-[11px]"
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-semibold text-[#22D3EE]">{a.action}</span>
                          <span className="font-mono-tabular text-[10px] text-[#9FB0C7]">
                            {a.audit_id}
                          </span>
                        </div>
                        <div className="mt-1 text-[#F4F7FB]">{a.rationale}</div>
                        <div className="mt-1 flex justify-between text-[10px] text-[#9FB0C7]">
                          <span>{a.actor}</span>
                          <span className="font-mono-tabular">{a.timestamp}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {webhookLog && (
                  <div>
                    <h4 className="text-xs font-semibold text-[#F4F7FB] mb-2 flex items-center gap-1.5">
                      <Webhook className="h-3.5 w-3.5 text-[#22C55E]" /> Partner Webhooks
                    </h4>
                    <div className="space-y-1.5 text-[10px]">
                      {webhookLog.subscriptions.map((w) => (
                        <div
                          key={w.webhook_id}
                          className="rounded border border-[#1E2A40] bg-[#0E1524] p-2 flex items-center justify-between"
                        >
                          <div>
                            <div className="font-semibold text-[#F4F7FB]">{w.partner_name}</div>
                            <div className="font-mono-tabular text-[#9FB0C7] truncate max-w-[220px]">
                              {w.url}
                            </div>
                          </div>
                          <span className="rounded bg-[#22C55E]/15 px-1.5 py-0.5 text-[#22C55E]">
                            Active
                          </span>
                        </div>
                      ))}
                      {webhookLog.recent_deliveries.map((d) => (
                        <div
                          key={d.delivery_id}
                          className="rounded border border-[#2563EB]/40 bg-[#141C2E] p-2 flex items-center justify-between"
                        >
                          <span className="font-mono-tabular text-[#22D3EE]">
                            {d.event_type} ({d.delivery_id})
                          </span>
                          <span className="text-[#22C55E]">{d.status}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </>
        )}

        {/* TAB 3: ASK ANALYST */}
        {activeTab === 'analyst' && (
          <div className="flex h-full flex-col space-y-3">
            <div className="rounded-xl border border-[#8B5CF6]/40 bg-[#0E1524] p-3 text-[11px] text-[#9FB0C7]">
              <div className="font-semibold text-[#F4F7FB] flex items-center gap-1.5 mb-1">
                <Bot className="h-3.5 w-3.5 text-[#8B5CF6]" /> Ask ClaimPulse Analyst
              </div>
              Answers retrieve strictly from computed layers, timestamps, and INR formulas.
            </div>

            <div className="flex flex-wrap gap-1.5">
              {STARTER_PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  disabled={asking}
                  onClick={() => void handleAsk(prompt)}
                  className="rounded-full border border-[#1E2A40] bg-[#141C2E] px-2.5 py-1 text-left text-[10px] text-[#F4F7FB] hover:border-[#8B5CF6] transition-colors duration-150"
                >
                  {prompt}
                </button>
              ))}
            </div>

            <div className="flex-1 space-y-3 overflow-y-auto pr-1">
              {chatHistory.length === 0 && (
                <div className="rounded-xl border border-dashed border-[#1E2A40] p-5 text-center text-xs text-[#9FB0C7]">
                  Select a prompt above or enter a question to query computed losses, claims, and depth stress tests.
                </div>
              )}
              {chatHistory.map((turn) => (
                <div key={turn.id} className="space-y-2">
                  <div className="ml-auto max-w-[90%] rounded-lg bg-[#2563EB]/25 border border-[#2563EB]/40 px-3 py-2 text-xs text-[#F4F7FB]">
                    {turn.question}
                  </div>
                  <div className="rounded-xl border border-[#1E2A40] bg-[#0E1524] p-3 text-xs space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-semibold uppercase tracking-wider text-[#8B5CF6]">
                        ClaimPulse Analyst
                      </span>
                      <span className="font-mono-tabular text-[10px] text-[#22C55E]">
                        {Math.round(turn.response.confidence * 100)}% confidence
                      </span>
                    </div>
                    <p className="leading-relaxed text-[#F4F7FB]">{turn.response.answer}</p>
                    <div className="rounded bg-[#070B14] p-2 font-mono-tabular text-[10px] text-[#9FB0C7] border border-[#1E2A40]">
                      <b className="text-[#F4F7FB]">Formula:</b> {turn.response.formula_trace}
                    </div>
                    {turn.response.citations.length > 0 && (
                      <div className="space-y-1">
                        {turn.response.citations.map((cit, idx) => (
                          <div
                            key={`${cit.layer}-${idx}`}
                            className="flex items-center justify-between rounded bg-[#141C2E] px-2 py-1 text-[10px] border border-[#1E2A40]"
                          >
                            <span className="text-[#22D3EE] truncate max-w-[190px]">
                              {cit.layer} ({cit.timestamp.slice(0, 10)})
                            </span>
                            <span className="font-mono-tabular text-[#F4F7FB]">{cit.value}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                void handleAsk(question);
              }}
              className="flex items-center gap-2 border-t border-[#1E2A40] pt-2"
            >
              <input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask ClaimPulse (e.g. +20% flood depth)..."
                aria-label="Ask ClaimPulse analyst question"
                className="flex-1 rounded-lg border border-[#1E2A40] bg-[#141C2E] px-3 py-2 text-xs text-[#F4F7FB] placeholder:text-[#9FB0C7]"
              />
              <button
                type="submit"
                disabled={asking || !question.trim()}
                className="rounded-lg bg-[#2563EB] p-2 text-[#F4F7FB] hover:bg-[#1D4ED8] disabled:opacity-50"
                aria-label="Submit analyst question"
              >
                <Send className="h-4 w-4" />
              </button>
            </form>
          </div>
        )}
      </div>
    </aside>
  );
};
