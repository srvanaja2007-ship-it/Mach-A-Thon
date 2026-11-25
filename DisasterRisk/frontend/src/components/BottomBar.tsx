import React, { useState } from 'react';
import {
  Clock,
  Database,
  Download,
  FileJson,
  FileSpreadsheet,
  FileText,
  ShieldCheck,
  X,
} from 'lucide-react';
import { getExportUrl, usingLocalFallback } from '../api/client';
import { triggerClientSideDownload } from '../api/localFallback';
import type { EventFullResponse } from '../types/claimpulse';

interface BottomBarProps {
  data: EventFullResponse;
  timelineIndex: number;
  onChangeTimeline: (idx: number) => void;
}

export const BottomBar: React.FC<BottomBarProps> = ({
  data,
  timelineIndex,
  onChangeTimeline,
}) => {
  const [showProvenance, setShowProvenance] = useState(false);
  const dates = data.damage.metrics.timeline_dates;
  const activeDate = dates[timelineIndex] || dates[dates.length - 1];
  const confidencePct = Math.round(data.decisions.overall_confidence * 100);
  const allDatasets = [
    ...data.damage.metrics.datasets,
    ...data.exposure.summary.datasets,
  ];

  const handleExport = (fmt: 'pdf' | 'csv' | 'geojson') => {
    if (!usingLocalFallback) {
      window.open(getExportUrl(data.event.id, fmt), '_blank');
      return;
    }
    if (fmt === 'pdf') {
      window.print();
    } else {
      triggerClientSideDownload(data, fmt);
    }
  };

  return (
    <footer
      className="relative flex flex-wrap items-center justify-between gap-3 border-t border-[#1E2A40] bg-[#0E1524] px-4 py-2 text-xs text-[#9FB0C7]"
      aria-label="Event timeline, provenance, and export controls"
    >
      {/* Timeline Scrubber */}
      <div className="flex items-center gap-3 min-w-[240px]">
        <div className="flex items-center gap-1.5 text-[#F4F7FB] font-medium whitespace-nowrap">
          <Clock className="h-3.5 w-3.5 text-[#22D3EE]" />
          <span className="text-[11px] uppercase tracking-wider">Event Timeline</span>
        </div>
        <div className="flex items-center gap-1.5" role="group" aria-label="Select event date">
          {dates.map((d, idx) => {
            const active = idx === timelineIndex;
            return (
              <button
                key={d}
                type="button"
                onClick={() => onChangeTimeline(idx)}
                className={`rounded px-2 py-0.5 font-mono-tabular text-[11px] transition-colors duration-150 ${
                  active
                    ? 'bg-[#2563EB] text-[#F4F7FB] font-semibold'
                    : 'bg-[#141C2E] border border-[#1E2A40] text-[#9FB0C7] hover:text-[#F4F7FB]'
                }`}
              >
                {d.slice(5)}
              </button>
            );
          })}
        </div>
        <span className="hidden xl:inline font-mono-tabular text-[11px] text-[#9FB0C7]">
          {activeDate}
        </span>
      </div>

      {/* Freshness, Provenance & Confidence */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => setShowProvenance((v) => !v)}
          className="flex items-center gap-1.5 rounded-md border border-[#1E2A40] bg-[#141C2E] px-2.5 py-1 text-[11px] text-[#F4F7FB] hover:border-[#2563EB] transition-colors duration-150"
        >
          <Database className="h-3.5 w-3.5 text-[#22D3EE]" />
          <span>{allDatasets.length} Datasets</span>
          <span className="text-[#9FB0C7]">•</span>
          <span className="font-mono-tabular text-[#9FB0C7]">
            {data.event.data_freshness.slice(0, 10)}
          </span>
        </button>

        <div className="flex items-center gap-1.5 rounded-md border border-[#22C55E]/40 bg-[#22C55E]/10 px-2.5 py-1 text-[11px]">
          <ShieldCheck className="h-3.5 w-3.5 text-[#22C55E]" />
          <span className="text-[#F4F7FB]">Confidence:</span>
          <span className="font-mono-tabular font-semibold text-[#22C55E]">
            {confidencePct}%
          </span>
        </div>
      </div>

      {/* Export Actions */}
      <div className="flex items-center gap-1.5">
        <span className="hidden sm:inline-flex items-center gap-1 text-[11px] text-[#9FB0C7] mr-1">
          <Download className="h-3 w-3" /> Export:
        </span>
        <button
          type="button"
          onClick={() => handleExport('pdf')}
          className="flex items-center gap-1 rounded-md bg-[#2563EB] px-2.5 py-1 text-[11px] font-medium text-[#F4F7FB] hover:bg-[#1D4ED8] transition-colors duration-150"
        >
          <FileText className="h-3.5 w-3.5" /> PDF Brief
        </button>
        <button
          type="button"
          onClick={() => handleExport('csv')}
          className="flex items-center gap-1 rounded-md border border-[#1E2A40] bg-[#141C2E] px-2.5 py-1 text-[11px] font-medium text-[#F4F7FB] hover:border-[#22C55E] transition-colors duration-150"
        >
          <FileSpreadsheet className="h-3.5 w-3.5 text-[#22C55E]" /> CSV
        </button>
        <button
          type="button"
          onClick={() => handleExport('geojson')}
          className="flex items-center gap-1 rounded-md border border-[#1E2A40] bg-[#141C2E] px-2.5 py-1 text-[11px] font-medium text-[#F4F7FB] hover:border-[#F59E0B] transition-colors duration-150"
        >
          <FileJson className="h-3.5 w-3.5 text-[#F59E0B]" /> GeoJSON
        </button>
      </div>

      {/* Dataset Provenance Modal */}
      {showProvenance && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-[#070B14]/80 p-4 backdrop-blur-sm"
          role="dialog"
          aria-modal="true"
          aria-label="Satellite and socioeconomic data sources"
        >
          <div className="w-full max-w-2xl rounded-xl border border-[#1E2A40] bg-[#0E1524] p-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-[#1E2A40] pb-2.5">
              <div>
                <h3 className="text-sm font-semibold text-[#F4F7FB]">
                  Satellite &amp; Socioeconomic Data Provenance
                </h3>
                <p className="text-[11px] text-[#9FB0C7]">
                  Every calculated value traces to a named dataset, spatial resolution, timestamp, and license.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setShowProvenance(false)}
                className="text-[#9FB0C7] hover:text-[#F4F7FB]"
                aria-label="Close provenance dialog"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="mt-3 max-h-80 overflow-y-auto space-y-2">
              {allDatasets.map((ds) => (
                <div
                  key={ds.dataset_name}
                  className="rounded-lg border border-[#1E2A40] bg-[#141C2E] p-2.5 flex flex-wrap items-center justify-between gap-2"
                >
                  <div>
                    <div className="text-xs font-semibold text-[#F4F7FB]">{ds.dataset_name}</div>
                    <div className="text-[11px] text-[#9FB0C7]">
                      {ds.provider} • Resolution: <b className="text-[#F4F7FB]">{ds.spatial_resolution}</b> • {ds.license}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono-tabular text-[11px] text-[#22D3EE]">
                      {Math.round(ds.confidence * 100)}% conf
                    </div>
                    <div className="font-mono-tabular text-[10px] text-[#9FB0C7]">{ds.timestamp}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </footer>
  );
};
