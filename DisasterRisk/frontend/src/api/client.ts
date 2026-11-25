import type {
  AskResponse,
  BoundingBox,
  ConnectedModule,
  DisasterType,
  EventFullResponse,
  EventSummary,
} from '../types/claimpulse';
import {
  fallbackAskClaimPulse,
  fallbackGetEvent,
  fallbackGetModules,
  fallbackListEvents,
  fallbackRegisterModule,
  fallbackReviewDecision,
  fallbackRunEvent,
} from './localFallback';

const API_BASE = '/api';

export let usingLocalFallback = false;

async function fetchJsonWithTimeout<T>(url: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 3000);
  try {
    const res = await fetch(url, { ...init, signal: controller.signal });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    usingLocalFallback = false;
    return (await res.json()) as T;
  } finally {
    clearTimeout(timeout);
  }
}

export async function listEvents(): Promise<EventSummary[]> {
  try {
    return await fetchJsonWithTimeout<EventSummary[]>(`${API_BASE}/events`);
  } catch {
    usingLocalFallback = true;
    return fallbackListEvents();
  }
}

export async function getEvent(eventId: string): Promise<EventFullResponse> {
  try {
    return await fetchJsonWithTimeout<EventFullResponse>(
      `${API_BASE}/events/${encodeURIComponent(eventId)}`
    );
  } catch {
    usingLocalFallback = true;
    return fallbackGetEvent(eventId);
  }
}

export async function runEventAnalysis(payload: {
  preset_id?: string | null;
  disaster_type: DisasterType;
  title?: string;
  country?: string;
  state?: string;
  district?: string;
  start_date: string;
  end_date: string;
  bbox?: BoundingBox;
  intensity_multiplier?: number;
}): Promise<EventFullResponse> {
  try {
    return await fetchJsonWithTimeout<EventFullResponse>(`${API_BASE}/events`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch {
    usingLocalFallback = true;
    return fallbackRunEvent(payload);
  }
}

export async function reviewDecision(
  eventId: string,
  decisionId: string,
  action: 'Approve' | 'Escalate' | 'Request Field Verification',
  reviewer: string,
  notes: string
): Promise<EventFullResponse> {
  try {
    return await fetchJsonWithTimeout<EventFullResponse>(
      `${API_BASE}/events/${encodeURIComponent(eventId)}/decisions/${encodeURIComponent(
        decisionId
      )}/review`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, reviewer, notes }),
      }
    );
  } catch {
    usingLocalFallback = true;
    return fallbackReviewDecision(eventId, decisionId, action, reviewer, notes);
  }
}

export async function askClaimPulse(
  eventId: string,
  question: string
): Promise<AskResponse> {
  try {
    return await fetchJsonWithTimeout<AskResponse>(
      `${API_BASE}/events/${encodeURIComponent(eventId)}/ask`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      }
    );
  } catch {
    usingLocalFallback = true;
    return fallbackAskClaimPulse(eventId, question);
  }
}

export async function getModules(): Promise<ConnectedModule[]> {
  try {
    return await fetchJsonWithTimeout<ConnectedModule[]>(`${API_BASE}/modules`);
  } catch {
    usingLocalFallback = true;
    return fallbackGetModules();
  }
}

export async function registerModule(
  mod: Omit<ConnectedModule, 'module_id' | 'status'>
): Promise<ConnectedModule> {
  try {
    return await fetchJsonWithTimeout<ConnectedModule>(`${API_BASE}/modules/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...mod, status: 'active' }),
    });
  } catch {
    usingLocalFallback = true;
    return fallbackRegisterModule(mod);
  }
}

export async function getWebhooks(): Promise<{
  subscriptions: Array<{
    webhook_id: string;
    partner_name: string;
    url: string;
    events: string[];
    active: boolean;
  }>;
  recent_deliveries: Array<{
    delivery_id: string;
    webhook_id: string;
    event_type: string;
    timestamp: string;
    status: string;
  }>;
}> {
  try {
    return await fetchJsonWithTimeout(`${API_BASE}/webhooks`);
  } catch {
    return {
      subscriptions: [
        {
          webhook_id: 'WH-PARTNER-01',
          partner_name: 'Insurer Core Claims Bus',
          url: 'https://claims.partner-insurer.demo/hooks/claimpulse',
          events: ['event.processed', 'decision.updated'],
          active: true,
        },
      ],
      recent_deliveries: [],
    };
  }
}

export function getExportUrl(eventId: string, format: 'pdf' | 'csv' | 'geojson'): string {
  return `${API_BASE}/events/${encodeURIComponent(eventId)}/export?format=${format}`;
}
