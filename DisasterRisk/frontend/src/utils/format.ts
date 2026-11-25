import type { SeverityLevel } from '../types/claimpulse';

export function formatInr(amount: number, compact = true): string {
  if (!Number.isFinite(amount)) return '₹0';
  if (compact) {
    if (Math.abs(amount) >= 1e7) {
      return `₹${(amount / 1e7).toLocaleString('en-IN', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })} Cr`;
    }
    if (Math.abs(amount) >= 1e5) {
      return `₹${(amount / 1e5).toLocaleString('en-IN', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })} Lakh`;
    }
  }
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 0,
  }).format(amount);
}

export function formatNumber(val: number): string {
  return new Intl.NumberFormat('en-IN').format(Math.round(val));
}

export const SEVERITY_COLORS: Record<SeverityLevel, string> = {
  safe: '#22C55E',
  low: '#FACC15',
  medium: '#F97316',
  high: '#EF4444',
  extreme: '#B91C1C',
};

export const SEVERITY_LEFT_BORDERS: Record<SeverityLevel, string> = {
  safe: 'border-l-[#22C55E]',
  low: 'border-l-[#FACC15]',
  medium: 'border-l-[#F97316]',
  high: 'border-l-[#EF4444]',
  extreme: 'border-l-[#B91C1C]',
};

export const SEVERITY_BADGE_CLASSES: Record<SeverityLevel, string> = {
  safe: 'bg-[#22C55E]/15 text-[#4ADE80] border-[#22C55E]/40',
  low: 'bg-[#FACC15]/15 text-[#FDE047] border-[#FACC15]/40',
  medium: 'bg-[#F97316]/15 text-[#FB923C] border-[#F97316]/40',
  high: 'bg-[#EF4444]/15 text-[#F87171] border-[#EF4444]/40',
  extreme: 'bg-[#B91C1C]/30 text-[#FCA5A5] border-[#B91C1C]/60',
};
