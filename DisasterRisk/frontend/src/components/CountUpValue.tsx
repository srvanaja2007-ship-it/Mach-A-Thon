import React, { useEffect, useRef, useState } from 'react';

interface CountUpValueProps {
  value: number;
  formatter: (val: number) => string;
  durationMs?: number;
  className?: string;
}

export const CountUpValue: React.FC<CountUpValueProps> = ({
  value,
  formatter,
  durationMs = 220,
  className = '',
}) => {
  const [display, setDisplay] = useState(value);
  const prevRef = useRef(value);

  useEffect(() => {
    const startVal = prevRef.current;
    const endVal = value;
    if (Math.abs(endVal - startVal) < 0.01) {
      setDisplay(endVal);
      prevRef.current = endVal;
      return;
    }

    let rafId = 0;
    const startTime = performance.now();

    const tick = (now: number) => {
      const t = Math.min(1, (now - startTime) / durationMs);
      const eased = 1 - Math.pow(1 - t, 3);
      setDisplay(startVal + (endVal - startVal) * eased);
      if (t < 1) {
        rafId = requestAnimationFrame(tick);
      } else {
        prevRef.current = endVal;
      }
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [value, durationMs]);

  return <span className={`font-mono-tabular ${className}`}>{formatter(display)}</span>;
};
