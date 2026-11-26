"""Priority-based onboard transmission queue.

The queue is deliberately deterministic: high priority is sent first, then
medium and low, and an item is only completed when its whole JSON payload fits
inside the current contact budget.  This models defer-and-resume behaviour
without pretending to simulate a radio modem.
"""

from dataclasses import dataclass, field
from itertools import count
from typing import Any

PRIORITY_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


@dataclass(order=True)
class TransmissionItem:
    sort_key: tuple = field(init=False, repr=False)
    priority: str
    payload: bytes = field(compare=False)
    label: str = field(default="observation", compare=False)
    created_order: int = field(default=0, compare=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        normalized = self.priority.upper()
        if normalized not in PRIORITY_RANK:
            raise ValueError("priority must be HIGH, MEDIUM, or LOW")
        self.priority = normalized
        self.sort_key = (PRIORITY_RANK[normalized], self.created_order)


class TransmissionQueue:
    def __init__(self) -> None:
        self._items: list[TransmissionItem] = []
        self._counter = count()

    def enqueue(self, payload: bytes, priority: str, label: str = "observation",
                metadata: dict[str, Any] | None = None) -> TransmissionItem:
        item = TransmissionItem(priority=priority, payload=payload, label=label,
                                created_order=next(self._counter), metadata=metadata or {})
        self._items.append(item)
        return item

    def pending(self) -> list[TransmissionItem]:
        return sorted(self._items)

    def dispatch(self, visible: bool, bandwidth_mbps: float, contact_seconds: float) -> dict:
        if bandwidth_mbps <= 0 or contact_seconds < 0:
            raise ValueError("bandwidth must be positive and contact_seconds non-negative")
        bytes_per_second = bandwidth_mbps * 1_000_000 / 8.0
        budget_bytes = bytes_per_second * contact_seconds
        sent, deferred = [], []
        remaining_seconds = contact_seconds
        elapsed_contact_seconds = 0.0
        for item in self.pending():
            required = len(item.payload)
            duration = required / bytes_per_second
            if visible and duration <= remaining_seconds:
                item.metadata["simulated_transmit_duration_s"] = duration
                item.metadata["simulated_transmit_start_offset_s"] = elapsed_contact_seconds
                item.metadata["simulated_receive_offset_s"] = elapsed_contact_seconds + duration
                sent.append(item)
                remaining_seconds -= duration
                elapsed_contact_seconds += duration
            else:
                deferred.append(item)
        sent_ids = {id(item) for item in sent}
        self._items = [item for item in self._items if id(item) not in sent_ids]
        return {
            "visible": visible,
            "budget_bytes": round(budget_bytes, 3),
            "sent": sent,
            "deferred": deferred,
            "remaining_bytes": round(remaining_seconds * bytes_per_second, 3),
            "remaining_contact_seconds": round(remaining_seconds, 6),
        }

    def snapshot(self) -> list[dict[str, Any]]:
        return [{"label": i.label, "priority": i.priority, "size_bytes": len(i.payload),
                 "metadata": i.metadata} for i in self.pending()]
