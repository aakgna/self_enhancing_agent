"""Local sample EHR store — plain Python + CSV, no credential (per
agent_maker/planner.py's own rule: "local computation never requires
credentials"). Adapted from FurHat's data/ehr/*.csv and app/ehr/__init__.py's
facade function names (find_patient, find_appointment_today, get_insurance,
get_provider), minus the calendar/notification integration columns and the
TTL caching layer this local/offline port doesn't need.

This is the full sample dataset (patients, appointments, insurance,
providers, slots) — not just the two files the checkin agent itself uses —
so future work (hand-written or self-extension-generated) has real data to
build against: e.g. an insurance-on-file lookup, a "which provider do I see"
question, or a next-available-appointment-slot check. checkin/agent.py only
calls find_patient/find_appointment_today today; the rest is here to be
built on.

This module *is* the hospital DB for this project, not a cache in front of a
real one: book_slot()/cancel_appointment() write straight back to
appointments.csv/slots.csv, immediately, no separate commit step. Every
generated agent that touches appointments/scheduling must go through these
functions (or its own new Create/Read/Update functions against the same raw
CSVs — see agent_maker/generator.py's EHR-store grounding section) rather
than inventing its own synthetic data.

Deliberately read-through, never cached: every function here re-reads its
CSV(s) from disk on every call rather than loading once into a module-level
table. This used to load once at import time, but that went stale the
moment anything else wrote to the same files — and since additive-only
generation means a generated agent's own new CRUD function must be a
separate module (it can never edit this file), there is no way to keep a
single in-process cache in sync with every writer that might touch these
CSVs. Re-reading a few hundred rows on every call costs nothing that
matters here; staying wrong silently would have.

Deliberately excluded from the port: rows containing real personal data that
had been mixed into the source project's otherwise-fictional sample rows
(a real name/address/phone/insurance-member-ID, and several garbled
manual-test-entry patient rows) — this file only ever ships clearly
fictional sample data.

"Today's appointment" (find_appointment_today) is simplified to "this
patient's scheduled-status row" rather than a real date match — a documented
simplification, not a hidden shortcut. Patient P001 has two scheduled rows
on purpose (a realistic "more than one appointment on file" case);
find_appointment_today returns the first one found.
"""

import csv
import os
from pathlib import Path

# EHR_CSV_DATA_DIR lets the generation sandbox redirect writes to a
# throwaway copy (see agent_maker/sandbox_runner.py) instead of the real,
# git-tracked CSVs. Unset in every real path (the REPL, this repo's own
# tests) — this only exists so a generated agent's own test_agent.py can
# call book_slot()/cancel_appointment() for real, against realistic data,
# without permanently corrupting the shared "hospital DB" on every sandbox
# run (and every later regression re-run of that same test).
_DATA_DIR = Path(os.environ["EHR_CSV_DATA_DIR"]) if os.environ.get("EHR_CSV_DATA_DIR") else Path(__file__).parent / "data"


def _table(name: str) -> tuple[list[dict], list[str]]:
    """Fresh read of one CSV table. Always call this at the top of a
    function rather than caching the result across calls — see module
    docstring for why a cache would go stale."""
    with (_DATA_DIR / name).open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _write_table(name: str, rows: list[dict], fieldnames: list[str]) -> None:
    with (_DATA_DIR / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_patient(first_name: str, last_name: str, dob: str) -> dict | None:
    """Case-insensitive match on first name, last name, and date of birth
    (YYYY-MM-DD)."""
    patients, _ = _table("patients.csv")
    for row in patients:
        if (
            row["first_name"].lower() == first_name.strip().lower()
            and row["last_name"].lower() == last_name.strip().lower()
            and row["dob"] == dob.strip()
        ):
            return dict(row)
    return None


def find_appointment_today(patient_id: str) -> dict | None:
    """This patient's scheduled appointment, if any (see module docstring for
    why this doesn't do a real date match against sample data)."""
    appointments, _ = _table("appointments.csv")
    for row in appointments:
        if row["patient_id"] == patient_id and row["status"] == "scheduled":
            return dict(row)
    return None


def get_insurance(patient_id: str) -> dict | None:
    """This patient's insurance on file, if any."""
    insurance, _ = _table("insurance.csv")
    for row in insurance:
        if row["patient_id"] == patient_id:
            return dict(row)
    return None


def get_provider(provider_id: str) -> dict | None:
    """A provider's name and specialty by id."""
    providers, _ = _table("providers.csv")
    for row in providers:
        if row["provider_id"] == provider_id:
            return dict(row)
    return None


def list_available_slots(provider_id: str | None = None, date: str | None = None) -> list[dict]:
    """Open (available=true) appointment slots, optionally filtered by
    provider and/or date (YYYY-MM-DD)."""
    slots, _ = _table("slots.csv")
    return [
        dict(row)
        for row in slots
        if row["available"] == "true"
        and (provider_id is None or row["provider_id"] == provider_id)
        and (date is None or row["date"] == date)
    ]


def _next_id(rows: list[dict], id_field: str, prefix: str) -> str:
    """The next `{prefix}NNN` id given existing rows, based on the true max
    found — never len(rows) + 1. Real appointments.csv has ids A001-A004,
    A006 (A005 is a gap on purpose in the sample data); assuming no gaps
    would generate a colliding, already-used id."""
    existing = [int(row[id_field][len(prefix):]) for row in rows if row[id_field][len(prefix):].isdigit()]
    return f"{prefix}{(max(existing, default=0) + 1):03d}"


def book_slot(slot_id: str, patient_id: str, reason: str) -> dict | None:
    """Book an open slot for a patient: marks the slot unavailable and
    creates a new 'scheduled' appointment row, persisting both slots.csv and
    appointments.csv immediately. Returns the created appointment dict, or
    None if slot_id doesn't exist or is already booked (available=false) —
    callers must re-check list_available_slots() rather than retry the same
    slot_id, since someone else may have taken it since it was listed."""
    slots, slots_fields = _table("slots.csv")
    slot = next((row for row in slots if row["slot_id"] == slot_id and row["available"] == "true"), None)
    if slot is None:
        return None

    slot["available"] = "false"
    _write_table("slots.csv", slots, slots_fields)

    appointments, appt_fields = _table("appointments.csv")
    appointment = {field: "" for field in appt_fields}
    appointment.update(
        appointment_id=_next_id(appointments, "appointment_id", "A"),
        patient_id=patient_id,
        provider_id=slot["provider_id"],
        date=slot["date"],
        time=slot["time"],
        reason=reason,
        status="scheduled",
    )
    appointments.append(appointment)
    _write_table("appointments.csv", appointments, appt_fields)
    return dict(appointment)


def cancel_appointment(appointment_id: str) -> bool:
    """Cancel a scheduled appointment: marks it 'cancelled' in
    appointments.csv and frees its matching slot (provider_id/date/time)
    back to available in slots.csv, if a matching slot record exists. Never
    deletes the appointment row — a cancellation is a status change, not a
    disappearance, so the visit stays in the record. Returns True if the
    appointment existed and was scheduled, False if it doesn't exist or was
    already cancelled — idempotent, not an error, since "cancel a cancelled
    appointment" is a plausible caller retry."""
    appointments, appt_fields = _table("appointments.csv")
    appointment = next((row for row in appointments if row["appointment_id"] == appointment_id), None)
    if appointment is None or appointment["status"] == "cancelled":
        return False

    appointment["status"] = "cancelled"
    _write_table("appointments.csv", appointments, appt_fields)

    slots, slots_fields = _table("slots.csv")
    slot = next(
        (
            row for row in slots
            if row["provider_id"] == appointment["provider_id"]
            and row["date"] == appointment["date"]
            and row["time"] == appointment["time"]
        ),
        None,
    )
    if slot is not None:
        slot["available"] = "true"
        _write_table("slots.csv", slots, slots_fields)
    return True
