"""Offline tests for the local sample EHR store used by the checkin agent —
plain CSV reads, no network, no credential.

book_slot/cancel_appointment tests below run against a temp-directory copy
of the real CSVs (via the _reload_all() test seam), never the real,
git-tracked sample data — writing straight to the real files on every test
run would permanently mutate them, the same local-state test-pollution
pattern documented on the credential-gate tests."""

import csv
import shutil
from pathlib import Path

import pytest

from tools.shared.ehr_csv import store


def test_find_patient_matches_case_insensitively():
    patient = store.find_patient("sarah", "JOHNSON", "1985-03-12")
    assert patient is not None
    assert patient["patient_id"] == "P001"


def test_find_patient_no_match_returns_none():
    assert store.find_patient("Nobody", "Real", "1900-01-01") is None


def test_find_patient_wrong_dob_returns_none():
    assert store.find_patient("Sarah", "Johnson", "1999-01-01") is None


def test_find_appointment_today_returns_scheduled_appointment():
    appt = store.find_appointment_today("P001")
    assert appt is not None
    assert appt["status"] == "scheduled"
    assert appt["reason"] == "Annual checkup"


def test_find_appointment_today_ignores_non_scheduled_status():
    # P004's only appointment is status=completed, not scheduled.
    assert store.find_appointment_today("P004") is None


def test_find_appointment_today_no_appointment_returns_none():
    # P005 has no appointment row at all.
    assert store.find_appointment_today("P005") is None


def test_find_appointment_today_returns_first_when_patient_has_multiple():
    # P001 has two scheduled rows on purpose (A001, A006) — first in file wins.
    appt = store.find_appointment_today("P001")
    assert appt["appointment_id"] == "A001"


def test_get_insurance_returns_patients_plan():
    insurance = store.get_insurance("P001")
    assert insurance is not None
    assert insurance["payer"] == "Blue Cross Blue Shield"


def test_get_insurance_no_record_returns_none():
    assert store.get_insurance("P999") is None


def test_get_provider_returns_name_and_specialty():
    provider = store.get_provider("PR001")
    assert provider is not None
    assert provider["name"] == "Dr. Amanda Foster"
    assert provider["specialty"] == "Family Medicine"


def test_get_provider_unknown_id_returns_none():
    assert store.get_provider("PR999") is None


def test_list_available_slots_filters_out_unavailable():
    slots = store.list_available_slots(provider_id="PR001")
    assert slots  # PR001 has open slots in the sample data
    assert all(s["available"] == "true" for s in slots)
    assert all(s["provider_id"] == "PR001" for s in slots)


def test_list_available_slots_by_date():
    slots = store.list_available_slots(date="2026-07-09")
    assert slots
    assert all(s["date"] == "2026-07-09" for s in slots)


def test_list_available_slots_no_match_returns_empty_list():
    assert store.list_available_slots(provider_id="PR999") == []


@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    """Points the store at a scratch copy of the real CSVs for the duration
    of one test — nothing to reload before/after, since store.py reads fresh
    from _DATA_DIR on every call rather than caching; monkeypatch reverts
    _DATA_DIR automatically once the test ends."""
    for csv_file in store._DATA_DIR.glob("*.csv"):
        shutil.copy(csv_file, tmp_path / csv_file.name)
    monkeypatch.setattr(store, "_DATA_DIR", tmp_path)
    yield tmp_path


def _read_csv(path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_controlled_appointments(path, rows: list[dict]) -> None:
    """Overwrites appointments.csv at `path` with an explicit set of rows
    (keeping whatever fieldnames are already there), independent of how
    many real bookings have accumulated in the live-use data since a test
    was written — see test_book_slot_generates_a_unique_appointment_id_no_
    gaps_assumed for why this matters."""
    with path.open(newline="") as f:
        fieldnames = list(csv.DictReader(f).fieldnames)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: "" for k in fieldnames}, **row})


def test_book_slot_marks_slot_unavailable_and_creates_appointment(isolated_store):
    before = store.list_available_slots(provider_id="PR001")
    slot_id = before[0]["slot_id"]

    appointment = store.book_slot(slot_id, patient_id="P001", reason="Annual physical")

    assert appointment is not None
    assert appointment["patient_id"] == "P001"
    assert appointment["provider_id"] == before[0]["provider_id"]
    assert appointment["date"] == before[0]["date"]
    assert appointment["time"] == before[0]["time"]
    assert appointment["reason"] == "Annual physical"
    assert appointment["status"] == "scheduled"
    assert appointment["appointment_id"]

    # No longer offered as available, in memory...
    after = store.list_available_slots(provider_id="PR001")
    assert slot_id not in [s["slot_id"] for s in after]
    # ...and persisted to disk, not just in the in-memory table.
    on_disk_slots = _read_csv(isolated_store / "slots.csv")
    booked = next(r for r in on_disk_slots if r["slot_id"] == slot_id)
    assert booked["available"] == "false"
    on_disk_appointments = _read_csv(isolated_store / "appointments.csv")
    assert any(r["appointment_id"] == appointment["appointment_id"] for r in on_disk_appointments)


def test_book_slot_generates_a_unique_appointment_id_no_gaps_assumed(isolated_store):
    # Ids end at A004, A006 — A005 is a deliberate gap, so the next
    # generated id must be based on the true max, not len(rows)+1, or it
    # would collide with an already-used id. Written explicitly here rather
    # than relying on the real appointments.csv's current row count: that
    # file is the live hospital DB now and grows with real bookings made
    # during manual testing, so a hardcoded expected id tied to a past
    # snapshot of it was exactly the kind of fragility that broke this test.
    _write_controlled_appointments(isolated_store / "appointments.csv", [
        {"appointment_id": "A001", "patient_id": "P001", "status": "scheduled"},
        {"appointment_id": "A004", "patient_id": "P004", "status": "completed"},
        {"appointment_id": "A006", "patient_id": "P001", "status": "scheduled"},
    ])

    slots = store.list_available_slots(provider_id="PR002")
    appointment = store.book_slot(slots[0]["slot_id"], patient_id="P002", reason="Follow-up")
    assert appointment["appointment_id"] == "A007"


def test_book_slot_unknown_id_returns_none(isolated_store):
    assert store.book_slot("NOT_A_REAL_SLOT", patient_id="P001", reason="x") is None


def test_book_slot_already_booked_slot_returns_none(isolated_store):
    slots = store.list_available_slots(provider_id="PR001")
    slot_id = slots[0]["slot_id"]
    assert store.book_slot(slot_id, patient_id="P001", reason="first booking") is not None
    # Second attempt on the same (now-unavailable) slot must not double-book.
    assert store.book_slot(slot_id, patient_id="P002", reason="second booking") is None


def test_cancel_appointment_frees_the_slot_and_marks_cancelled(isolated_store):
    slots = store.list_available_slots(provider_id="PR003")
    slot_id = slots[0]["slot_id"]
    appointment = store.book_slot(slot_id, patient_id="P003", reason="Consultation")

    result = store.cancel_appointment(appointment["appointment_id"])

    assert result is True
    on_disk_appointments = _read_csv(isolated_store / "appointments.csv")
    cancelled = next(r for r in on_disk_appointments if r["appointment_id"] == appointment["appointment_id"])
    assert cancelled["status"] == "cancelled"
    on_disk_slots = _read_csv(isolated_store / "slots.csv")
    freed = next(r for r in on_disk_slots if r["slot_id"] == slot_id)
    assert freed["available"] == "true"
    assert slot_id in [s["slot_id"] for s in store.list_available_slots(provider_id="PR003")]


def test_cancel_appointment_unknown_id_returns_false(isolated_store):
    assert store.cancel_appointment("NOT_A_REAL_APPOINTMENT") is False


def test_cancel_appointment_already_cancelled_returns_false(isolated_store):
    slots = store.list_available_slots(provider_id="PR004")
    appointment = store.book_slot(slots[0]["slot_id"], patient_id="P004", reason="x")
    assert store.cancel_appointment(appointment["appointment_id"]) is True
    assert store.cancel_appointment(appointment["appointment_id"]) is False


def test_isolated_store_fixture_does_not_touch_real_data(isolated_store):
    """Meta-test: proves the fixture actually isolates writes — the real,
    git-tracked slots.csv must be byte-identical before and after booking a
    slot inside the fixture (isolated_store already pointed store._DATA_DIR
    at a tmp copy; the real file lives at the path captured here, from the
    module's own source location)."""
    real_data_dir = Path(store.__spec__.origin).parent / "data"
    before = (real_data_dir / "slots.csv").read_bytes()

    slots = store.list_available_slots(provider_id="PR001")
    store.book_slot(slots[0]["slot_id"], patient_id="P001", reason="isolation check")

    after = (real_data_dir / "slots.csv").read_bytes()
    assert before == after

