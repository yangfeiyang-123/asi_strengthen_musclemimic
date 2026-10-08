from __future__ import annotations


def classify_failure(info: dict) -> str:
    if bool(info.get("fell", False)):
        return "fall"
    if float(info.get("grip_slip_m", 0.0)) > 0.05:
        return "grip_slip"
    if not bool(info.get("stringbed_contact", False)):
        return "missed_contact"
    if not bool(info.get("shuttle_crossed_net", False)):
        return "bad_flight"
    return "unknown"
