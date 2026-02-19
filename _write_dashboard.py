import pathlib

CONTENT = r'''import os
import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

DEFAULT_API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Investment App", layout="wide")
st.title("Personal Portfolio Intelligence")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_target_weights(text: str) -> dict[str, float]:
    target_weights: dict[str, float] = {}
    for entry in text.split(","):
        chunk = entry.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid entry {chunk!r}. Use format TICKER:WEIGHT")
        ticker, value = chunk.split(":", 1)
        target_weights[ticker.strip().upper()] = float(value.strip())
    if not target_weights:
        raise ValueError("At least one ticker:weight pair is required")
    return target_weights


def build_auth_headers(api_key: str) -> dict[str, str]:
    if not api_key.strip():
        return {}
    return {"X-API-Key": api_key.strip()}


def safe_get(url: str, **kwargs):
    try:
        return requests.get(url, **kwargs), None
    except requests.RequestException as exc:
        return None, str(exc)


def safe_post(url: str, **kwargs):
    try:
        return requests.post(url, **kwargs), None
    except requests.RequestException as exc:
        return None, str(exc)


def allocs_to_weight_string(allocations: list) -> str:
    """Convert allocation list to TICKER:weight string for rebalance input."""
    parts = [f"{a[chr(116)+chr(105)+chr(99)+chr(107)+chr(101)+chr(114)]}:{a['final_weight']:.4f}" for a in allocations]
    return ", ".join(parts)

'''
pathlib.Path("dashboard/streamlit_app.py").write_text(CONTENT, encoding="utf-8")
print("wrote", len(CONTENT), "chars")
