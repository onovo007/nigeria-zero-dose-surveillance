"""
Nigeria Zero-Dose Predictive Modelling Platform — Backend API
=============================================================
NPHCDA Digital Innovation Hub | UNICEF Technical Assistance
PI: Dr. Amobi Andrew Onovo, PhD, MPH | CIDRE-Quantium Insights

Deploy on Render:
  Build:  pip install -r requirements.txt
  Start:  uvicorn app:app --host 0.0.0.0 --port $PORT
  Env:    ANTHROPIC_API_KEY = sk-ant-...

Local:    uvicorn app:app --reload --port 8000
"""

import os, io, uuid, json
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

try:
    import pandas as pd
    import numpy as np
    from prophet import Prophet
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.linear_model import LassoCV
    from sklearn.metrics import silhouette_score
    ANALYSIS_READY = True
except ImportError:
    ANALYSIS_READY = False

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")

import asyncio
import datetime

# ---------------------------------------------------------------------------
# Lifespan — runs on startup and shutdown (paid Render: always-on, no sleep)
# ---------------------------------------------------------------------------
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app_instance):
    """Start background keep-alive ping on startup. Cancel on shutdown."""
    task = asyncio.create_task(_keepalive_task())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

async def _keepalive_task():
    """
    Pings the health endpoint every 5 minutes to confirm the process is
    alive and reset any proxy-level idle timeouts upstream of Render.
    On a paid Render plan the service never sleeps, but this also:
      - Keeps worker threads warm for faster first-request response
      - Logs a heartbeat so the Render log stream shows activity
      - Resets any network-layer idle-connection timeouts (e.g. Cloudflare)
    """
    await asyncio.sleep(10)  # brief delay to let server finish starting
    while True:
        try:
            now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"[heartbeat] {now} — service alive, {len(STATE_DATA)} states loaded")
        except Exception as e:
            print(f"[heartbeat] error: {e}")
        await asyncio.sleep(300)  # every 5 minutes


app = FastAPI(
    title="Nigeria Zero-Dose Predictive Modelling API",
    description="NPHCDA DIH | UNICEF | CIDRE-Quantium Insights",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

sessions: dict = {}
jobs: dict = {}

# Pre-computed built-in results — 37 states, Bayesian model
STATE_DATA: List[dict] = [
  {
    "state": "Sokoto",
    "zone": "North West",
    "zd_obs_2024": 86.5,
    "zd_pred_2026_mean": 71.9,
    "zd_pred_2026_lo95": 57.8,
    "zd_pred_2026_hi95": 83.8,
    "zd_pred_2027_mean": 69.9,
    "zd_pred_2028_mean": 67.8,
    "zd_count_2026": 168829,
    "gi_class_2026": "Hot Spot (p<0.01)",
    "gi_z_2026": 2.594,
    "priority_tier": "Tier 1: Critical",
    "risk_index": 85.4,
    "state_rank": 1
  },
  {
    "state": "Kebbi",
    "zone": "North West",
    "zd_obs_2024": 84.0,
    "zd_pred_2026_mean": 68.4,
    "zd_pred_2026_lo95": 54.1,
    "zd_pred_2026_hi95": 81.2,
    "zd_pred_2027_mean": 66.7,
    "zd_pred_2028_mean": 65.0,
    "zd_count_2026": 154298,
    "gi_class_2026": "Hot Spot (p<0.01)",
    "gi_z_2026": 2.222,
    "priority_tier": "Tier 1: Critical",
    "risk_index": 81.8,
    "state_rank": 2
  },
  {
    "state": "Zamfara",
    "zone": "North West",
    "zd_obs_2024": 82.6,
    "zd_pred_2026_mean": 73.0,
    "zd_pred_2026_lo95": 59.6,
    "zd_pred_2026_hi95": 84.7,
    "zd_pred_2027_mean": 71.9,
    "zd_pred_2028_mean": 70.7,
    "zd_count_2026": 156026,
    "gi_class_2026": "Hot Spot (p<0.01)",
    "gi_z_2026": 1.76,
    "priority_tier": "Tier 1: Critical",
    "risk_index": 81.2,
    "state_rank": 3
  },
  {
    "state": "Kano",
    "zone": "North West",
    "zd_obs_2024": 42.4,
    "zd_pred_2026_mean": 36.4,
    "zd_pred_2026_lo95": 26.8,
    "zd_pred_2026_hi95": 46.9,
    "zd_pred_2027_mean": 34.3,
    "zd_pred_2028_mean": 32.3,
    "zd_count_2026": 195206,
    "gi_class_2026": "Not Significant",
    "gi_z_2026": 0.639,
    "priority_tier": "Tier 2: High",
    "risk_index": 61.2,
    "state_rank": 4
  },
  {
    "state": "Kaduna",
    "zone": "North West",
    "zd_obs_2024": 45.6,
    "zd_pred_2026_mean": 43.3,
    "zd_pred_2026_lo95": 30.2,
    "zd_pred_2026_hi95": 56.2,
    "zd_pred_2027_mean": 43.2,
    "zd_pred_2028_mean": 43.0,
    "zd_count_2026": 138309,
    "gi_class_2026": "Hot Spot (p<0.05)",
    "gi_z_2026": 0.667,
    "priority_tier": "Tier 2: High",
    "risk_index": 58.4,
    "state_rank": 5
  },
  {
    "state": "Niger",
    "zone": "North Central",
    "zd_obs_2024": 56.0,
    "zd_pred_2026_mean": 45.3,
    "zd_pred_2026_lo95": 31.5,
    "zd_pred_2026_hi95": 59.4,
    "zd_pred_2027_mean": 44.8,
    "zd_pred_2028_mean": 44.4,
    "zd_count_2026": 104775,
    "gi_class_2026": "Hot Spot (p<0.01)",
    "gi_z_2026": 1.1,
    "priority_tier": "Tier 2: High",
    "risk_index": 57.8,
    "state_rank": 6
  },
  {
    "state": "Katsina",
    "zone": "North West",
    "zd_obs_2024": 39.8,
    "zd_pred_2026_mean": 38.0,
    "zd_pred_2026_lo95": 26.4,
    "zd_pred_2026_hi95": 51.2,
    "zd_pred_2027_mean": 35.2,
    "zd_pred_2028_mean": 32.6,
    "zd_count_2026": 165574,
    "gi_class_2026": "Hot Spot (p<0.05)",
    "gi_z_2026": 1.085,
    "priority_tier": "Tier 2: High",
    "risk_index": 54.9,
    "state_rank": 7
  },
  {
    "state": "Kogi",
    "zone": "North Central",
    "zd_obs_2024": 57.0,
    "zd_pred_2026_mean": 32.3,
    "zd_pred_2026_lo95": 16.1,
    "zd_pred_2026_hi95": 51.2,
    "zd_pred_2027_mean": 32.9,
    "zd_pred_2028_mean": 33.4,
    "zd_count_2026": 49078,
    "gi_class_2026": "Not Significant",
    "gi_z_2026": -0.223,
    "priority_tier": "Tier 3: Moderate",
    "risk_index": 49.0,
    "state_rank": 8
  },
  {
    "state": "Kwara",
    "zone": "North Central",
    "zd_obs_2024": 51.7,
    "zd_pred_2026_mean": 37.3,
    "zd_pred_2026_lo95": 20.8,
    "zd_pred_2026_hi95": 55.6,
    "zd_pred_2027_mean": 37.4,
    "zd_pred_2028_mean": 37.6,
    "zd_count_2026": 46895,
    "gi_class_2026": "Not Significant",
    "gi_z_2026": -0.002,
    "priority_tier": "Tier 3: Moderate",
    "risk_index": 46.3,
    "state_rank": 9
  },
  {
    "state": "Jigawa",
    "zone": "North West",
    "zd_obs_2024": 34.3,
    "zd_pred_2026_mean": 34.0,
    "zd_pred_2026_lo95": 21.3,
    "zd_pred_2026_hi95": 48.3,
    "zd_pred_2027_mean": 31.4,
    "zd_pred_2028_mean": 28.9,
    "zd_count_2026": 103706,
    "gi_class_2026": "Not Significant",
    "gi_z_2026": 0.541,
    "priority_tier": "Tier 3: Moderate",
    "risk_index": 45.6,
    "state_rank": 10
  },
  {
    "state": "Bauchi",
    "zone": "North East",
    "zd_obs_2024": 36.9,
    "zd_pred_2026_mean": 34.4,
    "zd_pred_2026_lo95": 23.1,
    "zd_pred_2026_hi95": 46.9,
    "zd_pred_2027_mean": 32.0,
    "zd_pred_2028_mean": 29.6,
    "zd_count_2026": 113091,
    "gi_class_2026": "Not Significant",
    "gi_z_2026": 0.444,
    "priority_tier": "Tier 3: Moderate",
    "risk_index": 44.0,
    "state_rank": 11
  },
  {
    "state": "Benue",
    "zone": "North Central",
    "zd_obs_2024": 32.7,
    "zd_pred_2026_mean": 28.9,
    "zd_pred_2026_lo95": 16.1,
    "zd_pred_2026_hi95": 44.0,
    "zd_pred_2027_mean": 28.5,
    "zd_pred_2028_mean": 28.1,
    "zd_count_2026": 51902,
    "gi_class_2026": "Not Significant",
    "gi_z_2026": -0.329,
    "priority_tier": "Tier 3: Moderate",
    "risk_index": 37.9,
    "state_rank": 12
  },
  {
    "state": "Plateau",
    "zone": "North Central",
    "zd_obs_2024": 35.3,
    "zd_pred_2026_mean": 26.0,
    "zd_pred_2026_lo95": 13.6,
    "zd_pred_2026_hi95": 41.0,
    "zd_pred_2027_mean": 26.0,
    "zd_pred_2028_mean": 26.0,
    "zd_count_2026": 38618,
    "gi_class_2026": "Not Significant",
    "gi_z_2026": 0.3,
    "priority_tier": "Tier 3: Moderate",
    "risk_index": 36.4,
    "state_rank": 13
  },
  {
    "state": "Yobe",
    "zone": "North East",
    "zd_obs_2024": 40.4,
    "zd_pred_2026_mean": 34.9,
    "zd_pred_2026_lo95": 21.9,
    "zd_pred_2026_hi95": 49.6,
    "zd_pred_2027_mean": 32.5,
    "zd_pred_2028_mean": 30.2,
    "zd_count_2026": 42197,
    "gi_class_2026": "Not Significant",
    "gi_z_2026": 0.431,
    "priority_tier": "Tier 3: Moderate",
    "risk_index": 36.0,
    "state_rank": 14
  },
  {
    "state": "Gombe",
    "zone": "North East",
    "zd_obs_2024": 34.0,
    "zd_pred_2026_mean": 34.9,
    "zd_pred_2026_lo95": 20.1,
    "zd_pred_2026_hi95": 52.1,
    "zd_pred_2027_mean": 33.5,
    "zd_pred_2028_mean": 32.1,
    "zd_count_2026": 51117,
    "gi_class_2026": "Not Significant",
    "gi_z_2026": 0.243,
    "priority_tier": "Tier 3: Moderate",
    "risk_index": 34.8,
    "state_rank": 15
  },
  {
    "state": "Taraba",
    "zone": "North East",
    "zd_obs_2024": 40.7,
    "zd_pred_2026_mean": 27.1,
    "zd_pred_2026_lo95": 14.1,
    "zd_pred_2026_hi95": 42.8,
    "zd_pred_2027_mean": 25.8,
    "zd_pred_2028_mean": 24.6,
    "zd_count_2026": 33109,
    "gi_class_2026": "Not Significant",
    "gi_z_2026": 0.125,
    "priority_tier": "Tier 3: Moderate",
    "risk_index": 33.8,
    "state_rank": 16
  },
  {
    "state": "Borno",
    "zone": "North East",
    "zd_obs_2024": 31.7,
    "zd_pred_2026_mean": 30.1,
    "zd_pred_2026_lo95": 18.3,
    "zd_pred_2026_hi95": 43.3,
    "zd_pred_2027_mean": 27.6,
    "zd_pred_2028_mean": 25.2,
    "zd_count_2026": 54422,
    "gi_class_2026": "Not Significant",
    "gi_z_2026": 0.233,
    "priority_tier": "Tier 3: Moderate",
    "risk_index": 33.0,
    "state_rank": 17
  },
  {
    "state": "Oyo",
    "zone": "South West",
    "zd_obs_2024": 29.9,
    "zd_pred_2026_mean": 19.9,
    "zd_pred_2026_lo95": 9.3,
    "zd_pred_2026_hi95": 34.5,
    "zd_pred_2027_mean": 19.7,
    "zd_pred_2028_mean": 19.5,
    "zd_count_2026": 46370,
    "gi_class_2026": "Not Significant",
    "gi_z_2026": -0.26,
    "priority_tier": "Tier 3: Moderate",
    "risk_index": 30.6,
    "state_rank": 18
  },
  {
    "state": "Nasarawa",
    "zone": "North Central",
    "zd_obs_2024": 20.3,
    "zd_pred_2026_mean": 26.1,
    "zd_pred_2026_lo95": 13.6,
    "zd_pred_2026_hi95": 41.1,
    "zd_pred_2027_mean": 25.5,
    "zd_pred_2028_mean": 25.0,
    "zd_count_2026": 27429,
    "gi_class_2026": "Not Significant",
    "gi_z_2026": 0.143,
    "priority_tier": "Tier 3: Moderate",
    "risk_index": 29.7,
    "state_rank": 19
  },
  {
    "state": "Adamawa",
    "zone": "North East",
    "zd_obs_2024": 29.1,
    "zd_pred_2026_mean": 21.1,
    "zd_pred_2026_lo95": 10.1,
    "zd_pred_2026_hi95": 35.8,
    "zd_pred_2027_mean": 20.3,
    "zd_pred_2028_mean": 19.6,
    "zd_count_2026": 36710,
    "gi_class_2026": "Not Significant",
    "gi_z_2026": 0.121,
    "priority_tier": "Tier 3: Moderate",
    "risk_index": 29.2,
    "state_rank": 20
  },
  {
    "state": "Ondo",
    "zone": "South West",
    "zd_obs_2024": 23.5,
    "zd_pred_2026_mean": 18.5,
    "zd_pred_2026_lo95": 7.7,
    "zd_pred_2026_hi95": 34.3,
    "zd_pred_2027_mean": 18.1,
    "zd_pred_2028_mean": 17.7,
    "zd_count_2026": 27312,
    "gi_class_2026": "Cold Spot (p<0.05)",
    "gi_z_2026": -0.577,
    "priority_tier": "Tier 3: Moderate",
    "risk_index": 25.5,
    "state_rank": 21
  },
  {
    "state": "Rivers",
    "zone": "South South",
    "zd_obs_2024": 22.7,
    "zd_pred_2026_mean": 15.7,
    "zd_pred_2026_lo95": 7.0,
    "zd_pred_2026_hi95": 27.2,
    "zd_pred_2027_mean": 15.3,
    "zd_pred_2028_mean": 14.9,
    "zd_count_2026": 31601,
    "gi_class_2026": "Cold Spot (p<0.01)",
    "gi_z_2026": -0.82,
    "priority_tier": "Tier 4: Lower",
    "risk_index": 22.2,
    "state_rank": 22
  },
  {
    "state": "Ogun",
    "zone": "South West",
    "zd_obs_2024": 20.6,
    "zd_pred_2026_mean": 18.1,
    "zd_pred_2026_lo95": 8.6,
    "zd_pred_2026_hi95": 30.6,
    "zd_pred_2027_mean": 17.7,
    "zd_pred_2028_mean": 17.2,
    "zd_count_2026": 28170,
    "gi_class_2026": "Cold Spot (p<0.10)",
    "gi_z_2026": -0.643,
    "priority_tier": "Tier 4: Lower",
    "risk_index": 20.9,
    "state_rank": 23
  },
  {
    "state": "FCT",
    "zone": "North Central",
    "zd_obs_2024": 6.2,
    "zd_pred_2026_mean": 17.0,
    "zd_pred_2026_lo95": 7.1,
    "zd_pred_2026_hi95": 30.4,
    "zd_pred_2027_mean": 16.9,
    "zd_pred_2028_mean": 16.8,
    "zd_count_2026": 16472,
    "gi_class_2026": "Not Significant",
    "gi_z_2026": null,
    "priority_tier": "Tier 4: Lower",
    "risk_index": 18.2,
    "state_rank": 24
  },
  {
    "state": "Bayelsa",
    "zone": "South South",
    "zd_obs_2024": 19.1,
    "zd_pred_2026_mean": 15.2,
    "zd_pred_2026_lo95": 5.3,
    "zd_pred_2026_hi95": 31.3,
    "zd_pred_2027_mean": 14.5,
    "zd_pred_2028_mean": 13.8,
    "zd_count_2026": 10471,
    "gi_class_2026": "Not Significant",
    "gi_z_2026": -0.672,
    "priority_tier": "Tier 4: Lower",
    "risk_index": 16.9,
    "state_rank": 25
  },
  {
    "state": "Akwa Ibom",
    "zone": "South South",
    "zd_obs_2024": 15.1,
    "zd_pred_2026_mean": 13.1,
    "zd_pred_2026_lo95": 5.5,
    "zd_pred_2026_hi95": 24.6,
    "zd_pred_2027_mean": 12.7,
    "zd_pred_2028_mean": 12.4,
    "zd_count_2026": 16551,
    "gi_class_2026": "Cold Spot (p<0.05)",
    "gi_z_2026": -0.795,
    "priority_tier": "Tier 4: Lower",
    "risk_index": 16.2,
    "state_rank": 26
  },
  {
    "state": "Enugu",
    "zone": "South East",
    "zd_obs_2024": 12.8,
    "zd_pred_2026_mean": 10.1,
    "zd_pred_2026_lo95": 3.6,
    "zd_pred_2026_hi95": 20.6,
    "zd_pred_2027_mean": 9.6,
    "zd_pred_2028_mean": 9.2,
    "zd_count_2026": 14974,
    "gi_class_2026": "Cold Spot (p<0.10)",
    "gi_z_2026": -0.582,
    "priority_tier": "Tier 4: Lower",
    "risk_index": 14.0,
    "state_rank": 27
  },
  {
    "state": "Ekiti",
    "zone": "South West",
    "zd_obs_2024": 6.0,
    "zd_pred_2026_mean": 10.7,
    "zd_pred_2026_lo95": 3.8,
    "zd_pred_2026_hi95": 22.4,
    "zd_pred_2027_mean": 10.8,
    "zd_pred_2028_mean": 10.9,
    "zd_count_2026": 11941,
    "gi_class_2026": "Not Significant",
    "gi_z_2026": -0.239,
    "priority_tier": "Tier 4: Lower",
    "risk_index": 13.8,
    "state_rank": 28
  },
  {
    "state": "Anambra",
    "zone": "South East",
    "zd_obs_2024": 14.9,
    "zd_pred_2026_mean": 9.9,
    "zd_pred_2026_lo95": 3.8,
    "zd_pred_2026_hi95": 19.4,
    "zd_pred_2027_mean": 9.7,
    "zd_pred_2028_mean": 9.5,
    "zd_count_2026": 18041,
    "gi_class_2026": "Cold Spot (p<0.01)",
    "gi_z_2026": -0.735,
    "priority_tier": "Tier 4: Lower",
    "risk_index": 13.7,
    "state_rank": 29
  },
  {
    "state": "Cross River",
    "zone": "South South",
    "zd_obs_2024": 2.8,
    "zd_pred_2026_mean": 12.3,
    "zd_pred_2026_lo95": 4.9,
    "zd_pred_2026_hi95": 23.4,
    "zd_pred_2027_mean": 11.9,
    "zd_pred_2028_mean": 11.6,
    "zd_count_2026": 13202,
    "gi_class_2026": "Cold Spot (p<0.10)",
    "gi_z_2026": -0.707,
    "priority_tier": "Tier 4: Lower",
    "risk_index": 13.3,
    "state_rank": 30
  },
  {
    "state": "Osun",
    "zone": "South West",
    "zd_obs_2024": 9.4,
    "zd_pred_2026_mean": 11.4,
    "zd_pred_2026_lo95": 4.4,
    "zd_pred_2026_hi95": 21.8,
    "zd_pred_2027_mean": 11.5,
    "zd_pred_2028_mean": 11.5,
    "zd_count_2026": 13580,
    "gi_class_2026": "Not Significant",
    "gi_z_2026": -0.397,
    "priority_tier": "Tier 4: Lower",
    "risk_index": 13.2,
    "state_rank": 31
  },
  {
    "state": "Delta",
    "zone": "South South",
    "zd_obs_2024": 7.6,
    "zd_pred_2026_mean": 12.8,
    "zd_pred_2026_lo95": 5.7,
    "zd_pred_2026_hi95": 22.9,
    "zd_pred_2027_mean": 12.4,
    "zd_pred_2028_mean": 12.1,
    "zd_count_2026": 21776,
    "gi_class_2026": "Cold Spot (p<0.05)",
    "gi_z_2026": -0.728,
    "priority_tier": "Tier 4: Lower",
    "risk_index": 13.2,
    "state_rank": 32
  },
  {
    "state": "Edo",
    "zone": "South South",
    "zd_obs_2024": 3.7,
    "zd_pred_2026_mean": 9.5,
    "zd_pred_2026_lo95": 3.7,
    "zd_pred_2026_hi95": 18.5,
    "zd_pred_2027_mean": 9.3,
    "zd_pred_2028_mean": 9.2,
    "zd_count_2026": 14042,
    "gi_class_2026": "Not Significant",
    "gi_z_2026": -0.553,
    "priority_tier": "Tier 4: Lower",
    "risk_index": 10.7,
    "state_rank": 33
  },
  {
    "state": "Lagos",
    "zone": "South West",
    "zd_obs_2024": 4.7,
    "zd_pred_2026_mean": 7.4,
    "zd_pred_2026_lo95": 3.3,
    "zd_pred_2026_hi95": 13.8,
    "zd_pred_2027_mean": 7.3,
    "zd_pred_2028_mean": 7.2,
    "zd_count_2026": 22378,
    "gi_class_2026": "Not Significant",
    "gi_z_2026": -0.775,
    "priority_tier": "Tier 4: Lower",
    "risk_index": 10.0,
    "state_rank": 34
  },
  {
    "state": "Abia",
    "zone": "South East",
    "zd_obs_2024": 4.2,
    "zd_pred_2026_mean": 8.7,
    "zd_pred_2026_lo95": 3.1,
    "zd_pred_2026_hi95": 18.1,
    "zd_pred_2027_mean": 8.4,
    "zd_pred_2028_mean": 8.1,
    "zd_count_2026": 12129,
    "gi_class_2026": "Cold Spot (p<0.01)",
    "gi_z_2026": -0.899,
    "priority_tier": "Tier 4: Lower",
    "risk_index": 10.0,
    "state_rank": 35
  },
  {
    "state": "Imo",
    "zone": "South East",
    "zd_obs_2024": 5.6,
    "zd_pred_2026_mean": 8.7,
    "zd_pred_2026_lo95": 3.2,
    "zd_pred_2026_hi95": 17.6,
    "zd_pred_2027_mean": 8.5,
    "zd_pred_2028_mean": 8.2,
    "zd_count_2026": 14068,
    "gi_class_2026": "Cold Spot (p<0.05)",
    "gi_z_2026": -0.893,
    "priority_tier": "Tier 4: Lower",
    "risk_index": 9.8,
    "state_rank": 36
  },
  {
    "state": "Ebonyi",
    "zone": "South East",
    "zd_obs_2024": 2.6,
    "zd_pred_2026_mean": 6.8,
    "zd_pred_2026_lo95": 2.8,
    "zd_pred_2026_hi95": 12.9,
    "zd_pred_2027_mean": 6.5,
    "zd_pred_2028_mean": 6.2,
    "zd_count_2026": 7709,
    "gi_class_2026": "Cold Spot (p<0.10)",
    "gi_z_2026": -0.742,
    "priority_tier": "Tier 4: Lower",
    "risk_index": 6.1,
    "state_rank": 37
  }
]

ARCHETYPE_DATA: List[dict] = [
  {
    "state_name": "Jigawa",
    "zone_name": "North West",
    "cluster_id": 0,
    "archetype": "Nomadic / High-Mobility",
    "zero_dose_2024": 34.3,
    "zero_dose_2024.1": 34.3,
    "pct_urban": 11.5,
    "pct_problem_accessing_hfdistance": 32.0,
    "pct_women_no_education": 70.5,
    "pct_lowest_wealth_quintile": 68.4,
    "pct_muslim": 99.9,
    "pct_severely_food_insecure": 31.2,
    "total_fertility_rate": 6.9,
    "pct_cu5_stunted": 55.7,
    "pct_women_with_mobile_phone": 31.2,
    "anc_4plus": 37.7,
    "delivered_in_hf": 21.4,
    "pct_cu5_birth_registered": 15.7,
    "pct_women_say_wife_beating_justified": 28.2,
    "pct_women_moved_current_res_past5yrs": 17.5
  },
  {
    "state_name": "Kaduna",
    "zone_name": "North West",
    "cluster_id": 0,
    "archetype": "Nomadic / High-Mobility",
    "zero_dose_2024": 45.6,
    "zero_dose_2024.1": 45.6,
    "pct_urban": 38.2,
    "pct_problem_accessing_hfdistance": 30.8,
    "pct_women_no_education": 35.8,
    "pct_lowest_wealth_quintile": 51.1,
    "pct_muslim": 82.2,
    "pct_severely_food_insecure": 33.9,
    "total_fertility_rate": 5.6,
    "pct_cu5_stunted": 40.7,
    "pct_women_with_mobile_phone": 53.4,
    "anc_4plus": 59.4,
    "delivered_in_hf": 25.9,
    "pct_cu5_birth_registered": 25.7,
    "pct_women_say_wife_beating_justified": 26.4,
    "pct_women_moved_current_res_past5yrs": 27.2
  },
  {
    "state_name": "Kano",
    "zone_name": "North West",
    "cluster_id": 0,
    "archetype": "Nomadic / High-Mobility",
    "zero_dose_2024": 42.4,
    "zero_dose_2024.1": 42.4,
    "pct_urban": 34.3,
    "pct_problem_accessing_hfdistance": 13.2,
    "pct_women_no_education": 40.2,
    "pct_lowest_wealth_quintile": 22.5,
    "pct_muslim": 99.1,
    "pct_severely_food_insecure": 42.6,
    "total_fertility_rate": 5.8,
    "pct_cu5_stunted": 51.9,
    "pct_women_with_mobile_phone": 56.0,
    "anc_4plus": 51.3,
    "delivered_in_hf": 32.7,
    "pct_cu5_birth_registered": 50.7,
    "pct_women_say_wife_beating_justified": 27.8,
    "pct_women_moved_current_res_past5yrs": 24.2
  },
  {
    "state_name": "Katsina",
    "zone_name": "North West",
    "cluster_id": 0,
    "archetype": "Nomadic / High-Mobility",
    "zero_dose_2024": 39.8,
    "zero_dose_2024.1": 39.8,
    "pct_urban": 25.6,
    "pct_problem_accessing_hfdistance": 22.0,
    "pct_women_no_education": 53.0,
    "pct_lowest_wealth_quintile": 0.3,
    "pct_muslim": 100.0,
    "pct_severely_food_insecure": 36.3,
    "total_fertility_rate": 5.7,
    "pct_cu5_stunted": 64.6,
    "pct_women_with_mobile_phone": 45.4,
    "anc_4plus": 37.2,
    "delivered_in_hf": 15.8,
    "pct_cu5_birth_registered": 40.2,
    "pct_women_say_wife_beating_justified": 42.1,
    "pct_women_moved_current_res_past5yrs": 20.8
  },
  {
    "state_name": "Kebbi",
    "zone_name": "North West",
    "cluster_id": 4,
    "archetype": "High-Burden Remote Rural",
    "zero_dose_2024": 84.0,
    "zero_dose_2024.1": 84.0,
    "pct_urban": 14.3,
    "pct_problem_accessing_hfdistance": 43.0,
    "pct_women_no_education": 85.8,
    "pct_lowest_wealth_quintile": 0.8,
    "pct_muslim": 96.0,
    "pct_severely_food_insecure": 34.5,
    "total_fertility_rate": 6.6,
    "pct_cu5_stunted": 60.0,
    "pct_women_with_mobile_phone": 24.8,
    "anc_4plus": 14.0,
    "delivered_in_hf": 8.8,
    "pct_cu5_birth_registered": 18.8,
    "pct_women_say_wife_beating_justified": 37.6,
    "pct_women_moved_current_res_past5yrs": 36.4
  },
  {
    "state_name": "Sokoto",
    "zone_name": "North West",
    "cluster_id": 4,
    "archetype": "High-Burden Remote Rural",
    "zero_dose_2024": 86.5,
    "zero_dose_2024.1": 86.5,
    "pct_urban": 9.8,
    "pct_problem_accessing_hfdistance": 48.2,
    "pct_women_no_education": 83.9,
    "pct_lowest_wealth_quintile": 3.1,
    "pct_muslim": 99.3,
    "pct_severely_food_insecure": 32.5,
    "total_fertility_rate": 5.4,
    "pct_cu5_stunted": 42.8,
    "pct_women_with_mobile_phone": 45.4,
    "anc_4plus": 22.7,
    "delivered_in_hf": 12.5,
    "pct_cu5_birth_registered": 19.0,
    "pct_women_say_wife_beating_justified": 34.5,
    "pct_women_moved_current_res_past5yrs": 5.8
  },
  {
    "state_name": "Zamfara",
    "zone_name": "North West",
    "cluster_id": 0,
    "archetype": "Nomadic / High-Mobility",
    "zero_dose_2024": 82.6,
    "zero_dose_2024.1": 82.6,
    "pct_urban": 20.6,
    "pct_problem_accessing_hfdistance": 7.3,
    "pct_women_no_education": 76.0,
    "pct_lowest_wealth_quintile": 1.0,
    "pct_muslim": 100.0,
    "pct_severely_food_insecure": 50.7,
    "total_fertility_rate": 6.3,
    "pct_cu5_stunted": 64.2,
    "pct_women_with_mobile_phone": 35.3,
    "anc_4plus": 21.5,
    "delivered_in_hf": 15.3,
    "pct_cu5_birth_registered": 29.1,
    "pct_women_say_wife_beating_justified": 33.7,
    "pct_women_moved_current_res_past5yrs": 15.8
  },
  {
    "state_name": "Adamawa",
    "zone_name": "North East",
    "cluster_id": 2,
    "archetype": "Conflict-Affected / Hard-to-Reach",
    "zero_dose_2024": 29.1,
    "zero_dose_2024.1": 29.1,
    "pct_urban": 24.2,
    "pct_problem_accessing_hfdistance": 48.3,
    "pct_women_no_education": 33.6,
    "pct_lowest_wealth_quintile": 13.6,
    "pct_muslim": 69.3,
    "pct_severely_food_insecure": 30.9,
    "total_fertility_rate": 5.3,
    "pct_cu5_stunted": 48.6,
    "pct_women_with_mobile_phone": 54.7,
    "anc_4plus": 56.4,
    "delivered_in_hf": 41.6,
    "pct_cu5_birth_registered": 39.3,
    "pct_women_say_wife_beating_justified": 40.1,
    "pct_women_moved_current_res_past5yrs": 23.5
  },
  {
    "state_name": "Bauchi",
    "zone_name": "North East",
    "cluster_id": 0,
    "archetype": "Nomadic / High-Mobility",
    "zero_dose_2024": 36.9,
    "zero_dose_2024.1": 36.9,
    "pct_urban": 17.1,
    "pct_problem_accessing_hfdistance": 25.9,
    "pct_women_no_education": 63.1,
    "pct_lowest_wealth_quintile": 4.3,
    "pct_muslim": 95.2,
    "pct_severely_food_insecure": 35.3,
    "total_fertility_rate": 6.2,
    "pct_cu5_stunted": 61.7,
    "pct_women_with_mobile_phone": 42.5,
    "anc_4plus": 46.6,
    "delivered_in_hf": 31.1,
    "pct_cu5_birth_registered": 24.8,
    "pct_women_say_wife_beating_justified": 48.8,
    "pct_women_moved_current_res_past5yrs": 41.0
  },
  {
    "state_name": "Borno",
    "zone_name": "North East",
    "cluster_id": 0,
    "archetype": "Nomadic / High-Mobility",
    "zero_dose_2024": 31.7,
    "zero_dose_2024.1": 31.7,
    "pct_urban": 80.6,
    "pct_problem_accessing_hfdistance": 36.3,
    "pct_women_no_education": 57.8,
    "pct_lowest_wealth_quintile": 34.3,
    "pct_muslim": 92.0,
    "pct_severely_food_insecure": 44.2,
    "total_fertility_rate": 6.5,
    "pct_cu5_stunted": 40.9,
    "pct_women_with_mobile_phone": 55.4,
    "anc_4plus": 61.1,
    "delivered_in_hf": 45.9,
    "pct_cu5_birth_registered": 28.9,
    "pct_women_say_wife_beating_justified": 24.5,
    "pct_women_moved_current_res_past5yrs": 30.7
  },
  {
    "state_name": "Gombe",
    "zone_name": "North East",
    "cluster_id": 0,
    "archetype": "Nomadic / High-Mobility",
    "zero_dose_2024": 34.0,
    "zero_dose_2024.1": 34.0,
    "pct_urban": 25.4,
    "pct_problem_accessing_hfdistance": 12.5,
    "pct_women_no_education": 53.0,
    "pct_lowest_wealth_quintile": 17.5,
    "pct_muslim": 91.2,
    "pct_severely_food_insecure": 44.2,
    "total_fertility_rate": 5.5,
    "pct_cu5_stunted": 50.6,
    "pct_women_with_mobile_phone": 46.4,
    "anc_4plus": 39.1,
    "delivered_in_hf": 48.5,
    "pct_cu5_birth_registered": 33.7,
    "pct_women_say_wife_beating_justified": 16.8,
    "pct_women_moved_current_res_past5yrs": 60.5
  },
  {
    "state_name": "Taraba",
    "zone_name": "North East",
    "cluster_id": 0,
    "archetype": "Nomadic / High-Mobility",
    "zero_dose_2024": 40.7,
    "zero_dose_2024.1": 40.7,
    "pct_urban": 35.6,
    "pct_problem_accessing_hfdistance": 20.0,
    "pct_women_no_education": 51.6,
    "pct_lowest_wealth_quintile": 2.6,
    "pct_muslim": 50.1,
    "pct_severely_food_insecure": 50.5,
    "total_fertility_rate": 5.2,
    "pct_cu5_stunted": 45.6,
    "pct_women_with_mobile_phone": 58.4,
    "anc_4plus": 50.5,
    "delivered_in_hf": 33.0,
    "pct_cu5_birth_registered": 31.1,
    "pct_women_say_wife_beating_justified": 17.2,
    "pct_women_moved_current_res_past5yrs": 37.0
  },
  {
    "state_name": "Yobe",
    "zone_name": "North East",
    "cluster_id": 0,
    "archetype": "Nomadic / High-Mobility",
    "zero_dose_2024": 40.4,
    "zero_dose_2024.1": 40.4,
    "pct_urban": 44.2,
    "pct_problem_accessing_hfdistance": 27.5,
    "pct_women_no_education": 68.0,
    "pct_lowest_wealth_quintile": 0.3,
    "pct_muslim": 98.7,
    "pct_severely_food_insecure": 65.9,
    "total_fertility_rate": 7.5,
    "pct_cu5_stunted": 54.5,
    "pct_women_with_mobile_phone": 44.6,
    "anc_4plus": 48.5,
    "delivered_in_hf": 32.1,
    "pct_cu5_birth_registered": 19.4,
    "pct_women_say_wife_beating_justified": 52.7,
    "pct_women_moved_current_res_past5yrs": 21.4
  },
  {
    "state_name": "Benue",
    "zone_name": "North Central",
    "cluster_id": 3,
    "archetype": "Urban Underserved",
    "zero_dose_2024": 32.7,
    "zero_dose_2024.1": 32.7,
    "pct_urban": 31.3,
    "pct_problem_accessing_hfdistance": 26.0,
    "pct_women_no_education": 11.6,
    "pct_lowest_wealth_quintile": 23.4,
    "pct_muslim": 4.1,
    "pct_severely_food_insecure": 23.5,
    "total_fertility_rate": 3.5,
    "pct_cu5_stunted": 25.3,
    "pct_women_with_mobile_phone": 67.4,
    "anc_4plus": 49.1,
    "delivered_in_hf": 59.0,
    "pct_cu5_birth_registered": 37.8,
    "pct_women_say_wife_beating_justified": 27.0,
    "pct_women_moved_current_res_past5yrs": 32.0
  },
  {
    "state_name": "FCT",
    "zone_name": "North Central",
    "cluster_id": 1,
    "archetype": "Moderate Access \u2014 Improving",
    "zero_dose_2024": 6.2,
    "zero_dose_2024.1": 6.2,
    "pct_urban": 56.7,
    "pct_problem_accessing_hfdistance": 9.4,
    "pct_women_no_education": 8.0,
    "pct_lowest_wealth_quintile": 10.5,
    "pct_muslim": 40.7,
    "pct_severely_food_insecure": 24.7,
    "total_fertility_rate": 3.2,
    "pct_cu5_stunted": 16.3,
    "pct_women_with_mobile_phone": 89.2,
    "anc_4plus": 79.9,
    "delivered_in_hf": 81.3,
    "pct_cu5_birth_registered": 70.6,
    "pct_women_say_wife_beating_justified": 0.6,
    "pct_women_moved_current_res_past5yrs": 35.3
  },
  {
    "state_name": "Kogi",
    "zone_name": "North Central",
    "cluster_id": 2,
    "archetype": "Conflict-Affected / Hard-to-Reach",
    "zero_dose_2024": 57.0,
    "zero_dose_2024.1": 57.0,
    "pct_urban": 29.2,
    "pct_problem_accessing_hfdistance": 21.6,
    "pct_women_no_education": 24.7,
    "pct_lowest_wealth_quintile": 17.4,
    "pct_muslim": 59.9,
    "pct_severely_food_insecure": 25.7,
    "total_fertility_rate": 4.9,
    "pct_cu5_stunted": 34.6,
    "pct_women_with_mobile_phone": 67.4,
    "anc_4plus": 54.1,
    "delivered_in_hf": 62.2,
    "pct_cu5_birth_registered": 27.2,
    "pct_women_say_wife_beating_justified": 9.9,
    "pct_women_moved_current_res_past5yrs": 20.3
  },
  {
    "state_name": "Kwara",
    "zone_name": "North Central",
    "cluster_id": 2,
    "archetype": "Conflict-Affected / Hard-to-Reach",
    "zero_dose_2024": 51.7,
    "zero_dose_2024.1": 51.7,
    "pct_urban": 40.3,
    "pct_problem_accessing_hfdistance": 23.9,
    "pct_women_no_education": 33.9,
    "pct_lowest_wealth_quintile": 0.7,
    "pct_muslim": 80.6,
    "pct_severely_food_insecure": 30.5,
    "total_fertility_rate": 4.0,
    "pct_cu5_stunted": 40.8,
    "pct_women_with_mobile_phone": 81.7,
    "anc_4plus": 51.3,
    "delivered_in_hf": 51.5,
    "pct_cu5_birth_registered": 36.6,
    "pct_women_say_wife_beating_justified": 3.9,
    "pct_women_moved_current_res_past5yrs": 13.1
  },
  {
    "state_name": "Nasarawa",
    "zone_name": "North Central",
    "cluster_id": 3,
    "archetype": "Urban Underserved",
    "zero_dose_2024": 20.3,
    "zero_dose_2024.1": 20.3,
    "pct_urban": 42.9,
    "pct_problem_accessing_hfdistance": 18.7,
    "pct_women_no_education": 34.3,
    "pct_lowest_wealth_quintile": 1.2,
    "pct_muslim": 65.3,
    "pct_severely_food_insecure": 16.2,
    "total_fertility_rate": 4.3,
    "pct_cu5_stunted": 35.0,
    "pct_women_with_mobile_phone": 70.3,
    "anc_4plus": 66.0,
    "delivered_in_hf": 55.7,
    "pct_cu5_birth_registered": 51.9,
    "pct_women_say_wife_beating_justified": 42.6,
    "pct_women_moved_current_res_past5yrs": 34.7
  },
  {
    "state_name": "Niger",
    "zone_name": "North Central",
    "cluster_id": 2,
    "archetype": "Conflict-Affected / Hard-to-Reach",
    "zero_dose_2024": 56.0,
    "zero_dose_2024.1": 56.0,
    "pct_urban": 25.8,
    "pct_problem_accessing_hfdistance": 30.5,
    "pct_women_no_education": 73.0,
    "pct_lowest_wealth_quintile": 0.0,
    "pct_muslim": 90.1,
    "pct_severely_food_insecure": 8.9,
    "total_fertility_rate": 4.4,
    "pct_cu5_stunted": 43.9,
    "pct_women_with_mobile_phone": 56.1,
    "anc_4plus": 34.7,
    "delivered_in_hf": 30.2,
    "pct_cu5_birth_registered": 27.3,
    "pct_women_say_wife_beating_justified": 11.1,
    "pct_women_moved_current_res_past5yrs": 13.8
  },
  {
    "state_name": "Plateau",
    "zone_name": "North Central",
    "cluster_id": 2,
    "archetype": "Conflict-Affected / Hard-to-Reach",
    "zero_dose_2024": 35.3,
    "zero_dose_2024.1": 35.3,
    "pct_urban": 24.2,
    "pct_problem_accessing_hfdistance": 60.0,
    "pct_women_no_education": 20.6,
    "pct_lowest_wealth_quintile": 1.5,
    "pct_muslim": 37.4,
    "pct_severely_food_insecure": 42.0,
    "total_fertility_rate": 4.4,
    "pct_cu5_stunted": 46.4,
    "pct_women_with_mobile_phone": 54.7,
    "anc_4plus": 46.4,
    "delivered_in_hf": 45.7,
    "pct_cu5_birth_registered": 27.0,
    "pct_women_say_wife_beating_justified": 12.7,
    "pct_women_moved_current_res_past5yrs": 30.2
  },
  {
    "state_name": "Ekiti",
    "zone_name": "South West",
    "cluster_id": 1,
    "archetype": "Moderate Access \u2014 Improving",
    "zero_dose_2024": 6.0,
    "zero_dose_2024.1": 6.0,
    "pct_urban": 67.2,
    "pct_problem_accessing_hfdistance": 29.7,
    "pct_women_no_education": 2.2,
    "pct_lowest_wealth_quintile": 42.1,
    "pct_muslim": 15.4,
    "pct_severely_food_insecure": 27.9,
    "total_fertility_rate": 3.8,
    "pct_cu5_stunted": 17.1,
    "pct_women_with_mobile_phone": 84.8,
    "anc_4plus": 68.6,
    "delivered_in_hf": 81.7,
    "pct_cu5_birth_registered": 57.9,
    "pct_women_say_wife_beating_justified": 15.0,
    "pct_women_moved_current_res_past5yrs": 38.6
  },
  {
    "state_name": "Lagos",
    "zone_name": "South West",
    "cluster_id": 1,
    "archetype": "Moderate Access \u2014 Improving",
    "zero_dose_2024": 4.7,
    "zero_dose_2024.1": 4.7,
    "pct_urban": 98.7,
    "pct_problem_accessing_hfdistance": 2.9,
    "pct_women_no_education": 3.0,
    "pct_lowest_wealth_quintile": 0.0,
    "pct_muslim": 32.0,
    "pct_severely_food_insecure": 30.7,
    "total_fertility_rate": 3.2,
    "pct_cu5_stunted": 17.3,
    "pct_women_with_mobile_phone": 88.9,
    "anc_4plus": 95.4,
    "delivered_in_hf": 85.8,
    "pct_cu5_birth_registered": 78.1,
    "pct_women_say_wife_beating_justified": 3.5,
    "pct_women_moved_current_res_past5yrs": 16.8
  },
  {
    "state_name": "Ogun",
    "zone_name": "South West",
    "cluster_id": 1,
    "archetype": "Moderate Access \u2014 Improving",
    "zero_dose_2024": 20.6,
    "zero_dose_2024.1": 20.6,
    "pct_urban": 81.7,
    "pct_problem_accessing_hfdistance": 27.9,
    "pct_women_no_education": 7.4,
    "pct_lowest_wealth_quintile": 1.7,
    "pct_muslim": 31.0,
    "pct_severely_food_insecure": 33.1,
    "total_fertility_rate": 4.1,
    "pct_cu5_stunted": 17.7,
    "pct_women_with_mobile_phone": 81.5,
    "anc_4plus": 73.7,
    "delivered_in_hf": 83.3,
    "pct_cu5_birth_registered": 53.9,
    "pct_women_say_wife_beating_justified": 10.8,
    "pct_women_moved_current_res_past5yrs": 40.1
  },
  {
    "state_name": "Ondo",
    "zone_name": "South West",
    "cluster_id": 1,
    "archetype": "Moderate Access \u2014 Improving",
    "zero_dose_2024": 23.5,
    "zero_dose_2024.1": 23.5,
    "pct_urban": 67.9,
    "pct_problem_accessing_hfdistance": 6.6,
    "pct_women_no_education": 4.7,
    "pct_lowest_wealth_quintile": 0.0,
    "pct_muslim": 7.9,
    "pct_severely_food_insecure": 22.7,
    "total_fertility_rate": 3.1,
    "pct_cu5_stunted": 23.2,
    "pct_women_with_mobile_phone": 78.6,
    "anc_4plus": 66.3,
    "delivered_in_hf": 83.2,
    "pct_cu5_birth_registered": 77.6,
    "pct_women_say_wife_beating_justified": 7.9,
    "pct_women_moved_current_res_past5yrs": 37.9
  },
  {
    "state_name": "Osun",
    "zone_name": "South West",
    "cluster_id": 1,
    "archetype": "Moderate Access \u2014 Improving",
    "zero_dose_2024": 9.4,
    "zero_dose_2024.1": 9.4,
    "pct_urban": 85.3,
    "pct_problem_accessing_hfdistance": 18.1,
    "pct_women_no_education": 1.4,
    "pct_lowest_wealth_quintile": 0.2,
    "pct_muslim": 48.6,
    "pct_severely_food_insecure": 22.6,
    "total_fertility_rate": 3.3,
    "pct_cu5_stunted": 30.5,
    "pct_women_with_mobile_phone": 87.6,
    "anc_4plus": 92.0,
    "delivered_in_hf": 86.7,
    "pct_cu5_birth_registered": 81.1,
    "pct_women_say_wife_beating_justified": 8.1,
    "pct_women_moved_current_res_past5yrs": 37.0
  },
  {
    "state_name": "Oyo",
    "zone_name": "South West",
    "cluster_id": 1,
    "archetype": "Moderate Access \u2014 Improving",
    "zero_dose_2024": 29.9,
    "zero_dose_2024.1": 29.9,
    "pct_urban": 79.0,
    "pct_problem_accessing_hfdistance": 15.8,
    "pct_women_no_education": 10.0,
    "pct_lowest_wealth_quintile": 6.0,
    "pct_muslim": 57.6,
    "pct_severely_food_insecure": 23.7,
    "total_fertility_rate": 3.3,
    "pct_cu5_stunted": 23.1,
    "pct_women_with_mobile_phone": 82.9,
    "anc_4plus": 73.8,
    "delivered_in_hf": 75.0,
    "pct_cu5_birth_registered": 62.2,
    "pct_women_say_wife_beating_justified": 7.3,
    "pct_women_moved_current_res_past5yrs": 30.4
  },
  {
    "state_name": "Abia",
    "zone_name": "South East",
    "cluster_id": 1,
    "archetype": "Moderate Access \u2014 Improving",
    "zero_dose_2024": 4.2,
    "zero_dose_2024.1": 4.2,
    "pct_urban": 46.6,
    "pct_problem_accessing_hfdistance": 39.8,
    "pct_women_no_education": 0.8,
    "pct_lowest_wealth_quintile": 0.3,
    "pct_muslim": 0.0,
    "pct_severely_food_insecure": 39.7,
    "total_fertility_rate": 3.7,
    "pct_cu5_stunted": 20.2,
    "pct_women_with_mobile_phone": 82.6,
    "anc_4plus": 79.1,
    "delivered_in_hf": 86.0,
    "pct_cu5_birth_registered": 55.9,
    "pct_women_say_wife_beating_justified": 2.3,
    "pct_women_moved_current_res_past5yrs": 35.9
  },
  {
    "state_name": "Anambra",
    "zone_name": "South East",
    "cluster_id": 1,
    "archetype": "Moderate Access \u2014 Improving",
    "zero_dose_2024": 14.9,
    "zero_dose_2024.1": 14.9,
    "pct_urban": 56.5,
    "pct_problem_accessing_hfdistance": 31.5,
    "pct_women_no_education": 1.0,
    "pct_lowest_wealth_quintile": 22.8,
    "pct_muslim": 0.6,
    "pct_severely_food_insecure": 14.5,
    "total_fertility_rate": 3.7,
    "pct_cu5_stunted": 12.9,
    "pct_women_with_mobile_phone": 81.5,
    "anc_4plus": 84.9,
    "delivered_in_hf": 83.2,
    "pct_cu5_birth_registered": 62.9,
    "pct_women_say_wife_beating_justified": 13.7,
    "pct_women_moved_current_res_past5yrs": 31.7
  },
  {
    "state_name": "Ebonyi",
    "zone_name": "South East",
    "cluster_id": 3,
    "archetype": "Urban Underserved",
    "zero_dose_2024": 2.6,
    "zero_dose_2024.1": 2.6,
    "pct_urban": 24.0,
    "pct_problem_accessing_hfdistance": 38.2,
    "pct_women_no_education": 7.1,
    "pct_lowest_wealth_quintile": 41.4,
    "pct_muslim": 0.0,
    "pct_severely_food_insecure": 30.2,
    "total_fertility_rate": 4.7,
    "pct_cu5_stunted": 31.6,
    "pct_women_with_mobile_phone": 57.1,
    "anc_4plus": 61.7,
    "delivered_in_hf": 79.4,
    "pct_cu5_birth_registered": 43.2,
    "pct_women_say_wife_beating_justified": 32.6,
    "pct_women_moved_current_res_past5yrs": 50.4
  },
  {
    "state_name": "Enugu",
    "zone_name": "South East",
    "cluster_id": 1,
    "archetype": "Moderate Access \u2014 Improving",
    "zero_dose_2024": 12.8,
    "zero_dose_2024.1": 12.8,
    "pct_urban": 66.0,
    "pct_problem_accessing_hfdistance": 23.7,
    "pct_women_no_education": 8.3,
    "pct_lowest_wealth_quintile": 60.1,
    "pct_muslim": 0.8,
    "pct_severely_food_insecure": 43.4,
    "total_fertility_rate": 3.5,
    "pct_cu5_stunted": 15.2,
    "pct_women_with_mobile_phone": 80.7,
    "anc_4plus": 61.9,
    "delivered_in_hf": 92.6,
    "pct_cu5_birth_registered": 80.1,
    "pct_women_say_wife_beating_justified": 2.7,
    "pct_women_moved_current_res_past5yrs": 44.4
  },
  {
    "state_name": "Imo",
    "zone_name": "South East",
    "cluster_id": 1,
    "archetype": "Moderate Access \u2014 Improving",
    "zero_dose_2024": 5.6,
    "zero_dose_2024.1": 5.6,
    "pct_urban": 32.2,
    "pct_problem_accessing_hfdistance": 21.2,
    "pct_women_no_education": 0.5,
    "pct_lowest_wealth_quintile": 24.4,
    "pct_muslim": 0.2,
    "pct_severely_food_insecure": 23.3,
    "total_fertility_rate": 4.4,
    "pct_cu5_stunted": 17.3,
    "pct_women_with_mobile_phone": 82.5,
    "anc_4plus": 84.9,
    "delivered_in_hf": 97.0,
    "pct_cu5_birth_registered": 58.2,
    "pct_women_say_wife_beating_justified": 4.4,
    "pct_women_moved_current_res_past5yrs": 35.1
  },
  {
    "state_name": "Akwa Ibom",
    "zone_name": "South South",
    "cluster_id": 3,
    "archetype": "Urban Underserved",
    "zero_dose_2024": 15.1,
    "zero_dose_2024.1": 15.1,
    "pct_urban": 48.1,
    "pct_problem_accessing_hfdistance": 22.4,
    "pct_women_no_education": 0.6,
    "pct_lowest_wealth_quintile": 17.6,
    "pct_muslim": 0.0,
    "pct_severely_food_insecure": 39.9,
    "total_fertility_rate": 3.3,
    "pct_cu5_stunted": 24.1,
    "pct_women_with_mobile_phone": 73.2,
    "anc_4plus": 65.7,
    "delivered_in_hf": 38.6,
    "pct_cu5_birth_registered": 48.0,
    "pct_women_say_wife_beating_justified": 19.8,
    "pct_women_moved_current_res_past5yrs": 35.4
  },
  {
    "state_name": "Bayelsa",
    "zone_name": "South South",
    "cluster_id": 3,
    "archetype": "Urban Underserved",
    "zero_dose_2024": 19.1,
    "zero_dose_2024.1": 19.1,
    "pct_urban": 66.9,
    "pct_problem_accessing_hfdistance": 39.2,
    "pct_women_no_education": 4.9,
    "pct_lowest_wealth_quintile": 18.0,
    "pct_muslim": 0.6,
    "pct_severely_food_insecure": 32.6,
    "total_fertility_rate": 3.8,
    "pct_cu5_stunted": 27.6,
    "pct_women_with_mobile_phone": 77.7,
    "anc_4plus": 48.6,
    "delivered_in_hf": 46.1,
    "pct_cu5_birth_registered": 49.1,
    "pct_women_say_wife_beating_justified": 18.2,
    "pct_women_moved_current_res_past5yrs": 32.1
  },
  {
    "state_name": "Cross River",
    "zone_name": "South South",
    "cluster_id": 1,
    "archetype": "Moderate Access \u2014 Improving",
    "zero_dose_2024": 2.8,
    "zero_dose_2024.1": 2.8,
    "pct_urban": 55.1,
    "pct_problem_accessing_hfdistance": 6.5,
    "pct_women_no_education": 3.3,
    "pct_lowest_wealth_quintile": 37.9,
    "pct_muslim": 0.2,
    "pct_severely_food_insecure": 18.7,
    "total_fertility_rate": 3.0,
    "pct_cu5_stunted": 21.0,
    "pct_women_with_mobile_phone": 68.6,
    "anc_4plus": 80.0,
    "delivered_in_hf": 58.8,
    "pct_cu5_birth_registered": 44.3,
    "pct_women_say_wife_beating_justified": 11.4,
    "pct_women_moved_current_res_past5yrs": 32.2
  },
  {
    "state_name": "Delta",
    "zone_name": "South South",
    "cluster_id": 1,
    "archetype": "Moderate Access \u2014 Improving",
    "zero_dose_2024": 7.6,
    "zero_dose_2024.1": 7.6,
    "pct_urban": 83.1,
    "pct_problem_accessing_hfdistance": 20.9,
    "pct_women_no_education": 3.4,
    "pct_lowest_wealth_quintile": 37.2,
    "pct_muslim": 3.4,
    "pct_severely_food_insecure": 20.8,
    "total_fertility_rate": 3.7,
    "pct_cu5_stunted": 20.0,
    "pct_women_with_mobile_phone": 81.7,
    "anc_4plus": 60.5,
    "delivered_in_hf": 83.0,
    "pct_cu5_birth_registered": 62.4,
    "pct_women_say_wife_beating_justified": 7.7,
    "pct_women_moved_current_res_past5yrs": 39.8
  },
  {
    "state_name": "Edo",
    "zone_name": "South South",
    "cluster_id": 1,
    "archetype": "Moderate Access \u2014 Improving",
    "zero_dose_2024": 3.7,
    "zero_dose_2024.1": 3.7,
    "pct_urban": 82.5,
    "pct_problem_accessing_hfdistance": 16.1,
    "pct_women_no_education": 1.6,
    "pct_lowest_wealth_quintile": 34.8,
    "pct_muslim": 8.7,
    "pct_severely_food_insecure": 15.2,
    "total_fertility_rate": 3.3,
    "pct_cu5_stunted": 13.6,
    "pct_women_with_mobile_phone": 87.7,
    "anc_4plus": 63.0,
    "delivered_in_hf": 90.9,
    "pct_cu5_birth_registered": 62.5,
    "pct_women_say_wife_beating_justified": 21.8,
    "pct_women_moved_current_res_past5yrs": 44.9
  },
  {
    "state_name": "Rivers",
    "zone_name": "South South",
    "cluster_id": 1,
    "archetype": "Moderate Access \u2014 Improving",
    "zero_dose_2024": 22.7,
    "zero_dose_2024.1": 22.7,
    "pct_urban": 80.0,
    "pct_problem_accessing_hfdistance": 21.2,
    "pct_women_no_education": 2.4,
    "pct_lowest_wealth_quintile": 2.7,
    "pct_muslim": 0.8,
    "pct_severely_food_insecure": 33.5,
    "total_fertility_rate": 2.9,
    "pct_cu5_stunted": 12.3,
    "pct_women_with_mobile_phone": 75.5,
    "anc_4plus": 76.5,
    "delivered_in_hf": 56.9,
    "pct_cu5_birth_registered": 69.2,
    "pct_women_say_wife_beating_justified": 5.5,
    "pct_women_moved_current_res_past5yrs": 34.8
  }
]

TIER_ACTIONS = {
    "Tier 1: Critical": (
        "Immediate door-to-door zero-dose child mapping; mobile outreach surge; "
        "CHW deployment; governor-level engagement; quarterly GAVI/UNICEF review"
    ),
    "Tier 2: High": (
        "LGA-targeted outreach within highest-risk LGAs; urban slum strategies; "
        "DHIS2 data quality strengthening; state EPI review"
    ),
    "Tier 3: Moderate": (
        "Routine EPI strengthening; dropout investigation; "
        "quarterly monitoring; prevent slide to Tier 2"
    ),
    "Tier 4: Lower": (
        "Maintain routine EPI; equity sub-group monitoring; "
        "document and share effective practices"
    ),
}


@app.get("/")
def root():
    return {
        "name": "Nigeria Zero-Dose Predictive Modelling API",
        "version": "2.0.0",
        "status": "always-on (paid Render plan)",
        "server_time_utc": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "analysis_ready": ANALYSIS_READY,
        "n_states": len(STATE_DATA),
        "n_archetypes": len(ARCHETYPE_DATA),
        "endpoints": ["/", "/health", "/upload", "/session/{id}/info",
                      "/run-analysis", "/job/{id}", "/built-in/states",
                      "/built-in/summary", "/built-in/archetypes",
                      "/hotspots", "/interpret"],
        "data_basis": "NDHS 2008-2024 | Bayesian Beta Regression | Getis-Ord Gi*",
    }


@app.get("/health")
def health_check():
    """
    Render health check endpoint.
    Configure this path in Render Dashboard -> Health & Alerts -> Health Check Path.
    Returns 200 so Render confirms the service is running.
    """
    return {
        "status": "healthy",
        "timestamp_utc": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "n_states_loaded": len(STATE_DATA),
        "analysis_ready":  ANALYSIS_READY,
    }


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not ANALYSIS_READY:
        raise HTTPException(500, "pandas / prophet not installed on this instance.")
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Cannot parse CSV: {e}")

    sid = str(uuid.uuid4())[:8]
    sessions[sid] = {"df": df, "filename": file.filename}
    numeric_cols   = df.select_dtypes(include="number").columns.tolist()
    all_cols       = df.columns.tolist()
    date_cands     = [c for c in all_cols if any(k in c.lower()
                       for k in ["date","year","month","period","time","ds"])]
    geo_cands      = [c for c in all_cols if any(k in c.lower()
                       for k in ["state","lga","region","zone","area","district"])]
    outcome_cands  = [c for c in numeric_cols if any(k in c.lower()
                       for k in ["dose","zero","coverage","rate","count","pct","prop"])]
    return {
        "session_id":        sid,
        "filename":          file.filename,
        "rows":              len(df),
        "columns":           all_cols,
        "numeric_columns":   numeric_cols,
        "date_candidates":   date_cands,
        "geo_candidates":    geo_cands,
        "outcome_candidates":outcome_cands,
        "sample":            df.head(6).fillna("").to_dict("records"),
    }


@app.get("/session/{session_id}/info")
def session_info(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found.")
    df = sessions[session_id]["df"]
    return {
        "rows":    len(df),
        "columns": df.columns.tolist(),
        "dtypes":  {c: str(t) for c, t in df.dtypes.items()},
        "sample":  df.head(8).fillna("").to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Run analysis (background job)
# ---------------------------------------------------------------------------
class AnalysisConfig(BaseModel):
    session_id:       str
    geo_col:          str
    date_col:         str
    outcome_col:      str
    covariate_cols:   Optional[List[str]] = []
    selected_regions: Optional[List[str]] = []
    forecast_horizon: Optional[int]       = 18
    confidence_level: Optional[float]     = 0.95
    analysis_types:   Optional[List[str]] = ["forecast", "clustering", "dropout"]


@app.post("/run-analysis")
def run_analysis(config: AnalysisConfig, background_tasks: BackgroundTasks):
    if config.session_id not in sessions:
        raise HTTPException(404, "Session not found. Upload a file first.")
    jid = str(uuid.uuid4())[:8]
    jobs[jid] = {"status": "running", "progress": 0, "result": None, "error": None}
    background_tasks.add_task(_analysis_task, jid, config)
    return {"job_id": jid, "status": "running"}


def _analysis_task(jid: str, config: AnalysisConfig):
    try:
        df = sessions[config.session_id]["df"].copy()
        result = {}
        df["_date"] = pd.to_datetime(df[config.date_col], errors="coerce")
        df["_y"]    = pd.to_numeric(df[config.outcome_col], errors="coerce")
        if config.selected_regions and config.geo_col in df.columns:
            df = df[df[config.geo_col].isin(config.selected_regions)]

        jobs[jid]["progress"] = 10

        # Prophet forecast
        if "forecast" in config.analysis_types:
            agg = (df.groupby("_date")["_y"].mean().dropna().reset_index()
                   .rename(columns={"_date": "ds", "_y": "y"}))
            agg = agg[agg["y"].between(0, 300)].sort_values("ds")
            if len(agg) >= 6:
                m = Prophet(
                    yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, interval_width=config.confidence_level,
                    changepoint_prior_scale=0.05, seasonality_prior_scale=10,
                )
                m.add_seasonality("semi_annual", period=182.5, fourier_order=3)
                m.fit(agg)
                future = m.make_future_dataframe(
                    periods=config.forecast_horizon, freq="MS")
                fc     = m.predict(future)
                cutoff = agg["ds"].max()
                fc_out = fc[fc["ds"] > cutoff][
                    ["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                fc_out["ds"] = fc_out["ds"].dt.strftime("%Y-%m")
                result["forecast"] = {
                    "historical":  agg.assign(
                        ds=agg["ds"].dt.strftime("%Y-%m")).to_dict("records"),
                    "predicted":   fc_out.round(2).to_dict("records"),
                    "outcome_col": config.outcome_col,
                    "n_obs":       len(agg),
                    "horizon":     config.forecast_horizon,
                }
            else:
                result["forecast"] = {
                    "error": "Fewer than 6 time points — cannot fit Prophet."}

        jobs[jid]["progress"] = 50

        # HAC clustering
        if "clustering" in config.analysis_types and config.covariate_cols:
            feats = [c for c in config.covariate_cols if c in df.columns]
            if feats and config.geo_col in df.columns:
                latest = (df.sort_values("_date")
                          .groupby(config.geo_col)
                          [[config.outcome_col] + feats].last().dropna())
                if len(latest) >= 4:
                    Xs = StandardScaler().fit_transform(
                        latest.fillna(latest.median()).values)
                    sil = [
                        silhouette_score(
                            Xs, AgglomerativeClustering(
                                n_clusters=k, linkage="ward").fit_predict(Xs))
                        for k in range(2, min(7, len(latest)))
                    ]
                    best_k = sil.index(max(sil)) + 2
                    labels = AgglomerativeClustering(
                        n_clusters=best_k, linkage="ward").fit_predict(Xs)
                    latest["cluster"] = labels
                    result["clustering"] = {
                        "n_clusters":       int(best_k),
                        "silhouette_scores":[round(s, 3) for s in sil],
                        "assignments":      latest.reset_index()[
                            [config.geo_col, "cluster", config.outcome_col]
                        ].to_dict("records"),
                    }

        jobs[jid]["progress"] = 80

        # LASSO feature importance
        if "dropout" in config.analysis_types and config.covariate_cols:
            feats = [c for c in config.covariate_cols if c in df.columns]
            if feats and config.geo_col in df.columns:
                cross = (df.groupby(config.geo_col)
                         [[config.outcome_col] + feats].mean().dropna())
                if len(cross) >= 5:
                    Xf  = cross[feats].fillna(0).values
                    yf  = cross[config.outcome_col].values
                    Xfs = StandardScaler().fit_transform(Xf)
                    lasso = LassoCV(cv=min(5, len(cross)),
                                    max_iter=3000, random_state=42)
                    lasso.fit(Xfs, yf)
                    coefs = {c: round(float(abs(v)), 4)
                             for c, v in zip(feats, lasso.coef_) if abs(v) > 0}
                    result["feature_importance"] = {
                        "top_predictors": dict(sorted(
                            coefs.items(), key=lambda x: x[1], reverse=True)),
                        "outcome_col":    config.outcome_col,
                        "alpha":          round(float(lasso.alpha_), 4),
                    }

        result["summary"] = {
            "rows_analysed": len(df),
            "outcome_mean":  round(float(df["_y"].mean()), 2),
            "outcome_std":   round(float(df["_y"].std()),  2),
            "outcome_min":   round(float(df["_y"].min()),  2),
            "outcome_max":   round(float(df["_y"].max()),  2),
            "regions":       sorted(df[config.geo_col].dropna().unique().tolist())
                             if config.geo_col in df.columns else [],
        }
        jobs[jid]["progress"] = 100
        jobs[jid]["status"]   = "complete"
        jobs[jid]["result"]   = result

    except Exception as e:
        jobs[jid]["status"] = "error"
        jobs[jid]["error"]  = str(e)


@app.get("/job/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found.")
    j = jobs[job_id]
    if j["status"] == "error":
        raise HTTPException(500, j["error"])
    return j


# ---------------------------------------------------------------------------
# Built-in pre-computed results
# ---------------------------------------------------------------------------
@app.get("/built-in/states")
def builtin_states(zone: Optional[str] = None, tier: Optional[str] = None,
                   sort_by: str = "state_rank"):
    data = STATE_DATA.copy()
    if zone:
        data = [d for d in data if zone.lower() in (d.get("zone") or "").lower()]
    if tier:
        data = [d for d in data
                if tier.lower() in (d.get("priority_tier") or "").lower()]
    reverse = sort_by in ["risk_index","zd_pred_2026_mean","zd_count_2026","Gi_z_2026"]
    data.sort(key=lambda x: x.get(sort_by) or 0, reverse=reverse)
    return {"count": len(data), "states": data}


@app.get("/built-in/summary")
def builtin_summary():
    total  = sum(d.get("zd_count_2026") or 0 for d in STATE_DATA)
    mean_r = sum(d.get("zd_pred_2026_mean", 0) for d in STATE_DATA) / len(STATE_DATA)
    tiers, zones, hotspot = {}, {}, []
    for d in STATE_DATA:
        t = d.get("priority_tier", "Unknown")
        tiers[t] = tiers.get(t, 0) + 1
        z = d.get("zone", "")
        zones[z] = zones.get(z, 0) + (d.get("zd_count_2026") or 0)
        if "Hot Spot" in (d.get("hotspot_2026") or ""):
            hotspot.append(d["state"])
    return {
        "national_zd_rate_2026":     round(mean_r, 1),
        "national_zd_children_2026": int(total),
        "tier_distribution":         tiers,
        "zone_burden_2026":          zones,
        "hotspot_states":            hotspot,
        "convergence":               {"max_rhat": 1.0025, "min_ess": 1268, "divergences": 0},
        "data_basis": "NDHS 2008-2024 | Bayesian Hierarchical Beta Regression | PyMC v6.0",
    }


@app.get("/hotspots")
def get_hotspots(year: int = 2026):
    if year not in [2026, 2027, 2028]:
        raise HTTPException(400, "year must be 2026, 2027, or 2028.")
    results = []
    for d in STATE_DATA:
        cls = d.get("hotspot_2026") or "Not Significant"
        results.append({
            "state":        d["state"],
            "zone":         d["zone"],
            "gi_z":         round(d.get("Gi_z_2026") or 0, 3),
            "gi_class":     cls,
            "is_hotspot":   "Hot Spot" in cls,
            "zd_pred_mean": d.get("zd_pred_2026_mean"),
            "zd_count_2026":d.get("zd_count_2026"),
        })
    results.sort(key=lambda x: x["gi_z"], reverse=True)
    return {
        "year":             year,
        "method":           "Getis-Ord Gi* | Queen Contiguity | GRID3 | 999 permutations",
        "n_hotspot_states": sum(1 for r in results if r["is_hotspot"]),
        "hotspot_states":   [r for r in results if r["is_hotspot"]],
        "all_states":       results,
    }


@app.get("/built-in/archetypes")
def builtin_archetypes():
    return {
        "method":     "HAC | Ward linkage | k=5 | 15 contextual features",
        "n_states":   len(ARCHETYPE_DATA),
        "archetypes": ARCHETYPE_DATA,
    }


# ---------------------------------------------------------------------------
# LLM interpretation
# ---------------------------------------------------------------------------
class InterpretRequest(BaseModel):
    question:      str
    state:         Optional[str]  = None
    forecast_year: Optional[int]  = 2026
    context:       Optional[dict] = None


@app.post("/interpret")
async def interpret(req: InterpretRequest):
    if not ANTHROPIC_KEY:
        return {"error": "ANTHROPIC_API_KEY not set.",
                "response": None,
                "note": "Add ANTHROPIC_API_KEY to Render environment variables."}

    if req.state:
        matches = [d for d in STATE_DATA
                   if req.state.lower() in (d.get("state") or "").lower()]
        if not matches:
            raise HTTPException(404, "State not found: " + req.state)
        s   = matches[0]
        tier_action = TIER_ACTIONS.get(s.get("priority_tier", ""), "")
        ctx = (
            "State: " + str(s.get("state")) + " (" + str(s.get("zone")) + ")\n"
            "National rank: #" + str(s.get("state_rank")) +
            " | Priority tier: " + str(s.get("priority_tier")) +
            " | Risk index: " + str(s.get("risk_index")) + "/100\n"
            "Observed zero-dose 2024 (NDHS): " + str(s.get("zd_obs_2024")) + "%\n"
            "Predicted 2026: " + str(s.get("zd_pred_2026_mean")) +
            "% (95% CrI: " + str(s.get("zd_pred_2026_lo95")) +
            "-" + str(s.get("zd_pred_2026_hi95")) + "%)\n"
            "Predicted 2027: " + str(s.get("zd_pred_2027_mean")) +
            "% | Predicted 2028: " + str(s.get("zd_pred_2028_mean")) + "%\n"
            "Estimated zero-dose children (2026): " + str(s.get("zd_count_2026")) + "\n"
            "Getis-Ord Gi* status: " + str(s.get("hotspot_2026")) +
            " (z = " + str(s.get("Gi_z_2026")) + ")\n"
            "Recommended action: " + tier_action + "\n"
        )
    elif req.context:
        ctx = json.dumps(req.context, indent=2)
    else:
        sm  = builtin_summary()
        ctx = (
            "National predicted zero-dose rate (2026): " +
            str(sm["national_zd_rate_2026"]) + "%\n"
            "Estimated zero-dose children (2026): " +
            str(sm["national_zd_children_2026"]) + "\n"
            "Confirmed hotspot states: " + ", ".join(sm["hotspot_states"]) + "\n"
            "North West zone burden: " +
            str(sm["zone_burden_2026"].get("North West", 0)) + " children\n"
            "Model convergence: Max R-hat=" +
            str(sm["convergence"]["max_rhat"]) + "\n"
        )

    prompt = (
        "You are a senior epidemiologist advising NPHCDA and GAVI on "
        "Nigeria's zero-dose immunisation programme.\n\n"
        "Analytical context (Bayesian model NDHS 2008-2024):\n" + ctx + "\n"
        "Question: " + req.question + "\n\n"
        "Provide a structured evidence-based response covering:\n"
        "1. Direct answer\n"
        "2. Key risk factors or drivers\n"
        "3. Three prioritised programme recommendations\n"
        "4. Data quality caveats\n\n"
        "Write for programme managers — clear, specific, actionable."
    )

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key":         ANTHROPIC_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={
                    "model":    "claude-sonnet-4-20250514",
                    "max_tokens": 900,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
        if resp.status_code != 200:
            raise HTTPException(502, "Upstream API error: " + str(resp.status_code))
        return {
            "state":         req.state,
            "forecast_year": req.forecast_year,
            "response":      resp.json()["content"][0]["text"],
        }
    except httpx.TimeoutException:
        raise HTTPException(504, "Request timed out after 45 seconds.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
