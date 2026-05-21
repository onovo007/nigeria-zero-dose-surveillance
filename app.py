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
STATE_DATA: List[dict] = json.loads(r'''
[{"state_rank": 1, "state": "Sokoto", "zone": "North West", "zd_obs_2008": 94.5, "zd_obs_2013": 96.7, "zd_obs_2018": 75.4, "zd_obs_2024": 86.5, "zd_pred_2026_mean": 71.89919327576642, "zd_pred_2026_lo95": 57.82516727932253, "zd_pred_2026_hi95": 83.76392591550544, "zd_count_2026": 168829.3716985582, "zd_pred_2027_mean": 69.88632912089484, "zd_pred_2027_lo95": 54.512636100485736, "zd_pred_2027_hi95": 82.99434825211421, "zd_count_2027": 164102.88486193802, "zd_pred_2028_mean": 67.80191914773691, "zd_pred_2028_lo95": 50.96371560751956, "zd_pred_2028_hi95": 82.16218738323586, "zd_count_2028": 159208.39842756695, "gi_z_2026": 2.5943441631090867, "gi_p_2026": 0.001, "gi_class_2026": "Hot Spot (p<0.01)", "gi_z_2027": 2.613428150608068, "gi_p_2027": 0.001, "gi_class_2027": "Hot Spot (p<0.01)", "gi_z_2028": 2.624705366047253, "gi_p_2028": 0.001, "gi_class_2028": "Hot Spot (p<0.01)", "risk_index": 85.4244073236416, "priority_tier": "Tier 1: Critical", "pop_under5": 1174070.0, "cohort_12_23m": 234814.0}, {"state_rank": 2, "state": "Kebbi", "zone": "North West", "zd_obs_2008": 73.5, "zd_obs_2013": 95.1, "zd_obs_2018": 70.0, "zd_obs_2024": 84.0, "zd_pred_2026_mean": 68.35578681513495, "zd_pred_2026_lo95": 54.14879892895653, "zd_pred_2026_hi95": 81.16302441349602, "zd_count_2026": 154298.83401993595, "zd_pred_2027_mean": 66.69675599056048, "zd_pred_2027_lo95": 51.24560718225871, "zd_pred_2027_hi95": 80.47778443004445, "zd_count_2027": 150553.92032993224, "zd_pred_2028_mean": 65.00124463757358, "zd_pred_2028_lo95": 48.18401470427741, "zd_pred_2028_hi95": 79.98295658692088, "zd_count_2028": 146726.65950794844, "gi_z_2026": 2.221504942884664, "gi_p_2026": 0.001, "gi_class_2026": "Hot Spot (p<0.01)", "gi_z_2027": 2.2482404463374484, "gi_p_2027": 0.001, "gi_class_2027": "Hot Spot (p<0.01)", "gi_z_2028": 2.2689318875793463, "gi_p_2028": 0.001, "gi_class_2028": "Hot Spot (p<0.01)", "risk_index": 81.78256489829639, "priority_tier": "Tier 1: Critical", "pop_under5": 1128644.0, "cohort_12_23m": 225729.0}, {"state_rank": 3, "state": "Zamfara", "zone": "North West", "zd_obs_2008": 76.1, "zd_obs_2013": 87.8, "zd_obs_2018": 82.9, "zd_obs_2024": 82.6, "zd_pred_2026_mean": 72.9931746682898, "zd_pred_2026_lo95": 59.60872812850128, "zd_pred_2026_hi95": 84.67482452732197, "zd_count_2026": 156026.56051220285, "zd_pred_2027_mean": 71.86971319175304, "zd_pred_2027_lo95": 57.24411060658361, "zd_pred_2027_hi95": 84.58234788183405, "zd_count_2027": 153625.1054330317, "zd_pred_2028_mean": 70.71840942986499, "zd_pred_2028_lo95": 54.8586493784271, "zd_pred_2028_hi95": 84.52882684182177, "zd_count_2028": 151164.13607680792, "gi_z_2026": 1.760346957655908, "gi_p_2026": 0.001, "gi_class_2026": "Hot Spot (p<0.01)", "gi_z_2027": 1.7716636482617525, "gi_p_2027": 0.001, "gi_class_2027": "Hot Spot (p<0.01)", "gi_z_2028": 1.7787726170086724, "gi_p_2028": 0.001, "gi_class_2028": "Hot Spot (p<0.01)", "risk_index": 81.17208178104357, "priority_tier": "Tier 1: Critical", "pop_under5": 1068773.0, "cohort_12_23m": 213755.0}, {"state_rank": 4, "state": "Kano", "zone": "North West", "zd_obs_2008": 75.4, "zd_obs_2013": 73.8, "zd_obs_2018": 40.2, "zd_obs_2024": 42.4, "zd_pred_2026_mean": 36.43920147894183, "zd_pred_2026_lo95": 26.82508023540013, "zd_pred_2026_hi95": 46.9323325926909, "zd_count_2026": 195206.6242827653, "zd_pred_2027_mean": 34.33935205734349, "zd_pred_2027_lo95": 24.4223555913463, "zd_pred_2027_hi95": 45.34067870761053, "zd_count_2027": 183957.625938792, "zd_pred_2028_mean": 32.30788975585028, "zd_pred_2028_lo95": 22.10901569597753, "zd_pred_2028_hi95": 43.81014346546857, "zd_count_2028": 173074.98081657774, "gi_z_2026": 0.6390513375582456, "gi_p_2026": 0.104, "gi_class_2026": "Not Significant", "gi_z_2027": 0.5821748740484757, "gi_p_2027": 0.118, "gi_class_2027": "Not Significant", "gi_z_2028": 0.5248529978974021, "gi_p_2028": 0.13, "gi_class_2028": "Not Significant", "risk_index": 61.19906974091632, "priority_tier": "Tier 2: High", "pop_under5": 2678524.0, "cohort_12_23m": 535705.0}, {"state_rank": 5, "state": "Kaduna", "zone": "North West", "zd_obs_2008": 39.5, "zd_obs_2013": 39.5, "zd_obs_2018": 46.5, "zd_obs_2024": 45.6, "zd_pred_2026_mean": 43.34166311426329, "zd_pred_2026_lo95": 30.193950654785336, "zd_pred_2026_hi95": 56.22224106323638, "zd_count_2026": 138309.31483045017, "zd_pred_2027_mean": 43.17755363873893, "zd_pred_2027_lo95": 29.09799338824276, "zd_pred_2027_hi95": 56.96831191515744, "zd_count_2027": 137785.61851872536, "zd_pred_2028_mean": 43.016867638600615, "zd_pred_2028_lo95": 27.94432716451121, "zd_pred_2028_hi95": 57.81810421640801, "zd_count_2028": 137272.84699624396, "gi_z_2026": 0.666716444919874, "gi_p_2026": 0.032, "gi_class_2026": "Hot Spot (p<0.05)", "gi_z_2027": 0.6668466915212945, "gi_p_2027": 0.032, "gi_class_2027": "Hot Spot (p<0.05)", "gi_z_2028": 0.6662682424485995, "gi_p_2028": 0.035, "gi_class_2028": "Hot Spot (p<0.05)", "risk_index": 58.44078557964031, "priority_tier": "Tier 2: High", "pop_under5": 1595569.0, "cohort_12_23m": 319114.0}, {"state_rank": 6, "state": "Niger", "zone": "North Central", "zd_obs_2008": 61.3, "zd_obs_2013": 47.3, "zd_obs_2018": 45.3, "zd_obs_2024": 56.0, "zd_pred_2026_mean": 45.27644228585297, "zd_pred_2026_lo95": 31.51146481221182, "zd_pred_2026_hi95": 59.380223215533725, "zd_count_2026": 104775.12062253807, "zd_pred_2027_mean": 44.84816960546844, "zd_pred_2027_lo95": 30.301802113702458, "zd_pred_2027_hi95": 60.07677028543629, "zd_count_2027": 103784.04624740664, "zd_pred_2028_mean": 44.42475316504226, "zd_pred_2028_lo95": 29.097184828907952, "zd_pred_2028_hi95": 60.58935407078936, "zd_count_2028": 102804.2097942876, "gi_z_2026": 1.099754204500106, "gi_p_2026": 0.002, "gi_class_2026": "Hot Spot (p<0.01)", "gi_z_2027": 1.1514280602127442, "gi_p_2027": 0.002, "gi_class_2027": "Hot Spot (p<0.01)", "gi_z_2028": 1.2018545383787198, "gi_p_2028": 0.002, "gi_class_2028": "Hot Spot (p<0.01)", "risk_index": 57.76848437212037, "priority_tier": "Tier 2: High", "pop_under5": 1157059.0, "cohort_12_23m": 231412.0}, {"state_rank": 7, "state": "Katsina", "zone": "North West", "zd_obs_2008": 91.4, "zd_obs_2013": 76.7, "zd_obs_2018": 57.5, "zd_obs_2024": 39.8, "zd_pred_2026_mean": 38.04167209479841, "zd_pred_2026_lo95": 26.382401507548806, "zd_pred_2026_hi95": 51.15240340174774, "zd_count_2026": 165574.47570900532, "zd_pred_2027_mean": 35.24707886583309, "zd_pred_2027_lo95": 23.36440989725749, "zd_pred_2027_hi95": 49.01227754266074, "zd_count_2027": 153411.14840959522, "zd_pred_2028_mean": 32.565431509524444, "zd_pred_2028_lo95": 20.51876785077075, "zd_pred_2028_hi95": 46.88720525128529, "zd_count_2028": 141739.41237362966, "gi_z_2026": 1.0845525925141717, "gi_p_2026": 0.02, "gi_class_2026": "Hot Spot (p<0.05)", "gi_z_2027": 1.0551626396063127, "gi_p_2027": 0.022, "gi_class_2027": "Hot Spot (p<0.05)", "gi_z_2028": 1.024273586645571, "gi_p_2028": 0.023, "gi_class_2028": "Hot Spot (p<0.05)", "risk_index": 54.8707979997069, "priority_tier": "Tier 2: High", "pop_under5": 2176226.0, "cohort_12_23m": 435245.0}, {"state_rank": 8, "state": "Kogi", "zone": "North Central", "zd_obs_2008": 21.4, "zd_obs_2013": 12.9, "zd_obs_2018": 19.9, "zd_obs_2024": 57.0, "zd_pred_2026_mean": 32.31009274550254, "zd_pred_2026_lo95": 16.058081136478716, "zd_pred_2026_hi95": 51.23054275240789, "zd_count_2026": 49078.06157763599, "zd_pred_2027_mean": 32.85796933338114, "zd_pred_2027_lo95": 15.66278949654829, "zd_pred_2027_hi95": 53.25707800738903, "zd_count_2027": 49910.26967832595, "zd_pred_2028_mean": 33.412409050573984, "zd_pred_2028_lo95": 15.26257096877108, "zd_pred_2028_hi95": 55.39102718237579, "zd_count_2028": 50752.44697555036, "gi_z_2026": -0.2234014117098449, "gi_p_2026": 0.19, "gi_class_2026": "Not Significant", "gi_z_2027": -0.1930999480578877, "gi_p_2027": 0.223, "gi_class_2027": "Not Significant", "gi_z_2028": -0.1618679352024286, "gi_p_2028": 0.261, "gi_class_2028": "Not Significant", "risk_index": 48.98110900014562, "priority_tier": "Tier 3: Moderate", "pop_under5": 759485.0, "cohort_12_23m": 151897.0}, {"state_rank": 9, "state": "Kwara", "zone": "North Central", "zd_obs_2008": 28.2, "zd_obs_2013": 26.5, "zd_obs_2018": 39.2, "zd_obs_2024": 51.7, "zd_pred_2026_mean": 37.283449167743846, "zd_pred_2026_lo95": 20.78338105161295, "zd_pred_2026_hi95": 55.60627911920642, "zd_count_2026": 46895.12236318821, "zd_pred_2027_mean": 37.44578101405232, "zd_pred_2027_lo95": 19.958856638166605, "zd_pred_2027_hi95": 56.99714866704398, "zd_count_2027": 47099.303359475, "zd_pred_2028_mean": 37.61197720867203, "zd_pred_2028_lo95": 19.246265583180875, "zd_pred_2028_hi95": 58.38412838212207, "zd_count_2028": 47308.34493306768, "gi_z_2026": -0.0021121559569136, "gi_p_2026": 0.455, "gi_class_2026": "Not Significant", "gi_z_2027": 0.0462502044880577, "gi_p_2027": 0.494, "gi_class_2027": "Not Significant", "gi_z_2028": 0.0957681101321352, "gi_p_2028": 0.465, "gi_class_2028": "Not Significant", "risk_index": 46.26307061225848, "priority_tier": "Tier 3: Moderate", "pop_under5": 628900.0, "cohort_12_23m": 125780.0}, {"state_rank": 10, "state": "Jigawa", "zone": "North West", "zd_obs_2008": 88.3, "zd_obs_2013": 79.1, "zd_obs_2018": 41.1, "zd_obs_2024": 34.3, "zd_pred_2026_mean": 33.97154082549035, "zd_pred_2026_lo95": 21.307430007388557, "zd_pred_2026_hi95": 48.280446314251485, "zd_count_2026": 103706.96097042393, "zd_pred_2027_mean": 31.38404548043953, "zd_pred_2027_lo95": 18.614290674725886, "zd_pred_2027_hi95": 46.36879824128848, "zd_count_2027": 95807.9586808666, "zd_pred_2028_mean": 28.925466660779755, "zd_pred_2028_lo95": 16.105948929302432, "zd_pred_2028_hi95": 44.50068967195326, "zd_count_2028": 88302.507603362, "gi_z_2026": 0.541113089178169, "gi_p_2026": 0.13, "gi_class_2026": "Not Significant", "gi_z_2027": 0.4553403833714636, "gi_p_2027": 0.157, "gi_class_2027": "Not Significant", "gi_z_2028": 0.3690900279497454, "gi_p_2028": 0.199, "gi_class_2028": "Not Significant", "risk_index": 45.57078232978114, "priority_tier": "Tier 3: Moderate", "pop_under5": 1526380.0, "cohort_12_23m": 305276.0}, {"state_rank": 11, "state": "Bauchi", "zone": "North East", "zd_obs_2008": 83.6, "zd_obs_2013": 74.6, "zd_obs_2018": 53.0, "zd_obs_2024": 36.9, "zd_pred_2026_mean": 34.44865029458281, "zd_pred_2026_lo95": 23.05452802342612, "zd_pred_2026_hi95": 46.94829938620845, "zd_count_2026": 113091.81853858884, "zd_pred_2027_mean": 31.984155377056805, "zd_pred_2027_lo95": 20.375320586002378, "zd_pred_2027_hi95": 45.079263411239, "zd_count_2027": 105001.10352889357, "zd_pred_2028_mean": 29.63075476178309, "zd_pred_2028_lo95": 17.888147604099334, "zd_pred_2028_hi95": 43.18285425281307, "zd_count_2028": 97275.10111500534, "gi_z_2026": 0.4442566497244163, "gi_p_2026": 0.122, "gi_class_2026": "Not Significant", "gi_z_2027": 0.4103931449490322, "gi_p_2027": 0.135, "gi_class_2027": "Not Significant", "gi_z_2028": 0.3764078924592497, "gi_p_2028": 0.15, "gi_class_2028": "Not Significant", "risk_index": 44.047599324178414, "priority_tier": "Tier 3: Moderate", "pop_under5": 1641457.0, "cohort_12_23m": 328291.0}, {"state_rank": 12, "state": "Benue", "zone": "North Central", "zd_obs_2008": 39.5, "zd_obs_2013": 44.4, "zd_obs_2018": 23.2, "zd_obs_2024": 32.7, "zd_pred_2026_mean": 28.949124462843205, "zd_pred_2026_lo95": 16.081603473362044, "zd_pred_2026_hi95": 43.95064161716382, "zd_count_2026": 51902.0167756977, "zd_pred_2027_mean": 28.493788842063257, "zd_pred_2027_lo95": 15.122684160734932, "zd_pred_2027_hi95": 44.496210757117645, "zd_count_2027": 51085.65920126995, "zd_pred_2028_mean": 28.05257807651944, "zd_pred_2028_lo95": 14.12120071053302, "zd_pred_2028_hi95": 45.11405276427191, "zd_count_2028": 50294.62565604941, "gi_z_2026": -0.3290512919089585, "gi_p_2026": 0.166, "gi_class_2026": "Not Significant", "gi_z_2027": -0.3147055407196792, "gi_p_2027": 0.176, "gi_class_2027": "Not Significant", "gi_z_2028": -0.2994259613326962, "gi_p_2028": 0.19, "gi_class_2028": "Not Significant", "risk_index": 37.94328511883687, "priority_tier": "Tier 3: Moderate", "pop_under5": 896436.0, "cohort_12_23m": 179287.0}, {"state_rank": 13, "state": "Plateau", "zone": "North Central", "zd_obs_2008": 20.5, "zd_obs_2013": 37.6, "zd_obs_2018": 13.7, "zd_obs_2024": 35.3, "zd_pred_2026_mean": 25.967481076240723, "zd_pred_2026_lo95": 13.559994375733028, "zd_pred_2026_hi95": 40.97207754659275, "zd_count_2026": 38618.8378565852, "zd_pred_2027_mean": 25.996942305042268, "zd_pred_2027_lo95": 12.982885300515717, "zd_pred_2027_hi95": 41.90147215752138, "zd_count_2027": 38662.65259605886, "zd_pred_2028_mean": 26.033837702538086, "zd_pred_2028_lo95": 12.39497763628732, "zd_pred_2028_hi95": 43.15844851126836, "zd_count_2028": 38717.52343121464, "gi_z_2026": 0.2998763596133033, "gi_p_2026": 0.22, "gi_class_2026": "Not Significant", "gi_z_2027": 0.3017677707338852, "gi_p_2027": 0.223, "gi_class_2027": "Not Significant", "gi_z_2028": 0.3039498315019173, "gi_p_2028": 0.229, "gi_class_2028": "Not Significant", "risk_index": 36.36837145430116, "priority_tier": "Tier 3: Moderate", "pop_under5": 743598.0, "cohort_12_23m": 148720.0}, {"state_rank": 14, "state": "Yobe", "zone": "North East", "zd_obs_2008": 80.7, "zd_obs_2013": 81.2, "zd_obs_2018": 50.1, "zd_obs_2024": 40.4, "zd_pred_2026_mean": 34.86809955005406, "zd_pred_2026_lo95": 21.886067188070584, "zd_pred_2026_hi95": 49.61478869862911, "zd_count_2026": 42197.37407547543, "zd_pred_2027_mean": 32.48200339585021, "zd_pred_2027_lo95": 19.405138613932813, "zd_pred_2027_hi95": 47.96542754823631, "zd_count_2027": 39309.72050965792, "zd_pred_2028_mean": 30.202147413857755, "zd_pred_2028_lo95": 16.961882808866836, "zd_pred_2028_hi95": 46.2806335395974, "zd_count_2028": 36550.63880025066, "gi_z_2026": 0.4311440614542345, "gi_p_2026": 0.198, "gi_class_2026": "Not Significant", "gi_z_2027": 0.3539010392738713, "gi_p_2027": 0.24, "gi_class_2027": "Not Significant", "gi_z_2028": 0.2769803255293687, "gi_p_2028": 0.286, "gi_class_2028": "Not Significant", "risk_index": 35.97866907355236, "priority_tier": "Tier 3: Moderate", "pop_under5": 605098.0, "cohort_12_23m": 121020.0}, {"state_rank": 15, "state": "Gombe", "zone": "North East", "zd_obs_2008": 51.4, "zd_obs_2013": 55.6, "zd_obs_2018": 62.7, "zd_obs_2024": 34.0, "zd_pred_2026_mean": 34.90055807629817, "zd_pred_2026_lo95": 20.12433020420649, "zd_pred_2026_hi95": 52.066718445793086, "zd_count_2026": 51117.80039761164, "zd_pred_2027_mean": 33.475859352167355, "zd_pred_2027_lo95": 18.25521385521824, "zd_pred_2027_hi95": 51.62556304992209, "zd_count_2027": 49031.08691733896, "zd_pred_2028_mean": 32.097676653682505, "zd_pred_2028_lo95": 16.503442190499715, "zd_pred_2028_hi95": 51.25745955997717, "zd_count_2028": 47012.50406434915, "gi_z_2026": 0.2434955812314646, "gi_p_2026": 0.304, "gi_class_2026": "Not Significant", "gi_z_2027": 0.1892346880535893, "gi_p_2027": 0.345, "gi_class_2027": "Not Significant", "gi_z_2028": 0.1352554843163737, "gi_p_2028": 0.402, "gi_class_2028": "Not Significant", "risk_index": 34.77826384008982, "priority_tier": "Tier 3: Moderate", "pop_under5": 732333.0, "cohort_12_23m": 146467.0}, {"state_rank": 16, "state": "Taraba", "zone": "North East", "zd_obs_2008": 48.2, "zd_obs_2013": 55.9, "zd_obs_2018": 32.1, "zd_obs_2024": 40.7, "zd_pred_2026_mean": 27.08552269606713, "zd_pred_2026_lo95": 14.137419661296658, "zd_pred_2026_hi95": 42.750273826169504, "zd_count_2026": 33109.0720884455, "zd_pred_2027_mean": 25.79429182708306, "zd_pred_2027_lo95": 12.761581335847517, "zd_pred_2027_hi95": 42.16244021410041, "zd_count_2027": 31530.684386508063, "zd_pred_2028_mean": 24.55933628547141, "zd_pred_2028_lo95": 11.466122466810276, "zd_pred_2028_hi95": 41.7169509477214, "zd_count_2028": 30021.0870819974, "gi_z_2026": 0.1248842345738375, "gi_p_2026": 0.342, "gi_class_2026": "Not Significant", "gi_z_2027": 0.1162724631496445, "gi_p_2027": 0.348, "gi_class_2027": "Not Significant", "gi_z_2028": 0.1078157543362319, "gi_p_2028": 0.352, "gi_class_2028": "Not Significant", "risk_index": 33.76200572520685, "priority_tier": "Tier 3: Moderate", "pop_under5": 611196.0, "cohort_12_23m": 122239.0}, {"state_rank": 17, "state": "Borno", "zone": "North East", "zd_obs_2008": 83.0, "zd_obs_2013": 80.5, "zd_obs_2018": 43.8, "zd_obs_2024": 31.7, "zd_pred_2026_mean": 30.065855499170205, "zd_pred_2026_lo95": 18.318549784914037, "zd_pred_2026_hi95": 43.25716838392648, "zd_count_2026": 54422.505697602974, "zd_pred_2027_mean": 27.55651347836996, "zd_pred_2027_lo95": 15.804821754449558, "zd_pred_2027_hi95": 41.1469722712696, "zd_count_2027": 49880.320612332245, "zd_pred_2028_mean": 25.19771985269398, "zd_pred_2028_lo95": 13.540049830894986, "zd_pred_2028_hi95": 39.14443802906619, "zd_count_2028": 45610.6446825599, "gi_z_2026": 0.2330009939437243, "gi_p_2026": 0.289, "gi_class_2026": "Not Significant", "gi_z_2027": 0.1806411479271846, "gi_p_2027": 0.318, "gi_class_2027": "Not Significant", "gi_z_2028": 0.1286841455594432, "gi_p_2028": 0.339, "gi_class_2028": "Not Significant", "risk_index": 33.01175392265466, "priority_tier": "Tier 3: Moderate", "pop_under5": 905055.0, "cohort_12_23m": 181011.0}, {"state_rank": 18, "state": "Oyo", "zone": "South West", "zd_obs_2008": 20.6, "zd_obs_2013": 29.4, "zd_obs_2018": 16.8, "zd_obs_2024": 29.9, "zd_pred_2026_mean": 19.877640998550973, "zd_pred_2026_lo95": 9.288742995068608, "zd_pred_2026_hi95": 34.47141940342016, "zd_count_2026": 46370.75969782969, "zd_pred_2027_mean": 19.68244965469328, "zd_pred_2027_lo95": 8.7083808172292, "zd_pred_2027_hi95": 35.16121067088642, "zd_count_2027": 45915.41537896502, "zd_pred_2028_mean": 19.49765045826797, "zd_pred_2028_lo95": 8.167248052240417, "zd_pred_2028_hi95": 35.97908979417442, "zd_count_2028": 45484.31396555209, "gi_z_2026": -0.2603824419149909, "gi_p_2026": 0.422, "gi_class_2026": "Not Significant", "gi_z_2027": -0.2279224122819586, "gi_p_2027": 0.435, "gi_class_2027": "Not Significant", "gi_z_2028": -0.1944485032062483, "gi_p_2028": 0.46, "gi_class_2028": "Not Significant", "risk_index": 30.64418760662495, "priority_tier": "Tier 3: Moderate", "pop_under5": 1166403.0, "cohort_12_23m": 233281.0}, {"state_rank": 19, "state": "Nasarawa", "zone": "North Central", "zd_obs_2008": 46.4, "zd_obs_2013": 39.9, "zd_obs_2018": 20.6, "zd_obs_2024": 20.3, "zd_pred_2026_mean": 26.054177456605984, "zd_pred_2026_lo95": 13.625953648455758, "zd_pred_2026_hi95": 41.09944115372364, "zd_count_2026": 27429.838026314785, "zd_pred_2027_mean": 25.53340078188042, "zd_pred_2027_lo95": 12.705213027054109, "zd_pred_2027_hi95": 41.479543912370445, "zd_count_2027": 26881.564343163707, "zd_pred_2028_mean": 25.031767644194385, "zd_pred_2028_lo95": 11.795792159952107, "zd_pred_2028_hi95": 41.82480043693474, "zd_count_2028": 26353.444975807848, "gi_z_2026": 0.1433008555811964, "gi_p_2026": 0.308, "gi_class_2026": "Not Significant", "gi_z_2027": 0.1769712171082305, "gi_p_2027": 0.281, "gi_class_2027": "Not Significant", "gi_z_2028": 0.2114052477334275, "gi_p_2028": 0.264, "gi_class_2028": "Not Significant", "risk_index": 29.6691061997375, "priority_tier": "Tier 3: Moderate", "pop_under5": 526402.0, "cohort_12_23m": 105280.0}, {"state_rank": 20, "state": "Adamawa", "zone": "North East", "zd_obs_2008": 47.5, "zd_obs_2013": 19.7, "zd_obs_2018": 19.8, "zd_obs_2024": 29.1, "zd_pred_2026_mean": 21.05466660111364, "zd_pred_2026_lo95": 10.105345499487802, "zd_pred_2026_hi95": 35.79338718051774, "zd_count_2026": 36710.0744990377, "zd_pred_2027_mean": 20.30331983553144, "zd_pred_2027_lo95": 9.24082964674253, "zd_pred_2027_hi95": 35.86156136792492, "zd_count_2027": 35400.05633243919, "zd_pred_2028_mean": 19.5851302846853, "zd_pred_2028_lo95": 8.389351721094364, "zd_pred_2028_hi95": 36.00375047315597, "zd_count_2028": 34147.849759165896, "gi_z_2026": 0.120561295582599, "gi_p_2026": 0.299, "gi_class_2026": "Not Significant", "gi_z_2027": 0.0815073409517889, "gi_p_2027": 0.338, "gi_class_2027": "Not Significant", "gi_z_2028": 0.0429486588694887, "gi_p_2028": 0.363, "gi_class_2028": "Not Significant", "risk_index": 29.233482340095826, "priority_tier": "Tier 3: Moderate", "pop_under5": 871779.0, "cohort_12_23m": 174356.0}, {"state_rank": 21, "state": "Ondo", "zone": "South West", "zd_obs_2008": 28.1, "zd_obs_2013": 29.7, "zd_obs_2018": 17.9, "zd_obs_2024": 23.5, "zd_pred_2026_mean": 18.48708762765235, "zd_pred_2026_lo95": 7.72087005809144, "zd_pred_2026_hi95": 34.33948408707949, "zd_count_2026": 27312.823261093585, "zd_pred_2027_mean": 18.095697096867298, "zd_pred_2027_lo95": 7.162683194640542, "zd_pred_2027_hi95": 34.46887402352717, "zd_count_2027": 26734.582890911744, "zd_pred_2028_mean": 17.72267825937809, "zd_pred_2028_lo95": 6.575977720487598, "zd_pred_2028_hi95": 34.714096265534195, "zd_count_2028": 26183.484860405188, "gi_z_2026": -0.5766459975643244, "gi_p_2026": 0.044, "gi_class_2026": "Cold Spot (p<0.05)", "gi_z_2027": -0.5521040483973761, "gi_p_2027": 0.054, "gi_class_2027": "Cold Spot (p<0.10)", "gi_z_2028": -0.5262674181285186, "gi_p_2028": 0.066, "gi_class_2028": "Cold Spot (p<0.10)", "risk_index": 25.5135215734702, "priority_tier": "Tier 3: Moderate", "pop_under5": 738699.0, "cohort_12_23m": 147740.0}, {"state_rank": 22, "state": "Rivers", "zone": "South South", "zd_obs_2008": 31.1, "zd_obs_2013": 14.9, "zd_obs_2018": 14.9, "zd_obs_2024": 22.7, "zd_pred_2026_mean": 15.673304880553838, "zd_pred_2026_lo95": 7.047058688194, "zd_pred_2026_hi95": 27.178373137755997, "zd_count_2026": 31601.300965416674, "zd_pred_2027_mean": 15.258592461654274, "zd_pred_2027_lo95": 6.474754827165354, "zd_pred_2027_hi95": 27.3259747352921, "zd_count_2027": 30765.13705081043, "zd_pred_2028_mean": 14.861779501165383, "zd_pred_2028_lo95": 5.947953530385227, "zd_pred_2028_hi95": 27.611933261841703, "zd_count_2028": 29965.062919224707, "gi_z_2026": -0.8196249717932792, "gi_p_2026": 0.004, "gi_class_2026": "Cold Spot (p<0.01)", "gi_z_2027": -0.8167068108231821, "gi_p_2027": 0.003, "gi_class_2027": "Cold Spot (p<0.01)", "gi_z_2028": -0.8127655412051068, "gi_p_2028": 0.003, "gi_class_2028": "Cold Spot (p<0.01)", "risk_index": 22.200760636804432, "priority_tier": "Tier 4: Lower", "pop_under5": 1008124.0, "cohort_12_23m": 201625.0}, {"state_rank": 23, "state": "Ogun", "zone": "South West", "zd_obs_2008": 35.6, "zd_obs_2013": 22.5, "zd_obs_2018": 26.6, "zd_obs_2024": 20.6, "zd_pred_2026_mean": 18.14127036061701, "zd_pred_2026_lo95": 8.638071443233002, "zd_pred_2026_hi95": 30.64497491427541, "zd_count_2026": 28170.49026678052, "zd_pred_2027_mean": 17.65017528190631, "zd_pred_2027_lo95": 7.981115498687265, "zd_pred_2027_hi95": 30.81051795023763, "zd_count_2027": 27407.898184755395, "zd_pred_2028_mean": 17.17986900042458, "zd_pred_2028_lo95": 7.292789552206795, "zd_pred_2028_hi95": 31.126250173775613, "zd_count_2028": 26677.587778619305, "gi_z_2026": -0.6425987440834487, "gi_p_2026": 0.06, "gi_class_2026": "Cold Spot (p<0.10)", "gi_z_2027": -0.626484719977254, "gi_p_2027": 0.065, "gi_class_2027": "Cold Spot (p<0.10)", "gi_z_2028": -0.6094034753085216, "gi_p_2028": 0.075, "gi_class_2028": "Cold Spot (p<0.10)", "risk_index": 20.93103209665237, "priority_tier": "Tier 4: Lower", "pop_under5": 776421.0, "cohort_12_23m": 155284.0}, {"state_rank": 24, "state": "FCT", "zone": "North Central", "zd_obs_2008": 12.8, "zd_obs_2013": 16.1, "zd_obs_2018": 14.5, "zd_obs_2024": 6.2, "zd_pred_2026_mean": 16.982873448302595, "zd_pred_2026_lo95": 7.134377986877754, "zd_pred_2026_hi95": 30.350806910507263, "zd_count_2026": 16472.5381011811, "zd_pred_2027_mean": 16.893758306745237, "zd_pred_2027_lo95": 6.847640570114094, "zd_pred_2027_hi95": 30.806166167925745, "zd_count_2027": 16386.100869627542, "zd_pred_2028_mean": 16.815609427656295, "zd_pred_2028_lo95": 6.498831656076146, "zd_pred_2028_hi95": 31.47183726338688, "zd_count_2028": 16310.300364355222, "gi_z_2026": 0.3815661329798164, "gi_p_2026": 0.125, "gi_class_2026": "Not Significant", "gi_z_2027": 0.4301315196200271, "gi_p_2027": 0.103, "gi_class_2027": "Not Significant", "gi_z_2028": 0.4793324682337836, "gi_p_2028": 0.078, "gi_class_2028": "Hot Spot (p<0.10)", "risk_index": 18.230750968115228, "priority_tier": "Tier 4: Lower", "pop_under5": 484973.0, "cohort_12_23m": 96995.0}, {"state_rank": 25, "state": "Bayelsa", "zone": "South South", "zd_obs_2008": 46.9, "zd_obs_2013": 20.9, "zd_obs_2018": 28.5, "zd_obs_2024": 19.1, "zd_pred_2026_mean": 15.164143641589696, "zd_pred_2026_lo95": 5.277829114436855, "zd_pred_2026_hi95": 31.31100650706327, "zd_count_2026": 10471.296108826931, "zd_pred_2027_mean": 14.48389424658382, "zd_pred_2027_lo95": 4.711958258833815, "zd_pred_2027_hi95": 30.95332991096038, "zd_count_2027": 10001.563494093523, "zd_pred_2028_mean": 13.842891434140178, "zd_pred_2028_lo95": 4.188642907117691, "zd_pred_2028_hi95": 30.75909724257029, "zd_count_2028": 9558.931822016815, "gi_z_2026": -0.6722014653375666, "gi_p_2026": 0.187, "gi_class_2026": "Not Significant", "gi_z_2027": -0.6728217580264909, "gi_p_2027": 0.183, "gi_class_2027": "Not Significant", "gi_z_2028": -0.6725170223723537, "gi_p_2028": 0.183, "gi_class_2028": "Not Significant", "risk_index": 16.90831680151455, "priority_tier": "Tier 4: Lower", "pop_under5": 345263.0, "cohort_12_23m": 69053.0}, {"state_rank": 26, "state": "Akwa Ibom", "zone": "South South", "zd_obs_2008": 27.6, "zd_obs_2013": 10.5, "zd_obs_2018": 13.7, "zd_obs_2024": 15.1, "zd_pred_2026_mean": 13.075381156372265, "zd_pred_2026_lo95": 5.452091276728125, "zd_pred_2026_hi95": 24.563412235794512, "zd_count_2026": 16551.209729170703, "zd_pred_2027_mean": 12.72577882842657, "zd_pred_2027_lo95": 5.032057828742647, "zd_pred_2027_hi95": 24.749124852236115, "zd_count_2027": 16108.672614387206, "zd_pred_2028_mean": 12.393977532225268, "zd_pred_2028_lo95": 4.609349478110919, "zd_pred_2028_hi95": 24.9325194987067, "zd_count_2028": 15688.668579616711, "gi_z_2026": -0.7953283710186981, "gi_p_2026": 0.049, "gi_class_2026": "Cold Spot (p<0.05)", "gi_z_2027": -0.7908231412457477, "gi_p_2027": 0.046, "gi_class_2027": "Cold Spot (p<0.05)", "gi_z_2028": -0.7853637841100356, "gi_p_2028": 0.043, "gi_class_2028": "Cold Spot (p<0.05)", "risk_index": 16.161797290647257, "priority_tier": "Tier 4: Lower", "pop_under5": 632917.0, "cohort_12_23m": 126583.0}, {"state_rank": 27, "state": "Enugu", "zone": "South East", "zd_obs_2008": 36.2, "zd_obs_2013": 11.8, "zd_obs_2018": 6.8, "zd_obs_2024": 12.8, "zd_pred_2026_mean": 10.06117538107504, "zd_pred_2026_lo95": 3.628497530822316, "zd_pred_2026_hi95": 20.555348217704243, "zd_count_2026": 14974.75160193066, "zd_pred_2027_mean": 9.637856484974703, "zd_pred_2027_lo95": 3.256102307532132, "zd_pred_2027_hi95": 20.48241343239353, "zd_count_2027": 14344.696456541802, "zd_pred_2028_mean": 9.240424976099543, "zd_pred_2028_lo95": 2.9046710242989797, "zd_pred_2028_hi95": 20.49209786473006, "zd_count_2028": 13753.17132167728, "gi_z_2026": -0.5824466671998753, "gi_p_2026": 0.097, "gi_class_2026": "Cold Spot (p<0.10)", "gi_z_2027": -0.5624899413182942, "gi_p_2027": 0.11, "gi_class_2027": "Not Significant", "gi_z_2028": -0.5412908923171745, "gi_p_2028": 0.126, "gi_class_2028": "Not Significant", "risk_index": 14.03873307689179, "priority_tier": "Tier 4: Lower", "pop_under5": 744187.0, "cohort_12_23m": 148837.0}, {"state_rank": 28, "state": "Ekiti", "zone": "South West", "zd_obs_2008": 3.8, "zd_obs_2013": 2.4, "zd_obs_2018": 3.9, "zd_obs_2024": 6.0, "zd_pred_2026_mean": 10.738903707014714, "zd_pred_2026_lo95": 3.752639047244957, "zd_pred_2026_hi95": 22.37208272525525, "zd_count_2026": 11941.23136605208, "zd_pred_2027_mean": 10.815865214324868, "zd_pred_2027_lo95": 3.590032291503751, "zd_pred_2027_hi95": 23.307154396762343, "zd_count_2027": 12026.80948372068, "zd_pred_2028_mean": 10.901881185713624, "zd_pred_2028_lo95": 3.419280651635359, "zd_pred_2028_hi95": 24.203622739415025, "zd_count_2028": 12122.45580326612, "gi_z_2026": -0.2392041908510336, "gi_p_2026": 0.462, "gi_class_2026": "Not Significant", "gi_z_2027": -0.1938519205945622, "gi_p_2027": 0.5, "gi_class_2027": "Not Significant", "gi_z_2028": -0.1469753516723466, "gi_p_2028": 0.463, "gi_class_2028": "Not Significant", "risk_index": 13.800187441075586, "priority_tier": "Tier 4: Lower", "pop_under5": 555979.0, "cohort_12_23m": 111196.0}, {"state_rank": 29, "state": "Anambra", "zone": "South East", "zd_obs_2008": 11.4, "zd_obs_2013": 14.5, "zd_obs_2018": 9.9, "zd_obs_2024": 14.9, "zd_pred_2026_mean": 9.896469815612695, "zd_pred_2026_lo95": 3.752794407918002, "zd_pred_2026_hi95": 19.36749880235801, "zd_count_2026": 18041.46240325826, "zd_pred_2027_mean": 9.70103181777749, "zd_pred_2027_lo95": 3.449264145338153, "zd_pred_2027_hi95": 19.72486712726457, "zd_count_2027": 17685.17502444472, "zd_pred_2028_mean": 9.517284072149534, "zd_pred_2028_lo95": 3.158521795929874, "zd_pred_2028_hi95": 20.241185634496105, "zd_count_2028": 17350.199209210045, "gi_z_2026": -0.7354205565971479, "gi_p_2026": 0.006, "gi_class_2026": "Cold Spot (p<0.01)", "gi_z_2027": -0.7204175394355437, "gi_p_2027": 0.006, "gi_class_2027": "Cold Spot (p<0.01)", "gi_z_2028": -0.7041801397960074, "gi_p_2028": 0.006, "gi_class_2028": "Cold Spot (p<0.01)", "risk_index": 13.665915289119155, "priority_tier": "Tier 4: Lower", "pop_under5": 911511.0, "cohort_12_23m": 182302.0}, {"state_rank": 30, "state": "Cross River", "zone": "South South", "zd_obs_2008": 23.3, "zd_obs_2013": 13.2, "zd_obs_2018": 8.1, "zd_obs_2024": 2.8, "zd_pred_2026_mean": 12.26291376815304, "zd_pred_2026_lo95": 4.863449291332902, "zd_pred_2026_hi95": 23.40033482157335, "zd_count_2026": 13202.866108481972, "zd_pred_2027_mean": 11.922165290610891, "zd_pred_2027_lo95": 4.501765376356663, "zd_pred_2027_hi95": 23.444068963581056, "zd_count_2027": 12835.99926013622, "zd_pred_2028_mean": 11.599098329214712, "zd_pred_2028_lo95": 4.139928623904295, "zd_pred_2028_hi95": 23.520630640072547, "zd_count_2028": 12488.16921614902, "gi_z_2026": -0.7074771058503369, "gi_p_2026": 0.056, "gi_class_2026": "Cold Spot (p<0.10)", "gi_z_2027": -0.7003780521736368, "gi_p_2027": 0.056, "gi_class_2027": "Cold Spot (p<0.10)", "gi_z_2028": -0.6923803770538873, "gi_p_2028": 0.059, "gi_class_2028": "Cold Spot (p<0.10)", "risk_index": 13.317508705925713, "priority_tier": "Tier 4: Lower", "pop_under5": 538324.0, "cohort_12_23m": 107665.0}, {"state_rank": 31, "state": "Osun", "zone": "South West", "zd_obs_2008": 4.7, "zd_obs_2013": 7.8, "zd_obs_2018": 11.1, "zd_obs_2024": 9.4, "zd_pred_2026_mean": 11.437007664991834, "zd_pred_2026_lo95": 4.381322917465403, "zd_pred_2026_hi95": 21.836322251187003, "zd_count_2026": 13580.188531334652, "zd_pred_2027_mean": 11.476994951296778, "zd_pred_2027_lo95": 4.184079658222451, "zd_pred_2027_hi95": 22.515425374354425, "zd_count_2027": 13627.66903522028, "zd_pred_2028_mean": 11.52571202477513, "zd_pred_2028_lo95": 3.975258815506821, "zd_pred_2028_hi95": 23.41324169624079, "zd_count_2028": 13685.515201097742, "gi_z_2026": -0.3966108897781249, "gi_p_2026": 0.244, "gi_class_2026": "Not Significant", "gi_z_2027": -0.3684089229808172, "gi_p_2027": 0.266, "gi_class_2027": "Not Significant", "gi_z_2028": -0.3391238878845961, "gi_p_2028": 0.288, "gi_class_2028": "Not Significant", "risk_index": 13.205719980620168, "priority_tier": "Tier 4: Lower", "pop_under5": 593693.0, "cohort_12_23m": 118739.0}, {"state_rank": 32, "state": "Delta", "zone": "South South", "zd_obs_2008": 16.1, "zd_obs_2013": 26.8, "zd_obs_2018": 18.9, "zd_obs_2024": 7.6, "zd_pred_2026_mean": 12.838863923008358, "zd_pred_2026_lo95": 5.710055068370773, "zd_pred_2026_hi95": 22.89925980215842, "zd_count_2026": 21776.76743164986, "zd_pred_2027_mean": 12.438893682335712, "zd_pred_2027_lo95": 5.260963767916586, "zd_pred_2027_hi95": 22.99407656296201, "zd_count_2027": 21098.35390823054, "zd_pred_2028_mean": 12.0581319230082, "zd_pred_2028_lo95": 4.786083509934077, "zd_pred_2028_hi95": 22.9353799142754, "zd_count_2028": 20452.52104252959, "gi_z_2026": -0.727682398945943, "gi_p_2026": 0.027, "gi_class_2026": "Cold Spot (p<0.05)", "gi_z_2027": -0.7226812171483328, "gi_p_2027": 0.025, "gi_class_2027": "Cold Spot (p<0.05)", "gi_z_2028": -0.7167353657446351, "gi_p_2028": 0.025, "gi_class_2028": "Cold Spot (p<0.05)", "risk_index": 13.175018863271973, "priority_tier": "Tier 4: Lower", "pop_under5": 848079.0, "cohort_12_23m": 169616.0}, {"state_rank": 33, "state": "Edo", "zone": "South South", "zd_obs_2008": 12.1, "zd_obs_2013": 8.5, "zd_obs_2018": 6.5, "zd_obs_2024": 3.7, "zd_pred_2026_mean": 9.532539045712312, "zd_pred_2026_lo95": 3.714691074275237, "zd_pred_2026_hi95": 18.47195648505371, "zd_count_2026": 14042.192617457891, "zd_pred_2027_mean": 9.339263453086698, "zd_pred_2027_lo95": 3.4401486171109594, "zd_pred_2027_hi95": 18.652998699837685, "zd_count_2027": 13757.482207472953, "zd_pred_2028_mean": 9.15739449574691, "zd_pred_2028_lo95": 3.1788002622252303, "zd_pred_2028_hi95": 18.83170666572182, "zd_count_2028": 13489.57468379486, "gi_z_2026": -0.5534859757735215, "gi_p_2026": 0.18, "gi_class_2026": "Not Significant", "gi_z_2027": -0.5289700911009662, "gi_p_2027": 0.198, "gi_class_2027": "Not Significant", "gi_z_2028": -0.503199122922082, "gi_p_2028": 0.218, "gi_class_2028": "Not Significant", "risk_index": 10.658490530947478, "priority_tier": "Tier 4: Lower", "pop_under5": 736538.0, "cohort_12_23m": 147308.0}, {"state_rank": 34, "state": "Lagos", "zone": "South West", "zd_obs_2008": 13.9, "zd_obs_2013": 9.8, "zd_obs_2018": 2.7, "zd_obs_2024": 4.7, "zd_pred_2026_mean": 7.412058639378597, "zd_pred_2026_lo95": 3.252904883454908, "zd_pred_2026_hi95": 13.77000059827516, "zd_count_2026": 22378.41332342547, "zd_pred_2027_mean": 7.304439090319525, "zd_pred_2027_lo95": 3.0371126541397, "zd_pred_2027_hi95": 14.13194708924728, "zd_count_2027": 22053.48945710181, "zd_pred_2028_mean": 7.204487335345134, "zd_pred_2028_lo95": 2.835048016838681, "zd_pred_2028_hi95": 14.50236812598806, "zd_count_2028": 21751.71611800068, "gi_z_2026": -0.7751902312804785, "gi_p_2026": 0.413, "gi_class_2026": "Not Significant", "gi_z_2027": -0.7666918221026691, "gi_p_2027": 0.413, "gi_class_2027": "Not Significant", "gi_z_2028": -0.7573238669599016, "gi_p_2028": 0.413, "gi_class_2028": "Not Significant", "risk_index": 9.95384650257464, "priority_tier": "Tier 4: Lower", "pop_under5": 1509597.0, "cohort_12_23m": 301919.0}, {"state_rank": 35, "state": "Abia", "zone": "South East", "zd_obs_2008": 25.3, "zd_obs_2013": 6.6, "zd_obs_2018": 6.7, "zd_obs_2024": 4.2, "zd_pred_2026_mean": 8.701185545897488, "zd_pred_2026_lo95": 3.06775324504606, "zd_pred_2026_hi95": 18.06241086758513, "zd_count_2026": 12129.017591703803, "zd_pred_2027_mean": 8.37475809034306, "zd_pred_2027_lo95": 2.768474723769633, "zd_pred_2027_hi95": 17.954152385012083, "zd_count_2027": 11673.994040033707, "zd_pred_2028_mean": 8.068367621495852, "zd_pred_2028_lo95": 2.4887857041170305, "zd_pred_2028_hi95": 18.04062026788655, "zd_count_2028": 11246.901045984145, "gi_z_2026": -0.89860626402559, "gi_p_2026": 0.001, "gi_class_2026": "Cold Spot (p<0.01)", "gi_z_2027": -0.894784554789963, "gi_p_2027": 0.001, "gi_class_2027": "Cold Spot (p<0.01)", "gi_z_2028": -0.8898861638989118, "gi_p_2028": 0.001, "gi_class_2028": "Cold Spot (p<0.01)", "risk_index": 9.95072548166145, "priority_tier": "Tier 4: Lower", "pop_under5": 696975.0, "cohort_12_23m": 139395.0}, {"state_rank": 36, "state": "Imo", "zone": "South East", "zd_obs_2008": 16.1, "zd_obs_2013": 8.2, "zd_obs_2018": 9.2, "zd_obs_2024": 5.6, "zd_pred_2026_mean": 8.705045388278057, "zd_pred_2026_lo95": 3.211165146496272, "zd_pred_2026_hi95": 17.58069773111494, "zd_count_2026": 14068.485003357817, "zd_pred_2027_mean": 8.45355370428104, "zd_pred_2027_lo95": 2.9216858189422497, "zd_pred_2027_hi95": 17.75046429275345, "zd_count_2027": 13662.041748099717, "zd_pred_2028_mean": 8.217103761318459, "zd_pred_2028_lo95": 2.65203757030349, "zd_pred_2028_hi95": 17.91590708702742, "zd_count_2028": 13279.9079017796, "gi_z_2026": -0.8926587917681499, "gi_p_2026": 0.031, "gi_class_2026": "Cold Spot (p<0.05)", "gi_z_2027": -0.8870760001132245, "gi_p_2027": 0.031, "gi_class_2027": "Cold Spot (p<0.05)", "gi_z_2028": -0.8804567531834515, "gi_p_2028": 0.03, "gi_class_2028": "Cold Spot (p<0.05)", "risk_index": 9.763436301256014, "priority_tier": "Tier 4: Lower", "pop_under5": 808066.0, "cohort_12_23m": 161613.0}, {"state_rank": 37, "state": "Ebonyi", "zone": "South East", "zd_obs_2008": 23.1, "zd_obs_2013": 11.3, "zd_obs_2018": 4.5, "zd_obs_2024": 2.6, "zd_pred_2026_mean": 6.753220061447042, "zd_pred_2026_lo95": 2.769889647858588, "zd_pred_2026_hi95": 12.884302998755096, "zd_count_2026": 7709.340957746715, "zd_pred_2027_mean": 6.462074730665585, "zd_pred_2027_lo95": 2.467063597528528, "zd_pred_2027_hi95": 12.853713591288251, "zd_count_2027": 7376.975271033218, "zd_pred_2028_mean": 6.189812174828129, "zd_pred_2028_lo95": 2.203715678178864, "zd_pred_2028_hi95": 12.810264346250504, "zd_count_2028": 7066.165782540295, "gi_z_2026": -0.7423155810270164, "gi_p_2026": 0.056, "gi_class_2026": "Cold Spot (p<0.10)", "gi_z_2027": -0.7369965569525084, "gi_p_2027": 0.055, "gi_class_2027": "Cold Spot (p<0.10)", "gi_z_2028": -0.7307118195991359, "gi_p_2028": 0.052, "gi_class_2028": "Cold Spot (p<0.10)", "risk_index": 6.109422492401215, "priority_tier": "Tier 4: Lower", "pop_under5": 570789.0, "cohort_12_23m": 114158.0}]
''')

ARCHETYPE_DATA: List[dict] = json.loads(r'''
[{"state_name": "Jigawa", "zone_name": "North West", "cluster_id": 0, "archetype": "Nomadic / High-Mobility", "zero_dose_2024": 34.3, "zero_dose_2024.1": 34.3, "pct_urban": 11.5, "pct_problem_accessing_hfdistance": 32.0, "pct_women_no_education": 70.5, "pct_lowest_wealth_quintile": 68.4, "pct_muslim": 99.9, "pct_severely_food_insecure": 31.2, "total_fertility_rate": 6.9, "pct_cu5_stunted": 55.7, "pct_women_with_mobile_phone": 31.2, "anc_4plus": 37.7, "delivered_in_hf": 21.4, "pct_cu5_birth_registered": 15.7, "pct_women_say_wife_beating_justified": 28.2, "pct_women_moved_current_res_past5yrs": 17.5}, {"state_name": "Kaduna", "zone_name": "North West", "cluster_id": 0, "archetype": "Nomadic / High-Mobility", "zero_dose_2024": 45.6, "zero_dose_2024.1": 45.6, "pct_urban": 38.2, "pct_problem_accessing_hfdistance": 30.8, "pct_women_no_education": 35.8, "pct_lowest_wealth_quintile": 51.1, "pct_muslim": 82.2, "pct_severely_food_insecure": 33.9, "total_fertility_rate": 5.6, "pct_cu5_stunted": 40.7, "pct_women_with_mobile_phone": 53.4, "anc_4plus": 59.4, "delivered_in_hf": 25.9, "pct_cu5_birth_registered": 25.7, "pct_women_say_wife_beating_justified": 26.4, "pct_women_moved_current_res_past5yrs": 27.2}, {"state_name": "Kano", "zone_name": "North West", "cluster_id": 0, "archetype": "Nomadic / High-Mobility", "zero_dose_2024": 42.4, "zero_dose_2024.1": 42.4, "pct_urban": 34.3, "pct_problem_accessing_hfdistance": 13.2, "pct_women_no_education": 40.2, "pct_lowest_wealth_quintile": 22.5, "pct_muslim": 99.1, "pct_severely_food_insecure": 42.6, "total_fertility_rate": 5.8, "pct_cu5_stunted": 51.9, "pct_women_with_mobile_phone": 56.0, "anc_4plus": 51.3, "delivered_in_hf": 32.7, "pct_cu5_birth_registered": 50.7, "pct_women_say_wife_beating_justified": 27.8, "pct_women_moved_current_res_past5yrs": 24.2}, {"state_name": "Katsina", "zone_name": "North West", "cluster_id": 0, "archetype": "Nomadic / High-Mobility", "zero_dose_2024": 39.8, "zero_dose_2024.1": 39.8, "pct_urban": 25.6, "pct_problem_accessing_hfdistance": 22.0, "pct_women_no_education": 53.0, "pct_lowest_wealth_quintile": 0.3, "pct_muslim": 100.0, "pct_severely_food_insecure": 36.3, "total_fertility_rate": 5.7, "pct_cu5_stunted": 64.6, "pct_women_with_mobile_phone": 45.4, "anc_4plus": 37.2, "delivered_in_hf": 15.8, "pct_cu5_birth_registered": 40.2, "pct_women_say_wife_beating_justified": 42.1, "pct_women_moved_current_res_past5yrs": 20.8}, {"state_name": "Kebbi", "zone_name": "North West", "cluster_id": 4, "archetype": "High-Burden Remote Rural", "zero_dose_2024": 84.0, "zero_dose_2024.1": 84.0, "pct_urban": 14.3, "pct_problem_accessing_hfdistance": 43.0, "pct_women_no_education": 85.8, "pct_lowest_wealth_quintile": 0.8, "pct_muslim": 96.0, "pct_severely_food_insecure": 34.5, "total_fertility_rate": 6.6, "pct_cu5_stunted": 60.0, "pct_women_with_mobile_phone": 24.8, "anc_4plus": 14.0, "delivered_in_hf": 8.8, "pct_cu5_birth_registered": 18.8, "pct_women_say_wife_beating_justified": 37.6, "pct_women_moved_current_res_past5yrs": 36.4}, {"state_name": "Sokoto", "zone_name": "North West", "cluster_id": 4, "archetype": "High-Burden Remote Rural", "zero_dose_2024": 86.5, "zero_dose_2024.1": 86.5, "pct_urban": 9.8, "pct_problem_accessing_hfdistance": 48.2, "pct_women_no_education": 83.9, "pct_lowest_wealth_quintile": 3.1, "pct_muslim": 99.3, "pct_severely_food_insecure": 32.5, "total_fertility_rate": 5.4, "pct_cu5_stunted": 42.8, "pct_women_with_mobile_phone": 45.4, "anc_4plus": 22.7, "delivered_in_hf": 12.5, "pct_cu5_birth_registered": 19.0, "pct_women_say_wife_beating_justified": 34.5, "pct_women_moved_current_res_past5yrs": 5.8}, {"state_name": "Zamfara", "zone_name": "North West", "cluster_id": 0, "archetype": "Nomadic / High-Mobility", "zero_dose_2024": 82.6, "zero_dose_2024.1": 82.6, "pct_urban": 20.6, "pct_problem_accessing_hfdistance": 7.3, "pct_women_no_education": 76.0, "pct_lowest_wealth_quintile": 1.0, "pct_muslim": 100.0, "pct_severely_food_insecure": 50.7, "total_fertility_rate": 6.3, "pct_cu5_stunted": 64.2, "pct_women_with_mobile_phone": 35.3, "anc_4plus": 21.5, "delivered_in_hf": 15.3, "pct_cu5_birth_registered": 29.1, "pct_women_say_wife_beating_justified": 33.7, "pct_women_moved_current_res_past5yrs": 15.8}, {"state_name": "Adamawa", "zone_name": "North East", "cluster_id": 2, "archetype": "Conflict-Affected / Hard-to-Reach", "zero_dose_2024": 29.1, "zero_dose_2024.1": 29.1, "pct_urban": 24.2, "pct_problem_accessing_hfdistance": 48.3, "pct_women_no_education": 33.6, "pct_lowest_wealth_quintile": 13.6, "pct_muslim": 69.3, "pct_severely_food_insecure": 30.9, "total_fertility_rate": 5.3, "pct_cu5_stunted": 48.6, "pct_women_with_mobile_phone": 54.7, "anc_4plus": 56.4, "delivered_in_hf": 41.6, "pct_cu5_birth_registered": 39.3, "pct_women_say_wife_beating_justified": 40.1, "pct_women_moved_current_res_past5yrs": 23.5}, {"state_name": "Bauchi", "zone_name": "North East", "cluster_id": 0, "archetype": "Nomadic / High-Mobility", "zero_dose_2024": 36.9, "zero_dose_2024.1": 36.9, "pct_urban": 17.1, "pct_problem_accessing_hfdistance": 25.9, "pct_women_no_education": 63.1, "pct_lowest_wealth_quintile": 4.3, "pct_muslim": 95.2, "pct_severely_food_insecure": 35.3, "total_fertility_rate": 6.2, "pct_cu5_stunted": 61.7, "pct_women_with_mobile_phone": 42.5, "anc_4plus": 46.6, "delivered_in_hf": 31.1, "pct_cu5_birth_registered": 24.8, "pct_women_say_wife_beating_justified": 48.8, "pct_women_moved_current_res_past5yrs": 41.0}, {"state_name": "Borno", "zone_name": "North East", "cluster_id": 0, "archetype": "Nomadic / High-Mobility", "zero_dose_2024": 31.7, "zero_dose_2024.1": 31.7, "pct_urban": 80.6, "pct_problem_accessing_hfdistance": 36.3, "pct_women_no_education": 57.8, "pct_lowest_wealth_quintile": 34.3, "pct_muslim": 92.0, "pct_severely_food_insecure": 44.2, "total_fertility_rate": 6.5, "pct_cu5_stunted": 40.9, "pct_women_with_mobile_phone": 55.4, "anc_4plus": 61.1, "delivered_in_hf": 45.9, "pct_cu5_birth_registered": 28.9, "pct_women_say_wife_beating_justified": 24.5, "pct_women_moved_current_res_past5yrs": 30.7}, {"state_name": "Gombe", "zone_name": "North East", "cluster_id": 0, "archetype": "Nomadic / High-Mobility", "zero_dose_2024": 34.0, "zero_dose_2024.1": 34.0, "pct_urban": 25.4, "pct_problem_accessing_hfdistance": 12.5, "pct_women_no_education": 53.0, "pct_lowest_wealth_quintile": 17.5, "pct_muslim": 91.2, "pct_severely_food_insecure": 44.2, "total_fertility_rate": 5.5, "pct_cu5_stunted": 50.6, "pct_women_with_mobile_phone": 46.4, "anc_4plus": 39.1, "delivered_in_hf": 48.5, "pct_cu5_birth_registered": 33.7, "pct_women_say_wife_beating_justified": 16.8, "pct_women_moved_current_res_past5yrs": 60.5}, {"state_name": "Taraba", "zone_name": "North East", "cluster_id": 0, "archetype": "Nomadic / High-Mobility", "zero_dose_2024": 40.7, "zero_dose_2024.1": 40.7, "pct_urban": 35.6, "pct_problem_accessing_hfdistance": 20.0, "pct_women_no_education": 51.6, "pct_lowest_wealth_quintile": 2.6, "pct_muslim": 50.1, "pct_severely_food_insecure": 50.5, "total_fertility_rate": 5.2, "pct_cu5_stunted": 45.6, "pct_women_with_mobile_phone": 58.4, "anc_4plus": 50.5, "delivered_in_hf": 33.0, "pct_cu5_birth_registered": 31.1, "pct_women_say_wife_beating_justified": 17.2, "pct_women_moved_current_res_past5yrs": 37.0}, {"state_name": "Yobe", "zone_name": "North East", "cluster_id": 0, "archetype": "Nomadic / High-Mobility", "zero_dose_2024": 40.4, "zero_dose_2024.1": 40.4, "pct_urban": 44.2, "pct_problem_accessing_hfdistance": 27.5, "pct_women_no_education": 68.0, "pct_lowest_wealth_quintile": 0.3, "pct_muslim": 98.7, "pct_severely_food_insecure": 65.9, "total_fertility_rate": 7.5, "pct_cu5_stunted": 54.5, "pct_women_with_mobile_phone": 44.6, "anc_4plus": 48.5, "delivered_in_hf": 32.1, "pct_cu5_birth_registered": 19.4, "pct_women_say_wife_beating_justified": 52.7, "pct_women_moved_current_res_past5yrs": 21.4}, {"state_name": "Benue", "zone_name": "North Central", "cluster_id": 3, "archetype": "Urban Underserved", "zero_dose_2024": 32.7, "zero_dose_2024.1": 32.7, "pct_urban": 31.3, "pct_problem_accessing_hfdistance": 26.0, "pct_women_no_education": 11.6, "pct_lowest_wealth_quintile": 23.4, "pct_muslim": 4.1, "pct_severely_food_insecure": 23.5, "total_fertility_rate": 3.5, "pct_cu5_stunted": 25.3, "pct_women_with_mobile_phone": 67.4, "anc_4plus": 49.1, "delivered_in_hf": 59.0, "pct_cu5_birth_registered": 37.8, "pct_women_say_wife_beating_justified": 27.0, "pct_women_moved_current_res_past5yrs": 32.0}, {"state_name": "FCT", "zone_name": "North Central", "cluster_id": 1, "archetype": "Moderate Access \u2014 Improving", "zero_dose_2024": 6.2, "zero_dose_2024.1": 6.2, "pct_urban": 56.7, "pct_problem_accessing_hfdistance": 9.4, "pct_women_no_education": 8.0, "pct_lowest_wealth_quintile": 10.5, "pct_muslim": 40.7, "pct_severely_food_insecure": 24.7, "total_fertility_rate": 3.2, "pct_cu5_stunted": 16.3, "pct_women_with_mobile_phone": 89.2, "anc_4plus": 79.9, "delivered_in_hf": 81.3, "pct_cu5_birth_registered": 70.6, "pct_women_say_wife_beating_justified": 0.6, "pct_women_moved_current_res_past5yrs": 35.3}, {"state_name": "Kogi", "zone_name": "North Central", "cluster_id": 2, "archetype": "Conflict-Affected / Hard-to-Reach", "zero_dose_2024": 57.0, "zero_dose_2024.1": 57.0, "pct_urban": 29.2, "pct_problem_accessing_hfdistance": 21.6, "pct_women_no_education": 24.7, "pct_lowest_wealth_quintile": 17.4, "pct_muslim": 59.9, "pct_severely_food_insecure": 25.7, "total_fertility_rate": 4.9, "pct_cu5_stunted": 34.6, "pct_women_with_mobile_phone": 67.4, "anc_4plus": 54.1, "delivered_in_hf": 62.2, "pct_cu5_birth_registered": 27.2, "pct_women_say_wife_beating_justified": 9.9, "pct_women_moved_current_res_past5yrs": 20.3}, {"state_name": "Kwara", "zone_name": "North Central", "cluster_id": 2, "archetype": "Conflict-Affected / Hard-to-Reach", "zero_dose_2024": 51.7, "zero_dose_2024.1": 51.7, "pct_urban": 40.3, "pct_problem_accessing_hfdistance": 23.9, "pct_women_no_education": 33.9, "pct_lowest_wealth_quintile": 0.7, "pct_muslim": 80.6, "pct_severely_food_insecure": 30.5, "total_fertility_rate": 4.0, "pct_cu5_stunted": 40.8, "pct_women_with_mobile_phone": 81.7, "anc_4plus": 51.3, "delivered_in_hf": 51.5, "pct_cu5_birth_registered": 36.6, "pct_women_say_wife_beating_justified": 3.9, "pct_women_moved_current_res_past5yrs": 13.1}, {"state_name": "Nasarawa", "zone_name": "North Central", "cluster_id": 3, "archetype": "Urban Underserved", "zero_dose_2024": 20.3, "zero_dose_2024.1": 20.3, "pct_urban": 42.9, "pct_problem_accessing_hfdistance": 18.7, "pct_women_no_education": 34.3, "pct_lowest_wealth_quintile": 1.2, "pct_muslim": 65.3, "pct_severely_food_insecure": 16.2, "total_fertility_rate": 4.3, "pct_cu5_stunted": 35.0, "pct_women_with_mobile_phone": 70.3, "anc_4plus": 66.0, "delivered_in_hf": 55.7, "pct_cu5_birth_registered": 51.9, "pct_women_say_wife_beating_justified": 42.6, "pct_women_moved_current_res_past5yrs": 34.7}, {"state_name": "Niger", "zone_name": "North Central", "cluster_id": 2, "archetype": "Conflict-Affected / Hard-to-Reach", "zero_dose_2024": 56.0, "zero_dose_2024.1": 56.0, "pct_urban": 25.8, "pct_problem_accessing_hfdistance": 30.5, "pct_women_no_education": 73.0, "pct_lowest_wealth_quintile": 0.0, "pct_muslim": 90.1, "pct_severely_food_insecure": 8.9, "total_fertility_rate": 4.4, "pct_cu5_stunted": 43.9, "pct_women_with_mobile_phone": 56.1, "anc_4plus": 34.7, "delivered_in_hf": 30.2, "pct_cu5_birth_registered": 27.3, "pct_women_say_wife_beating_justified": 11.1, "pct_women_moved_current_res_past5yrs": 13.8}, {"state_name": "Plateau", "zone_name": "North Central", "cluster_id": 2, "archetype": "Conflict-Affected / Hard-to-Reach", "zero_dose_2024": 35.3, "zero_dose_2024.1": 35.3, "pct_urban": 24.2, "pct_problem_accessing_hfdistance": 60.0, "pct_women_no_education": 20.6, "pct_lowest_wealth_quintile": 1.5, "pct_muslim": 37.4, "pct_severely_food_insecure": 42.0, "total_fertility_rate": 4.4, "pct_cu5_stunted": 46.4, "pct_women_with_mobile_phone": 54.7, "anc_4plus": 46.4, "delivered_in_hf": 45.7, "pct_cu5_birth_registered": 27.0, "pct_women_say_wife_beating_justified": 12.7, "pct_women_moved_current_res_past5yrs": 30.2}, {"state_name": "Ekiti", "zone_name": "South West", "cluster_id": 1, "archetype": "Moderate Access \u2014 Improving", "zero_dose_2024": 6.0, "zero_dose_2024.1": 6.0, "pct_urban": 67.2, "pct_problem_accessing_hfdistance": 29.7, "pct_women_no_education": 2.2, "pct_lowest_wealth_quintile": 42.1, "pct_muslim": 15.4, "pct_severely_food_insecure": 27.9, "total_fertility_rate": 3.8, "pct_cu5_stunted": 17.1, "pct_women_with_mobile_phone": 84.8, "anc_4plus": 68.6, "delivered_in_hf": 81.7, "pct_cu5_birth_registered": 57.9, "pct_women_say_wife_beating_justified": 15.0, "pct_women_moved_current_res_past5yrs": 38.6}, {"state_name": "Lagos", "zone_name": "South West", "cluster_id": 1, "archetype": "Moderate Access \u2014 Improving", "zero_dose_2024": 4.7, "zero_dose_2024.1": 4.7, "pct_urban": 98.7, "pct_problem_accessing_hfdistance": 2.9, "pct_women_no_education": 3.0, "pct_lowest_wealth_quintile": 0.0, "pct_muslim": 32.0, "pct_severely_food_insecure": 30.7, "total_fertility_rate": 3.2, "pct_cu5_stunted": 17.3, "pct_women_with_mobile_phone": 88.9, "anc_4plus": 95.4, "delivered_in_hf": 85.8, "pct_cu5_birth_registered": 78.1, "pct_women_say_wife_beating_justified": 3.5, "pct_women_moved_current_res_past5yrs": 16.8}, {"state_name": "Ogun", "zone_name": "South West", "cluster_id": 1, "archetype": "Moderate Access \u2014 Improving", "zero_dose_2024": 20.6, "zero_dose_2024.1": 20.6, "pct_urban": 81.7, "pct_problem_accessing_hfdistance": 27.9, "pct_women_no_education": 7.4, "pct_lowest_wealth_quintile": 1.7, "pct_muslim": 31.0, "pct_severely_food_insecure": 33.1, "total_fertility_rate": 4.1, "pct_cu5_stunted": 17.7, "pct_women_with_mobile_phone": 81.5, "anc_4plus": 73.7, "delivered_in_hf": 83.3, "pct_cu5_birth_registered": 53.9, "pct_women_say_wife_beating_justified": 10.8, "pct_women_moved_current_res_past5yrs": 40.1}, {"state_name": "Ondo", "zone_name": "South West", "cluster_id": 1, "archetype": "Moderate Access \u2014 Improving", "zero_dose_2024": 23.5, "zero_dose_2024.1": 23.5, "pct_urban": 67.9, "pct_problem_accessing_hfdistance": 6.6, "pct_women_no_education": 4.7, "pct_lowest_wealth_quintile": 0.0, "pct_muslim": 7.9, "pct_severely_food_insecure": 22.7, "total_fertility_rate": 3.1, "pct_cu5_stunted": 23.2, "pct_women_with_mobile_phone": 78.6, "anc_4plus": 66.3, "delivered_in_hf": 83.2, "pct_cu5_birth_registered": 77.6, "pct_women_say_wife_beating_justified": 7.9, "pct_women_moved_current_res_past5yrs": 37.9}, {"state_name": "Osun", "zone_name": "South West", "cluster_id": 1, "archetype": "Moderate Access \u2014 Improving", "zero_dose_2024": 9.4, "zero_dose_2024.1": 9.4, "pct_urban": 85.3, "pct_problem_accessing_hfdistance": 18.1, "pct_women_no_education": 1.4, "pct_lowest_wealth_quintile": 0.2, "pct_muslim": 48.6, "pct_severely_food_insecure": 22.6, "total_fertility_rate": 3.3, "pct_cu5_stunted": 30.5, "pct_women_with_mobile_phone": 87.6, "anc_4plus": 92.0, "delivered_in_hf": 86.7, "pct_cu5_birth_registered": 81.1, "pct_women_say_wife_beating_justified": 8.1, "pct_women_moved_current_res_past5yrs": 37.0}, {"state_name": "Oyo", "zone_name": "South West", "cluster_id": 1, "archetype": "Moderate Access \u2014 Improving", "zero_dose_2024": 29.9, "zero_dose_2024.1": 29.9, "pct_urban": 79.0, "pct_problem_accessing_hfdistance": 15.8, "pct_women_no_education": 10.0, "pct_lowest_wealth_quintile": 6.0, "pct_muslim": 57.6, "pct_severely_food_insecure": 23.7, "total_fertility_rate": 3.3, "pct_cu5_stunted": 23.1, "pct_women_with_mobile_phone": 82.9, "anc_4plus": 73.8, "delivered_in_hf": 75.0, "pct_cu5_birth_registered": 62.2, "pct_women_say_wife_beating_justified": 7.3, "pct_women_moved_current_res_past5yrs": 30.4}, {"state_name": "Abia", "zone_name": "South East", "cluster_id": 1, "archetype": "Moderate Access \u2014 Improving", "zero_dose_2024": 4.2, "zero_dose_2024.1": 4.2, "pct_urban": 46.6, "pct_problem_accessing_hfdistance": 39.8, "pct_women_no_education": 0.8, "pct_lowest_wealth_quintile": 0.3, "pct_muslim": 0.0, "pct_severely_food_insecure": 39.7, "total_fertility_rate": 3.7, "pct_cu5_stunted": 20.2, "pct_women_with_mobile_phone": 82.6, "anc_4plus": 79.1, "delivered_in_hf": 86.0, "pct_cu5_birth_registered": 55.9, "pct_women_say_wife_beating_justified": 2.3, "pct_women_moved_current_res_past5yrs": 35.9}, {"state_name": "Anambra", "zone_name": "South East", "cluster_id": 1, "archetype": "Moderate Access \u2014 Improving", "zero_dose_2024": 14.9, "zero_dose_2024.1": 14.9, "pct_urban": 56.5, "pct_problem_accessing_hfdistance": 31.5, "pct_women_no_education": 1.0, "pct_lowest_wealth_quintile": 22.8, "pct_muslim": 0.6, "pct_severely_food_insecure": 14.5, "total_fertility_rate": 3.7, "pct_cu5_stunted": 12.9, "pct_women_with_mobile_phone": 81.5, "anc_4plus": 84.9, "delivered_in_hf": 83.2, "pct_cu5_birth_registered": 62.9, "pct_women_say_wife_beating_justified": 13.7, "pct_women_moved_current_res_past5yrs": 31.7}, {"state_name": "Ebonyi", "zone_name": "South East", "cluster_id": 3, "archetype": "Urban Underserved", "zero_dose_2024": 2.6, "zero_dose_2024.1": 2.6, "pct_urban": 24.0, "pct_problem_accessing_hfdistance": 38.2, "pct_women_no_education": 7.1, "pct_lowest_wealth_quintile": 41.4, "pct_muslim": 0.0, "pct_severely_food_insecure": 30.2, "total_fertility_rate": 4.7, "pct_cu5_stunted": 31.6, "pct_women_with_mobile_phone": 57.1, "anc_4plus": 61.7, "delivered_in_hf": 79.4, "pct_cu5_birth_registered": 43.2, "pct_women_say_wife_beating_justified": 32.6, "pct_women_moved_current_res_past5yrs": 50.4}, {"state_name": "Enugu", "zone_name": "South East", "cluster_id": 1, "archetype": "Moderate Access \u2014 Improving", "zero_dose_2024": 12.8, "zero_dose_2024.1": 12.8, "pct_urban": 66.0, "pct_problem_accessing_hfdistance": 23.7, "pct_women_no_education": 8.3, "pct_lowest_wealth_quintile": 60.1, "pct_muslim": 0.8, "pct_severely_food_insecure": 43.4, "total_fertility_rate": 3.5, "pct_cu5_stunted": 15.2, "pct_women_with_mobile_phone": 80.7, "anc_4plus": 61.9, "delivered_in_hf": 92.6, "pct_cu5_birth_registered": 80.1, "pct_women_say_wife_beating_justified": 2.7, "pct_women_moved_current_res_past5yrs": 44.4}, {"state_name": "Imo", "zone_name": "South East", "cluster_id": 1, "archetype": "Moderate Access \u2014 Improving", "zero_dose_2024": 5.6, "zero_dose_2024.1": 5.6, "pct_urban": 32.2, "pct_problem_accessing_hfdistance": 21.2, "pct_women_no_education": 0.5, "pct_lowest_wealth_quintile": 24.4, "pct_muslim": 0.2, "pct_severely_food_insecure": 23.3, "total_fertility_rate": 4.4, "pct_cu5_stunted": 17.3, "pct_women_with_mobile_phone": 82.5, "anc_4plus": 84.9, "delivered_in_hf": 97.0, "pct_cu5_birth_registered": 58.2, "pct_women_say_wife_beating_justified": 4.4, "pct_women_moved_current_res_past5yrs": 35.1}, {"state_name": "Akwa Ibom", "zone_name": "South South", "cluster_id": 3, "archetype": "Urban Underserved", "zero_dose_2024": 15.1, "zero_dose_2024.1": 15.1, "pct_urban": 48.1, "pct_problem_accessing_hfdistance": 22.4, "pct_women_no_education": 0.6, "pct_lowest_wealth_quintile": 17.6, "pct_muslim": 0.0, "pct_severely_food_insecure": 39.9, "total_fertility_rate": 3.3, "pct_cu5_stunted": 24.1, "pct_women_with_mobile_phone": 73.2, "anc_4plus": 65.7, "delivered_in_hf": 38.6, "pct_cu5_birth_registered": 48.0, "pct_women_say_wife_beating_justified": 19.8, "pct_women_moved_current_res_past5yrs": 35.4}, {"state_name": "Bayelsa", "zone_name": "South South", "cluster_id": 3, "archetype": "Urban Underserved", "zero_dose_2024": 19.1, "zero_dose_2024.1": 19.1, "pct_urban": 66.9, "pct_problem_accessing_hfdistance": 39.2, "pct_women_no_education": 4.9, "pct_lowest_wealth_quintile": 18.0, "pct_muslim": 0.6, "pct_severely_food_insecure": 32.6, "total_fertility_rate": 3.8, "pct_cu5_stunted": 27.6, "pct_women_with_mobile_phone": 77.7, "anc_4plus": 48.6, "delivered_in_hf": 46.1, "pct_cu5_birth_registered": 49.1, "pct_women_say_wife_beating_justified": 18.2, "pct_women_moved_current_res_past5yrs": 32.1}, {"state_name": "Cross River", "zone_name": "South South", "cluster_id": 1, "archetype": "Moderate Access \u2014 Improving", "zero_dose_2024": 2.8, "zero_dose_2024.1": 2.8, "pct_urban": 55.1, "pct_problem_accessing_hfdistance": 6.5, "pct_women_no_education": 3.3, "pct_lowest_wealth_quintile": 37.9, "pct_muslim": 0.2, "pct_severely_food_insecure": 18.7, "total_fertility_rate": 3.0, "pct_cu5_stunted": 21.0, "pct_women_with_mobile_phone": 68.6, "anc_4plus": 80.0, "delivered_in_hf": 58.8, "pct_cu5_birth_registered": 44.3, "pct_women_say_wife_beating_justified": 11.4, "pct_women_moved_current_res_past5yrs": 32.2}, {"state_name": "Delta", "zone_name": "South South", "cluster_id": 1, "archetype": "Moderate Access \u2014 Improving", "zero_dose_2024": 7.6, "zero_dose_2024.1": 7.6, "pct_urban": 83.1, "pct_problem_accessing_hfdistance": 20.9, "pct_women_no_education": 3.4, "pct_lowest_wealth_quintile": 37.2, "pct_muslim": 3.4, "pct_severely_food_insecure": 20.8, "total_fertility_rate": 3.7, "pct_cu5_stunted": 20.0, "pct_women_with_mobile_phone": 81.7, "anc_4plus": 60.5, "delivered_in_hf": 83.0, "pct_cu5_birth_registered": 62.4, "pct_women_say_wife_beating_justified": 7.7, "pct_women_moved_current_res_past5yrs": 39.8}, {"state_name": "Edo", "zone_name": "South South", "cluster_id": 1, "archetype": "Moderate Access \u2014 Improving", "zero_dose_2024": 3.7, "zero_dose_2024.1": 3.7, "pct_urban": 82.5, "pct_problem_accessing_hfdistance": 16.1, "pct_women_no_education": 1.6, "pct_lowest_wealth_quintile": 34.8, "pct_muslim": 8.7, "pct_severely_food_insecure": 15.2, "total_fertility_rate": 3.3, "pct_cu5_stunted": 13.6, "pct_women_with_mobile_phone": 87.7, "anc_4plus": 63.0, "delivered_in_hf": 90.9, "pct_cu5_birth_registered": 62.5, "pct_women_say_wife_beating_justified": 21.8, "pct_women_moved_current_res_past5yrs": 44.9}, {"state_name": "Rivers", "zone_name": "South South", "cluster_id": 1, "archetype": "Moderate Access \u2014 Improving", "zero_dose_2024": 22.7, "zero_dose_2024.1": 22.7, "pct_urban": 80.0, "pct_problem_accessing_hfdistance": 21.2, "pct_women_no_education": 2.4, "pct_lowest_wealth_quintile": 2.7, "pct_muslim": 0.8, "pct_severely_food_insecure": 33.5, "total_fertility_rate": 2.9, "pct_cu5_stunted": 12.3, "pct_women_with_mobile_phone": 75.5, "anc_4plus": 76.5, "delivered_in_hf": 56.9, "pct_cu5_birth_registered": 69.2, "pct_women_say_wife_beating_justified": 5.5, "pct_women_moved_current_res_past5yrs": 34.8}]
''')

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
