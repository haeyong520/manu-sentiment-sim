import os
import json
import uuid
import math
import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


# =============================
# 0) USER SETTINGS
# =============================
# Resolve all paths relative to this script's directory so the script runs
# correctly regardless of which working directory it is launched from.
# Previously EXCEL_PATH and OUTDIR used bare relative paths (e.g. "outputs"),
# which caused FileNotFoundError when the script was executed from a different
# directory (e.g. outputs/step_excels/).
_SCRIPT_DIR = Path(__file__).resolve().parent

EXCEL_PATH = _SCRIPT_DIR / "ManU_match_stock_merged.xlsx"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1"                            # change to your installed Ollama model
OUTDIR = _SCRIPT_DIR / "outputs"

# Excel step exports (one file per step)
EXPORT_STEP_EXCELS = True
STEP_EXCEL_DIR = OUTDIR / "step_excels"

# Step2 (reactions) Excel export options
# REACTIONS_EXCEL_MODE:
#   - 'full'        : dump all reactions into one workbook, one sheet
#   - 'recent_n'    : dump only the most recent N matches into one workbook, one sheet
#   - 'by_match'    : one workbook, one sheet per match (optionally limited by recent_n)
#   - 'match_files' : one workbook per match (optionally limited by recent_n)
#   - 'chunked'     : multiple workbooks, each containing sheets for a chunk of matches
REACTIONS_EXCEL_MODE = "by_match"
REACTIONS_EXCEL_RECENT_N = 20  # used for mode='recent_n' or to limit matches in other modes
REACTIONS_EXCEL_CHUNK_SIZE = 20  # used for mode='chunked'


# Persona panel config
N_PERSONAS = 200
PERSONA_MIX = {  # proportions (sum ~ 1.0)
    "TEMPORARY_FAN": 0.45,
    "LOCAL_FAN": 0.15,
    "FANATICAL_FAN": 0.25,
    "INVESTOR_FAN": 0.15,
}
# REGION_CONTEXT sets the overall panel region context for LLM-mode persona generation.
# For sampled-mode, each persona is assigned a region individually via REGION_MIX below,
# so REGION_CONTEXT is only used as a fallback label.
REGION_CONTEXT = "US"

# Region distribution for sampled-mode persona generation.
# Each persona draws a region independently from this distribution.
# Reflects a US-heavy but internationally diverse MANU shareholder base.
# Adjust proportions to match your target investor population.
REGION_MIX = {
    "UK":     0.20,
    "EU":     0.30,
    "GLOBAL": 0.50,
}

# --- Persona generation mode ---
# 'sampled' : generate diverse persona traits via Latin Hypercube sampling (fast + guaranteed diversity)
# 'llm'     : use LLM to generate personas (slower; enable temperature + targets to improve diversity)
PERSONA_GENERATION_MODE = "sampled"

# LLM sampling controls (used only when PERSONA_GENERATION_MODE='llm')
PERSONA_TEMPERATURE = 0.7
PERSONA_TOP_P = 0.9
PERSONA_TOP_K = 40

# Reaction generation controls (LLM).
# [FIX-1] 0.2 → 0.6: At temperature 0.2 the LLM always collapses to the single
# most-probable token, producing only SELL/HOLD even after a WIN. At 0.6, lower-ranked
# choices (e.g. BUY) can be sampled when contextually appropriate, while the risk of
# JSON format corruption (typically seen above 0.8) remains low.
REACTION_TEMPERATURE = 0.6
REACTION_TOP_P = 0.9
REACTION_TOP_K = 40


# Post-processing: reconcile action from net_demand when the LLM outputs HOLD/NO_TRADE too often
# [FIX-2] 0.15 → 0.05: With the old threshold of 0.15, reconciliation never triggered
# when the LLM also output a low net_demand. Lowering to 0.05 allows even weak net_demand
# signals to override HOLD with BUY/SELL, acting as a second safety net alongside FIX-1.
ACTION_FROM_NET_THRESHOLD = 0.05
# Debug controls
MAX_MATCHES = 190     # None = all matches. e.g., 30 for quick test
MAX_PERSONAS = 200    # None = all personas. e.g., 20 for quick test

# Regeneration toggles
FORCE_REGEN_PERSONAS = False  # False = load existing personas.jsonl (skip Step1)
SKIP_TO_STEP = 2              # 1=full run, 2=skip persona gen, 3=skip reactions, 4=skip features

# Parallel reaction generation
# N_WORKERS: number of concurrent Ollama requests.
# RTX 3070 Laptop 8GB recommendation: 2~3
#   - llama3.1:8b uses ~4.7GB base + ~0.4GB per concurrent request
#   - 2 workers: ~5.5GB total (safe)
#   - 3 workers: ~6.3GB total (recommended sweet spot)
#   - 4+ workers: risk of VRAM overflow → slowdown or OOM
N_WORKERS = 3

# Labels
LABEL_COL = "return_manu_tplus1"              # or "abnormal_return_tplus1"


# =============================
# 1) PROMPTS (JSON-safe)
# =============================
SYSTEM = "You MUST output ONLY a single valid JSON object and nothing else. No markdown. No prose. No extra keys."

PROMPT_A = r"""
TARGET SCHEMA: PersonaProfile_v1
Return ONE JSON object only.

You MUST output ONLY a single valid JSON object and nothing else.
No markdown. No prose. No comments. No trailing commas.
Do NOT include extra keys beyond the schema.

TASK
Create one persona profile for a simulation linking Manchester United match outcomes to next-trading-day trading intentions.

INPUTS (provided below)
- persona_type: "TEMPORARY_FAN" or "LOCAL_FAN" or "FANATICAL_FAN" or "INVESTOR_FAN"
- region_context: e.g., "US", "UK", "Korea", "EU", "Global"
- optional_demographic_overrides: JSON object (may be empty). Apply overrides exactly.

PERSONA TYPE GUIDELINES (encode in numeric traits)
- TEMPORARY_FAN: situational attachment, high overreaction, high CORFing after losses, higher trading frequency.
- LOCAL_FAN: place/community-driven, higher home_bias, moderate trading.
- FANATICAL_FAN: enduring identity attachment, low CORFing, rarely panic sell, long horizon.
- INVESTOR_FAN: financially-motivated, low emotional_return, low overreaction, very low trading frequency, mostly HOLD.

NUMERIC CONSTRAINTS
- Any field ending with "_0_10" must be a number between 0 and 10.
- Any field ending with "_0_1" must be a number between 0 and 1.
- fandom_tenure_years must be a non-negative integer.
- primary_info_channels must be a non-empty list from:
  ["traditional_media","twitter_x","tiktok","instagram","reddit","friends","financial_news","youtube"]

OUTPUT JSON KEYS (MUST MATCH EXACTLY; no extra keys)
{
  "schema": "PersonaProfile_v1",
  "persona_id": "TEMPORARY_FAN_001",
  "persona_type": "TEMPORARY_FAN",
  "demographics": {
    "age_band": "25-34",
    "gender": "male",
    "income_band": "mid",
    "education": "mid",
    "region": "US",
    "local_to_manchester": false,
    "timezone": "America/Detroit"
  },
  "fan_identity": {
    "fandom_tenure_years": 3,
    "match_following_intensity_0_10": 6.0,
    "primary_info_channels": ["twitter_x","youtube"]
  },
  "psychology": {
    "club_attachment_0_10": 5.5,
    "emotional_return_weight_0_10": 6.5,
    "financial_return_weight_0_10": 4.0,
    "risk_aversion_0_10": 4.5,
    "loss_aversion_0_10": 6.0,
    "overreaction_0_10": 7.0,
    "corfing_tendency_0_10": 7.5,
    "herding_0_10": 6.0
  },
  "trading_style": {
    "brokerage_access": true,
    "typical_trade_frequency": "weekly",
    "typical_order_size_0_1": 0.25,
    "time_horizon": "1w",
    "default_action_no_shock": "HOLD"
  },
  "model_params": {
    "home_bias_0_10": 3.0,
    "importance_sensitivity_0_10": 6.0,
    "odds_surprise_sensitivity_0_10": 7.0,
    "liquidity_price_impact_sensitivity_0_10": 6.0
  }
}

IMPORTANT
- Replace values to match the provided inputs (persona_type, region_context, overrides).
- Keep the same keys and structure exactly.
- Return ONLY JSON.
"""

PROMPT_B = r"""
TARGET SCHEMA: AgentReaction_v1
Return ONE JSON object only.

You MUST output ONLY a single valid JSON object and nothing else.
No markdown. No prose. No comments. No trailing commas.
Do NOT include extra keys beyond the schema.

INPUTS (provided below)
1) persona_profile_json: a valid PersonaProfile_v1 JSON
2) match_event_json: a valid MatchEvent_v1 JSON
3) market_context_json: a valid MarketContext_v1 JSON

STEP 1 — READ THE MATCH RESULT FIRST
Before anything else, read match_event_json["result"]. It is one of:
  "WIN"  → Manchester United WON this match
  "DRAW" → The match ended in a draw
  "LOSS" → Manchester United LOST this match

STEP 2 — COMPUTE SHOCK (signed: positive = good news, negative = bad news)
- Let p = odds_implied_prob[result], clamp p >= 0.000001
- surprise = -ln(p)   ← always positive; larger = more unexpected
- signed_surprise:
    WIN  → +surprise   (POSITIVE shock: good news, stock price likely UP)
    DRAW →  0
    LOSS → -surprise   (NEGATIVE shock: bad news, stock price likely DOWN)
- shock = signed_surprise
  shock *= (1 + 0.60 * importance_0_1)
  shock *= (1 + 0.30 * opponent_strength_0_1)
  shock *= (1 + 0.10 * min(abs(goal_diff), 4))
  if result is LOSS: shock *= (1 + 0.05 * loss_aversion_0_10)
  shock *= (1 + 0.05 * overreaction_0_10)

  CRITICAL: shock > 0 means WIN/good-news. shock < 0 means LOSS/bad-news.

- net_demand: positive when buying pressure, negative when selling pressure
    net_demand = tanh(shock * odds_surprise_sensitivity_0_10 / 10)
    clamp to [-1, 1]

- liquidity_amp = 1 + liquidity_thinness_0_1 * (liquidity_price_impact_sensitivity_0_10 / 10)
  clamp liquidity_amp to [1, 3]

STEP 3 — DECIDE ACTION (based on result AND persona type)
[WIN — shock > 0: good news]
  TEMPORARY_FAN : BUY if shock is large and unexpected (high surprise); else HOLD
  LOCAL_FAN     : BUY if home game with high importance; else HOLD
  FANATICAL_FAN : BUY (supportive buying is natural after a win)
  INVESTOR_FAN  : HOLD (wins are already priced in for long-term investors)

[LOSS — shock < 0: bad news]
  TEMPORARY_FAN : SELL (high corfing, panic reaction)
  LOCAL_FAN     : SELL if home game; else HOLD
  FANATICAL_FAN : HOLD or small BUY (loyal, does not panic sell)
  INVESTOR_FAN  : HOLD (long-term view, ignore short-term shock)

[DRAW — shock ~ 0]
  All types     : HOLD in most cases; small BUY/SELL only if shock is notable

ADDITIONAL RULES
- If brokerage_access is false OR typical_trade_frequency is "never":
  action must be "NO_TRADE" and size_0_1 must be 0
- If action is "HOLD" or "NO_TRADE", size_0_1 must be 0
- size_0_1 scales with abs(shock): small shock → 0.05~0.15, large shock → 0.30~0.60

NOTE ON ACTION LABELS
- Your output 'action' MUST be one of: BUY, SELL, HOLD, NO_TRADE.
- Use size_0_1 to express small vs large trades (e.g., 0.05 = small, 0.40 = large).

NUMERIC CONSTRAINTS
- size_0_1 and confidence_0_1 must be between 0 and 1
- net_demand_-1_to_1 must be between -1 and 1
- liquidity_amp_1_to_3 must be between 1 and 3
- emotion valence is integer between -3 and 3; arousal integer between 0 and 3

OUTPUT JSON KEYS (MUST MATCH EXACTLY; no extra keys)
{
  "schema": "AgentReaction_v1",
  "match_id": "MANU_001",
  "persona_id": "TEMPORARY_FAN_001",
  "action": "HOLD",
  "size_0_1": 0.0,
  "horizon": "1w",
  "confidence_0_1": 0.5,
  "signals": {
    "surprise": 0.0,
    "shock": 0.0,
    "net_demand_-1_to_1": 0.0,
    "liquidity_amp_1_to_3": 1.0
  },
  "emotion": {
    "valence_-3_to_3": 0,
    "arousal_0_to_3": 0,
    "dominant": "indifference"
  },
  "rationale_short": "string"
}

IMPORTANT
- Use match_id from match_event_json and persona_id from persona_profile_json.
- Keep the same keys and structure exactly.
- Return ONLY JSON.
"""


# =============================
# 2) OLLAMA CALL + SAFE JSON
# =============================
def ollama_chat(system: str, user: str, temperature: float = 0.0, format_schema=None) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": temperature},
        "stream": False,
        "format": format_schema if format_schema is not None else "json"
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"]


def ollama_chat_with_options(system: str, user: str, options: Dict[str, Any], format_schema=None) -> str:
    """Ollama chat with additional sampling options (top_p, top_k, etc.)."""
    opts = dict(options or {})
    if "temperature" not in opts:
        opts["temperature"] = 0.0
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": opts,
        "stream": False,
        "format": format_schema if format_schema is not None else "json",
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"]


def extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    raise json.JSONDecodeError("Unclosed JSON object", text, start)


def safe_parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(extract_first_json_object(text))


def safe_llm_json(system: str, user_prompt: str, schema: dict, *, options: Dict[str, Any] = None) -> Dict[str, Any]:
    """LLM -> JSON (schema-validated). Uses a repair pass if needed."""
    if options is None:
        raw = ollama_chat(system, user_prompt, temperature=0.0, format_schema=schema)
    else:
        raw = ollama_chat_with_options(system, user_prompt, options=options, format_schema=schema)
    try:
        return safe_parse_json(raw)
    except json.JSONDecodeError:
        repair_prompt = (
            "Your previous output was invalid JSON.\n"
            "Return ONLY valid JSON that matches the schema exactly. No extra keys.\n\n"
            f"Previous output:\n{raw}"
        )
        if options is None:
            raw2 = ollama_chat(system, repair_prompt, temperature=0.0, format_schema=schema)
        else:
            # In repair, lower temperature to maximize compliance
            repair_opts = dict(options)
            repair_opts["temperature"] = min(float(repair_opts.get("temperature", 0.0)), 0.2)
            raw2 = ollama_chat_with_options(system, repair_prompt, options=repair_opts, format_schema=schema)
        return safe_parse_json(raw2)


# =============================
# 3) JSON SCHEMAS (format)
# =============================
PERSONA_PROFILE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema","persona_id","persona_type","demographics","fan_identity","psychology","trading_style","model_params"],
    "properties": {
        "schema": {"const": "PersonaProfile_v1"},
        "persona_id": {"type": "string"},
        "persona_type": {"enum": ["TEMPORARY_FAN","LOCAL_FAN","FANATICAL_FAN","INVESTOR_FAN"]},
        "demographics": {
            "type": "object",
            "additionalProperties": False,
            "required": ["age_band","gender","income_band","education","region","local_to_manchester","timezone"],
            "properties": {
                "age_band": {"enum": ["18-24","25-34","35-44","45-54","55+"]},
                "gender": {"enum": ["male","female","nonbinary","prefer_not_say"]},
                "income_band": {"enum": ["low","mid","high"]},
                "education": {"enum": ["low","mid","high"]},
                "region": {"type": "string"},
                "local_to_manchester": {"type": "boolean"},
                "timezone": {"type": "string"}
            }
        },
        "fan_identity": {
            "type": "object",
            "additionalProperties": False,
            "required": ["fandom_tenure_years","match_following_intensity_0_10","primary_info_channels"],
            "properties": {
                "fandom_tenure_years": {"type": "integer", "minimum": 0, "maximum": 60},
                "match_following_intensity_0_10": {"type": "number", "minimum": 0.0, "maximum": 10.0},
                "primary_info_channels": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"enum": ["traditional_media","twitter_x","tiktok","instagram","reddit","friends","financial_news","youtube"]}
                }
            }
        },
        "psychology": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "club_attachment_0_10","emotional_return_weight_0_10","financial_return_weight_0_10",
                "risk_aversion_0_10","loss_aversion_0_10","overreaction_0_10",
                "corfing_tendency_0_10","herding_0_10"
            ],
            "properties": {
                "club_attachment_0_10": {"type":"number","minimum":0.0,"maximum":10.0},
                "emotional_return_weight_0_10": {"type":"number","minimum":0.0,"maximum":10.0},
                "financial_return_weight_0_10": {"type":"number","minimum":0.0,"maximum":10.0},
                "risk_aversion_0_10": {"type":"number","minimum":0.0,"maximum":10.0},
                "loss_aversion_0_10": {"type":"number","minimum":0.0,"maximum":10.0},
                "overreaction_0_10": {"type":"number","minimum":0.0,"maximum":10.0},
                "corfing_tendency_0_10": {"type":"number","minimum":0.0,"maximum":10.0},
                "herding_0_10": {"type":"number","minimum":0.0,"maximum":10.0}
            }
        },
        "trading_style": {
            "type": "object",
            "additionalProperties": False,
            "required": ["brokerage_access","typical_trade_frequency","typical_order_size_0_1","time_horizon","default_action_no_shock"],
            "properties": {
                "brokerage_access": {"type":"boolean"},
                "typical_trade_frequency": {"enum":["never","rare","monthly","weekly","daily"]},
                "typical_order_size_0_1": {"type":"number","minimum":0.0,"maximum":1.0},
                "time_horizon": {"enum":["intraday","1w","1m","6m","multi_year"]},
                "default_action_no_shock": {"enum":["HOLD","BUY_SMALL","SELL_SMALL"]}
            }
        },
        "model_params": {
            "type": "object",
            "additionalProperties": False,
            "required": ["home_bias_0_10","importance_sensitivity_0_10","odds_surprise_sensitivity_0_10","liquidity_price_impact_sensitivity_0_10"],
            "properties": {
                "home_bias_0_10": {"type":"number","minimum":0.0,"maximum":10.0},
                "importance_sensitivity_0_10": {"type":"number","minimum":0.0,"maximum":10.0},
                "odds_surprise_sensitivity_0_10": {"type":"number","minimum":0.0,"maximum":10.0},
                "liquidity_price_impact_sensitivity_0_10": {"type":"number","minimum":0.0,"maximum":10.0}
            }
        }
    }
}

AGENT_REACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema","match_id","persona_id","action","size_0_1","horizon","confidence_0_1","signals","emotion","rationale_short"],
    "properties": {
        "schema": {"const": "AgentReaction_v1"},
        "match_id": {"type": "string"},
        "persona_id": {"type": "string"},
        "action": {"enum": ["BUY","SELL","HOLD","NO_TRADE"]},
        "size_0_1": {"type":"number","minimum":0.0,"maximum":1.0},
        "horizon": {"enum":["intraday","1w","1m","6m","multi_year"]},
        "confidence_0_1": {"type":"number","minimum":0.0,"maximum":1.0},
        "signals": {
            "type":"object",
            "additionalProperties": False,
            "required":["surprise","shock","net_demand_-1_to_1","liquidity_amp_1_to_3"],
            "properties":{
                "surprise":{"type":"number"},
                "shock":{"type":"number"},
                "net_demand_-1_to_1":{"type":"number","minimum":-1.0,"maximum":1.0},
                "liquidity_amp_1_to_3":{"type":"number","minimum":1.0,"maximum":3.0}
            }
        },
        "emotion": {
            "type":"object",
            "additionalProperties": False,
            "required":["valence_-3_to_3","arousal_0_to_3","dominant"],
            "properties":{
                "valence_-3_to_3":{"type":"integer","minimum":-3,"maximum":3},
                "arousal_0_to_3":{"type":"integer","minimum":0,"maximum":3},
                "dominant":{"enum":["pride","joy","relief","anger","sadness","disappointment","anxiety","indifference"]}
            }
        },
        "rationale_short":{"type":"string","maxLength":120}
    }
}


# =============================
# 4) MATCH ROW -> MatchEvent / MarketContext
# =============================
def implied_probs(odds_home, odds_draw, odds_away) -> Tuple[float,float,float]:
    inv = np.array([1/odds_home, 1/odds_draw, 1/odds_away], dtype=float)
    s = float(inv.sum())
    p = inv / s
    return float(p[0]), float(p[1]), float(p[2])


def make_match_event(row: pd.Series) -> Dict[str, Any]:
    p_home, p_draw, p_away = implied_probs(row["Odds_Home_Win"], row["Odds_Draw"], row["Odds_Away_Win"])
    venue = str(row["Venue"]).strip().title()  # Home/Away

    # Convert to MANU perspective
    if venue == "Home":
        pW, pD, pL = p_home, p_draw, p_away
    else:
        pW, pD, pL = p_away, p_draw, p_home

    result_map = {"Win": "WIN", "Draw": "DRAW", "Loss": "LOSS"}
    result = result_map.get(str(row["Result"]).strip().title(), "DRAW")

    goal_diff = int(row["GF"] - row["GA"])
    opponent_strength = float(np.clip(1 - pW, 0.0, 1.0))  # proxy

    return {
        "schema": "MatchEvent_v1",
        "match_id": f"MANU_{int(row['Match_ID']):03d}",
        "date_local": pd.to_datetime(row["match_date"]).date().isoformat(),
        "competition": "PL",
        "venue": venue.upper(),
        "importance_0_1": float(row.get("importance_0_1", 0.5)) if not pd.isna(row.get("importance_0_1", 0.5)) else 0.5,
        "result": result,
        "goal_diff": goal_diff,
        "opponent_strength_0_1": opponent_strength,
        "odds_implied_prob": {"WIN": pW, "DRAW": pD, "LOSS": pL}
    }


def make_market_context(row: pd.Series) -> Dict[str, Any]:
    asof_date = row.get("pre_trade_date_t0")
    if pd.isna(asof_date):
        asof_date = row.get("pre_trade_date")
    if pd.isna(asof_date):
        asof_date = row["match_date"]

    rr = row.get("recent_return_5d", 0.0)
    rv = row.get("recent_volatility_20d", 0.0)
    mkt = row.get("return_market_t0", 0.0)

    rr = float(0.0 if pd.isna(rr) else rr)
    rv = float(0.0 if pd.isna(rv) else rv)
    mkt = float(0.0 if pd.isna(mkt) else mkt)

    return {
        "schema": "MarketContext_v1",
        "asof_date": pd.to_datetime(asof_date).date().isoformat(),
        "recent_return_5d": rr,
        "recent_volatility_20d": rv,
        "market_index_return_1d": mkt,
        "liquidity_thinness_0_1": 0.6
    }


def compute_realized_surprise(match_event: Dict[str, Any]) -> float:
    res = match_event["result"]
    p = float(match_event["odds_implied_prob"][res])
    p = max(p, 1e-6)
    return float(-math.log(p))


# =============================
# 5) PERSONA PANEL (once) + SAVE/LOAD
# =============================
def ensure_outdir():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if EXPORT_STEP_EXCELS:
        STEP_EXCEL_DIR.mkdir(parents=True, exist_ok=True)

def _flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    """Make JSON-normalized frames Excel-friendly."""
    out = df.copy()
    # Convert list-like cells to pipe-joined strings
    for c in out.columns:
        if out[c].apply(lambda x: isinstance(x, (list, tuple))).any():
            out[c] = out[c].apply(lambda x: "|".join(map(str, x)) if isinstance(x, (list, tuple)) else x)
    return out

def export_step1_personas_excel(personas: List[Dict[str, Any]]):
    """Step1: personas.jsonl -> Step1_Personas.xlsx"""
    if not EXPORT_STEP_EXCELS:
        return
    ensure_outdir()
    df = pd.json_normalize(personas, sep="__") if personas else pd.DataFrame()
    df = _flatten_df(df)
    out = STEP_EXCEL_DIR / "Step1_Personas.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Personas")
    print(f"[INFO] Step1 Excel -> {out}")

def export_step2_reactions_excel(reactions_jsonl_path: Path):
    """Step2: reactions.jsonl Excel exports.
    Controlled by:
      REACTIONS_EXCEL_MODE, REACTIONS_EXCEL_RECENT_N, REACTIONS_EXCEL_CHUNK_SIZE
    """
    if not EXPORT_STEP_EXCELS:
        return
    ensure_outdir()
    if not reactions_jsonl_path.exists():
        print("[WARN] reactions.jsonl not found; skipping Step2 Excel export.")
        return

    recs = jsonl_read(reactions_jsonl_path)
    if not recs:
        print("[WARN] reactions.jsonl is empty; skipping Step2 Excel export.")
        return

    df = pd.json_normalize(recs, sep="__")
    df = _flatten_df(df)

    # Sort matches by numeric suffix if available (MANU_001 -> 1)
    if "match_id" in df.columns:
        def _mid_num(x):
            try:
                return int(str(x).split("_")[-1])
            except Exception:
                return -1
        df["__match_num"] = df["match_id"].apply(_mid_num)
        sort_cols = ["__match_num"]
        if "persona_id" in df.columns:
            sort_cols.append("persona_id")
        df = df.sort_values(sort_cols).reset_index(drop=True)
    else:
        df["__match_num"] = -1

    mode = str(REACTIONS_EXCEL_MODE).lower().strip()
    recent_n = REACTIONS_EXCEL_RECENT_N
    recent_n = int(recent_n) if recent_n is not None else None
    chunk_size = int(REACTIONS_EXCEL_CHUNK_SIZE) if REACTIONS_EXCEL_CHUNK_SIZE is not None else 20

    match_ids = df["match_id"].dropna().unique().tolist() if "match_id" in df.columns else []
    # match_ids are in df sort order
    match_ids_recent = match_ids[-recent_n:] if (recent_n is not None and len(match_ids) > 0) else match_ids

    # Helper: safe sheet/workbook identifiers
    def _sheet_name(mid: str) -> str:
        s = str(mid)[:31]
        return s

    base_df = df.drop(columns=["__match_num"], errors="ignore")

    # ---- MODE: full / recent_n (single sheet) ----
    if mode in ["full", "recent_n"]:
        out = STEP_EXCEL_DIR / "Step2_Reactions.xlsx"
        if mode == "recent_n" and len(match_ids_recent) > 0:
            df2 = base_df[base_df["match_id"].isin(match_ids_recent)].copy()
        else:
            df2 = base_df.copy()
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            df2.to_excel(w, index=False, sheet_name="Reactions")
        print(f"[INFO] Step2 Excel ({mode}) -> {out}")
        return

    # ---- MODE: by_match (one workbook, multiple sheets) ----
    if mode == "by_match":
        out = STEP_EXCEL_DIR / "Step2_Reactions.xlsx"
        if len(match_ids_recent) == 0:
            with pd.ExcelWriter(out, engine="openpyxl") as w:
                base_df.to_excel(w, index=False, sheet_name="Reactions")
            print(f"[INFO] Step2 Excel (fallback full) -> {out}")
            return
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            for mid in match_ids_recent:
                sub = base_df[base_df["match_id"] == mid]
                sub.to_excel(w, index=False, sheet_name=_sheet_name(mid))
        print(f"[INFO] Step2 Excel (by_match, sheets={len(match_ids_recent)}) -> {out}")
        return

    # ---- MODE: match_files (one workbook per match) ----
    if mode == "match_files":
        out_dir = STEP_EXCEL_DIR / "Step2_Reactions_by_match"
        out_dir.mkdir(parents=True, exist_ok=True)
        if len(match_ids_recent) == 0:
            # fallback single file
            out = out_dir / "Step2_Reactions_ALL.xlsx"
            with pd.ExcelWriter(out, engine="openpyxl") as w:
                base_df.to_excel(w, index=False, sheet_name="Reactions")
            print(f"[INFO] Step2 Excel (fallback full) -> {out}")
            return
        for mid in match_ids_recent:
            sub = base_df[base_df["match_id"] == mid]
            out = out_dir / f"Step2_Reactions_{mid}.xlsx"
            with pd.ExcelWriter(out, engine="openpyxl") as w:
                sub.to_excel(w, index=False, sheet_name=_sheet_name(mid))
        print(f"[INFO] Step2 Excel (match_files, files={len(match_ids_recent)}) -> {out_dir}")
        return

    # ---- MODE: chunked (multiple workbooks, each with multiple match sheets) ----
    if mode == "chunked":
        out_dir = STEP_EXCEL_DIR / "Step2_Reactions_chunks"
        out_dir.mkdir(parents=True, exist_ok=True)
        mids = match_ids_recent if len(match_ids_recent) > 0 else match_ids
        if len(mids) == 0:
            out = out_dir / "Step2_Reactions_ALL.xlsx"
            with pd.ExcelWriter(out, engine="openpyxl") as w:
                base_df.to_excel(w, index=False, sheet_name="Reactions")
            print(f"[INFO] Step2 Excel (fallback full) -> {out}")
            return
        # chunk into groups
        for i in range(0, len(mids), chunk_size):
            chunk = mids[i:i+chunk_size]
            first = chunk[0]
            last = chunk[-1]
            out = out_dir / f"Step2_Reactions_{first}_to_{last}.xlsx"
            with pd.ExcelWriter(out, engine="openpyxl") as w:
                for mid in chunk:
                    sub = base_df[base_df["match_id"] == mid]
                    sub.to_excel(w, index=False, sheet_name=_sheet_name(mid))
        print(f"[INFO] Step2 Excel (chunked, chunk_size={chunk_size}) -> {out_dir}")
        return

    # Unknown mode fallback
    out = STEP_EXCEL_DIR / "Step2_Reactions.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        base_df.to_excel(w, index=False, sheet_name="Reactions")
    print(f"[WARN] Unknown REACTIONS_EXCEL_MODE='{REACTIONS_EXCEL_MODE}'. Wrote fallback -> {out}")

def export_step3_features_excel(feat_df: pd.DataFrame):
    """Step3: match_features.csv -> Step3_MatchFeatures.xlsx"""
    if not EXPORT_STEP_EXCELS:
        return
    ensure_outdir()
    out = STEP_EXCEL_DIR / "Step3_MatchFeatures.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        feat_df.to_excel(w, index=False, sheet_name="MatchFeatures")
    print(f"[INFO] Step3 Excel -> {out}")

def export_step3_predictions_excel(pred_df: pd.DataFrame):
    """Step3: rolling-origin predictions -> Step3_Predictions.xlsx

    This file is required for downstream comparisons against realized market returns
    (e.g., compare_predictions_to_market.py).
    """
    if not EXPORT_STEP_EXCELS:
        return
    ensure_outdir()
    out = STEP_EXCEL_DIR / "Step3_Predictions.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        pred_df.to_excel(w, index=False, sheet_name="Predictions")
    print(f"[INFO] Step3 Predictions Excel -> {out}")


def export_step4_validation_excel(rows: List[Dict[str, Any]]):
    """Step4: validation summary -> Step4_Validation.xlsx"""
    if not EXPORT_STEP_EXCELS:
        return
    ensure_outdir()
    df = pd.DataFrame(rows)
    out = STEP_EXCEL_DIR / "Step4_Validation.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Validation")
    print(f"[INFO] Step4 Excel -> {out}")

    
def _sanitize_str(obj):
    """
    Recursively replace surrogate characters in strings.
    LLM responses occasionally contain lone surrogate code points (e.g. \ud83d
    from broken emoji) that cause UnicodeEncodeError when writing to utf-8 files.
    """
    if isinstance(obj, str):
        return obj.encode('utf-8', 'surrogatepass').decode('utf-8', 'replace')
    elif isinstance(obj, dict):
        return {k: _sanitize_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_str(v) for v in obj]
    return obj


def jsonl_write(path: Path, records: List[Dict[str, Any]], mode="a"):
    with path.open(mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def jsonl_read(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def allocate_counts(n: int, mix: Dict[str, float]) -> Dict[str, int]:
    types = list(mix.keys())
    raw = np.array([mix[t] for t in types], dtype=float)
    raw = raw / raw.sum()
    counts = np.floor(raw * n).astype(int)
    # distribute remainder
    rem = n - int(counts.sum())
    if rem > 0:
        frac = raw * n - np.floor(raw * n)
        order = np.argsort(-frac)
        for i in range(rem):
            counts[order[i % len(types)]] += 1
    return {t: int(c) for t, c in zip(types, counts)}


# =============================
# 5A) DIVERSE PERSONA SAMPLING (no LLM)
# =============================
def lhs(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube Sampling in (0, 1) to enforce diversity coverage."""
    cut = np.linspace(0.0, 1.0, n + 1)
    u = rng.random((n, k))
    a = cut[:n]
    b = cut[1:n + 1]
    rd = u * (b - a)[:, None] + a[:, None]
    H = np.zeros_like(rd)
    for j in range(k):
        H[:, j] = rd[rng.permutation(n), j]
    return H


def _clip_round_0_10(x: float) -> float:
    return float(np.clip(np.round(x, 1), 0.0, 10.0))


def _clip_0_1(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _lin_map(u: float, lo: float, hi: float) -> float:
    return lo + (hi - lo) * float(u)


def _sample_channels(persona_type: str, rng: np.random.Generator) -> List[str]:
    pool = {
        "TEMPORARY_FAN": ["twitter_x", "instagram", "tiktok", "youtube", "friends"],
        "LOCAL_FAN": ["traditional_media", "twitter_x", "youtube", "friends"],
        "FANATICAL_FAN": ["twitter_x", "reddit", "youtube", "friends", "traditional_media"],
        "INVESTOR_FAN": ["financial_news", "twitter_x", "reddit", "youtube"],
    }
    base = pool[persona_type]
    k = int(rng.integers(2, 5))  # 2-4 channels
    return rng.choice(base, size=k, replace=False).tolist()


def generate_personas_sampled(rng: np.random.Generator, counts: Dict[str, int], region: str) -> List[Dict[str, Any]]:
    """Generate a diverse persona panel without calling an LLM.

    Why this exists:
    - LLM persona generation at temperature=0 collapses to near-identical profiles.
    - Even at higher temperature, strict JSON schemas can still reduce diversity.
    - Sampling numeric traits directly guarantees diversity and speeds up Step1.
    """

    # Ranges per persona type (0-10 unless order_size 0-1)
    spec = {
        "TEMPORARY_FAN": dict(
            match_intensity=(2, 6), attachment=(1, 5), emo_w=(3, 7), fin_w=(4, 9),
            risk=(3, 8), loss=(4, 9), overreact=(6, 10), corfing=(6, 10), herd=(5, 10),
            order=(0.05, 0.35), home_bias=(0, 4), imp=(3, 8), odds=(4, 9), liq=(3, 8),
            freq=["daily", "weekly", "weekly"], horizon=["intraday", "1w", "1w"], default=["HOLD", "SELL_SMALL"],
        ),
        "LOCAL_FAN": dict(
            match_intensity=(6, 10), attachment=(6, 10), emo_w=(5, 10), fin_w=(2, 7),
            risk=(2, 7), loss=(3, 8), overreact=(2, 6), corfing=(0, 4), herd=(2, 6),
            order=(0.10, 0.50), home_bias=(6, 10), imp=(5, 10), odds=(4, 8), liq=(3, 7),
            freq=["weekly", "weekly", "monthly"], horizon=["1w", "1m", "1m"], default=["HOLD", "BUY_SMALL"],
        ),
        "FANATICAL_FAN": dict(
            match_intensity=(8, 10), attachment=(8, 10), emo_w=(7, 10), fin_w=(0, 4),
            risk=(0, 5), loss=(2, 7), overreact=(4, 9), corfing=(0, 3), herd=(0, 4),
            order=(0.10, 0.55), home_bias=(7, 10), imp=(7, 10), odds=(5, 10), liq=(4, 9),
            freq=["weekly", "monthly"], horizon=["1m", "6m"], default=["BUY_SMALL", "HOLD"],
        ),
        "INVESTOR_FAN": dict(
            match_intensity=(3, 8), attachment=(0, 4), emo_w=(0, 3), fin_w=(7, 10),
            risk=(4, 10), loss=(5, 10), overreact=(0, 4), corfing=(6, 10), herd=(0, 5),
            order=(0.10, 0.80), home_bias=(0, 3), imp=(4, 9), odds=(6, 10), liq=(4, 10),
            freq=["rare", "monthly"], horizon=["1m", "6m", "multi_year"], default=["HOLD"],
        ),
    }

    personas: List[Dict[str, Any]] = []

    for pt, n in counts.items():
        if n <= 0:
            continue
        s = spec[pt]

        # 14 dimensions for LHS
        U = lhs(n, 14, rng)

        for i in range(n):
            u = U[i]

            match_intensity = _clip_round_0_10(_lin_map(u[0], *s["match_intensity"]))
            attachment = _clip_round_0_10(_lin_map(u[1], *s["attachment"]))

            # Couple emotional vs financial weights (weak inverse relationship)
            emo_w = _clip_round_0_10(_lin_map(u[2], *s["emo_w"]))
            fin_w = _clip_round_0_10(_lin_map(1 - 0.8 * float(u[2]), *s["fin_w"]))

            risk = _clip_round_0_10(_lin_map(u[3], *s["risk"]))
            loss = _clip_round_0_10(_lin_map(u[4], *s["loss"]))
            overreact = _clip_round_0_10(_lin_map(u[5], *s["overreact"]))
            corfing = _clip_round_0_10(_lin_map(u[6], *s["corfing"]))
            herd = _clip_round_0_10(_lin_map(u[7], *s["herd"]))
            order = _clip_0_1(_lin_map(u[8], *s["order"]))
            home_bias = _clip_round_0_10(_lin_map(u[9], *s["home_bias"]))
            imp = _clip_round_0_10(_lin_map(u[10], *s["imp"]))
            odds = _clip_round_0_10(_lin_map(u[11], *s["odds"]))
            liq = _clip_round_0_10(_lin_map(u[12], *s["liq"]))

            # Sample region independently per persona from REGION_MIX distribution.
            # This gives the panel geographic diversity rather than collapsing all
            # personas to a single region (e.g. all US) when REGION_CONTEXT is fixed.
            region_keys = list(REGION_MIX.keys())
            region_probs = [REGION_MIX[k] for k in region_keys]
            persona_region = str(rng.choice(region_keys, p=region_probs))
            overrides = sample_demographic_overrides(rng, pt, persona_region)
            demo = overrides["demographics"]

            # Simple tenure rule (keeps non-negative integer)
            if pt == "TEMPORARY_FAN":
                tenure = int(rng.integers(0, 6))
            elif pt == "LOCAL_FAN":
                tenure = int(rng.integers(2, 21))
            elif pt == "FANATICAL_FAN":
                tenure = int(rng.integers(5, 31))
            else:
                tenure = int(rng.integers(0, 9))

            persona = {
                "schema": "PersonaProfile_v1",
                "persona_id": f"{pt}_{i+1:03d}",
                "persona_type": pt,
                "demographics": demo,
                "fan_identity": {
                    "fandom_tenure_years": tenure,
                    "match_following_intensity_0_10": match_intensity,
                    "primary_info_channels": _sample_channels(pt, rng),
                },
                "psychology": {
                    "club_attachment_0_10": attachment,
                    "emotional_return_weight_0_10": emo_w,
                    "financial_return_weight_0_10": fin_w,
                    "risk_aversion_0_10": risk,
                    "loss_aversion_0_10": loss,
                    "overreaction_0_10": overreact,
                    "corfing_tendency_0_10": corfing,
                    "herding_0_10": herd,
                },
                "trading_style": {
                    "brokerage_access": True,
                    "typical_trade_frequency": str(rng.choice(s["freq"])),
                    "typical_order_size_0_1": order,
                    "time_horizon": str(rng.choice(s["horizon"])),
                    "default_action_no_shock": str(rng.choice(s["default"])),
                },
                "model_params": {
                    "home_bias_0_10": home_bias,
                    "importance_sensitivity_0_10": imp,
                    "odds_surprise_sensitivity_0_10": odds,
                    "liquidity_price_impact_sensitivity_0_10": liq,
                },
            }
            personas.append(persona)

    return personas


def sample_demographic_overrides(rng: np.random.Generator, persona_type: str, region: str) -> Dict[str, Any]:
    """
    Simple heuristic demographic sampler (you can replace with empirically-derived distributions later).
    """
    if persona_type == "TEMPORARY_FAN":
        age = rng.choice(["18-24","25-34","35-44"], p=[0.45,0.40,0.15])
        income = rng.choice(["low","mid","high"], p=[0.45,0.45,0.10])
        edu = rng.choice(["low","mid","high"], p=[0.20,0.60,0.20])
    elif persona_type == "LOCAL_FAN":
        age = rng.choice(["25-34","35-44","45-54","55+"], p=[0.25,0.35,0.25,0.15])
        income = rng.choice(["low","mid","high"], p=[0.20,0.60,0.20])
        edu = rng.choice(["low","mid","high"], p=[0.15,0.60,0.25])
    elif persona_type == "FANATICAL_FAN":
        age = rng.choice(["18-24","25-34","35-44","45-54"], p=[0.25,0.40,0.25,0.10])
        income = rng.choice(["low","mid","high"], p=[0.20,0.60,0.20])
        edu = rng.choice(["low","mid","high"], p=[0.10,0.60,0.30])
    else:  # INVESTOR_FAN
        age = rng.choice(["25-34","35-44","45-54","55+"], p=[0.15,0.30,0.30,0.25])
        income = rng.choice(["low","mid","high"], p=[0.05,0.45,0.50])
        edu = rng.choice(["low","mid","high"], p=[0.05,0.35,0.60])

    gender = rng.choice(["male","female","nonbinary","prefer_not_say"], p=[0.55,0.40,0.02,0.03])

    # local_to_manchester and timezone vary by region and persona type.
    # UK fans have the highest probability of being local to Manchester.
    # Other regions have small but non-zero probabilities for LOCAL_FAN /
    # FANATICAL_FAN to represent expats and heritage supporters.
    region_upper = region.upper()
    if region_upper == "UK":
        if persona_type in ["LOCAL_FAN", "FANATICAL_FAN"]:
            local = bool(rng.choice([True, False], p=[0.35, 0.65]))
        else:
            local = bool(rng.choice([True, False], p=[0.10, 0.90]))
        tz = "Europe/London"
    elif region_upper == "US":
        if persona_type == "LOCAL_FAN":
            local = bool(rng.choice([True, False], p=[0.10, 0.90]))
        elif persona_type == "FANATICAL_FAN":
            local = bool(rng.choice([True, False], p=[0.05, 0.95]))
        else:
            local = False
        tz = "America/Detroit"
    elif region_upper == "EU":
        if persona_type in ["LOCAL_FAN", "FANATICAL_FAN"]:
            local = bool(rng.choice([True, False], p=[0.08, 0.92]))
        else:
            local = False
        tz = "Europe/Paris"
    elif region_upper == "KOREA":
        if persona_type in ["LOCAL_FAN", "FANATICAL_FAN"]:
            local = bool(rng.choice([True, False], p=[0.03, 0.97]))
        else:
            local = False
        tz = "Asia/Seoul"
    else:  # Global or unknown
        if persona_type in ["LOCAL_FAN", "FANATICAL_FAN"]:
            local = bool(rng.choice([True, False], p=[0.03, 0.97]))
        else:
            local = False
        tz = "Etc/UTC"

    return {
        "demographics": {
            "age_band": age,
            "gender": gender,
            "income_band": income,
            "education": edu,
            "region": region,
            "local_to_manchester": local,
            "timezone": tz
        }
    }


def build_prompt_a(persona_type: str, region: str, overrides: Dict[str, Any]) -> str:
    return (
        PROMPT_A
        + "\n\npersona_type: " + json.dumps(persona_type)
        + "\nregion_context: " + json.dumps(region)
        + "\noptional_demographic_overrides: " + json.dumps(overrides, ensure_ascii=False)
    )

persona_path = OUTDIR / "personas.jsonl"

def _check_persona_status():
    n_existing = sum(1 for _ in persona_path.open("r", encoding="utf-8")) if persona_path.exists() else 0
    print(f"[CHECK] personas.jsonl exists={persona_path.exists()} lines={n_existing} | N_PERSONAS={N_PERSONAS} | MAX_PERSONAS={MAX_PERSONAS}")


def generate_or_load_personas(force_regen: bool = False, seed: int = 2026) -> List[Dict[str, Any]]:
    ensure_outdir()
    persona_path = OUTDIR / "personas.jsonl"

    if persona_path.exists() and (not force_regen):
        personas = jsonl_read(persona_path)
        print(f"[INFO] Loaded personas: {len(personas)} from {persona_path}")
        return personas

    rng = np.random.default_rng(seed)
    counts = allocate_counts(N_PERSONAS, PERSONA_MIX)
    personas: List[Dict[str, Any]] = []

    print("[INFO] Generating personas with counts:", counts)

    mode = str(PERSONA_GENERATION_MODE).lower().strip()
    if mode == "sampled":
        personas = generate_personas_sampled(rng, counts, REGION_CONTEXT)
    else:
        # LLM mode: enable temperature and enforce unique, deterministic persona_id sequence per type
        persona_opts = {
            "temperature": float(PERSONA_TEMPERATURE),
            "top_p": float(PERSONA_TOP_P),
            "top_k": int(PERSONA_TOP_K),
        }
        for pt, n in counts.items():
            for i in range(n):
                overrides = sample_demographic_overrides(rng, pt, REGION_CONTEXT)
                user_prompt = build_prompt_a(pt, REGION_CONTEXT, overrides)
                p = safe_llm_json(SYSTEM, user_prompt, PERSONA_PROFILE_SCHEMA, options=persona_opts)

                # Hard override persona_id + persona_type to prevent duplicates/collisions.
                p["persona_id"] = f"{pt}_{i+1:03d}"
                p["persona_type"] = pt
                # Also hard-apply demographics overrides (LLM can drift otherwise).
                if isinstance(overrides, dict) and "demographics" in overrides:
                    p["demographics"] = overrides["demographics"]

                personas.append(p)

    # save
    jsonl_write(persona_path, personas, mode="w")
    print(f"[INFO] Saved personas: {len(personas)} to {persona_path}")
    export_step1_personas_excel(personas)
    return personas


# =============================
# 6) REACTIONS: match × persona (append-only, resumable)
# =============================
def build_prompt_b(persona_json: Dict[str, Any], match_event: Dict[str, Any], market_ctx: Dict[str, Any]) -> str:
    # [FIX-4] Pre-compute shock direction in Python and inject it as an explicit hint in the prompt.
    # Rationale: even when the LLM understands "surprise = -ln(p)", it frequently confuses the
    # sign of signed_surprise (+/-). This was confirmed empirically: 89 WIN reactions were
    # labelled "High shock (loss)" in the rationale. Providing the pre-computed value removes
    # any ambiguity about direction before the LLM decides on an action.
    result = match_event.get("result", "DRAW")
    p = float(match_event["odds_implied_prob"].get(result, 0.33))
    p = max(p, 1e-6)
    import math as _math
    surprise_val = -_math.log(p)
    signed_val = surprise_val if result == "WIN" else (0.0 if result == "DRAW" else -surprise_val)
    direction_hint = (
        f"\n\n[PRE-COMPUTED HINT — do NOT override]\n"
        f"match result = {result}\n"
        f"surprise (unsigned) = {surprise_val:.4f}\n"
        f"signed_surprise = {signed_val:.4f}  "
        f"({'POSITIVE → buying pressure expected' if signed_val > 0 else 'NEGATIVE → selling pressure expected' if signed_val < 0 else 'NEUTRAL'})\n"
        f"Use this signed_surprise as your starting shock before applying multipliers."
    )
    return (
        PROMPT_B
        + "\n\npersona_profile_json:\n" + json.dumps(persona_json, ensure_ascii=False)
        + "\n\nmatch_event_json:\n" + json.dumps(match_event, ensure_ascii=False)
        + "\n\nmarket_context_json:\n" + json.dumps(market_ctx, ensure_ascii=False)
        + direction_hint
    )


def clamp_reaction(r: Dict[str, Any]) -> Dict[str, Any]:
    # Enforce action-size consistency regardless of model behavior
    action = r.get("action", "HOLD")
    if action in ["HOLD", "NO_TRADE"]:
        r["size_0_1"] = 0.0

    # clamp numeric ranges
    r["size_0_1"] = float(np.clip(float(r.get("size_0_1", 0.0)), 0.0, 1.0))
    r["confidence_0_1"] = float(np.clip(float(r.get("confidence_0_1", 0.5)), 0.0, 1.0))

    sig = r.get("signals", {})
    sig["net_demand_-1_to_1"] = float(np.clip(float(sig.get("net_demand_-1_to_1", 0.0)), -1.0, 1.0))
    sig["liquidity_amp_1_to_3"] = float(np.clip(float(sig.get("liquidity_amp_1_to_3", 1.0)), 1.0, 3.0))
    sig["surprise"] = float(sig.get("surprise", 0.0))
    sig["shock"] = float(sig.get("shock", 0.0))
    r["signals"] = sig
    return r



def reconcile_action_from_net(r: Dict[str, Any], persona: Dict[str, Any], threshold: float = ACTION_FROM_NET_THRESHOLD) -> Dict[str, Any]:
    """
    If the LLM outputs HOLD/NO_TRADE but net_demand suggests meaningful pressure,
    override action to BUY/SELL (pilot-friendly, improves aggregation signals).

    This is intentionally simple: it uses the sign/magnitude of net_demand and
    scales size from the persona's typical_order_size_0_1.
    """
    try:
        trading = persona.get("trading_style", {})
        brokerage = bool(trading.get("brokerage_access", True))
        freq = str(trading.get("typical_trade_frequency", "weekly")).lower()

        if (not brokerage) or (freq == "never"):
            r["action"] = "NO_TRADE"
            r["size_0_1"] = 0.0
            return r

        sig = r.get("signals", {})
        net = float(sig.get("net_demand_-1_to_1", 0.0))

        action = str(r.get("action", "HOLD")).upper()
        if action in ["HOLD", "NO_TRADE"] and abs(net) >= float(threshold):
            r["action"] = "BUY" if net > 0 else "SELL"

            base = float(trading.get("typical_order_size_0_1", 0.20))
            # Scale size smoothly with |net|; ensure > 0 for BUY/SELL
            scale = min(1.0, abs(net))
            size = base * (0.5 + 0.5 * scale)
            r["size_0_1"] = float(np.clip(size, 0.01, 1.0))

            # Confidence follows scale (but keep any higher model confidence)
            conf = float(r.get("confidence_0_1", 0.5))
            r["confidence_0_1"] = float(np.clip(max(conf, 0.55 + 0.35 * scale), 0.0, 1.0))

        return r
    except Exception:
        # Never fail the pipeline due to a reconciliation issue
        return r

def load_existing_pairs(reaction_path: Path) -> set:
    existing = set()
    if not reaction_path.exists():
        return existing
    with reaction_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            existing.add((obj.get("match_id"), obj.get("persona_id")))
    return existing


def _process_one_reaction(
    p: Dict[str, Any],
    match_event: Dict[str, Any],
    market_ctx: Dict[str, Any],
    reaction_opts: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate a single (match, persona) reaction.
    Called from worker threads — must be thread-safe (no shared mutable state).
    Each call is a fully independent Ollama HTTP request, so thread safety is
    guaranteed as long as we don't mutate shared objects.
    """
    user_prompt = build_prompt_b(p, match_event, market_ctx)
    r = safe_llm_json(SYSTEM, user_prompt, AGENT_REACTION_SCHEMA, options=reaction_opts)
    r = clamp_reaction(r)
    r = reconcile_action_from_net(r, p)
    r["match_id"] = match_event["match_id"]
    r["persona_id"] = p["persona_id"]
    return r


def generate_reactions(df: pd.DataFrame, personas: List[Dict[str, Any]]) -> None:
    """
    Generate match × persona reaction pairs using a thread pool.

    Thread safety design:
    - _process_one_reaction() is pure (no shared mutable state).
    - File writes and existing-set updates are protected by a threading.Lock.
    - reactions.jsonl is opened in append mode; each batch is flushed immediately
      so progress is not lost if the process is interrupted.

    N_WORKERS controls concurrency. Recommended for RTX 3070 8GB: N_WORKERS=3.
    Ollama server should be started with OLLAMA_NUM_PARALLEL=2 or 3.
    """
    ensure_outdir()
    reaction_path = OUTDIR / "reactions.jsonl"

    existing = load_existing_pairs(reaction_path)
    print(f"[INFO] Existing reactions: {len(existing)}")

    personas_use = personas[:MAX_PERSONAS] if MAX_PERSONAS else personas

    reaction_opts = {
        "temperature": float(REACTION_TEMPERATURE),
        "top_p":       float(REACTION_TOP_P),
        "top_k":       int(REACTION_TOP_K),
    }

    # Build the full todo list (match, persona) pairs that have not been done yet
    todo: List[Tuple[Dict, Dict, Dict]] = []
    for _, row in df.iterrows():
        match_event = make_match_event(row)
        market_ctx  = make_market_context(row)
        for p in personas_use:
            if (match_event["match_id"], p["persona_id"]) not in existing:
                todo.append((p, match_event, market_ctx))

    total_todo  = len(todo)
    total_exist = len(existing)
    print(f"[INFO] Total pairs: {len(personas_use) * len(df)} | "
          f"Already done: {total_exist} | To generate: {total_todo}")

    if total_todo == 0:
        print("[INFO] All reactions already generated. Skipping.")
        export_step2_reactions_excel(reaction_path)
        return

    # Shared state protected by a lock
    write_lock   = threading.Lock()
    done_count   = 0
    start_time   = time.time()

    def worker(args):
        p, match_event, market_ctx = args
        return _process_one_reaction(p, match_event, market_ctx, reaction_opts)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(worker, args): args for args in todo}

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                args = futures[future]
                p, match_event, _ = args
                print(f"[WARN] Failed {match_event['match_id']} / {p['persona_id']}: {exc}")
                continue

            with write_lock:
                jsonl_write(reaction_path, [result], mode="a")
                existing.add((result["match_id"], result["persona_id"]))
                done_count += 1

                # Progress log every 100 completions
                if done_count % 100 == 0 or done_count == total_todo:
                    elapsed   = time.time() - start_time
                    rate      = done_count / elapsed if elapsed > 0 else 0
                    remaining = (total_todo - done_count) / rate if rate > 0 else float("inf")
                    print(
                        f"[PROGRESS] {done_count}/{total_todo} "
                        f"({done_count/total_todo*100:.1f}%) | "
                        f"{rate:.1f} req/s | "
                        f"ETA {remaining/3600:.1f}h"
                    )

    print(f"[INFO] Newly generated reactions: {done_count}")
    print(f"[INFO] reactions.jsonl -> {reaction_path}")
    export_step2_reactions_excel(reaction_path)


# =============================
# 7) BUILD MATCH FEATURES (overall + by persona_type)
# =============================
def aggregate_group(reactions: List[Dict[str, Any]]) -> Dict[str, float]:
    actions = [r["action"] for r in reactions]
    net = np.array([r["signals"]["net_demand_-1_to_1"] for r in reactions], dtype=float)
    liq = np.array([r["signals"]["liquidity_amp_1_to_3"] for r in reactions], dtype=float)
    size = np.array([r["size_0_1"] for r in reactions], dtype=float)

    buy = np.array([a == "BUY" for a in actions])
    sell = np.array([a == "SELL" for a in actions])
    trade = buy | sell

    mean_net = float(np.mean(net)) if len(net) else 0.0
    buy_share = float(np.mean(buy)) if len(buy) else 0.0
    sell_share = float(np.mean(sell)) if len(sell) else 0.0
    trade_share = float(np.mean(trade)) if len(trade) else 0.0
    mean_size_trade = float(np.mean(size[trade])) if np.any(trade) else 0.0
    liq_weighted_pressure = float(np.mean(net * liq)) if len(net) else 0.0

    return {
        "mean_net_demand": mean_net,
        "buy_share": buy_share,
        "sell_share": sell_share,
        "trade_share": trade_share,
        "mean_size_trade": mean_size_trade,
        "liq_weighted_pressure": liq_weighted_pressure
    }


def build_match_features(df: pd.DataFrame, personas: List[Dict[str, Any]]) -> pd.DataFrame:
    ensure_outdir()
    reaction_path = OUTDIR / "reactions.jsonl"
    if not reaction_path.exists():
        raise FileNotFoundError(f"Missing {reaction_path}. Run generate_reactions() first.")

    # persona lookup: persona_id -> persona_type
    pid_to_type = {p["persona_id"]: p["persona_type"] for p in personas}

    reactions = jsonl_read(reaction_path)
    # attach persona_type
    for r in reactions:
        r["persona_type"] = pid_to_type.get(r["persona_id"], "UNKNOWN")

    # group by match_id
    by_match: Dict[str, List[Dict[str, Any]]] = {}
    for r in reactions:
        by_match.setdefault(r["match_id"], []).append(r)

    rows = []
    for _, row in df.iterrows():
        match_event = make_match_event(row)
        match_id = match_event["match_id"]

        # deterministic features
        surprise = compute_realized_surprise(match_event)
        venue_home = 1.0 if match_event["venue"] == "HOME" else 0.0
        goal_diff = float(match_event["goal_diff"])
        importance = float(match_event["importance_0_1"])
        opp = float(match_event["opponent_strength_0_1"])

        market_ctx = make_market_context(row)

        # label
        y = row.get(LABEL_COL, np.nan)
        y = float(y) if not pd.isna(y) else np.nan

        # aggregates
        rlist = by_match.get(match_id, [])
        overall = aggregate_group(rlist)

        # type-specific
        type_feats = {}
        for t in ["TEMPORARY_FAN","LOCAL_FAN","FANATICAL_FAN","INVESTOR_FAN"]:
            sub = [r for r in rlist if r.get("persona_type") == t]
            agg = aggregate_group(sub)
            for k, v in agg.items():
                type_feats[f"{k}__{t.lower()}"] = v

        # Map source price columns to standard names expected by _detect_price_cols()
        # in compare_predictions_to_market.py (looks for close_t0 / close_tplus1).
        # Source data: pre_price = price before match (t0), post_price = price after (t+1).
        def _safe_float(val):
            try:
                v = float(val)
                return v if v == v else float("nan")  # nan check
            except (TypeError, ValueError):
                return float("nan")

        close_t0     = _safe_float(row.get("pre_price",  row.get("event_price_t0", float("nan"))))
        close_tplus1 = _safe_float(row.get("post_price", float("nan")))

        rows.append({
            "match_id": match_id,
            "match_date": pd.to_datetime(row["match_date"]),
            "venue_home": venue_home,
            "goal_diff": goal_diff,
            "importance_0_1": importance,
            "opponent_strength_0_1": opp,
            "surprise": surprise,
            "recent_return_5d": float(row.get("recent_return_5d", 0.0)) if not pd.isna(row.get("recent_return_5d", 0.0)) else 0.0,
            "recent_volatility_20d": float(row.get("recent_volatility_20d", 0.0)) if not pd.isna(row.get("recent_volatility_20d", 0.0)) else 0.0,
            "market_index_return_1d": float(row.get("return_market_t0", 0.0)) if not pd.isna(row.get("return_market_t0", 0.0)) else 0.0,
            "liquidity_thinness_0_1": float(market_ctx["liquidity_thinness_0_1"]),
            "close_t0": close_t0,
            "close_tplus1": close_tplus1,
            **overall,
            **type_feats,
            "y": y
        })

    feat_df = pd.DataFrame(rows).sort_values("match_date").reset_index(drop=True)
    out_path = OUTDIR / "match_features.csv"
    feat_df.to_csv(out_path, index=False)
    print(f"[INFO] match_features.csv -> {out_path}")
    export_step3_features_excel(feat_df)
    return feat_df


# =============================
# 8) MODELING + VALIDATION (time-aware)
# =============================
def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    # closed-form ridge: (X'X + lam I)^{-1} X'y
    XTX = X.T @ X
    p = XTX.shape[0]
    beta = np.linalg.solve(XTX + lam * np.eye(p), X.T @ y)
    return beta


def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return X @ beta


def rolling_origin_eval(df: pd.DataFrame, feature_cols: List[str], y_col: str = "y", min_train: int = 50, lam: float = 1e-3) -> Dict[str, float]:
    d = df.dropna(subset=[y_col]).reset_index(drop=True)
    n = len(d)
    if n < min_train + 5:
        print(f"[WARN] Not enough labeled rows for validation: n={n}, need >= {min_train+5}. Skipping validation.")
        return {"rmse": float("nan"), "mae": float("nan"), "hit_rate": float("nan"), "r2_oos": float("nan"), "n_test": 0}

    preds = []
    trues = []

    for t in range(min_train, n):
        train = d.iloc[:t]
        test = d.iloc[t:t+1]

        X_train = train[feature_cols].to_numpy(dtype=float)
        y_train = train[y_col].to_numpy(dtype=float)
        X_test = test[feature_cols].to_numpy(dtype=float)
        y_test = test[y_col].to_numpy(dtype=float)

        X_train_s, X_test_s, _, _ = standardize_train_test(X_train, X_test)
        # add intercept
        X_train_s = np.c_[np.ones(len(X_train_s)), X_train_s]
        X_test_s = np.c_[np.ones(len(X_test_s)), X_test_s]

        beta = fit_ridge(X_train_s, y_train, lam=lam)
        y_hat = float(predict(X_test_s, beta)[0])

        preds.append(y_hat)
        trues.append(float(y_test[0]))

    preds = np.array(preds, dtype=float)
    trues = np.array(trues, dtype=float)

    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    mae = float(np.mean(np.abs(preds - trues)))
    hit = float(np.mean((preds > 0) == (trues > 0)))

    # out-of-sample R2
    denom = np.sum((trues - trues.mean()) ** 2)
    r2 = float(1 - np.sum((trues - preds) ** 2) / denom) if denom > 0 else float("nan")

    return {"rmse": rmse, "mae": mae, "hit_rate": hit, "r2_oos": r2, "n_test": int(len(trues))}


def eval_metrics_from_pred_df(pred_df: pd.DataFrame, y_true_col: str, y_hat_col: str) -> Dict[str, float]:
    """Compute evaluation metrics from a prediction table (robust to NaNs)."""
    d = pred_df.dropna(subset=[y_true_col, y_hat_col]).copy()
    if d.empty:
        return {"rmse": float("nan"), "mae": float("nan"), "hit_rate": float("nan"), "r2_oos": float("nan"), "n_test": 0}

    trues = d[y_true_col].to_numpy(dtype=float)
    preds = d[y_hat_col].to_numpy(dtype=float)

    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    mae = float(np.mean(np.abs(preds - trues)))
    hit = float(np.mean((preds > 0) == (trues > 0)))

    denom = np.sum((trues - trues.mean()) ** 2)
    r2 = float(1 - np.sum((trues - preds) ** 2) / denom) if denom > 0 else float("nan")

    return {"rmse": rmse, "mae": mae, "hit_rate": hit, "r2_oos": r2, "n_test": int(len(trues))}


def rolling_origin_predict_table(
    df: pd.DataFrame,
    feature_cols: List[str],
    y_col: str = "y",
    min_train: int = 50,
    lam: float = 1e-3,
) -> pd.DataFrame:
    """Rolling-origin one-step-ahead prediction table.

    Returns a table aligned to labeled rows (rows where y_col is not missing):
      match_id, match_date, y_true, y_hat

    Notes
    - The first `min_train` labeled rows have y_hat = NaN (used for training only).
    - This uses the same standardize + intercept + ridge logic as rolling_origin_eval().
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    d = df.dropna(subset=[y_col]).sort_values("match_date").reset_index(drop=True)
    n = len(d)

    out = pd.DataFrame({
        "match_id": d["match_id"].astype(str),
        "match_date": pd.to_datetime(d["match_date"]),
        "y_true": d[y_col].astype(float),
        "y_hat": np.nan,
    })

    if n < min_train + 1:
        # Not enough data to produce any OOS predictions
        return out

    for t in range(min_train, n):
        train = d.iloc[:t]
        test = d.iloc[t:t+1]

        X_train = train[feature_cols].to_numpy(dtype=float)
        y_train = train[y_col].to_numpy(dtype=float)
        X_test = test[feature_cols].to_numpy(dtype=float)

        X_train_s, X_test_s, _, _ = standardize_train_test(X_train, X_test)
        X_train_s = np.c_[np.ones(len(X_train_s)), X_train_s]
        X_test_s = np.c_[np.ones(len(X_test_s)), X_test_s]

        beta = fit_ridge(X_train_s, y_train, lam=lam)
        out.loc[t, "y_hat"] = float(predict(X_test_s, beta)[0])

    return out




def validate_models(feat_df: pd.DataFrame):
    # Baseline: match + market only
    baseline_cols = [
        "venue_home","goal_diff","importance_0_1","opponent_strength_0_1","surprise",
        "recent_return_5d","recent_volatility_20d","market_index_return_1d","liquidity_thinness_0_1"
    ]

    # [FIX-5] Indentation bug fix: the llm_cols block was outside the function body,
    # causing a NameError on feat_df before main() was called.
    # Augmented: baseline + overall + type-specific aggregates
    llm_cols = [
        c for c in feat_df.columns
        if c.startswith(("mean_net_demand","buy_share","sell_share","trade_share","mean_size_trade","liq_weighted_pressure"))
    ]
    augmented_cols = baseline_cols + llm_cols

    # Choose min_train relative to data size
    # Goal: produce more than 1 OOS prediction even in small pilots (e.g., 10 matches)
    labeled_n = int(feat_df["y"].notna().sum())

    if labeled_n < 15:
        # Keep at least ~5 OOS points when possible (while ensuring >=3 training points)
        min_test = min(5, max(1, labeled_n - 3))
        min_train = max(3, labeled_n - min_test)
        print(
            f"[WARN] Small labeled sample (y={labeled_n}). "
            f"Using min_train={min_train} (n_test={labeled_n - min_train}). "
            "Metrics may be unstable."
        )
    else:
        # Larger samples: keep prior heuristic
        min_train = max(50, int(0.6 * labeled_n)) if labeled_n >= 80 else max(30, int(0.6 * labeled_n))

    # Safety cap (cannot exceed labeled_n-1)
    min_train = min(min_train, max(1, labeled_n - 1))

    print(f"[INFO] Validation labeled_n={labeled_n}, min_train={min_train}")

    # --- Generate rolling-origin prediction tables (Step3_Predictions.xlsx) ---
    pred_base = rolling_origin_predict_table(feat_df, baseline_cols, y_col="y", min_train=min_train, lam=1e-2)
    pred_aug  = rolling_origin_predict_table(feat_df, augmented_cols, y_col="y", min_train=min_train, lam=1e-2)

    pred_base = pred_base.rename(columns={"y_hat": "y_hat_baseline"})
    pred_aug  = pred_aug.rename(columns={"y_hat": "y_hat_augmented"})

    # Merge by match_id + match_date to be safe
    pred_df = pred_base.merge(
        pred_aug[["match_id","match_date","y_hat_augmented"]],
        on=["match_id","match_date"],
        how="left",
    )

    # Export for downstream comparisons (compare_predictions_to_market.py)
    export_step3_predictions_excel(pred_df)

    # --- Metrics from the same prediction tables (no duplicated computation) ---
    base = eval_metrics_from_pred_df(pred_df, "y_true", "y_hat_baseline")
    aug  = eval_metrics_from_pred_df(pred_df, "y_true", "y_hat_augmented")

    print("\n=== Rolling-origin OOS validation ===")
    print("[Baseline] ", base)
    print("[Augmented: +LLM features] ", aug)

    print("\nDelta (Aug - Base):")
    print({
        "rmse": aug["rmse"] - base["rmse"],
        "mae": aug["mae"] - base["mae"],
        "hit_rate": aug["hit_rate"] - base["hit_rate"],
        "r2_oos": aug["r2_oos"] - base["r2_oos"],
    })

    return base, aug




# =============================
# 9) MAIN: 1)~5) END-TO-END
# =============================
def main():
    ensure_outdir()
    _check_persona_status()

    # Load match_stock data
    df = pd.read_excel(EXCEL_PATH).sort_values("match_date").reset_index(drop=True)

    # Rolling market context from return_manu_t0
    if "return_manu_t0" in df.columns:
        df["recent_return_5d"] = df["return_manu_t0"].rolling(5, min_periods=1).mean()
        df["recent_volatility_20d"] = df["return_manu_t0"].rolling(20, min_periods=5).std()
    else:
        df["recent_return_5d"] = 0.0
        df["recent_volatility_20d"] = 0.0

    # Optional limit for debug
    if MAX_MATCHES is not None:
        df = df.head(int(MAX_MATCHES)).copy()

    # ── Step control ──────────────────────────────────────────────────────────
    # SKIP_TO_STEP: which step to start from
    #   1 = full run from persona generation
    #   2 = load existing personas.jsonl, run Step2 reactions onward  (current)
    #   3 = load personas + reactions, run Step3 feature aggregation onward
    #   4 = load existing match_features.csv, run Step4 validation only
    # ──────────────────────────────────────────────────────────────────────────

    # Step 1: Persona generation
    if SKIP_TO_STEP <= 1:
        personas = generate_or_load_personas(force_regen=True, seed=2026)
    else:
        print(f"[SKIP] Step1 (SKIP_TO_STEP={SKIP_TO_STEP}) — loading existing personas.jsonl")
        personas = generate_or_load_personas(force_regen=False, seed=2026)

    # Step 2: Reaction generation (parallel, resumable)
    if SKIP_TO_STEP <= 2:
        generate_reactions(df, personas)
    else:
        print(f"[SKIP] Step2 reaction generation (SKIP_TO_STEP={SKIP_TO_STEP})")

    # Step 3: Feature aggregation
    if SKIP_TO_STEP <= 3:
        feat_df = build_match_features(df, personas)
    else:
        feat_csv = OUTDIR / "match_features.csv"
        if not feat_csv.exists():
            raise FileNotFoundError(
                f"SKIP_TO_STEP={SKIP_TO_STEP} but match_features.csv not found: {feat_csv}"
            )
        feat_df = pd.read_csv(feat_csv, parse_dates=["match_date"])
        print(f"[SKIP] Step3 — loaded existing match_features.csv ({len(feat_df)} rows)")

    # Step 4: Model validation
    base, aug = validate_models(feat_df)
    export_step4_validation_excel([
        {"model":"Baseline","rmse":base["rmse"],"mae":base["mae"],"hit_rate":base["hit_rate"],"r2_oos":base["r2_oos"],"n_test":base["n_test"]},
        {"model":"Augmented_LLM","rmse":aug["rmse"],"mae":aug["mae"],"hit_rate":aug["hit_rate"],"r2_oos":aug["r2_oos"],"n_test":aug["n_test"]},
        {"model":"Delta(aug-base)","rmse":aug["rmse"]-base["rmse"],"mae":aug["mae"]-base["mae"],"hit_rate":aug["hit_rate"]-base["hit_rate"],"r2_oos":aug["r2_oos"]-base["r2_oos"],"n_test":base["n_test"]}
    ])


if __name__ == "__main__":
    main()