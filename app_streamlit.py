"""PayGuard Deployment Audit: Streamlit view of one Gemini 4 Carbon experiment.

streamlit run app_streamlit.py
"""

import base64
import html
import io
import json
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import dotenv_values
from PIL import Image

import config as cfg

# Google palette (Partner Marketing Hub). White text needs 4.5:1, so filled swatches are
# darkened with black until they pass:
BLUE_FILL = "#3974D4"   # Blue 500 #4285F4 darkened 13%: white text 4.54:1 (Blue 500 itself: 3.56:1)
RED_FILL = "#D73E31"    # Red 500 #EA4335 darkened 8%: white text 4.54:1 (Red 500 itself: 3.92:1)
GREEN_FILL = "#2A8642"  # Green 500 #34A853 darkened 20%: white text 4.58:1 (Green 500 itself: 3.06:1)
YELLOW = "#FBBC05"      # Yellow 500 keeps black text: 12.3:1
GREY_700, BORDER = "#5F6368", "rgba(154, 160, 166, 0.4)"  # Grey 500 at 40% opacity, borders only

RUNS = Path("runs")
RECORDED = Path("recorded_runs")  # the tutorial's attempts, shown until you save a run of your own
ICON = {"deployable": GREEN_FILL, "not deployable": RED_FILL, "inconclusive": YELLOW}

st.set_page_config(page_title="PayGuard Deployment Audit", page_icon=":material/policy:", layout="wide")
st.logo("assets/datacamp-logo.png", size="large", link="https://www.datacamp.com")

st.markdown(f"""
<style>
:root {{ --ui: "Google Sans", "Product Sans", Arial, sans-serif; --mono: ui-monospace, Menlo, Consolas, monospace; }}
.stApp, .stApp p, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp label, .stApp button p {{ font-family: var(--ui); }}
[data-testid="stIconMaterial"] {{ font-family: "Material Symbols Rounded" !important; }}
.stApp {{ background: #FFFFFF; color: #000000; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {BORDER}; }}
.block-container {{ max-width: 1180px; padding-top: 2rem; }}
.g-bar {{ position: fixed; top: 0; left: 0; right: 0; height: 3px; z-index: 999999;
          background: linear-gradient(90deg, #EA4335, #FBBC05, #34A853, #4285F4); }}
h1, h2, h3 {{ color: #000000 !important; }}
.stMarkdown p, .stMarkdown li, .stMarkdown strong {{ color: #000000; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *, [data-testid="stMetricLabel"] * {{ color: {GREY_700} !important; }}
[data-testid="stCaptionContainer"] {{ opacity: 1 !important; }}  /* Streamlit fades captions to 60%, which drops Grey 700 below 4.5:1 */
[data-testid="stMetricValue"] {{ font-family: var(--mono); color: #000000; font-size: 1.35rem; }}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{ font-family: var(--mono) !important; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: {BORDER} !important; border-radius: 0.6rem !important; background: #FFFFFF; }}
.brand {{ display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 0.25rem; }}
.brand img {{ height: 26px; width: auto; display: block; }}
.model-pill {{ font-family: var(--mono); font-size: 0.8rem; padding: 0.15rem 0.65rem; background: #FFFFFF;
               border: 1px solid {BORDER}; border-radius: 0.6rem; color: #000000; }}
.st-key-cards [data-testid="stColumn"] > [data-testid="stVerticalBlock"],
.st-key-cards [data-testid="stLayoutWrapper"], .st-key-cards [class*="st-key-card-"] {{ height: 100%; }}
.stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.6rem; margin: 0.6rem 0 1rem; font-family: var(--ui); }}
@media (max-width: 900px) {{ .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.stat {{ background: #FFFFFF; border: 1px solid {BORDER}; border-radius: 0.6rem; padding: 0.55rem 0.8rem; }}
.stat .label {{ font-size: 0.78rem; color: {GREY_700}; }}
.stat .value {{ font-size: 1.1rem; font-weight: 700; font-family: var(--mono); color: #000000; overflow-wrap: anywhere; }}
.kv, .side-head {{ font-family: var(--ui); }}
.side-head {{ font-size: 0.85rem; font-weight: 700; color: #000000; margin: 0.4rem 0 0.1rem; }}
.chip {{ display: inline-block; padding: 0.2rem 0.75rem; margin: 0 0.35rem 0.35rem 0; border-radius: 999px;
         font-size: 0.85rem; border: 1px solid {BORDER}; color: {GREY_700}; background: #FFFFFF; }}
.chip.on {{ color: #FFFFFF; border-color: transparent; }}
.chip.on.inconclusive {{ color: #000000; }}
.chip.level {{ font-family: var(--mono); color: #000000; }}
button[kind="primary"] {{ background: {BLUE_FILL} !important; border-color: {BLUE_FILL} !important; color: #FFFFFF !important; }}
button[kind="primary"] * {{ color: #FFFFFF !important; }}
button[kind="secondary"] {{ background: #FFFFFF !important; border: 1px solid {BORDER} !important; color: #000000 !important; }}
.kv {{ width: 100%; border-collapse: collapse; border: none; font-size: 0.85rem; margin: 0 0 0.8rem; display: table; }}
.kv tr, .kv td {{ border: none !important; background: transparent !important; }}
.kv td {{ padding: 0.28rem 0; border-bottom: 1px solid {BORDER} !important; vertical-align: top; }}
.kv td:first-child {{ color: {GREY_700}; padding-right: 0.75rem; }}
.kv td:last-child {{ font-family: var(--mono); color: #000000; text-align: right; white-space: nowrap; }}
.harness {{ font-size: 0.8rem; letter-spacing: 0.04em; text-transform: uppercase; color: {GREY_700}; }}
</style>
<div class="g-bar"></div>
""", unsafe_allow_html=True)

if "running" not in st.session_state:
    st.session_state.update(running=False, trigger=False, record=None, spent=0.0, spent_period=None)


def esc(value) -> str:
    return html.escape(str(value))


def pct(x) -> str:
    return "n/a" if x is None else f"{x:.2%}"


def rate_in_force() -> tuple[str, dict]:
    intro = cfg.PRICES["introductory"]
    return ("introductory", intro) if date.today() <= intro["until"] else ("standard", cfg.PRICES["standard"])


# Header ------------------------------------------------------------------------------------
def trimmed_logo(path: str) -> str:
    """The PNG has wide transparent margins; crop them so 26px is the height of the mark itself."""
    img = Image.open(path).convert("RGBA")
    img = img.crop(img.getchannel("A").getbbox() or (0, 0, *img.size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


logo = trimmed_logo("assets/gemini-logo.png")
st.markdown(f'<div class="brand"><img src="data:image/png;base64,{logo}" alt="Gemini logo">'
            f'<span class="model-pill">{esc(cfg.MODEL_ID)}</span></div>', unsafe_allow_html=True)
st.title("PayGuard Deployment Audit")
st.markdown("Carbon audits PayGuard's fraud classifier against three deployment rules, and Python checks the verdict.")

# Sidebar -----------------------------------------------------------------------------------
def kv(rows) -> str:
    return '<table class="kv">' + "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in rows) + "</table>"


with st.sidebar:
    st.subheader("Run settings")
    st.markdown(kv([
        ("Model", cfg.MODEL_ID),
        ("thinking_level", cfg.PRIMARY_THINKING_LEVEL),
        ("Clean rerun", cfg.RERUN_THINKING_LEVEL),
        ("Max turns", cfg.MAX_TURNS),
        ("Max tool calls", cfg.MAX_TOOL_CALLS),
        ("Max seconds", f"{cfg.MAX_ELAPSED_SECONDS:,}"),
        ("Max spend", f"${cfg.MAX_SPEND_USD:.2f}"),
    ]), unsafe_allow_html=True)
    st.caption("Limits apply to each attempt and are enforced in code.")
    st.subheader("Price per million tokens")
    for label, p in (("Introductory, through Nov 30, 2026", cfg.PRICES["introductory"]),
                     ("Standard, from Dec 1, 2026", cfg.PRICES["standard"])):
        st.markdown(f'<div class="side-head">{esc(label)}</div>', unsafe_allow_html=True)
        st.markdown(kv([("Input", f"${p['input']:.2f}"), ("Output + thinking", f"${p['output']:.2f}"),
                        ("Cached input", f"${p['cached']:.2f}")]), unsafe_allow_html=True)
    st.caption("Gemini 4 Carbon has no free tier. Every token is billed, thinking included.")
    if st.session_state.spent_period:
        st.metric(f"Spent this session ({st.session_state.spent_period} rate)",
                  f"${st.session_state.spent:.4f}", delta_color="off")

# What this run uses ------------------------------------------------------------------------
st.subheader("What this run uses")
CARDS = [
    (":material/sync_alt:", "Interactions API", "client.interactions.create(), with state carried by previous_interaction_id."),
    (":material/build:", "Three read-only tools", "Audit context, one global threshold, one permitted policy change."),
    (":material/rule:", "Hidden evaluator", "Exhaustive search over every attainable threshold, outside the agent's reach."),
    (":material/lock:", "Sealed holdout", "Four later weeks, opened once after the recommendation is frozen."),
    (":material/data_object:", "Structured report", "A final request with a JSON schema and no tools."),
    (":material/payments:", "Usage-based cost", "Tokens from every turn, priced at the rate in force."),
]
with st.container(key="cards"):
    for row in range(0, len(CARDS), 2):  # one st.columns per row, so each pair of cards lines up
        for col, (icon, name, caption) in zip(st.columns(2), CARDS[row:row + 2]):
            with col.container(border=True, key=f"card-{name.split()[0].lower()}"):
                st.markdown(f"{icon} **{name}**")
                st.caption(caption)

# Buttons -----------------------------------------------------------------------------------
def start_run():
    key = dotenv_values(".env").get("GEMINI_API_KEY") if Path(".env").exists() else None
    if key:
        st.session_state.update(running=True, trigger=True, key_error=False)
    else:
        st.session_state.key_error = True


b1, b2 = st.columns(2)
b1.button("Audit the deployment", icon=":material/fact_check:", type="primary", width="stretch",
          disabled=st.session_state.running, on_click=start_run)
show_latest = b2.button("Show the latest recorded run", icon=":material/history:", width="stretch",
                        disabled=st.session_state.running)
st.caption("A live run calls the Gemini API and is billed to GEMINI_API_KEY in .env.")

if st.session_state.get("key_error"):
    st.error("GEMINI_API_KEY is missing from .env, so no run was started. Add it and try again.")

if st.session_state.trigger:
    st.session_state.trigger = False
    key = dotenv_values(".env").get("GEMINI_API_KEY")
    if key:
        from agent import run_experiment
        with st.status("Running the audit: medium attempt, evaluator, holdout, then the clean high rerun",
                       expanded=True) as status:
            box = st.container(height=280)

            def log(line: str):
                box.text(line.replace(key, "***"))

            try:
                path = run_experiment(log)
                record = json.loads(path.read_text())
                st.session_state.record = record
                period, _ = rate_in_force()
                st.session_state.spent += sum(record[k]["cost"][f"{period}_usd"] for k in ("medium", "high"))
                st.session_state.spent_period = period
                status.update(label="Run saved", state="complete", expanded=False)
            except Exception as exc:  # show the failure, keep the page alive
                status.update(label="The run stopped", state="error")
                st.error(f"{type(exc).__name__}: {str(exc).replace(key, '***')}")
        st.session_state.running = False
        st.rerun()

if show_latest:
    files = sorted(RUNS.glob("*.json"), key=lambda p: p.stat().st_mtime) if RUNS.exists() else []
    if not files and RECORDED.exists():
        files = sorted(RECORDED.glob("*.json"))
    if not files:
        st.info("No recorded runs yet. Start a live audit to create one in runs/.")
        st.session_state.record = None
    else:
        st.session_state.record = json.loads(files[-1].read_text())
        st.session_state.record_name = files[-1].name


# Results -----------------------------------------------------------------------------------
def chips(status: str | None) -> str:
    out = ""
    for name, color in ICON.items():
        on = name == status
        style = f' style="background:{color}"' if on else ""
        out += f'<span class="chip{" on" if on else ""} {name.replace(" ", "-")}{" inconclusive" if on and name == "inconclusive" else ""}"{style}>{esc(name)}</span>'
    return out


def stats_html(items) -> str:
    cells = "".join(f'<div class="stat"><div class="label">{esc(a)}</div><div class="value">{esc(b)}</div></div>'
                    for a, b in items)
    return f'<div class="stats">{cells}</div>'


def tool_table(calls: list[dict]) -> pd.DataFrame:
    rows = []
    for c in calls:
        r, a = c["result"], c["arguments"]
        rows.append({
            "Call": c["n"], "Turn": c["turn"], "Tool": c["name"],
            "Arguments": ", ".join(f"{k}={v}" for k, v in a.items()) or "none",
            "Overall FVR": pct(r.get("overall_fraud_value_recall")),
            "Website FVR": pct(r.get("by_channel", {}).get("website", {}).get("fraud_value_recall")),
            "Mobile App FVR": pct(r.get("by_channel", {}).get("mobile_app", {}).get("fraud_value_recall")),
            "Busiest day": r.get("daily_flags", {}).get("max"),
            "Error": r.get("error", ""),
        })
    return pd.DataFrame(rows)


def requirement_rows(result: dict) -> pd.DataFrame:
    return pd.DataFrame([
        ("R1 overall fraud-value recall >= 85%", pct(result["overall_fraud_value_recall"]), result["R1_overall"]),
        ("R2 Website fraud-value recall >= 70%", pct(result["website_fraud_value_recall"]), result["R2_website"]),
        ("R2 Mobile App fraud-value recall >= 70%", pct(result["mobile_app_fraud_value_recall"]), result["R2_mobile_app"]),
        (f"R3 busiest day <= {result['capacity_used']}", f"{result['max_daily_flags']} on {result['peak_day']}",
         result["R3_capacity"]),
    ], columns=["Requirement", "Measured", "Pass"])


def render(record: dict):
    med, high = record["medium"], record.get("high", {})
    rep = med.get("report") or {}
    ev = med.get("evaluator", {})
    score = ev.get("score", {})
    hold = med.get("holdout")

    st.subheader("Result")
    st.markdown(chips(rep.get("status")) + f'<span class="chip level">{esc(med["thinking_level"])}</span>'
                + (f'<span class="chip level">{esc(high.get("thinking_level", ""))}</span>' if high else ""),
                unsafe_allow_html=True)

    change = {"channel_thresholds": "channel split", "capacity_increase": "more capacity"}
    st.markdown(stats_html([
        ("Carbon's verdict (medium)", rep.get("status", "no report")),
        ("Evaluator agrees", "yes" if score.get("correct_audit") else "no"),
        ("Holdout", "not run" if not hold else ("passes all" if hold["holdout"].get("passes_all") else "fails")),
        ("Recommended change", change.get(rep.get("recommended_change_type"), rep.get("recommended_change_type") or "n/a")),
        ("API turns", med["cost"]["api_turns"]),
        ("Tool calls", len(med["tool_calls"])),
        ("Elapsed", f"{med['elapsed_seconds']:.0f} s"),
        (f"Cost ({med['rate_period']} rate)", f"${med['cost'][med['rate_period'] + '_usd']:.4f}"),
    ]), unsafe_allow_html=True)

    tabs = st.tabs([":material/description: Report", ":material/tune: Thresholds", ":material/rule: Evaluator",
                    ":material/lock: Holdout", ":material/psychology: Thinking", ":material/payments: Cost"])

    with tabs[0]:
        st.caption("Written by Carbon in the final no-tools request. A formatted report is not proof.")
        if not rep:
            st.warning(f"No report. Stop reason: {med['stop_reason']}")
        else:
            st.markdown(f"**Summary.** {esc(rep['summary'])}")
            st.dataframe(pd.DataFrame([
                ("Status", rep["status"]),
                ("Current configuration passes", rep["current_configuration_passes"]),
                ("Failed by current configuration", ", ".join(rep["requirements_failed_by_current_configuration"])),
                ("Single global threshold feasible", rep["single_threshold_feasible"]),
                ("Conflicting requirements", ", ".join(rep["conflicting_requirements"])),
                ("Recommended change", rep["recommended_change_type"]),
                ("Global threshold", rep["recommended_global_threshold"]),
                ("Daily capacity", rep["recommended_daily_capacity"]),
                ("Website threshold", rep["recommended_website_threshold"]),
                ("Mobile App threshold", rep["recommended_mobile_app_threshold"]),
            ], columns=["Field", "Value"]).astype(str), hide_index=True, width="stretch")
            st.markdown("**Evidence cited**")
            st.dataframe(pd.DataFrame(rep["evidence"]), hide_index=True, width="stretch")
            st.markdown("**Required approvals**")
            st.dataframe(pd.DataFrame({"Approval": rep["required_approvals"]}), hide_index=True, width="stretch")
            st.markdown("**Unresolved issues**")
            st.dataframe(pd.DataFrame({"Issue": rep["unresolved_issues"]}), hide_index=True, width="stretch")
            st.markdown("**Harness checks on the report**")
            st.dataframe(pd.DataFrame(list(med["checks"].items()), columns=["Check", "Result"]).astype(str),
                         hide_index=True, width="stretch")

    with tabs[1]:
        st.caption("Every tool call in the medium attempt, in order. Numbers come from tools.py.")
        st.dataframe(tool_table(med["tool_calls"]), hide_index=True, width="stretch")
        curve = pd.DataFrame(ev.get("global_curve", []))
        if not curve.empty:
            tested = sorted({c["arguments"]["threshold"] for c in med["tool_calls"]
                             if c["name"] == "evaluate_global_threshold" and not c["is_error"]})
            view = curve[(curve["threshold"] >= 0.05) & (curve["threshold"] <= 0.4)]
            long = view.melt("threshold", ["overall", "website", "mobile_app"], "series", "recall")
            x = alt.X("threshold:Q", title="Global threshold", scale=alt.Scale(domain=[0.05, 0.4]))
            marks = alt.Chart(pd.DataFrame({"threshold": [t for t in tested if 0.05 <= t <= 0.4]})).mark_rule(
                color=GREEN_FILL, opacity=0.6).encode(x=x)
            top = alt.Chart(long).mark_line(clip=True).encode(
                x=x, y=alt.Y("recall:Q", title="Fraud-value recall", axis=alt.Axis(format="%", minExtent=44),
                             scale=alt.Scale(domain=[0.3, 1])),
                color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                                scale=alt.Scale(domain=["overall", "website", "mobile_app"],
                                                range=["#000000", BLUE_FILL, RED_FILL])),
                strokeDash=alt.StrokeDash("series:N", legend=None))
            rules = alt.Chart(pd.DataFrame({"y": [cfg.MIN_OVERALL_FRAUD_VALUE_RECALL,
                                                  cfg.MIN_CHANNEL_FRAUD_VALUE_RECALL]})).mark_rule(
                color=GREY_700, strokeDash=[4, 4]).encode(y="y:Q")
            st.altair_chart((marks + top + rules).properties(height=240), width="stretch")
            bottom = alt.Chart(view).mark_line(color="#000000", clip=True).encode(
                x=x, y=alt.Y("max_daily:Q", title="Busiest-day flags", axis=alt.Axis(minExtent=44), scale=alt.Scale(domain=[50, 300], clamp=True)))
            cap = alt.Chart(pd.DataFrame({"y": [cfg.MAX_DAILY_REVIEWS]})).mark_rule(
                color=GREY_700, strokeDash=[4, 4]).encode(y="y:Q")
            st.altair_chart((marks + bottom + cap).properties(height=200), width="stretch")
            st.caption("Lines: exhaustive evaluator curve. Green rules: global thresholds Carbon tested. Dashed: policy limits.")

    with tabs[2]:
        st.markdown('<div class="harness">Harness, not the model</div>', unsafe_allow_html=True)
        if not ev:
            st.info("The evaluator runs only after a report is frozen.")
        else:
            with st.container(border=True):
                ref = ev["reference"]
                st.markdown("**Exhaustive reference on validation**")
                st.dataframe(pd.DataFrame([
                    ("Search space", ref["search_space"]),
                    ("Global thresholds meeting R1, R2, R3", ref["feasible_global_thresholds"]),
                    ("Highest threshold meeting both recall rules", ref["highest_threshold_meeting_recall"]),
                    ("Lowest threshold meeting the daily cap", ref["lowest_threshold_meeting_capacity"]),
                    ("Smallest capacity for one global threshold", ref["min_capacity_single_threshold"]),
                    ("Feasible channel-threshold pairs at 200/day", ref["feasible_channel_pairs"]),
                    ("Preferred permitted change", ref["preferred_change"]),
                ], columns=["Item", "Value"]).astype(str), hide_index=True, width="stretch")
                st.markdown("**Carbon's recommendation, recomputed**")
                if "R1_overall" in ev["recommendation"]:
                    st.dataframe(requirement_rows(ev["recommendation"]), hide_index=True, width="stretch")
                else:
                    st.write(ev["recommendation"].get("reason", ""))
                st.markdown("**Score (rules fixed before the first API call)**")
                st.dataframe(pd.DataFrame(list(score.items()), columns=["Check", "Result"]).astype(str),
                             hide_index=True, width="stretch")
                st.markdown("**Evaluator self-test: configurations that must fail**")
                st.dataframe(pd.DataFrame([{
                    "Case": r["case"], "Website": r["website"], "Mobile App": r["mobile_app"],
                    "R1": r["R1_overall"], "R2 Website": r["R2_website"], "R2 Mobile App": r["R2_mobile_app"],
                    "R3": r["R3_capacity"], "Busiest day": r["max_daily_flags"], "Rejected": not r["passes_all"]}
                    for r in ev["self_test"]]), hide_index=True, width="stretch")

    with tabs[3]:
        st.markdown('<div class="harness">Harness, not the model</div>', unsafe_allow_html=True)
        if not hold:
            st.info("The holdout stays sealed until a recommendation is frozen.")
        else:
            with st.container(border=True):
                st.caption(f"Frozen report SHA-256 {hold['frozen_sha256'][:16]}... checked once. Nothing went back to the agent.")
                if "R1_overall" in hold["holdout"]:
                    v, h = requirement_rows(hold["validation"]), requirement_rows(hold["holdout"])
                    st.dataframe(pd.DataFrame({"Requirement": v["Requirement"],
                                               f"Validation ({hold['validation']['days']} days)": v["Measured"],
                                               "Validation pass": v["Pass"],
                                               f"Holdout ({hold['holdout']['days']} days)": h["Measured"],
                                               "Holdout pass": h["Pass"]}), hide_index=True, width="stretch")
                else:
                    st.write(hold["holdout"])

    with tabs[4]:
        rows = []
        for att in (med, high):
            if not att:
                continue
            sc = att.get("evaluator", {}).get("score", {})
            rows.append({"thinking_level": att["thinking_level"],
                         "Verdict": (att.get("report") or {}).get("status", "no report"),
                         "Stop": att["stop_reason"],
                         "Evaluator agrees": sc.get("correct_audit", "no report"),
                         "Recommended change": (att.get("report") or {}).get("recommended_change_type"),
                         "API turns": att["cost"]["api_turns"], "Tool calls": len(att["tool_calls"]),
                         "Thinking tokens": att["cost"]["tokens"]["thought"],
                         "Seconds": att["elapsed_seconds"],
                         "Cost intro": f"${att['cost']['introductory_usd']:.4f}"})
        st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True, width="stretch")
        st.caption("Thought summaries are condensed private reasoning. They are not evidence; the tool results are.")
        for att in (med, high):
            if att:
                with st.expander(f"{att['thinking_level']} thought summaries"):
                    for t_i, turn in enumerate(att["turns"], 1):
                        for line in turn["thought_summaries"]:
                            st.markdown(f"Turn {t_i}: {esc(line)}")

    with tabs[5]:
        for att in (med, high):
            if not att:
                continue
            st.markdown(f"**{esc(att['thinking_level'])}**")
            turns = pd.DataFrame([{"Turn": n, "Kind": t["kind"], "Status": t["status"], "Seconds": t["seconds"],
                                   **t["usage"]} for n, t in enumerate(att["turns"], 1)])
            st.dataframe(turns, hide_index=True, width="stretch", height=35 * (len(turns) + 1) + 3)  # every turn, no scroll
            tok = att["cost"]["tokens"]
            st.markdown(stats_html([
                ("Uncached input", f"{tok['input'] - tok['cached']:,}"),
                ("Cached input", f"{tok['cached']:,}"),
                ("Output + thinking", f"{tok['output'] + tok['thought']:,}"),
                ("Intro / standard", f"${att['cost']['introductory_usd']:.4f} / ${att['cost']['standard_usd']:.4f}"),
            ]), unsafe_allow_html=True)


if st.session_state.record:
    render(st.session_state.record)
