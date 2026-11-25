"""Visual layer: neon console theme, splash animation and presentational helpers.

All HTML here is static and authored by us. User/model text is escaped before it
reaches an HTML string; assistant answers are rendered with ``st.markdown`` so
markdown survives, and only bounded citation markers are turned into chips.
"""

from __future__ import annotations

import html
from typing import Any, Sequence

import streamlit as st

ACCENT = "#FF2E9F"
ACCENT_2 = "#22E4FF"
VIOLET = "#8B5CF6"
OK = "#34D399"
WARN = "#FBBF24"
BAD = "#F87171"

# Passage-card accents, cycled by citation index.
CARD_COLORS = ("#22E4FF", "#FF2E9F", "#8B5CF6", "#A3E635")

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

  :root {
    --bg: #07050B;
    --panel: #0E0A16;
    --line: rgba(255,255,255,.07);
    --text: #EFE7F6;
    --muted: #9A8FB3;
    --pink: #FF2E9F;
    --cyan: #22E4FF;
    --violet: #8B5CF6;
    --lime: #A3E635;
  }

  html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
  .stApp {
    background:
      radial-gradient(1000px 620px at -5% -12%, rgba(255,46,159,.13), transparent 55%),
      radial-gradient(980px 700px at 104% 108%, rgba(34,228,255,.10), transparent 58%),
      radial-gradient(700px 500px at 55% 0%, rgba(139,92,246,.07), transparent 60%),
      linear-gradient(165deg, #09060E 0%, #0A0611 50%, #07050B 100%);
    background-attachment: fixed;
  }
  #MainMenu, footer { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent; }
  .block-container { padding: 1.1rem 1.5rem .6rem 1.5rem; max-width: 100%; }

  /* ---------- sidebar ---------- */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0C0814 0%, #080510 100%);
    border-right: 1px solid var(--line);
    width: 296px !important;
    min-width: 296px !important;
  }
  section[data-testid="stSidebar"] > div:first-child { padding: 1rem .85rem 1.4rem .85rem; }

  .brand { display:flex; align-items:center; gap:.75rem; padding:.15rem 0 1.1rem 0; }
  .orb {
    width: 34px; height: 34px; border-radius: 50%; flex: 0 0 34px;
    background: conic-gradient(from 210deg, #FF2E9F, #8B5CF6 38%, #22E4FF 68%, #FF2E9F);
    box-shadow: 0 0 18px rgba(255,46,159,.55), inset 0 0 10px rgba(0,0,0,.65);
    position: relative;
  }
  .orb::after {
    content: ""; position: absolute; inset: 7px; border-radius: 50%;
    background: radial-gradient(circle at 32% 30%, rgba(255,255,255,.25), rgba(8,4,14,.9) 62%);
  }
  .brand-title {
    font-family: 'Space Grotesk', 'Inter', sans-serif;
    font-weight: 700; font-size: 1.22rem; letter-spacing: -.02em; color: #F7F2FC; line-height: 1.1;
  }
  .brand-sub { font-size: .65rem; color: #857A9E; letter-spacing: .13em; text-transform: uppercase; }

  .side-label {
    font-size: .66rem; letter-spacing: .15em; text-transform: uppercase;
    color: #7D7297; margin: 1.15rem 0 .45rem .2rem; font-weight: 600;
  }
  .coll-row {
    display:flex; align-items:center; gap:.6rem; padding:.48rem .6rem; border-radius:10px;
    color:#B7ABC9; font-size:.87rem; border:1px solid transparent; margin-bottom:.1rem;
  }
  .coll-row.active {
    color:#F6F0FB; border-color: rgba(255,46,159,.32);
    background: linear-gradient(90deg, rgba(255,46,159,.16), rgba(34,228,255,.05));
  }
  .coll-dot { width:9px; height:9px; border-radius:50%; flex:0 0 9px; box-shadow: 0 0 10px currentColor; }
  .coll-count { margin-left:auto; font-size:.76rem; color:#8B809F; }
  .recent-row {
    font-size:.83rem; color:#9B90AF; padding:.4rem .6rem; border-radius:9px;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  }
  .recent-row:hover { background: rgba(255,255,255,.04); color:#D9CFE8; }

  .user-card {
    display:flex; gap:.65rem; align-items:center; padding:.6rem .7rem; border-radius:12px;
    background: linear-gradient(140deg, rgba(255,255,255,.06), rgba(255,255,255,.015));
    border:1px solid var(--line);
  }
  .avatar {
    width:32px; height:32px; border-radius:50%; object-fit:cover; flex:0 0 32px;
    display:grid; place-items:center; font-weight:700; color:#0A0611; font-size:.78rem;
    background: conic-gradient(from 140deg, #FF2E9F, #8B5CF6, #22E4FF, #FF2E9F);
  }
  .user-name { font-size:.84rem; font-weight:600; color:#EDE6F5; line-height:1.2; }
  .user-meta { font-size:.69rem; color:#8B809F; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .badge {
    display:inline-block; padding:.06rem .42rem; border-radius:999px; font-size:.6rem;
    font-weight:700; letter-spacing:.07em; text-transform:uppercase;
    color:#FF9BD4; background: rgba(255,46,159,.14); border:1px solid rgba(255,46,159,.34);
  }

  /* sidebar nav (radio) */
  section[data-testid="stSidebar"] div[role="radiogroup"] { gap:.1rem; }
  section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding:.46rem .6rem; border-radius:10px; color:#B7ABC9; transition: all .15s ease;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: rgba(255,255,255,.045); }
  section[data-testid="stSidebar"] div[role="radiogroup"] label > div > div:first-child { display:none !important; }
  section[data-testid="stSidebar"] div[role="radiogroup"] label p { font-size:.87rem; font-weight:500; }
  section[data-testid="stSidebar"] div[role="radiogroup"] label > div { padding-left:.1rem; }
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    color:#F8F3FC; border:1px solid rgba(34,228,255,.38);
    background: linear-gradient(90deg, rgba(34,228,255,.14), rgba(255,46,159,.10));
    box-shadow: 0 0 22px -12px rgba(34,228,255,.9);
  }

  /* new chat button */
  .st-key-new_chat button {
    border-radius: 13px; font-weight: 600; font-size: .89rem; padding:.55rem .8rem;
    color: #F8F2FC !important;
    background: linear-gradient(#0F0A18, #0F0A18) padding-box,
                linear-gradient(120deg, rgba(255,46,159,.95), rgba(139,92,246,.85), rgba(34,228,255,.95)) border-box !important;
    border: 1px solid transparent !important;
    box-shadow: 0 0 26px -14px rgba(255,46,159,.95);
  }
  .st-key-new_chat button:hover {
    background: linear-gradient(#160E22, #160E22) padding-box,
                linear-gradient(120deg, #FF2E9F, #8B5CF6, #22E4FF) border-box !important;
    box-shadow: 0 0 28px -10px rgba(255,46,159,.85);
  }

  /* ---------- top bar ---------- */
  .topbar {
    display:flex; align-items:center; gap:.8rem; padding:0 0 .85rem 0;
    border-bottom:1px solid var(--line); margin-bottom:1.1rem;
  }
  .topbar-title {
    font-family:'Space Grotesk','Inter',sans-serif; font-size:1.32rem; font-weight:700;
    letter-spacing:-.015em; color:#F6F0FB;
  }
  .indexed-pill {
    display:inline-flex; align-items:center; gap:.42rem; font-size:.71rem; color:#C9BDD9;
    background: rgba(255,255,255,.045); border:1px solid var(--line);
    border-radius:999px; padding:.16rem .58rem;
  }
  .indexed-pill .dot { width:7px; height:7px; border-radius:50%; background:#4ADE80; box-shadow:0 0 9px #4ADE80; }

  /* ---------- chat ---------- */
  div[data-testid="stChatMessage"] {
    background: transparent; border: 0; padding: 0 0 .5rem 0; gap: .8rem;
  }
  div[data-testid="stChatMessageAvatarUser"] { display: none !important; }
  div[data-testid="stChatMessageAvatarAssistant"] {
    background: conic-gradient(from 210deg, #FF2E9F, #8B5CF6 38%, #22E4FF 68%, #FF2E9F) !important;
    border: 0 !important; border-radius: 50% !important;
    box-shadow: 0 0 20px rgba(255,46,159,.5), inset 0 0 9px rgba(0,0,0,.7);
  }
  div[data-testid="stChatMessageAvatarAssistant"] > * { opacity: 0 !important; }
  div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    justify-content: flex-end; padding-top: .2rem; padding-bottom: 1.1rem;
  }
  div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
    flex: 0 1 auto; max-width: 84%;
    background: linear-gradient(135deg, rgba(255,46,159,.18), rgba(139,92,246,.12) 55%, rgba(34,228,255,.10));
    border: 1px solid rgba(255,46,159,.34); border-radius: 18px 18px 5px 18px;
    padding: .55rem 1rem; box-shadow: 0 10px 34px -18px rgba(255,46,159,.9);
  }
  div[data-testid="stChatMessageContent"] p {
    font-size: .965rem; line-height: 1.78; color: #E6DFF0; margin: .3rem 0;
  }
  div[data-testid="stChatMessageContent"] li { color:#E6DFF0; line-height:1.7; }
  div[data-testid="stChatMessageContent"] h1,
  div[data-testid="stChatMessageContent"] h2,
  div[data-testid="stChatMessageContent"] h3 { color:#F6EFFB; }

  .cite {
    display:inline-flex; align-items:center; justify-content:center;
    min-width:19px; height:19px; padding:0 5px; margin:0 2px;
    border-radius:6px; font-size:.7rem; font-weight:700; vertical-align:1px;
    color:#FF7AC2; background: rgba(255,46,159,.13); border:1px solid rgba(255,46,159,.5);
  }
  .cite.c2 { color:#7DEBFF; background: rgba(34,228,255,.12); border-color: rgba(34,228,255,.5); }
  .cite.c3 { color:#BFA3FF; background: rgba(139,92,246,.14); border-color: rgba(139,92,246,.55); }
  .cite.c4 { color:#D3F58A; background: rgba(163,230,53,.12); border-color: rgba(163,230,53,.45); }

  .answer-meta { font-size:.71rem; color:#817792; margin:.1rem 0 .5rem 0; letter-spacing:.01em; }

  /* code blocks get a terminal chrome */
  div[data-testid="stMarkdownContainer"] pre {
    position: relative; background: #0A0712 !important;
    border: 1px solid var(--line); border-radius: 14px;
    padding: 1.65rem 1rem .85rem 1rem; margin: .7rem 0;
  }
  div[data-testid="stMarkdownContainer"] pre::before {
    content: ""; position:absolute; top:13px; left:15px; width:9px; height:9px; border-radius:50%;
    background:#FF5F57; box-shadow: 15px 0 #FEBC2E, 30px 0 #28C840;
  }
  div[data-testid="stMarkdownContainer"] code { color:#E9D9F7; }
  div[data-testid="stMarkdownContainer"] p code,
  div[data-testid="stMarkdownContainer"] li code {
    background: rgba(255,46,159,.10); border:1px solid rgba(255,46,159,.22);
    border-radius:6px; padding:.05rem .35rem; color:#FF9BD4;
  }

  /* greeting + suggestion chips */
  .greet {
    font-family:'Space Grotesk','Inter',sans-serif; font-size:1.9rem; font-weight:600;
    letter-spacing:-.025em; color:#F6EFFB; margin:1.1rem 0 .25rem 0;
  }
  .greet-sub { color:#948AA8; font-size:.9rem; line-height:1.6; margin-bottom:1rem; max-width:56ch; }
  .st-key-sugg_1 button, .st-key-sugg_2 button, .st-key-sugg_3 button {
    border-radius:999px !important; font-size:.8rem !important; font-weight:500 !important;
    padding:.34rem .85rem !important; color:#CDBFE0 !important;
    background: rgba(139,92,246,.07) !important; border:1px solid rgba(139,92,246,.42) !important;
  }
  .st-key-sugg_1 button:hover, .st-key-sugg_2 button:hover, .st-key-sugg_3 button:hover {
    color:#F6EFFB !important; border-color: rgba(255,46,159,.6) !important;
    box-shadow: 0 0 20px -10px rgba(255,46,159,.9);
  }

  /* ---------- sources panel ---------- */
  .sources-head {
    display:flex; align-items:baseline; justify-content:space-between;
    padding-bottom:.7rem; border-bottom:1px solid var(--line); margin-bottom:.9rem;
  }
  .sources-title { font-family:'Space Grotesk','Inter',sans-serif; font-size:1.05rem; font-weight:600; color:#F2EAF9; }
  .sources-sub { font-size:.73rem; color:#877C9C; }
  .src-card {
    position: relative; padding:.8rem .9rem; margin-bottom:.75rem; border-radius:13px;
    background: linear-gradient(160deg, rgba(255,255,255,.05), rgba(255,255,255,.012));
    border: 1px solid var(--line); border-left: 2px solid var(--c);
    box-shadow: 0 12px 34px -22px #000, 0 0 26px -18px var(--c);
  }
  .src-head { display:flex; align-items:center; gap:.55rem; }
  .src-num {
    width:22px; height:22px; flex:0 0 22px; border-radius:6px; display:grid; place-items:center;
    font-size:.71rem; font-weight:700; color: var(--c);
    background: color-mix(in srgb, var(--c) 14%, transparent);
    border: 1px solid color-mix(in srgb, var(--c) 70%, transparent);
  }
  .src-file { font-size:.85rem; font-weight:600; color:#EDE5F6; overflow-wrap:anywhere; }
  .src-meta { font-size:.71rem; color:#877C9C; margin:.38rem 0 .45rem 0; }
  .src-text { font-size:.82rem; line-height:1.62; color:#C2B7D4; margin:0 0 .7rem 0; }
  .src-foot { display:flex; align-items:center; gap:.65rem; }
  .src-bar { flex:1; height:4px; border-radius:99px; background: rgba(255,255,255,.08); overflow:hidden; }
  .src-bar span {
    display:block; height:100%; border-radius:99px;
    background: linear-gradient(90deg, var(--c), color-mix(in srgb, var(--c) 35%, #ffffff));
    box-shadow: 0 0 10px var(--c);
  }
  .src-score { font-size:.7rem; color:#D8CEE6; font-variant-numeric: tabular-nums; }
  .src-empty {
    border:1px dashed rgba(255,255,255,.10); border-radius:13px; padding:1.3rem 1rem;
    color:#847A98; font-size:.82rem; line-height:1.6; text-align:center;
    background: rgba(255,255,255,.018);
  }
  .rerank {
    display:flex; align-items:center; justify-content:space-between; margin-top:.9rem;
    padding:.7rem .95rem; border-radius:13px;
    background: linear-gradient(120deg, rgba(255,46,159,.12), rgba(34,228,255,.07));
    border:1px solid rgba(255,255,255,.08);
  }
  .rerank .label { font-size:.85rem; color:#E7DDF2; }
  .toggle {
    width:38px; height:20px; border-radius:99px; position:relative;
    background: linear-gradient(90deg, #A3E635, #4ADE80);
    box-shadow: 0 0 14px rgba(163,230,53,.55);
  }
  .toggle::after {
    content:""; position:absolute; top:2px; right:2px; width:16px; height:16px;
    border-radius:50%; background:#0A0710;
  }

  /* ---------- hero / notice / metrics ---------- */
  .hero {
    border-radius: 18px; padding: 1.15rem 1.35rem; margin-bottom: 1rem;
    background: linear-gradient(135deg, rgba(255,46,159,.13), rgba(139,92,246,.07) 55%, rgba(34,228,255,.09));
    border: 1px solid rgba(255,255,255,.08);
    box-shadow: 0 24px 70px -50px rgba(255,46,159,.9);
  }
  .hero h1 {
    font-family:'Space Grotesk','Inter',sans-serif; font-size:1.45rem; margin:0 0 .3rem 0;
    color:#F7F1FC; letter-spacing:-.02em;
  }
  .hero p { margin:0; color:#A99EBF; font-size:.88rem; line-height:1.55; max-width: 72ch; }
  .pill {
    display:inline-flex; align-items:center; gap:.35rem; font-size:.69rem; font-weight:600;
    padding:.2rem .58rem; border-radius:999px; margin-right:.35rem; margin-top:.65rem;
    border:1px solid rgba(255,255,255,.12); color:#CBC0DE; background: rgba(255,255,255,.04);
  }
  .dot { width:7px; height:7px; border-radius:50%; display:inline-block; }

  .metric-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap:.7rem; }
  .metric {
    border-radius:14px; padding:.8rem .95rem;
    background: linear-gradient(160deg, rgba(255,255,255,.05), rgba(255,255,255,.012));
    border:1px solid var(--line);
  }
  .metric .label { font-size:.66rem; text-transform:uppercase; letter-spacing:.1em; color:#877C9C; }
  .metric .value {
    font-family:'Space Grotesk','Inter',sans-serif; font-size:1.4rem; font-weight:700;
    color:#F7F1FC; margin-top:.15rem; letter-spacing:-.01em;
  }
  .metric .foot { font-size:.71rem; color:#877C9C; margin-top:.1rem; }

  .notice {
    display:flex; gap:.65rem; align-items:flex-start; padding:.72rem .9rem; border-radius:13px;
    background: rgba(251,191,36,.08); border:1px solid rgba(251,191,36,.26);
    color:#FBE3A8; font-size:.81rem; line-height:1.5; margin-bottom:.9rem;
  }
  .notice.info { background: rgba(34,228,255,.07); border-color: rgba(34,228,255,.26); color:#B7EEF9; }
  .notice.small { font-size:.75rem; background: rgba(255,255,255,.03); border-color: var(--line); color:#9C91B0; }
  .notice .ico { flex:0 0 auto; font-size:.95rem; line-height:1.25; }

  /* ---------- splash ---------- */
  .splash { display:flex; flex-direction:column; align-items:center; gap:1rem; padding: 4rem 1rem 2.5rem 1rem; }
  .orbit { position:relative; width:96px; height:96px; }
  .orbit .ring {
    position:absolute; inset:0; border-radius:50%;
    border:2px solid rgba(255,46,159,.16); border-top-color:#FF2E9F; border-right-color:#22E4FF;
    animation: spin 1.15s linear infinite;
  }
  .orbit .ring:nth-child(2) { inset:13px; animation-duration:1.9s; animation-direction:reverse;
    border-top-color:#22E4FF; border-right-color: rgba(139,92,246,.5); }
  .orbit .core {
    position:absolute; inset:31px; border-radius:50%;
    background: conic-gradient(from 200deg, #FF2E9F, #8B5CF6, #22E4FF, #FF2E9F);
    box-shadow: 0 0 26px rgba(255,46,159,.65);
    animation: pulse 1.9s ease-in-out infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes pulse { 0%,100% { transform: scale(.9); opacity:.85 } 50% { transform: scale(1.07); opacity:1 } }
  .splash h3 { margin:0; color:#F5EEFB; font-size:1.05rem; font-weight:600; }
  .splash p { margin:0; color:#8B809F; font-size:.83rem; text-align:center; max-width: 46ch; line-height:1.55; }
  .loading-bar { width:230px; height:3px; border-radius:99px; background: rgba(255,255,255,.07); overflow:hidden; }
  .loading-bar span {
    display:block; height:100%; width:40%; border-radius:99px;
    background: linear-gradient(90deg,#FF2E9F,#8B5CF6,#22E4FF);
    animation: slide 1.5s ease-in-out infinite;
  }
  @keyframes slide { 0% { margin-left:-42% } 100% { margin-left:104% } }

  /* ---------- auth ---------- */
  .auth-wrap { max-width: 430px; margin: 3rem auto 1rem auto; }
  .auth-card {
    border-radius: 18px; padding: 1.35rem 1.3rem;
    background: linear-gradient(160deg, rgba(255,46,159,.08), rgba(255,255,255,.02) 55%, rgba(34,228,255,.06));
    border: 1px solid rgba(255,255,255,.09);
    box-shadow: 0 30px 80px -50px rgba(255,46,159,.9);
  }
  .auth-title {
    font-family:'Space Grotesk','Inter',sans-serif; font-size:1.3rem; font-weight:700;
    color:#F7F1FC; margin:.25rem 0 .25rem 0; letter-spacing:-.02em;
  }
  .auth-sub { font-size:.83rem; color:#948AA8; margin-bottom:.9rem; line-height:1.5; }
  .fineprint { font-size:.71rem; color:#7C7291; line-height:1.55; }

  /* ---------- inputs & buttons ---------- */
  div[data-baseweb="input"] > div, div[data-baseweb="textarea"] > div {
    background: rgba(255,255,255,.04) !important;
    border-color: rgba(255,255,255,.10) !important;
    border-radius: 11px !important;
  }
  div[data-baseweb="input"]:focus-within > div,
  div[data-baseweb="textarea"]:focus-within > div { border-color: rgba(255,46,159,.6) !important; }

  /* pin the composer to the bottom of the chat column */
  div[data-testid="stColumn"]:has(div[data-testid="stChatInput"]) > div[data-testid="stVerticalBlock"] {
    min-height: calc(100vh - 132px); display: flex; flex-direction: column;
  }
  div[data-testid="stColumn"]:has(div[data-testid="stChatInput"]) div[data-testid="stElementContainer"]:has(div[data-testid="stChatInput"]) {
    margin-top: auto;
  }

  div[data-testid="stChatInput"] { background: transparent; padding-bottom: 0; }
  div[data-testid="stChatInput"] > div,
  div[data-testid="stChatInput"] div[data-baseweb="textarea"] {
    border-radius: 999px !important;
    background: linear-gradient(#100A1A, #100A1A) padding-box,
                linear-gradient(120deg, rgba(255,46,159,.9), rgba(139,92,246,.8), rgba(34,228,255,.95)) border-box !important;
    border: 1px solid transparent !important;
    box-shadow: 0 0 34px -14px rgba(255,46,159,.95) !important;
  }
  div[data-testid="stChatInput"] textarea { color:#EFE7F6 !important; }
  div[data-testid="stChatInput"] textarea::placeholder { color:#8B809F !important; }
  div[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #FF2E9F, #22E4FF) !important;
    border: 0 !important; border-radius: 50% !important; color:#08050D !important;
  }

  .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    border-radius: 11px; font-weight: 600; font-size: .86rem;
    border: 1px solid rgba(255,255,255,.10); background: rgba(255,255,255,.045);
    color:#E9E1F4; transition: all .15s ease;
  }
  .stButton > button:hover, .stDownloadButton > button:hover {
    border-color: rgba(255,46,159,.55); background: rgba(255,46,159,.10);
    box-shadow: 0 0 24px -14px rgba(255,46,159,.95);
  }
  .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(120deg, #FF2E9F, #B23BFF 60%, #22E4FF);
    border: 0; color:#FFFFFF;
    box-shadow: 0 12px 34px -20px rgba(255,46,159,.95);
  }
  .stButton > button[kind="primary"]:hover { filter: brightness(1.07); }

  div[data-testid="stDataFrame"] { border-radius: 14px; overflow:hidden; border:1px solid var(--line); }
  hr { border-color: rgba(255,255,255,.07); }
  div[data-testid="stExpander"] details {
    border: 1px solid var(--line); border-radius: 12px; background: rgba(255,255,255,.02);
  }
  div[data-testid="stExpander"] summary { font-size:.82rem; color:#A99EBF; }
  [data-testid="stCaptionContainer"] { color:#877C9C; }
</style>
"""

NOTICE_COLD_START = (
    "Free-tier hardware and free API tiers can cold-start in 30-60 seconds. "
    "The first question after a restart downloads/loads models and may be slow. "
    "When the evidence is weak, this assistant answers &ldquo;I don't know.&rdquo; instead of guessing."
)


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def brand_block() -> None:
    st.markdown(
        """
        <div class="brand">
          <div class="orb"></div>
          <div>
            <div class="brand-title">Northwind</div>
            <div class="brand-sub">Document intelligence</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def user_card(name: str, email: str, role: str, avatar_url: str | None = None) -> None:
    initials = "".join(p[0].upper() for p in name.replace("@", " ").split()[:2]) or "?"
    if avatar_url:
        avatar = f'<img class="avatar" src="{html.escape(avatar_url, quote=True)}" alt="Profile picture">'
    else:
        avatar = f'<div class="avatar">{html.escape(initials)}</div>'
    st.markdown(
        f"""
        <div class="user-card">
          {avatar}
          <div style="min-width:0;flex:1">
            <div class="user-name">{html.escape(name[:40])}</div>
            <div class="user-meta">{html.escape(email[:48])}</div>
          </div>
          <span class="badge">{html.escape(role)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, pills: Sequence[tuple[str, str]] = ()) -> None:
    chips = "".join(
        f'<span class="pill"><span class="dot" style="background:{colour}"></span>{html.escape(label)}</span>'
        for label, colour in pills
    )
    st.markdown(
        f"""
        <div class="hero">
          <h1>{html.escape(title)}</h1>
          <p>{html.escape(subtitle)}</p>
          <div>{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def notice(text: str, kind: str = "warn", icon: str = "⚠️") -> None:
    css_class = "notice" if kind == "warn" else f"notice {kind}"
    st.markdown(
        f'<div class="{css_class}"><span class="ico">{icon}</span><span>{text}</span></div>',
        unsafe_allow_html=True,
    )


def metric_cards(items: Sequence[tuple[str, str, str]], columns: int = 4) -> None:
    cells = "".join(
        f'<div class="metric"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>'
        f'<div class="foot">{html.escape(foot)}</div></div>'
        for label, value, foot in items
    )
    st.markdown(f'<div class="metric-grid">{cells}</div>', unsafe_allow_html=True)


def splash(placeholder: Any, message: str = "Warming up models and index…") -> None:
    placeholder.markdown(
        f"""
        <div class="splash">
          <div class="orbit"><div class="ring"></div><div class="ring"></div><div class="core"></div></div>
          <h3>{html.escape(message)}</h3>
          <p>This runs on free-tier hardware, so the first load can take 30-60 seconds.
             Models are downloaded once and cached afterwards.</p>
          <div class="loading-bar"><span></span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def topbar(title: str, indexed_label: str) -> None:
    st.markdown(
        f"""
        <div class="topbar">
          <span class="topbar-title">{html.escape(title)}</span>
          <span class="indexed-pill"><span class="dot"></span>{html.escape(indexed_label)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def greeting(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="greet">{html.escape(title)}</div><div class="greet-sub">{html.escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )


def sidebar_label(text: str) -> None:
    st.markdown(f'<div class="side-label">{html.escape(text)}</div>', unsafe_allow_html=True)


def collection_list(items: Sequence[tuple[str, str]], active: str | None = None) -> None:
    """items: (label, count) tuples; the first row is highlighted when active matches."""
    rows = []
    for index, (label, count) in enumerate(items):
        colour = CARD_COLORS[index % len(CARD_COLORS)]
        css = "coll-row active" if (active is not None and label == active) else "coll-row"
        rows.append(
            f'<div class="{css}">'
            f'<span class="coll-dot" style="background:{colour};color:{colour}"></span>'
            f'<span>{html.escape(label)}</span>'
            f'<span class="coll-count">{html.escape(count)}</span></div>'
        )
    if rows:
        st.markdown("".join(rows), unsafe_allow_html=True)


def recent_list(items: Sequence[str]) -> None:
    if not items:
        st.markdown('<div class="recent-row" style="color:#6F6584">No conversations yet</div>', unsafe_allow_html=True)
        return
    rows = "".join(
        f'<div class="recent-row">{html.escape(text[:64])}</div>' for text in items
    )
    st.markdown(rows, unsafe_allow_html=True)


def sources_header(count: int) -> None:
    label = f"{count} passage{'s' if count != 1 else ''} retrieved" if count else "no passages yet"
    st.markdown(
        f'<div class="sources-head"><span class="sources-title">Sources</span>'
        f'<span class="sources-sub">{html.escape(label)}</span></div>',
        unsafe_allow_html=True,
    )


def source_cards(citations: Sequence[dict[str, Any]], max_chars: int = 260) -> None:
    cards = []
    for position, citation in enumerate(citations, start=1):
        index = int(citation.get("index") or position)
        colour = CARD_COLORS[(index - 1) % len(CARD_COLORS)]
        source = str(citation.get("source", "unknown"))
        page = citation.get("page", "?")
        chunk = str(citation.get("chunk_id", ""))
        score = citation.get("score", 0.0) or 0.0
        try:
            score_value = max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            score_value = 0.0
        snippet = " ".join(str(citation.get("snippet", "")).split())[:max_chars]
        cards.append(
            f'<div class="src-card" style="--c:{colour}">'
            f'<div class="src-head"><span class="src-num">{index}</span>'
            f'<span class="src-file">{html.escape(source)}</span></div>'
            f'<div class="src-meta">page {html.escape(str(page))}'
            f'{" · chunk " + html.escape(chunk[:10]) if chunk else ""}</div>'
            f'<p class="src-text">{html.escape(snippet)}</p>'
            f'<div class="src-foot"><div class="src-bar">'
            f'<span style="width:{score_value * 100:.0f}%"></span></div>'
            f'<span class="src-score">{score_value:.2f}</span></div></div>'
        )
    if cards:
        st.markdown("".join(cards), unsafe_allow_html=True)


def sources_empty() -> None:
    st.markdown(
        '<div class="src-empty">Grounded passages from your documents appear here, '
        "with the re-ranker score for each one.</div>",
        unsafe_allow_html=True,
    )


def rerank_card(min_score: float, top_k: int) -> None:
    st.markdown(
        f'<div class="rerank"><span class="label">Rerank results</span>'
        f'<span class="toggle" title="Cross-encoder on · floor {min_score:.2f} · top {top_k}"></span></div>',
        unsafe_allow_html=True,
    )
