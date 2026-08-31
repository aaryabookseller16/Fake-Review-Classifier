"""Design system for the Streamlit app.

Streamlit's defaults are recognisably Streamlit. This module replaces them with
a small, deliberate design system so the app reads as a designed product:

* **One palette**, defined as CSS custom properties and reused by both the CSS
  and the Altair charts, so nothing drifts out of sync.
* **One type scale**, using Inter for text and JetBrains Mono for numerals --
  tabular figures matter when you are lining up four-decimal metrics.
* **One motion curve.** Every transition uses the same duration and easing;
  inconsistent motion is what makes an interface feel assembled rather than
  designed.
* **Full dark-mode support**, driven by ``prefers-color-scheme`` so the app
  follows the visitor's system setting.

Kept separate from ``streamlit_app.py`` so the page logic stays readable.
"""

from __future__ import annotations

import altair as alt

# ----------------------------------------------------------------------
# Palette. Single source of truth -- the CSS below and the Altair charts
# both read from these values.
# ----------------------------------------------------------------------
INDIGO = "#4f46e5"
INDIGO_SOFT = "#818cf8"
EMERALD = "#059669"
ROSE = "#e11d48"
AMBER = "#d97706"
SLATE_900 = "#0f172a"
SLATE_500 = "#64748b"
SLATE_200 = "#e2e8f0"

#: Categorical scale for the model comparison charts, ordered weakest to
#: strongest so the colour ramp reinforces the narrative of the results.
MODEL_COLORS = ["#cbd5e1", "#94a3b8", "#a5b4fc", INDIGO, EMERALD]

#: Motion: one duration, one easing curve, applied everywhere.
EASING = "cubic-bezier(0.4, 0.0, 0.2, 1)"
DURATION = "180ms"


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
    --brand:        {INDIGO};
    --brand-soft:   {INDIGO_SOFT};
    --success:      {EMERALD};
    --danger:       {ROSE};
    --warning:      {AMBER};

    --bg:           #f8fafc;
    --surface:      #ffffff;
    --surface-alt:  #f1f5f9;
    --border:       {SLATE_200};
    --text:         {SLATE_900};
    --text-muted:   {SLATE_500};

    --code:         {INDIGO};
    --radius:       14px;
    --radius-sm:    9px;
    --shadow-sm:    0 1px 2px rgba(15, 23, 42, .06);
    --shadow-md:    0 4px 14px rgba(15, 23, 42, .08);
    --shadow-lg:    0 12px 32px rgba(15, 23, 42, .12);
    --ease:         {EASING};
    --dur:          {DURATION};
}}

@media (prefers-color-scheme: dark) {{
    :root {{
        --bg:          #0b1120;
        --surface:     #111827;
        --surface-alt: #1e293b;
        --border:      #1e293b;
        --text:        #e2e8f0;
        --text-muted:  #94a3b8;
        --shadow-sm:   0 1px 2px rgba(0, 0, 0, .4);
        --shadow-md:   0 4px 14px rgba(0, 0, 0, .45);
        --shadow-lg:   0 12px 32px rgba(0, 0, 0, .55);
        /* Indigo on a dark surface is too low-contrast for inline code. */
        --code:        var(--brand-soft);
    }}
}}

/* ---------------------------------------------------------------- base */
html, body, [class*="css"], .stMarkdown, p, li, div {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
}}

[data-testid="stAppViewContainer"] {{ background: var(--bg); }}
[data-testid="stHeader"] {{ background: transparent; }}

.block-container {{
    padding-top: 2.6rem !important;
    padding-bottom: 4rem !important;
    max-width: 1180px;
}}

h1, h2, h3, h4 {{
    font-family: 'Inter', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -.021em !important;
    font-weight: 650 !important;
}}
h1 {{ font-size: 2.15rem !important; line-height: 1.15 !important; }}
h2 {{ font-size: 1.4rem  !important; margin-top: .4rem !important; }}
h3 {{ font-size: 1.08rem !important; }}

p, li {{ color: var(--text); line-height: 1.65; }}
code {{
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: .86em !important;
    background: var(--surface-alt) !important;
    color: var(--code) !important;
    padding: .12em .42em !important;
    border-radius: 5px !important;
}}

/* ------------------------------------------------------------- sidebar */
[data-testid="stSidebar"] {{
    background: var(--surface);
    border-right: 1px solid var(--border);
}}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem !important; }}

/* Nav radio rendered as a segmented list of pills. */
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 3px !important; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
    padding: .5rem .7rem !important;
    border-radius: var(--radius-sm) !important;
    transition: background var(--dur) var(--ease), color var(--dur) var(--ease);
    cursor: pointer;
    width: 100%;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
    background: var(--surface-alt) !important;
}}
[data-testid="stSidebar"] [role="radiogroup"] label p {{
    font-size: .9rem !important;
    font-weight: 500 !important;
}}

/* -------------------------------------------------------------- inputs */
.stButton button {{
    background: var(--brand) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: .58rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: .92rem !important;
    letter-spacing: .008em;
    box-shadow: var(--shadow-sm);
    transition: transform var(--dur) var(--ease),
                box-shadow var(--dur) var(--ease),
                background var(--dur) var(--ease);
}}
.stButton button:hover {{
    background: #4338ca !important;
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}}
.stButton button:active {{ transform: translateY(0); box-shadow: var(--shadow-sm); }}

.stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {{
    border-radius: var(--radius-sm) !important;
    border-color: var(--border) !important;
    background: var(--surface) !important;
    font-size: .93rem !important;
    transition: border-color var(--dur) var(--ease), box-shadow var(--dur) var(--ease);
}}
.stTextArea textarea:focus,
.stSelectbox div[data-baseweb="select"] > div:focus-within {{
    border-color: var(--brand) !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, .14) !important;
}}

/* ------------------------------------------------------------- metrics */
[data-testid="stMetric"] {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.15rem;
    box-shadow: var(--shadow-sm);
    transition: transform var(--dur) var(--ease), box-shadow var(--dur) var(--ease);
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}}
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-variant-numeric: tabular-nums;
    font-weight: 600 !important;
    font-size: 1.7rem !important;
    color: var(--text) !important;
}}
[data-testid="stMetricLabel"] {{
    font-size: .74rem !important;
    text-transform: uppercase;
    letter-spacing: .075em;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
}}

/* --------------------------------------------------------- data frames */
[data-testid="stDataFrame"] {{
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}}

/* --------------------------------------------------- custom components */
.card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.35rem 1.5rem;
    box-shadow: var(--shadow-sm);
    transition: transform var(--dur) var(--ease), box-shadow var(--dur) var(--ease);
}}
.card:hover {{ transform: translateY(-2px); box-shadow: var(--shadow-md); }}

/* Verdict banner shown after a classification. */
.verdict {{
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    display: flex;
    align-items: center;
    gap: 1.1rem;
    animation: rise var(--dur) var(--ease);
    border: 1px solid;
}}
.verdict-fake {{
    background: color-mix(in srgb, {ROSE} 8%, var(--surface));
    border-color: color-mix(in srgb, {ROSE} 32%, transparent);
}}
.verdict-genuine {{
    background: color-mix(in srgb, {EMERALD} 8%, var(--surface));
    border-color: color-mix(in srgb, {EMERALD} 32%, transparent);
}}
.verdict-icon {{
    width: 46px; height: 46px;
    border-radius: 12px;
    display: grid; place-items: center;
    font-size: 1.4rem;
    flex-shrink: 0;
}}
.verdict-fake .verdict-icon    {{ background: color-mix(in srgb, {ROSE} 15%, transparent); }}
.verdict-genuine .verdict-icon {{ background: color-mix(in srgb, {EMERALD} 15%, transparent); }}
.verdict-label {{
    font-size: .72rem; font-weight: 700; letter-spacing: .09em;
    text-transform: uppercase; color: var(--text-muted); margin-bottom: .18rem;
}}
.verdict-value {{ font-size: 1.35rem; font-weight: 650; letter-spacing: -.02em; }}
.verdict-fake    .verdict-value {{ color: {ROSE}; }}
.verdict-genuine .verdict-value {{ color: {EMERALD}; }}

/* Confidence meter: position of the score relative to the 0.5 threshold. */
.meter {{
    height: 7px; border-radius: 99px; margin-top: .85rem;
    background: linear-gradient(90deg,
        color-mix(in srgb, {EMERALD} 55%, transparent),
        var(--surface-alt) 50%,
        color-mix(in srgb, {ROSE} 55%, transparent));
    position: relative;
}}
.meter-pin {{
    position: absolute; top: -4px; width: 3px; height: 15px;
    border-radius: 99px; background: var(--text);
    transition: left 420ms var(--ease);
}}

.pill {{
    display: inline-block;
    padding: .2rem .6rem;
    border-radius: 99px;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .03em;
    background: var(--surface-alt);
    color: var(--text-muted);
    border: 1px solid var(--border);
}}

.stat-row {{ display: flex; gap: .9rem; flex-wrap: wrap; }}
.stat {{
    flex: 1 1 150px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: .95rem 1.1rem;
    box-shadow: var(--shadow-sm);
    transition: transform var(--dur) var(--ease), box-shadow var(--dur) var(--ease);
}}
.stat:hover {{ transform: translateY(-2px); box-shadow: var(--shadow-md); }}
.stat-label {{
    font-size: .71rem; text-transform: uppercase; letter-spacing: .075em;
    font-weight: 600; color: var(--text-muted); margin-bottom: .3rem;
}}
.stat-value {{
    font-family: 'JetBrains Mono', monospace;
    font-variant-numeric: tabular-nums;
    font-size: 1.5rem; font-weight: 600; color: var(--text); line-height: 1.1;
}}
.stat-sub {{ font-size: .78rem; color: var(--text-muted); margin-top: .2rem; }}

.hero-title {{
    font-size: 2.15rem; font-weight: 700; letter-spacing: -.028em;
    color: var(--text); margin: 0 0 .4rem 0; line-height: 1.12;
}}
.hero-sub {{
    font-size: 1.02rem; color: var(--text-muted);
    max-width: 62ch; line-height: 1.6; margin: 0;
}}

.callout {{
    border-left: 3px solid var(--brand);
    background: var(--surface-alt);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: .9rem 1.15rem;
    font-size: .9rem;
    color: var(--text);
}}
.callout-muted {{ border-left-color: var(--text-muted); }}

@keyframes rise {{
    from {{ opacity: 0; transform: translateY(7px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.rise {{ animation: rise var(--dur) var(--ease); }}

/* Streamlit chrome we do not want. */
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stDecoration"] {{ display: none; }}
hr {{ border-color: var(--border) !important; margin: 1.6rem 0 !important; }}
</style>
"""


def apply(st) -> None:
    """Inject the stylesheet. Call once, immediately after ``set_page_config``."""
    st.markdown(CSS, unsafe_allow_html=True)


def style_chart(
    chart: alt.Chart, dark: bool = False, y_grid: bool = False
) -> alt.Chart:
    """Apply consistent Altair styling.

    Configured per-chart rather than through ``alt.theme.register`` because the
    registration API differs between Altair 5 and 6, and pinning the behaviour
    matters more here than saving three lines.

    ``y_grid`` defaults to off, which suits the horizontal bar charts where a
    horizontal rule per category is just noise. Line charts (ROC, loss curves)
    need it to be readable, so they opt back in.
    """
    grid = "#e2e8f0" if not dark else "#1e293b"
    text = SLATE_500 if not dark else "#94a3b8"
    styled = (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            labelFont="Inter",
            titleFont="Inter",
            labelFontSize=11,
            titleFontSize=11,
            labelColor=text,
            titleColor=text,
            titleFontWeight=600,
            gridColor=grid,
            gridOpacity=0.7,
            domainColor=grid,
            tickColor=grid,
        )
        .configure_legend(
            labelFont="Inter",
            titleFont="Inter",
            labelFontSize=10,
            titleFontSize=10,
            labelColor=text,
            titleColor=text,
            labelLimit=260,
            symbolStrokeWidth=3,
            rowPadding=3,
        )
    )
    return styled if y_grid else styled.configure_axisY(grid=False)
