import streamlit as st
from typing import List, Dict, Any
from simulations.light_shadow import light_shadow_app

# Simple registry for simulations
SIMULATIONS: List[Dict[str, Any]] = [
    {
        "id": "light_shadow",
        "title": "Licht & Schatten",
        "topic": "Optik",
        "grades": [7, 8, 9],
        "component": light_shadow_app,
        "description": "Interaktive Simulation der Lichtausbreitung, penumbra/umbra, Punkt- und ausgedehnte Quellen.",
        "tags": ["Licht", "Schatten", "Optik", "Geometrische Optik"],
    }
]

st.set_page_config(
    page_title="Physik Simulationen",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Touch-friendly CSS
TOUCH_CSS = """
<style>
:root {
  --touch-target: 56px;
}

/* Bigger controls */
button, .stButton>button, .stSlider, input, select, textarea {
  min-height: var(--touch-target);
}

/* Card styling */
.sim-card {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 16px;
  background: #ffffffaa;
  transition: box-shadow 0.2s ease;
}
.sim-card:hover { box-shadow: 0 8px 24px rgba(0,0,0,0.08); }
.sim-title { font-weight: 700; font-size: 1.1rem; margin-bottom: 4px; }
.sim-meta { color: #6b7280; font-size: 0.9rem; }
.tag { display:inline-block; padding:2px 8px; border-radius:9999px; background:#eef2ff; color:#3730a3; margin-right:6px; font-size:0.75rem; }
</style>
"""

st.markdown(TOUCH_CSS, unsafe_allow_html=True)

if "active_sim" not in st.session_state:
    st.session_state.active_sim = None

st.sidebar.title("Physik-Simulationen")

# Filters
topics = sorted({s["topic"] for s in SIMULATIONS})
all_grades = sorted({g for s in SIMULATIONS for g in s["grades"]})

selected_topic = st.sidebar.multiselect("Themengebiet", options=topics, default=topics)
selected_grades = st.sidebar.multiselect("Jahrgangsstufe", options=all_grades, default=all_grades)
query = st.sidebar.text_input("Suche", "")

filtered = [
    s for s in SIMULATIONS
    if s["topic"] in selected_topic
    and any(g in selected_grades for g in s["grades"])
    and (query.lower() in s["title"].lower() or any(query.lower() in t.lower() for t in s.get("tags", [])))
]

# Navigation
if st.session_state.active_sim is None:
    st.title("🔬 Interaktive Physik-Simulationen")
    st.caption("Tippe auf eine Kachel, um die Simulation zu starten.")

    cols = st.columns(3)
    for idx, sim in enumerate(filtered):
        with cols[idx % 3]:
            with st.container(border=True):
                st.markdown(f"<div class='sim-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='sim-title'>{sim['title']}</div>")
                st.markdown(f"<div class='sim-meta'>Thema: {sim['topic']} · Klassen: {', '.join(map(str, sim['grades']))}</div>")
                st.markdown(" ".join([f"<span class='tag'>{t}</span>" for t in sim.get("tags", [])]), unsafe_allow_html=True)
                if st.button("Öffnen", key=f"open_{sim['id']}"):
                    st.session_state.active_sim = sim["id"]
                st.markdown("</div>", unsafe_allow_html=True)
else:
    # Render active simulation
    sim = next((s for s in SIMULATIONS if s["id"] == st.session_state.active_sim), None)
    if sim is None:
        st.session_state.active_sim = None
        st.rerun()

    top = st.columns([1,1,2])
    with top[0]:
        if st.button("← Zurück zur Übersicht"):
            st.session_state.active_sim = None
            st.rerun()
    with top[1]:
        st.caption(f"Thema: {sim['topic']} · Klassen: {', '.join(map(str, sim['grades']))}")
    with top[2]:
        st.title(sim["title"])

    sim["component"]()
