import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import os

# Set page config
st.set_page_config(
    page_title="FightIQ - UFC Predictor",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border-color: #ff3333;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    model = joblib.load('ufc_model_elo.pkl')
    with open('fighter_db.json', 'r') as f:
        fighter_db = json.load(f)
    with open('fighter_elo.json', 'r') as f:
        fighter_elo = json.load(f)
    with open('features_elo.json', 'r') as f:
        features = json.load(f)
    return model, fighter_db, fighter_elo, features

try:
    model, fighter_db, fighter_elo, features = load_resources()
    fighter_names = sorted(list(fighter_db.keys()))
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# --- Header ---
st.title("🥊 FightIQ: Advanced UFC Prediction")
st.markdown("### AI-Powered Fight Prediction Engine with Elo Ratings")

# --- Input Section ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔴 Red Corner (Fighter 1)")
    f1_name = st.selectbox("Select Fighter 1", fighter_names, index=fighter_names.index("Jon Jones") if "Jon Jones" in fighter_names else 0)
    f1_odds = st.number_input("Odds (Decimal)", min_value=1.01, value=1.50, step=0.01, key="o1")
    
    # Show basic stats if available
    if f1_name in fighter_elo:
        st.caption(f"Elo Rating: {int(fighter_elo[f1_name])}")

with col2:
    st.subheader("🔵 Blue Corner (Fighter 2)")
    f2_name = st.selectbox("Select Fighter 2", fighter_names, index=fighter_names.index("Stipe Miocic") if "Stipe Miocic" in fighter_names else 1)
    f2_odds = st.number_input("Odds (Decimal)", min_value=1.01, value=2.50, step=0.01, key="o2")
    
    if f2_name in fighter_elo:
        st.caption(f"Elo Rating: {int(fighter_elo[f2_name])}")

# --- Prediction Logic ---
def construct_features(f1, f2, o1, o2):
    stats_1 = fighter_db.get(f1, {})
    stats_2 = fighter_db.get(f2, {})
    elo_1 = fighter_elo.get(f1, 1500)
    elo_2 = fighter_elo.get(f2, 1500)
    
    row = {}
    for feat in features:
        val = np.nan
        
        # Elo Features
        if feat == 'f_1_elo': val = elo_1
        elif feat == 'f_2_elo': val = elo_2
        elif feat == 'diff_elo': val = elo_1 - elo_2
        
        # Odds
        elif feat == 'f_1_odds': val = o1
        elif feat == 'f_2_odds': val = o2
        elif feat == 'diff_odds': val = o1 - o2
        
        # Fighter Stats
        elif feat.startswith('f_1_'):
            base = feat[4:]
            if base in stats_1: val = stats_1[base]
        elif feat.startswith('f_2_'):
            base = feat[4:]
            if base in stats_2: val = stats_2[base]
        elif feat.endswith('_f_1'):
            base = feat[:-4]
            if base in stats_1: val = stats_1[base]
        elif feat.endswith('_f_2'):
            base = feat[:-4]
            if base in stats_2: val = stats_2[base]
            
        # Diffs
        elif feat.startswith('diff_'):
            base = feat[5:]
            v1 = stats_1.get(base)
            v2 = stats_2.get(base)
            if v1 is not None and v2 is not None:
                val = v1 - v2
        
        row[feat] = val
    
    return pd.DataFrame([row])

if st.button("PREDICT WINNER", type="primary"):
    if f1_name == f2_name:
        st.warning("Please select two different fighters.")
    else:
        with st.spinner("Analyzing matchup..."):
            try:
                X = construct_features(f1_name, f2_name, f1_odds, f2_odds)
                probs = model.predict_proba(X)[0]
                p_f1 = probs[1]
                p_f2 = probs[0]
                
                winner = f1_name if p_f1 > 0.5 else f2_name
                confidence = max(p_f1, p_f2)
                loser = f2_name if winner == f1_name else f1_name
                
                # --- Results Display ---
                st.markdown("---")
                res_col1, res_col2 = st.columns([2, 1])
                
                with res_col1:
                    st.markdown(f"## 🏆 Winner: **{winner}**")
                    st.progress(float(confidence))
                    st.markdown(f"**Confidence: {confidence:.1%}**")
                    
                    # Value Bet Analysis
                    implied_1 = 1/f1_odds
                    implied_2 = 1/f2_odds
                    
                    edge = 0
                    is_value = False
                    bet_target = ""
                    
                    if p_f1 > implied_1 + 0.05:
                        edge = p_f1 - implied_1
                        is_value = True
                        bet_target = f1_name
                    elif p_f2 > implied_2 + 0.05:
                        edge = p_f2 - implied_2
                        is_value = True
                        bet_target = f2_name
                        
                    if is_value:
                        st.success(f"💰 **VALUE BET DETECTED!**")
                        st.markdown(f"Bet on **{bet_target}** (Edge: +{edge:.1%})")
                    else:
                        st.info("No significant value found at these odds.")
                
                with res_col2:
                    st.markdown("### Tale of the Tape")
                    # Simple comparison table
                    # Extract some key stats if available
                    def get_stat(name, stat):
                        # Try exact match
                        val = fighter_db.get(name, {}).get(stat)
                        if val is not None: return val
                        
                        # Try finding keys that contain the stat name (heuristic)
                        # e.g. for 'wins', might be 'wins_15'
                        if stat == 'wins':
                            keys = [k for k in fighter_db.get(name, {}).keys() if 'wins' in k]
                            if keys: return fighter_db[name][keys[0]]
                            
                        return "N/A"
                    
                    comp_data = {
                        "Stat": ["Elo Rating", "Reach (cm)", "Age", "Wins (Recent)"],
                        f1_name: [
                            int(fighter_elo.get(f1_name, 0)),
                            get_stat(f1_name, "fighter_reach_cm"),
                            get_stat(f1_name, "fighter_age"),
                            get_stat(f1_name, "wins")
                        ],
                        f2_name: [
                            int(fighter_elo.get(f2_name, 0)),
                            get_stat(f2_name, "fighter_reach_cm"),
                            get_stat(f2_name, "fighter_age"),
                            get_stat(f2_name, "wins")
                        ]
                    }
                    st.table(pd.DataFrame(comp_data).set_index("Stat"))

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# --- Footer ---
st.markdown("---")
st.caption("FightIQ Model v2.0 (Elo Enhanced) | Built with XGBoost & Streamlit")
