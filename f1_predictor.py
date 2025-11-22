import streamlit as st
import fastf1
import pandas as pd
import logging
from google import genai
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential
import io 
from datetime import date 
import numpy as np 

# --- Initial Setup ---
pd.options.mode.chained_assignment = None
logging.getLogger('fastf1').setLevel(logging.ERROR)

# **Disable FastF1 Local Cache**
try:
    # Setting Cache Path to None disables the local FastF1 cache.
    fastf1.set_cache_path(None) 
except Exception:
    pass

# --- Constants ---
TRACKS = ["Bahrain", "Saudi Arabia", "Australia", "Imola", "Miami", "Monaco", 
          "Spain", "Canada", "Austria", "Great Britain", "Hungary", "Belgium", 
          "Netherlands", "Monza", "Singapore", "Japan", "Qatar", "United States", 
          "Mexico", "Brazil", "Las Vegas", "Abu Dhabi", "China", "Turkey", 
          "France"]
# Priority order for auto-detection: Race -> Qualifying -> FP3 -> FP2 -> FP1
SESSIONS_PRIORITY = ["R", "Q", "FP3", "FP2", "FP1"] 
YEARS = [2025, 2024, 2023, 2022, 2021, 2020]
MODEL_NAME = "gemini-2.5-flash"
# Custom Header Image URL
IMAGE_HEADER_URL = "https://raw.githubusercontent.com/ledout/F1-Strategy-Predictor/main/F1-App.png"


# --- Helper Functions ---

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def get_gemini_prediction(prompt):
    """Sends the prompt to Gemini Flash using the API key from Secrets."""
    
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
             raise ValueError("GEMINI_API_KEY not found in Streamlit Secrets. Please set it.")
    except Exception as e:
        raise ValueError(f"API Key Error: {e}")
        
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text


def load_and_process_data(year, event, session_key):
    """Loads data from FastF1 and performs initial processing."""
    
    try:
        session = fastf1.get_session(year, event, session_key)
        
        # Robust Session.load() attempt
        try:
            # Minimal load for speed and stability
            session.load(laps=True, telemetry=False, weather=False, messages=False, pit_stops=False)
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                 session.load()
            else:
                 raise e 
        except Exception as e:
            # General error fallback
            if "not loaded yet" in str(e):
                 session.load(telemetry=False, weather=False, messages=False, laps=True, pit_stops=False)
            else:
                 raise e
        
        # Check validity
        if session.laps is None or session.laps.empty or not isinstance(session.laps, pd.DataFrame):
            return None, f"Insufficient data for {year} {event} {session_key}."
            
    except Exception as e:
        # Clean error messages
        msg = str(e)
        if "schedule" in msg or "found" in msg:
             return None, f"Data not found/available for {event} {year}."
        return None, f"FastF1 Error: {msg}"

    laps = session.laps.reset_index(drop=True)
    
    # Filter laps
    laps_filtered = laps.loc[
        (laps['IsAccurate'] == True) & 
        (laps['LapTime'].notna()) & 
        (laps['Driver'] != 'OUT')
    ].copy()

    laps_filtered['LapTime_s'] = laps_filtered['LapTime'].dt.total_seconds()
    
    # Stats Calculation
    driver_stats = laps_filtered.groupby('Driver').agg(
        Best_Time=('LapTime', 'min'),
        Avg_Time=('LapTime', 'mean'),
        Var=('LapTime_s', lambda x: np.var(x) if len(x) >= 2 else np.nan),
        Laps=('LapTime', 'count')
    ).reset_index()

    driver_stats['Best_Time_s'] = driver_stats['Best_Time'].dt.total_seconds()
    driver_stats['Avg_Time_s'] = driver_stats['Avg_Time'].dt.total_seconds()
    
    # Filter drivers with too few laps (only relevant for Race pace analysis really)
    if session_key in ["R", "S"]:
        driver_stats = driver_stats[driver_stats['Laps'] >= 3]
    
    if driver_stats.empty:
        return None, "Insufficient driver data."

    # Determine ranking metric based on session type
    if session_key in ["R", "S"]:
        ranking_column = 'Avg_Time_s' # Average pace for Race
    else:
        ranking_column = 'Best_Time_s' # Single fastest lap for Quali/Practice
    
    driver_stats_ranked = driver_stats.sort_values(by=ranking_column, ascending=True).head(10)
    
    data_lines = []
    for index, row in driver_stats_ranked.iterrows():
        best = str(row['Best_Time']).split('0 days ')[-1][:10]
        avg = str(row['Avg_Time']).split('0 days ')[-1][:10]
        data_lines.append(
            f"POS {index+1}: {row['Driver']} | Best: {best} | Avg: {avg} | Var: {row['Var']:.3f} | Laps: {int(row['Laps'])}"
        )

    context_data = "\n".join(data_lines)
    return context_data, session.name

@st.cache_data(ttl=3600)
def get_latest_completed_race():
    """
    Finds the latest completed conventional F1 race to set defaults.
    """
    latest_date = pd.Timestamp.min
    latest_race = None
    
    # Loop backwards from current year
    for year in sorted(YEARS, reverse=True):
        try:
            schedule = fastf1.get_events(year=year)
            today = pd.Timestamp.now()
            
            if 'EventDate' in schedule.columns:
                # Find races that have already happened
                completed = schedule.loc[
                    (schedule['EventDate'] < today) & 
                    (schedule['EventFormat'] == 'conventional')
                ]
                
                if not completed.empty:
                    # Get the very last one
                    last_event = completed.sort_values(by='EventDate', ascending=False).iloc[0]
                    return (year, last_event['EventName'])
        except Exception:
            continue
            
    return (2024, 'Abu Dhabi') # Fallback

def find_last_three_races_data(current_year, event, expander_placeholder):
    """Finds last 3 races for seasonal context."""
    
    with expander_placeholder.container():
        st.info("🔄 Starting Seasonal Data Collection (Last 3 Races)")
        
        try:
            schedule = fastf1.get_event_schedule(current_year)
            if schedule.empty: return [], "Empty schedule."
            
            # Find current event date to filter backwards
            curr = schedule[schedule['EventName'] == event]
            if curr.empty:
                # If event not found (e.g. future year), use today's date
                ref_date = pd.Timestamp.now()
            else:
                ref_date = curr['EventDate'].iloc[0]

            # Filter: Conventional races BEFORE the reference date
            past_races = schedule.loc[
                (schedule['EventFormat'] == 'conventional') & 
                (schedule['EventDate'] < ref_date)
            ].sort_values(by='EventDate', ascending=False).head(3)
            
            if past_races.empty:
                return [], "No previous races found this season."

            race_reports = []
            for _, race in past_races.iterrows():
                ename = race['EventName']
                st.info(f"🔮 Loading data: {ename} {current_year}...")
                c_data, _ = load_and_process_data(current_year, ename, 'R')
                
                if c_data:
                    race_reports.append(f"--- {ename} {current_year} (Race) ---\n{c_data}\n")
                    st.success(f"✅ Loaded {ename}.")
                else:
                    st.warning(f"⚠️ Skipped {ename} (No data).")
            
            if not race_reports:
                return [], "No seasonal data available."
                
            return race_reports, "Seasonal data loaded."

        except Exception as e:
            return [], f"Error finding past races: {e}"

def create_prediction_prompt(context_data, year, event, session_name, session_type):
    """Builds prompt."""
    
    # Dynamic context based on session type
    if session_type in ['R', 'S']:
        focus = "This is RACE data. Drivers are ranked by AVERAGE PACE. Focus on consistency, tire management, and long-run pace."
    else:
        focus = "This is QUALIFYING/PRACTICE data. Drivers are ranked by FASTEST LAP. Focus on raw speed and one-lap performance."

    prompt = f"""
You are a Senior F1 Strategy Analyst.
Target: Predict winner for {event} {year}.
Data Source: {session_name} ({session_type}).

**DATA CONTEXT:** {focus}

**DRIVER DATA:**
{context_data}

--- Analysis Guidelines ---
1. **Immediate Prediction:** Winner name and 1-sentence reason (English).
2. **Performance Analysis:** Analyze the provided data (Pace vs Speed based on session type).
3. **Track Specifics:** Mention track characteristics ({event}) and how they suit the top drivers.
4. **Confidence Table:** A markdown table of top 5 drivers with winning probability %.

--- Output Format (Markdown, English) ---
🏎️ **Strategy Report: {event} {year}**

### 🥇 Immediate Prediction
...

### 📊 Performance Analysis
...

### 🏁 Confidence Table (D5)
| Driver | Probability |
| :--- | :--- |
...
"""
    return prompt

def get_preliminary_prediction(current_year, event):
    """Pre-race prediction based on history + season trends."""
    
    prev_year = current_year - 1
    st.subheader("🏁 Pre-Race Analysis (History + Season)")
    
    with st.expander("🛠️ Diagnostics (Data Loading)", expanded=False):
        holder = st.container()
        with holder:
            st.info(f"1. Loading historical data: {event} {prev_year}...")
            hist_data, _ = load_and_process_data(prev_year, event, 'R')
            if hist_data: st.success("✅ Historical data loaded.")
            else: st.warning("⚠️ Historical data missing.")
            
            st.markdown("---")
            
        # Load seasonal
        seasonal_reports, _ = find_last_three_races_data(current_year, event, holder)

    # Build Prompt
    data_text = ""
    if hist_data:
        data_text += f"### Historical Data ({event} {prev_year}):\n{hist_data}\n\n"
    if seasonal_reports:
        data_text += "### Recent Season Form (Last 3 Races):\n" + "\n".join(seasonal_reports)
        
    if not data_text:
        st.error("❌ No data available (Historical or Seasonal). Cannot predict.")
        return

    prompt = f"""
Act as F1 Strategist. Predict winner for **{event} {current_year}**.

**AVAILABLE DATA:**
{data_text}

**INSTRUCTIONS:**
1. **Weighting:** Give 65% weight to Recent Season Form (current car performance) and 35% to Historical Track Data (track suitability).
2. **Output:** English only. Markdown.
3. **Format:**
   - **Winner Prediction:** Name + Reason.
   - **Analysis:** Contrast historical track form vs current season form.
   - **Table:** Top 5 drivers probability.

**GO.**
"""
    return get_gemini_prediction(prompt)


# --- Main App ---

def main():
    st.set_page_config(page_title="F1 Strategy Predictor", layout="centered")

    # Header Image
    st.markdown(f"""
        <div style='text-align: center; margin-bottom: 20px;'>
            <img src='{IMAGE_HEADER_URL}' style='width: 100%; max-width: 800px; border-radius: 5px;'>
        </div>
    """, unsafe_allow_html=True)

    # **CENTERED TITLE**
    st.markdown("<h1 style='text-align: center;'>Who's on Pole?</h1>", unsafe_allow_html=True)
    
    # API Key check
    if not st.secrets.get("GEMINI_API_KEY"):
        st.error("❌ Missing GEMINI_API_KEY in Secrets.")
        return

    st.markdown("---")

    # Auto-detect latest race for defaults
    def_year, def_track = get_latest_completed_race()
    
    # Selectors
    c1, c2 = st.columns(2)
    
    # Safe index finding
    try:
        y_idx = YEARS.index(def_year)
    except: y_idx = 0
    
    try:
        t_idx = TRACKS.index(def_track)
    except: t_idx = 0

    sel_year = c1.selectbox("Year:", YEARS, index=y_idx)
    sel_track = c2.selectbox("Track:", TRACKS, index=t_idx)
    
    st.markdown("---")

    # --- BUTTON 1: Current Session (Auto-Detect) ---
    if st.button("🏎️ Predict Winner (Current Session Data)", type="primary", use_container_width=True):
        status = st.empty()
        status.info("🔍 Searching for latest session data...")
        
        found_data = None
        found_session = None
        
        # Try to load data in priority order: R -> Q -> FP3 -> FP2 -> FP1
        for s in SESSIONS_PRIORITY:
            data, msg = load_and_process_data(sel_year, sel_track, s)
            if data:
                found_data = data
                found_session = s
                break
        
        if found_data:
            status.success(f"✅ Found data for session: **{found_session}**")
            st.subheader(f"📊 Analysis: {sel_track} {sel_year} ({found_session})")
            
            with st.spinner("🤖 AI Analyst is thinking..."):
                # Generate Prompt based on the specific session type found
                prompt = create_prediction_prompt(found_data, sel_year, sel_track, f"{sel_track} {found_session}", found_session)
                result = get_gemini_prediction(prompt)
                st.markdown(result)
        else:
            status.error(f"❌ No data found for {sel_track} {sel_year} (Checked R, Q, FP3, FP2, FP1). The event might be in the future.")

    # --- BUTTON 2: Pre-Race ---
    if st.button("🔮 Preliminary Prediction (Past & Seasonal Context)", type="secondary", use_container_width=True):
        with st.spinner("🔮 Gathering historical and seasonal context..."):
            res = get_preliminary_prediction(sel_year, sel_track)
            if res:
                st.markdown("---")
                st.markdown(res)

if __name__ == "__main__":
    main()
