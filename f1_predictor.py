import streamlit as st
import fastf1
import pandas as pd
import logging
from google import genai
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential
import io 
from datetime import date, datetime 
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


# --- Helper Functions for Data Processing ---

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


# No Streamlit Caching
def load_and_process_data(year, event, session_key):
	"""Loads data from FastF1 and performs initial processing."""

	try:
		session = fastf1.get_session(year, event, session_key)
		
		# Robust Session.load() attempt
		try:
			# Basic load attempt (laps only)
			session.load(laps=True, telemetry=False, weather=False, messages=False, pit_stops=False)
		except TypeError as e:
			if "unexpected keyword argument" in str(e):
					session.load()
			else:
					raise e 
		except Exception as e:
			# General fallback
			if "not loaded yet" in str(e):
					session.load(telemetry=False, weather=False, messages=False, laps=True, pit_stops=False)
			else:
					raise e

		# Robustness Check
		if session.laps is None or session.laps.empty or not isinstance(session.laps, pd.DataFrame):
			return None, f"Insufficient data for {year} {event} {session_key}."

	except Exception as e:
		error_message = str(e)
		if "Failed to load" in error_message or "schedule" in error_message:
				return None, f"FastF1: Data not available/loaded for {event} {year} {session_key}."
		if "not found" in error_message:
				return None, f"Event not found: {year} {event}."
		return None, f"Error: {error_message}"

	laps = session.laps.reset_index(drop=True)

	# --- Filter Laps Correctly ---
	if 'IsAccurate' in laps.columns:
		clean_laps = laps.loc[laps['IsAccurate'] == True]
	elif 'IsGood' in laps.columns:
		clean_laps = laps.loc[laps['IsGood'] == True]
	else:
		# Fallback if no accuracy column found
		clean_laps = laps

	# Common filters
	clean_laps = clean_laps.loc[
		(clean_laps['LapTime'].notna()) &
		(clean_laps['Driver'] != 'OUT') &
		(clean_laps['Team'].notna())
	].copy()

	clean_laps['LapTime_s'] = clean_laps['LapTime'].dt.total_seconds()

	# --- Split Logic for Race vs Quali/Practice ---
	
	# CASE 1: RACE / SPRINT (Long Run Analysis)
	if session_key in ["R", "S"]:
		driver_stats = clean_laps.groupby('Driver').agg(
			Best_Time=('LapTime', 'min'),
			Avg_Time=('LapTime', 'mean'),
			Var=('LapTime_s', lambda x: np.var(x) if len(x) >= 2 else np.nan),
			Laps=('LapTime', 'count')
		).reset_index()
		
		driver_stats['Best_Time_s'] = driver_stats['Best_Time'].dt.total_seconds()
		driver_stats['Avg_Time_s'] = driver_stats['Avg_Time'].dt.total_seconds()
		
		# Filter drivers with too few laps for valid pace analysis
		driver_stats = driver_stats[driver_stats['Laps'] >= 3]
		driver_stats = driver_stats[driver_stats['Var'].notna()]
		
		# Rank by Average Pace
		driver_stats = driver_stats.sort_values(by='Avg_Time_s', ascending=True).head(10)

	# CASE 2: QUALIFYING / PRACTICE (Fastest Lap Analysis)
	else:
		# Just find the minimum lap time per driver
		driver_stats = clean_laps.groupby('Driver').agg(
			Best_Time=('LapTime', 'min'),
			Laps=('LapTime', 'count')
		).reset_index()
		
		driver_stats['Best_Time_s'] = driver_stats['Best_Time'].dt.total_seconds()
		driver_stats['Avg_Time'] = pd.NaT # Not relevant
		driver_stats['Var'] = 0.0 # Not relevant
		
		# Rank by Best Time (Fastest Lap)
		driver_stats = driver_stats.sort_values(by='Best_Time_s', ascending=True).head(10)


	if driver_stats.empty:
		return None, f"Insufficient data for analysis in {session_key}."

	# Format Output
	data_lines = []
	for index, row in driver_stats.iterrows():
		best_str = str(row['Best_Time']).split('0 days ')[-1][:11] # Trim
		if session_key in ["R", "S"]:
			# Show Avg for Race
			avg_str = str(row['Avg_Time']).split('0 days ')[-1][:11]
			data_lines.append(f"POS {index+1}: {row['Driver']} | Best: {best_str} | Avg: {avg_str} | Var: {row['Var']:.3f} | Laps: {int(row['Laps'])}")
		else:
			# Show only Best for Quali
			data_lines.append(f"POS {index+1}: {row['Driver']} | Best: {best_str} | Laps: {int(row['Laps'])}")

	context_data = "\n".join(data_lines)
	return context_data, session.name


@st.cache_data(ttl=3600)
def get_latest_completed_race():
    """
    Finds the most recent F1 event that has at least started (Session1Date passed).
    Returns: (year, track_name) or fallback.
    """
    now = pd.Timestamp.now()
    
    # Loop backwards from current year
    for year in sorted(YEARS, reverse=True):
        try:
            schedule = fastf1.get_event_schedule(year)
            
            # We want events where at least the weekend has started (Session1Date exists and is past)
            if 'Session1Date' in schedule.columns:
                 # Ensure Session1Date is datetime
                 if not pd.api.types.is_datetime64_any_dtype(schedule['Session1Date']):
                     schedule['Session1Date'] = pd.to_datetime(schedule['Session1Date'])

                 started_events = schedule.loc[
                    (schedule['Session1Date'] < now) &
                    (schedule['EventFormat'] == 'conventional')
                 ]
                 
                 if not started_events.empty:
                    # Get the last one that started
                    last_event = started_events.sort_values(by='Session1Date', ascending=False).iloc[0]
                    return (year, last_event['EventName'])
                    
        except Exception:
            continue
            
    return (2024, 'Bahrain') # Fallback


def find_last_three_races_data(current_year, event, expander_placeholder):
	"""Finds the last three 'conventional' races that should have occurred this season."""

	with expander_placeholder.container():
		st.info("🔄 Starting seasonal data collection (Last 3 Races)")
		
		try:
			schedule = fastf1.get_event_schedule(current_year)
			if schedule.empty: return [], "Error: Schedule empty."
		except Exception as e:
			return [], f"Error loading schedule: {e}" 
		
		# 1. Find current event
		current_event = schedule[schedule['EventName'] == event]
		
		if current_event.empty:
            # If event not in schedule, try to use today's date as reference
			ref_date = pd.Timestamp.now()
			# However, if we are in a future year with no schedule, stop
			if current_year > date.today().year:
				st.error("❌ Cannot analyze seasonal context for future year.")
				return [], "Future year error."
		else:
			ref_date = current_event['EventDate'].iloc[0]
		
		# 2. Filter previous races (using EventDate)
		try:
			potential_races = schedule.loc[
				(schedule['EventFormat'] == 'conventional') &
				(schedule['EventDate'] < ref_date)
			].sort_values(by='EventDate', ascending=False).head(3)
		except KeyError:
			return [], "FastF1: Missing date column."
		
		if potential_races.empty:
			st.warning(f"No previous races found in {current_year} before {event}.")
			return [], "No previous races."
		
		race_reports = []
		for index, race in potential_races.iterrows():
			event_name = race['EventName']
			st.info(f"🔮 Loading: {event_name} {current_year}...")
			
			# Always load 'R' (Race) for seasonal context
			context_data, _ = load_and_process_data(current_year, event_name, 'R')
			
			if context_data:
				report = f"--- {event_name} {current_year} (Race) ---\n{context_data}\n"
				race_reports.append(report)
				st.success(f"✅ Loaded {event_name}.")
			else:
				st.warning(f"⚠️ Could not load data for {event_name}.")

		if not race_reports:
			return [], "No seasonal data found." 
		
		st.success("✅ Seasonal data processed.")
		return race_reports, "Success"


def create_prediction_prompt(context_data, year, event, session_name, session_type):
	"""Builds the prompt."""

	prompt_data = f"--- Drivers Data ---\n{context_data}"
    
    # Dynamic instructions
	if session_type in ['R', 'S']:
		focus_instruction = "This is RACE data. Drivers are ranked by AVERAGE PACE. Focus on consistency, tire degradation, and long-run speed."
	else:
		focus_instruction = "This is QUALIFYING/PRACTICE data. Drivers are ranked by FASTEST LAP. Focus on raw one-lap speed and pole position potential. Ignore consistency."

	prompt = f"""
You are a Senior F1 Strategy Analyst.
Target: Predict winner for {event} {year}.
Context: {session_name} ({session_type}).

**DATA TYPE:** {focus_instruction}

**DATA:**
{prompt_data}

--- Analysis Guidelines ---
1. **Immediate Prediction:** Winner name + 1 sentence reason.
2. **Performance Analysis:** Analyze the data provided. For Quali/FP, focus on the fastest lap gaps. For Race, focus on avg pace.
3. **Track Specifics:** How does {event} suit the top car concepts?
4. **Confidence Table:** Top 5 drivers with %.

--- Output Format (Markdown, English) ---
🏎️ Strategy Report: {event} {year} ({session_type})

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
	"""Pre-race prediction combining history + season."""

	previous_year = current_year - 1
	st.subheader("🏁 Pre-Race Analysis Data Collection")

	with st.expander("🛠️ Data Loading Details", expanded=False):
		holder = st.container()
		with holder:
			st.info(f"Loading historical data: {event} {previous_year}...")
			hist_data, _ = load_and_process_data(previous_year, event, 'R')
			if hist_data: st.success("✅ Historical data loaded.")
			else: st.warning("⚠️ Historical data missing.")
			st.markdown("---")
		
		seasonal_reports, _ = find_last_three_races_data(current_year, event, holder)

	# Build prompt
	data_text = ""
	if hist_data:
		data_text += f"### Historical Data ({event} {previous_year}):\n{hist_data}\n\n"
	if seasonal_reports:
		data_text += "### Recent Season Form (Last 3 Races):\n" + "\n".join(seasonal_reports)
        
    # Allow running if at least one source exists
	if not data_text:
		st.error("❌ No data available (Historical or Seasonal). Cannot predict.")
		return None

	prompt = f"""
Act as F1 Strategist. Predict winner for **{event} {current_year}**.

**DATA AVAILABLE:**
{data_text}

**INSTRUCTIONS:**
1. **Weighting:** 65% Recent Season Form, 35% Historical Track Data.
2. **Output:** English. Markdown.
3. **Format:**
   - **Winner Prediction:** Name + Reason.
   - **Analysis:** Contrast history vs current form.
   - **Table:** Top 5 drivers probability.

**GO.**
"""
	return get_gemini_prediction(prompt)

# --- Main App ---

def main():
	st.set_page_config(page_title="F1 Strategy Predictor", layout="centered")

	# Header
	st.markdown(
		f"""
		<div style='text-align: center; margin-bottom: 20px;'>
			<img src='{IMAGE_HEADER_URL}' alt='F1 P1 Predict Header' style='width: 100%; max-width: 800px; height: auto; border-radius: 5px; object-fit: cover;'>
		</div>
		""",
		unsafe_allow_html=True
	)

	# Centered Title
	st.markdown("<h1 style='text-align: center;'>Who's on Pole?</h1>", unsafe_allow_html=True)
	st.markdown("---")

	# API Key
	if not st.secrets.get("GEMINI_API_KEY"):
		st.error("❌ Missing GEMINI_API_KEY in Secrets.")
		return

	# Auto-detect latest race
	def_year, def_track = get_latest_completed_race()
	
	try:
		y_idx = YEARS.index(def_year)
		t_idx = TRACKS.index(def_track)
	except:
		y_idx, t_idx = 0, 0

	# Selectors
	c1, c2 = st.columns(2)
	sel_year = c1.selectbox("Year:", YEARS, index=y_idx)
	sel_track = c2.selectbox("Track:", TRACKS, index=t_idx)
	
	st.markdown("---")

	# Button 1: Current Session
	if st.button("🏎️ Predict Winner (Current Session Data)", type="primary", use_container_width=True):
		
		# Auto-find session logic
		found_data = None
		found_session = None
		
		status = st.empty()
		status.info(f"🔍 Searching for latest data for {sel_track} {sel_year}...")

		# Loop through sessions to find the latest one with data
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
				prompt = create_prediction_prompt(found_data, sel_year, sel_track, f"{sel_track} {found_session}", found_session)
				res = get_gemini_prediction(prompt)
				st.markdown(res)
		else:
			status.error(f"❌ No data found for {sel_track} {sel_year} (Checked R, Q, FP3, FP2, FP1). The event might be in the future.")

	st.markdown("---")

	# Button 2: Pre-Race
	if st.button("🔮 Preliminary Prediction (Past & Seasonal Context)", type="secondary", use_container_width=True):
		with st.spinner("🔮 Gathering context..."):
			res = get_preliminary_prediction(sel_year, sel_track)
			if res:
				st.markdown("---")
				st.markdown(res)

if __name__ == "__main__":
	main()
