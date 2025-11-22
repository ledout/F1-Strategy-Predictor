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
# Order of priority for auto-detection: Race -> Qualifying -> FP3 -> FP2 -> FP1
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


# No Streamlit Caching to prevent stale data issues
def load_and_process_data(year, event, session_key):
	"""
    Loads data from FastF1. 
    Returns: (context_text, session_name_string) or (None, error_message)
    """

	try:
		session = fastf1.get_session(year, event, session_key)
		
		# Robust Session.load() attempt
		try:
			# 1. Basic load attempt (we only want laps)
			session.load(laps=True, telemetry=False, weather=False, messages=False, pit_stops=False)
		except TypeError as e:
			# 2. Fallback for older FastF1 versions
			if "unexpected keyword argument" in str(e):
					session.load()
			else:
					raise e 
		except Exception as e:
			# 3. General error handling
			error_message = str(e)
			if "not loaded yet" in error_message:
					session.load(telemetry=False, weather=False, messages=False, laps=True, pit_stops=False)
			else:
					raise e

		# Robustness Check
		if session.laps is None or session.laps.empty or not isinstance(session.laps, pd.DataFrame):
			return None, f"Insufficient data for {year} {event} {session_key}. FastF1 'load_laps' error."

	except Exception as e:
		error_message = str(e)
        # Filter out common FastF1 error messages for cleaner UI
		if "Failed to load any schedule data" in error_message or "schedule data" in error_message:
				return None, f"FastF1: Failed to load data. Check if the session has occurred yet."
		if "not found" in error_message:
				return None, f"Data not found for {year} {event} {session_key}."
		return None, f"General FastF1 Loading Error: {error_message}"

	laps = session.laps.reset_index(drop=True)

	# Required lap filtering
	laps_filtered = laps.loc[
		(laps['IsGood'] == True) & 
		(laps['LapTime'].notna()) & 
		(laps['Driver'] != 'OUT') & 
		(laps['Team'].notna()) & 
		(laps['Time'].notna()) 
	].copy()

	laps_filtered['LapTime_s'] = laps_filtered['LapTime'].dt.total_seconds()

	# 5. Calculate statistics
	
	# **V55 FIX: Determine the ranking metric based on session type**
	if session_key in ["R", "S"]:
		# For race/sprint sessions, prioritize average pace
		ranking_column = 'Avg_Time_s'
	else:
		# For practice/qualifying, prioritize fastest lap (Best_Time_s) - CRITICAL for Qualifying accuracy
		ranking_column = 'Best_Time_s'
		
	# Calculate necessary stats
	driver_stats = laps_filtered.groupby('Driver').agg(
		Best_Time=('LapTime', 'min'),
		Avg_Time=('LapTime', 'mean'),
		Var=('LapTime_s', lambda x: np.var(x) if len(x) >= 2 else np.nan),
		Laps=('LapTime', 'count')
	).reset_index()

	driver_stats['Best_Time_s'] = driver_stats['Best_Time'].dt.total_seconds()
	driver_stats['Avg_Time_s'] = driver_stats['Avg_Time'].dt.total_seconds()

    # Filter drivers with very few laps only for Race sessions to avoid skewing
	if session_key in ["R", "S"]:
		driver_stats = driver_stats[driver_stats['Laps'] >= 3] 
	
	if driver_stats.empty:
		return None, f"Insufficient data for statistical analysis in {session_key}."

	# Process data to text format (Top 10)
	data_lines = []
	
	# Rank the drivers based on the selected metric
	driver_stats_ranked = driver_stats.sort_values(by=ranking_column, ascending=True).head(10)
	
	for index, row in driver_stats_ranked.iterrows():
		# Format time strings
		best_time_str = str(row['Best_Time']).split('0 days ')[-1][:10] if row['Best_Time'] is not pd.NaT else 'N/A'
		avg_time_str = str(row['Avg_Time']).split('0 days ')[-1][:10] if row['Avg_Time'] is not pd.NaT else 'N/A'
		
		# Data is now sent based on the full statistical profile
		data_lines.append(
			f"POS {index+1}: {row['Driver']} | Best: {best_time_str} | Avg: {avg_time_str} | Var: {row['Var']:.3f} | Laps: {int(row['Laps'])}"
		)

	context_data = "\n".join(data_lines)

	return context_data, session.name


# --- NEW: Function to find the latest race (Added back!) ---
@st.cache_data(ttl=3600)
def get_latest_completed_race():
    """
    Finds the latest completed conventional F1 race across all years 
    to set the default dropdown state.
    Returns: (year, track_name) or (2024, 'Bahrain') as fallback default.
    """
    latest_date = pd.Timestamp.min
    latest_race = None
    
    # Start searching from the current year backwards
    for year in sorted(YEARS, reverse=True):
        try:
            # We don't need to load the full schedule, just get the event list
            schedule = fastf1.get_events(year=year)
            
            # Filter for completed conventional races
            if 'EventDate' in schedule.columns:
                 # We filter for events that have already happened based on today's date
                 today = pd.Timestamp.now()
                 completed_races = schedule.loc[
                    (schedule['EventDate'] < today) &
                    (schedule['EventFormat'] == 'conventional')
                 ]
                 
                 if not completed_races.empty:
                    last_event = completed_races.sort_values(by='EventDate', ascending=False).iloc[0]
                    return (year, last_event['EventName'])
                    
        except Exception:
            continue
            
    # Fallback to a common race if no data is found
    return (2024, 'Bahrain')


def find_last_three_races_data(current_year, event, expander_placeholder):
	"""Finds the last three 'conventional' races that should have occurred this season and returns their race data."""

	with expander_placeholder.container():
		st.info("🔄 Starting Seasonal Data Collection (Last 3 Races)")
		
		schedule = None
		current_event_date = pd.to_datetime(date.today()) # Set default to today's date
		
		try:
			schedule = fastf1.get_event_schedule(current_year)
			if schedule.empty:
				return [], "Error: Current year's schedule is empty." 

		except Exception as e:
			return [], f"Error: Failed to load current year's schedule. {e}" 
		
		
		# 1. Find the current event
		current_event = schedule[schedule['EventName'] == event]
		
		# Robust handling if the current event is missing from the Schedule
		if current_event.empty:
			st.warning(f"⚠️ Warning: Current event ({event}) not found in the full schedule. Using today's date.")
            # If future year, stop
			if current_year > date.today().year:
				return [], "❌ Cannot perform seasonal analysis for a future year."
			
		else:
			try:
				# Event found
				current_event_date = current_event['EventDate'].iloc[0]
				current_event_round = current_event['RoundNumber'].iloc[0]
				
				# 2. Check round number
				if current_event_round <= 4:
					st.warning(f"⚠️ Warning: Event is early in the season. Skipping seasonal context.")
					return [], "Seasonal skip." 
			except Exception:
                # Fallback if columns missing
				pass
		
		
		# 3. Filter races based on date
		try:
			potential_races = schedule.loc[
				(schedule['EventFormat'] == 'conventional') &
				(schedule['EventDate'] < current_event_date)
			].sort_values(by='EventDate', ascending=False).head(3) 
		except KeyError as e:
			return [], f"FastF1: Missing column ({e})."
		
		
		if potential_races.empty:
			st.warning(f"No previous conventional races found in {current_year}.")
			return [], f"No previous races." 
		
		race_reports = []
		
		for index, race in potential_races.iterrows():
			event_name = race['EventName']
			st.info(f"🔮 Attempting to load Race Data: {event_name} {current_year}...")
			
			# Attempt to load data (Always 'R' for seasonal context)
			context_data, session_name = load_and_process_data(current_year, event_name, 'R')
			
			if context_data:
				report = (
					f"--- Pace Report: {event_name} {current_year} Race (Seasonal Context) ---\n"
					f"{context_data}\n"
				)
				race_reports.append(report)
				st.success(f"✅ Race data for {event_name} loaded.")
			else:
				st.warning(f"⚠️ Could not load data for {event_name}.")

		if not race_reports:
			return [], f"No complete seasonal data found." 
		
		st.success("✅ Seasonal data processed successfully.")
		return race_reports, "Seasonal data loaded"


def create_prediction_prompt(context_data, year, event, session_name, session_type):
	"""Builds the complete prompt for the Gemini model for current data."""

	prompt_data = f"--- Raw Data for Analysis ---\n{context_data}"
    
    # Customize instructions based on session type
	if session_type in ['Q', 'FP1', 'FP2', 'FP3']:
		focus_text = "This data is from a Practice or Qualifying session. The drivers are ranked by FASTEST LAP. Focus your prediction on raw speed and qualifying performance. Note that Lando Norris or other top qualifiers should be prioritized if they are at the top."
	else:
		focus_text = "This data is from a Race session. The drivers are ranked by AVERAGE PACE. Focus on consistency and race strategy."

	prompt = f"""
You are a Senior F1 Analyst. Your task is to analyze the statistical data of the laps 
({session_name}, {event} {year}) and provide a complete strategic report and winner prediction.

{prompt_data}

**IMPORTANT CONTEXT:** {focus_text}

--- Analysis Guidelines (V33 - Combined R/Q/S Analysis and Context) ---
1. **Immediate Prediction (Executive Summary):** Select one winner and present the main justification (average pace or consistency) in a single sentence, **in English only**. (Mandatory)
2. **Overall Performance Summary:** Analyze the Average Pace (Avg Time) and Consistency (Var). Var < 1.0 is considered excellent consistency. Var > 5.0 may indicate inconsistency.
3. **Tire and Strategy Deep Dive:** Analyze the data relative to the track. Explain what kind of setup is reflected in the data.
4. **Weather/Track Influence:** Add general context on track conditions.
5. **Strategic Conclusions and Winner Justification:** Present a summary and clear justification.
6. **Confidence Score Table (D5):** Provide a Confidence Score table (in Markdown format) containing the top 5 candidates.

--- Mandatory Output Format (Markdown, English for the entire report) ---
🏎️ Strategy Report: {event} {year} ({session_name})

## Immediate Prediction (Executive Summary)
...

## Overall Performance Summary
...

## Tire and Strategy Deep Dive
...

## Weather/Track Influence
...

## Strategic Conclusions and Winner Justification
...

## 🏎️ Recommended Strategy & Pit-Stop Window
...

## 📊 Confidence Score Table (D5 - Visual Data)
| Driver | Confidence Score (%) |
|:--- | :--- |
| ... | :--- |
"""
	return prompt


def get_preliminary_prediction(current_year, event):
	"""Combines race data from the previous year and the last three races."""

	previous_year = current_year - 1

	# Translate Subheader
	st.subheader("🏁 Data Collection for Preliminary Prediction (Pre-Race Analysis)")

	with st.expander("🛠️ Show Historical and Seasonal Data Loading Details (Diagnostics)", expanded=False):
		expander_placeholder = st.container() 

		with expander_placeholder:
			st.info(f"🔮 Analyzing track dominance: Loading race data for {event} from {previous_year}...")

			# 1. Load Historical Data (Previous Year on the Same Track)
			context_data_prev, session_name_prev = load_and_process_data(previous_year, event, 'R')
			if context_data_prev:
				st.success(f"✅ Race data for {event} {previous_year} loaded successfully.")
			else:
				st.warning(f"⚠️ Warning: No complete historical data found for {event} {previous_year}.")

			st.markdown("---")

		# 2. Load Seasonal Data (Last 3 Completed Races)
		race_reports_current, status_msg = find_last_three_races_data(current_year, event, expander_placeholder)

	# 3. Data Check and Report Unification

	based_on_text = ""
	report_current = f"--- Seasonal Pace Report (No Seasonal Data Available) ---\n"

	if context_data_prev:
		report_prev = (
			f"--- Pace Report: {event} Race {previous_year} (Historical Track Context) ---\n"
			f"{context_data_prev}\n"
		)
		based_on_text += f"{event} {previous_year} Race Data"
	else:
		report_prev = f"--- Pace Report: {event} Race {previous_year} (No Historical Track Data Available) ---\n"

	if race_reports_current and isinstance(race_reports_current, list):
		report_current = "\n" + "\n".join(race_reports_current)
		num_races = len(race_reports_current)
		if based_on_text: based_on_text += " & "
		based_on_text += f"Analysis of the Last {num_races} Races of {current_year}."
	else:
		if not based_on_text: based_on_text = f"No Current Season Context or Historical Data Available."


	if not context_data_prev and not race_reports_current:
		st.error("❌ No historical or seasonal data available. Cannot perform analysis.")
		return None


	# 4. Build the prompt
	full_data_prompt = report_prev + report_current

	prompt = f"""
You are a Senior F1 Analyst. Analyze the following combined data to provide a Preliminary (Pre-Race) Prediction Report for **{event} {current_year} Race**.

{full_data_prompt}

--- Analysis Guidelines (V47 - Weight 65/35, Implicit Weather) ---
1. **Immediate Prediction (Executive Summary):** Select one winner.
2. **Past Performance Analysis:** Analyze the historical report.
3. **Current Season Trend Analysis:** Analyze the seasonal reports.
4. **Strategic Conclusions and Winner Justification:** Justify the winner choice based on seasonal capability (65% weight) and previous track dominance (35% weight).
5. **Weather & Tire Degradation:** Analyze the data.
6. **Confidence Score Table (D5):** Provide a Confidence Score table.

--- Mandatory Output Format (Markdown, English) ---
🔮 Pre-Race Strategy Report: {event} {year}

Based on: {based_on_text}

## Immediate Prediction
...
## Past Performance Analysis
...
## Current Season Trend Analysis
...
## Strategic Conclusions
...
## 📊 Confidence Score Table (D5 - Visual Data)
| Driver | Confidence Score (%) |
|:--- | :--- |
"""

	try:
		if not full_data_prompt:
			raise ValueError("Prompt failed: No base data for report creation.")

		report = get_gemini_prediction(prompt)
		return report
	except Exception as e:
		st.error(f"❌ Gemini API Error: {e}")
		return None

# --- Main Streamlit Function ---

def main():
	st.set_page_config(page_title="F1 Strategy Predictor", layout="centered")

	# Custom Header
	st.markdown(
		f"""
		<div style='text-align: center; margin-bottom: 20px;'>
			<img src='{IMAGE_HEADER_URL}' alt='F1 P1 Predict Header' style='width: 100%; max-width: 800px; height: auto; border-radius: 5px; object-fit: cover;'>
		</div>
		""",
		unsafe_allow_html=True
	)

	# Center Who's on Pole?
	st.markdown("<h1 style='text-align: center; font-size: 2em; font-weight: bold; margin-bottom: 10px;'>Who's on Pole?</h1>", unsafe_allow_html=True)
	st.markdown("---")

	# API Key Check
	try:
		api_key_check = st.secrets.get("GEMINI_API_KEY")
		if not api_key_check:
			st.error("❌ Error: Gemini API Key is not configured in Streamlit Secrets.")
	except Exception:
		st.error("❌ Error: Failed to read API key.")

	st.markdown("---")

	# --- V59 FIX: Auto-detect latest race and set initial defaults ---
	latest_year, latest_track = get_latest_completed_race()
	
	try:
	    year_index = YEARS.index(latest_year)
	    track_index = TRACKS.index(latest_track)
	except ValueError:
	    year_index = 0 
	    track_index = 0 
	    
	# Parameter Selection
	col1, col2 = st.columns(2) 

	with col1:
		selected_year = st.selectbox("Year:", YEARS, index=year_index, key="select_year")
	with col2:
		selected_event = st.selectbox("Track:", TRACKS, index=track_index, key="select_track")

	st.markdown("---")

	# 1. Current Data Analysis Button (Auto-finds R, Q, FP3...)
	if st.button("🏎️ Predict Winner (Current Session Data)", use_container_width=True, type="primary"):

		# Logic to automatically find the latest session (R -> Q -> FP3 -> FP2 -> FP1)
		selected_session = None
		context_data = None
		status_msg = ""
		
		for session_type in SESSIONS_PRIORITY:
			# Try to load the data for the most recent session
			temp_context_data, temp_status_msg = load_and_process_data(selected_year, selected_event, session_type)
			
			if temp_context_data:
				selected_session = session_type
				context_data = temp_context_data
				break # Found data, exit loop

		if not selected_session:
			status_msg = f"Error: Failed to find valid data for any session ({'/'.join(SESSIONS_PRIORITY)}) for {selected_event} {selected_year}."
			st.error(f"❌ {status_msg}")
			return

		# Start Analysis
		st.subheader(f"🔄 Starting Analysis: {selected_event} {selected_year} ({selected_session})")
		status_placeholder = st.empty()
		status_placeholder.info("...Loading and processing data from FastF1...")
		status_placeholder.success(f"✅ Data processed successfully for {selected_session}. Sending to AI for analysis...")

		try:
			# Pass selected_session to prompt creation for better context
			prompt = create_prediction_prompt(context_data, selected_year, selected_event, selected_session, selected_session)
			prediction_report = get_gemini_prediction(prompt)

			status_placeholder.success("🏆 Analysis completed successfully!")
			st.markdown("---")
			st.markdown(prediction_report)

		except APIError as e:
			status_placeholder.error(f"❌ Gemini API Error: {e}")
		except ValueError as e: 
			status_placeholder.error(f"❌ Critical Error: {e}")
		except Exception as e:
			status_placeholder.error(f"❌ Unexpected Error: {e}")

	st.markdown("---")

	# 2. Pre-Race Prediction Button
	if st.button("🔮 Preliminary Prediction (Past & Seasonal Context)", use_container_width=True, type="secondary"):
		st.subheader(f"🔮 Starting Preliminary Prediction: {selected_event} {selected_year}")

		prelim_report = get_preliminary_prediction(selected_year, selected_event)

		if prelim_report:
			st.markdown("---")
			st.markdown(prelim_report)


if __name__ == "__main__":
	main()
