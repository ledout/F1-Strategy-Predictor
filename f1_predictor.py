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
SESSIONS_PRIORITY = ["R", "Q", "FP3", "FP2", "FP1"] 
YEARS = [2025, 2024, 2023, 2022, 2021, 2020]
MODEL_NAME = "gemini-2.5-flash"
# Custom Header Image URL (Converted to RAW format for proper loading)
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
	"""Loads data from FastF1 and performs initial processing, with error handling for session.load()."""

	try:
		session = fastf1.get_session(year, event, session_key)
		
		# Robust Session.load() attempt for different FastF1 versions
		try:
			# 1. Basic load attempt (we only want laps)
			session.load(laps=True, telemetry=False, weather=False, messages=False, pit_stops=False)
		except TypeError as e:
			# 2. If it fails due to unexpected arguments, try loading without any arguments.
			if "unexpected keyword argument" in str(e):
					# Let FastF1 load everything if the arguments don't work
					session.load()
			else:
					# If it's another type error, re-raise it
					raise e 
		except Exception as e:
			# General loading error - explicit flag path
			error_message = str(e)
			if "not loaded yet" in error_message:
					# Explicit load attempt if there's a metadata issue
					session.load(telemetry=False, weather=False, messages=False, laps=True, pit_stops=False)
			else:
					raise e

		# Robustness Check: Ensure session.laps is a valid DataFrame
		if session.laps is None or session.laps.empty or not isinstance(session.laps, pd.DataFrame):
			return None, f"Insufficient data for {year} {event} {session_key}. FastF1 'load_laps' error."

	except Exception as e:
		error_message = str(e)

		if "Failed to load any schedule data" in error_message or "schedule data" in error_message:
				return None, f"FastF1: Failed to load any schedule data. Error loading FastF1: Possible network/connection issue or the year/track does not exist."

		if "not found" in error_message or "The data you are trying to access has not been loaded yet" in error_message:
				return None, f"Data missing for {year} {event} {session_key}. May be a cancelled or future event. Error: {error_message.split(':', 1)[-1].strip()}"

		if "unexpected keyword argument" in error_message:
				return None, f"FastF1 Version Error: Session.load() received an unexpected argument. (Error: {error_message})"

		return None, f"General FastF1 Loading Error: {error_message}"

	laps = session.laps.reset_index(drop=True)

	# Required lap filtering for accuracy
	laps_filtered = laps.loc[
		(laps['IsGood'] == True) & # Use IsGood for robust filtering
		(laps['LapTime'].notna()) &
		(laps['Driver'] != 'OUT') &
		(laps['Team'].notna()) &
		(laps['Time'].notna()) 
	].copy()

	laps_filtered['LapTime_s'] = laps_filtered['LapTime'].dt.total_seconds()

	# 5. Calculate statistics
	
	# Determine the ranking metric based on session type
	if session_key in ["R", "S"]:
		# For race/sprint sessions, prioritize average pace (lower is better)
		ranking_column = 'Avg_Time_s'
	else:
		# For practice/qualifying, prioritize fastest lap (Best_Time_s) - CRITICAL FOR Q/FP ACCURACY
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

	# Only include stats if 5 or more laps were completed and variance is calculated
	driver_stats = driver_stats[driver_stats['Laps'] >= 5]
	driver_stats = driver_stats[driver_stats['Var'].notna()] # Remove drivers with no variance

	
	if driver_stats.empty:
		return None, f"Insufficient data (fewer than 5 laps per driver) for statistical analysis in {session_key}."

	# Process data to text format (Top 10)
	data_lines = []
	
	# Rank the drivers based on the selected metric (Fastest Lap for Q/FP)
	driver_stats_ranked = driver_stats.sort_values(by=ranking_column, ascending=True).head(10)
	
	for index, row in driver_stats_ranked.iterrows():
		# Format time strings
		best_time_str = str(row['Best_Time']).split('0 days ')[-1][:10] if row['Best_Time'] is not pd.NaT else 'N/A'
		avg_time_str = str(row['Avg_Time']).split('0 days ')[-1][:10] if row['Avg_Time'] is not pd.NaT else 'N/A'
		
		# Data is now sent based on the full statistical profile
		data_lines.append(
			f"DRIVER: {row['Driver']} | Best: {best_time_str} | Avg: {avg_time_str} | Var: {row['Var']:.3f} | Laps: {int(row['Laps'])}"
		)

	context_data = "\n".join(data_lines)

	return context_data, session.name

# --- NEW: Function to find the latest race (V59/V60) ---

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
            completed_races = schedule.loc[
                (schedule['EventCompleted'] == True) &
                (schedule['EventFormat'] == 'conventional')
            ]
            
            if not completed_races.empty:
                last_event = completed_races.sort_values(by='EventDate', ascending=False).iloc[0]
                
                if last_event['EventDate'] > latest_date:
                    latest_date = last_event['EventDate']
                    latest_race = (year, last_event['EventName'])
                    # Since we sort by year descending, the first found is usually the latest
                    return latest_race 
                    
        except Exception:
            continue
            
    # Fallback to a common race if no data is found
    return (2024, 'Bahrain')


def find_last_three_races_data(current_year, event, expander_placeholder):
	"""Finds the last three 'conventional' races that should have occurred this season and returns their race data."""

	with expander_placeholder.container():
		st.info("🔄 Starting seasonal data collection (Last 3 Races)")
		
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
			st.warning(f"⚠️ Warning: Current event ({event}) not found in the full schedule. Using today's date ({current_event_date.strftime('%Y-%m-%d')}) as a seasonal reference point.")
			
			# If the selected year is in the future (e.g., 2025), this might fail.
			if current_year > date.today().year:
				st.error("❌ Cannot perform seasonal analysis for a future year without a defined event date.")
				return [], "❌ Cannot perform seasonal analysis for a future year."
			
		else:
			try:
				# Event found, use its information
				current_event_date = current_event['EventDate'].iloc[0]
				
				current_event_round = current_event['RoundNumber'].iloc[0]
				
				# 2. Check round number - only if the event was found
				if current_event_round <= 4:
					st.warning(f"⚠️ Warning: Current event ({event}) is one of the first 4 races of the season. Insufficient seasonal context. Skipping.")
					return [], "Seasonal skip (Race too early in the season)." 
			except KeyError as e:
				# If a column is missing in the Schedule
				st.error(f"FastF1 Schedule Error: Missing column ({e}). Using today's date.")
				# Continue with current_event_date = date.today()
			except Exception as e:
				# Another unexpected Schedule error
				st.error(f"Unexpected Schedule error: {e}")
				return [], "FastF1 Schedule Error."
		
		
		# 3. Filter races based on date (or today's date if the event was not found)
		try:
			# Filter based on the current event date
			potential_races = schedule.loc[
				(schedule['EventFormat'] == 'conventional') &
				(schedule['EventDate'] < current_event_date)
			].sort_values(by='EventDate', ascending=False).head(3) 
		except KeyError as e:
			return [], f"FastF1: Missing column ({e}). Cannot perform seasonal analysis."
		
		
		if potential_races.empty:
			st.warning(f"No previous conventional races found in the {current_year} schedule before {event}.")
			return [], f"No previous races in {current_year}." 
		
		race_reports = []
		
		for index, race in potential_races.iterrows():
			event_name = race['EventName']
			st.info(f"🔮 Attempting to load Race Data: {event_name} {current_year}...")
			
			# Attempt to load data
			context_data, session_name = load_and_process_data(current_year, event_name, 'R')
			
			if context_data:
				report = (
					f"--- Pace Report: {event_name} {current_year} Race (Seasonal Context) ---\n"
					f"{context_data}\n"
				)
				race_reports.append(report)
				st.success(f"✅ Race data for {event_name} loaded successfully.")
			else:
				# If load_and_process_data fails
				st.warning(f"⚠️ Could not load complete race data for {event_name}. AI will ignore this race. (Error: {session_name})") 

		if not race_reports:
			# Returns a seasonal failure status
			return [], f"No complete seasonal data found in {current_year}." 
		
		st.success("✅ Seasonal data processed successfully. Proceeding to AI.")
		return race_reports, "Seasonal data loaded"


def create_prediction_prompt(context_data, year, event, session_name):
	"""Builds the complete prompt for the Gemini model for current data."""

	prompt_data = f"--- Raw Data for Analysis (Top 10 Drivers, Race/Session Laps) ---\n{context_data}"

	prompt = f"""
You are a Senior F1 Analyst. Your task is to analyze the statistical data of the laps 
({session_name}, {event} {year}) and provide a complete strategic report and winner prediction.

{prompt_data}

--- Analysis Guidelines (V33 - Combined R/Q/S Analysis and Context) ---
1. **Immediate Prediction (Executive Summary):** Select one winner and present the main justification (average pace or consistency) in a single sentence, **in English only**. (Mandatory)
2. **Overall Performance Summary:** Analyze the Average Pace (Avg Time) and Consistency (Var). Var < 1.0 is considered excellent consistency. Var > 5.0 may indicate inconsistency or race disruptions (such as an accident or red flag).
3. **Tire and Strategy Deep Dive:** Analyze the data relative to the track. Explain what kind of setup ('High Downforce'/'Low Downforce') is reflected in the data, assuming the Max Speed data of the leading drivers is available in your analysis.
4. **Weather/Track Influence:** Add general context on track conditions and their effect on tires. Assume stable and warm conditions unless the high Var suggests the use of rain/intermediate tires. 
5. **Strategic Conclusions and Winner Justification:** Present a summary and clear justification for the winner choice based on data and strategic considerations.
6. **Confidence Score Table (D5):** Provide a Confidence Score table (in Markdown format) containing the top 5 candidates with a confidence percentage (total percentage must be 100%). **Ensure the table format appears correctly in Markdown**.

--- Mandatory Output Format (Markdown, English for the entire report) ---
🏎️ Strategy Report: {event} {year}

Based on: Specific Session Data ({session_name} Combined)

## Immediate Prediction (Executive Summary)
...

## Past Performance Analysis
...

## Current Season Trend Analysis
...

## Strategic Conclusions and Winner Justification
...

## 🏎️ Recommended Strategy & Pit-Stop Window
...

## 📊 Confidence Score Table (D5 - Visual Data)
| Driver | Confidence Score (%) |
|:--- | :--- |
| ... | :--- |
| ... | :--- |
| ... | ... |
| ... | ... |
| ... | ... |
"""

	try:
		# Final and strict check before calling the API
		if not full_data_prompt:
			raise ValueError("Prompt failed: No base data for report creation.")

		# Call the corrected function
		report = get_gemini_prediction(prompt)
		return report
	except Exception as e:
		# Translate Error Message
		st.error(f"❌ Gemini API Error during preliminary prediction creation: {e}")
		return None

# --- Main Streamlit Function ---

def main():
	"""
	Main function that runs the Streamlit application.
	Finds the latest available race and sets it as the default selection.
	"""

	st.set_page_config(page_title="F1 Strategy Predictor", layout="centered")

	# Custom Header Image from URL (Replacing st.title)
	st.markdown(
		f"""
		<div style='text-align: center; margin-bottom: 20px;'>
			<img src='{IMAGE_HEADER_URL}' alt='F1 P1 Predict Header' style='width: 100%; max-width: 800px; height: auto; border-radius: 5px; object-fit: cover;'>
		</div>
		""",
		unsafe_allow_html=True
	)

	# **FIXED V60: Centering and Styling Who's on Pole?**
	st.markdown("<h1 style='text-align: center; font-size: 2em; font-weight: bold; margin-bottom: 10px;'>Who's on Pole?</h1>", unsafe_allow_html=True)
	st.markdown("---")

	# API Key Check
	try:
		api_key_check = st.secrets.get("GEMINI_API_KEY")
		if not api_key_check:
			st.error("❌ Error: Gemini API Key is not configured in Streamlit Secrets. Please set it.")
		if not api_key_check:
			st.warning("⚠️ Note: API Key is missing. The analysis will fail when attempting to connect to Gemini.")

	except Exception:
		st.error("❌ Error: Failed to read API key. Ensure you have configured it correctly in Secrets.")

	st.markdown("---")

	# Parameter Selection
	col1, col2 = st.columns(2) 

	# --- V59 FIX: Auto-detect latest race and set initial defaults ---
	latest_year, latest_track = get_latest_completed_race()
	
	# Setting indexes for dropdowns based on detected values
	try:
	    year_index = YEARS.index(latest_year)
	    track_index = TRACKS.index(latest_track)
	except ValueError:
	    # Fallback in case the track/year are not in the predefined lists
	    latest_year = YEARS[2] # 2023
	    latest_track = TRACKS[5] # Monaco
	    year_index = 2 
	    track_index = 5 
	    
	with col1:
		selected_year = st.selectbox("Year:", YEARS, index=YEARS.index(latest_year), key="select_year")
	with col2:
		selected_event = st.selectbox("Track:", TRACKS, index=TRACKS.index(latest_track), key="select_track")

	# The session dropdown is removed, as requested, for full automation.

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
			status_msg = f"Error: Failed to find valid data for any session ({'/'.join(SESSIONS_PRIORITY)}) for this event. Try selecting a different year or track."
			st.error(f"❌ {status_msg}")
			return

		# Start Analysis
		st.subheader(f"🔄 Starting Analysis: {selected_event} {selected_year} ({selected_session})")

		status_placeholder = st.empty()
		status_placeholder.info("...Loading and processing data from FastF1...")

		# Load and process data (using the successful context_data loaded in the loop)
		
		status_placeholder.success(f"✅ Data processed successfully for {selected_session}. Sending to AI for analysis...")

		# Create prompt and get prediction
		try:
			prompt = create_prediction_prompt(context_data, selected_year, selected_event, selected_session)

			prediction_report = get_gemini_prediction(prompt)

			status_placeholder.success("🏆 Analysis completed successfully!")
			st.markdown("---")

			# Display Report
			st.markdown(prediction_report)

		except APIError as e:
			status_placeholder.error(f"❌ Gemini API Error: Failed to receive response. Details: {e}")
		except ValueError as e: # Catch API Key errors from get_gemini_prediction
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
