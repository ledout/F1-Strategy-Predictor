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
# Suppress pandas warning about chained assignment
pd.options.mode.chained_assignment = None
# Suppress FastF1 logging errors
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
SESSIONS = ["FP1", "FP2", "FP3", "Q", "S", "R"]
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
		(laps['IsAccurate'] == True) &
		(laps['LapTime'].notna()) &
		(laps['Driver'] != 'OUT') &
		(laps['Team'].notna()) &
		(laps['Time'].notna()) &
		(laps['Sector1SessionTime'].notna())
	].copy()

	laps_filtered['LapTime_s'] = laps_filtered['LapTime'].dt.total_seconds()

	# Calculate driver statistics (Best Time, Avg Time, Variance, Laps Count)
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
		return None, "Insufficient data (fewer than 5 laps per driver) for statistical analysis. Try a different session."

	# Process data to text format (Top 10)
	data_lines = []
	driver_stats = driver_stats.sort_values(by='Avg_Time_s', ascending=True).head(10)

	for index, row in driver_stats.iterrows():
		# Format time strings
		best_time_str = str(row['Best_Time']).split('0 days ')[-1][:10] if row['Best_Time'] is not pd.NaT else 'N/A'
		avg_time_str = str(row['Avg_Time']).split('0 days ')[-1][:10] if row['Best_Time'] is not pd.NaT else 'N/A'

		data_lines.append(
			f"DRIVER: {row['Driver']} | Best: {best_time_str} | Avg: {avg_time_str} | Var: {row['Var']:.3f} | Laps: {int(row['Laps'])}"
		)

	context_data = "\n".join(data_lines)

	return context_data, session.name

# --- Functions for Preliminary Prediction (Pre-Race) ---

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
You are a Senior Formula 1 Strategy Analyst. Your task is to analyze the statistical data of the laps
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

## Overall Performance Summary
...

## Tire and Strategy Deep Dive
...

## Weather/Track Influence
...

## Strategic Conclusions and Winner Justification
...

## 📊 Confidence Score Table (D5 - Visual Data)
| Driver | Confidence Score (%) |
|:--- | :--- |
| ... | ... |
| ... | ... |
| ... | ... |
| ... | ... |
| ... | ... |
"""
	return prompt


def get_preliminary_prediction(current_year, event):
	"""Combines race data from the previous year and the last three races of the current season to create a stronger pre-race prediction."""

	previous_year = current_year - 1

	# Translate Subheader
	st.subheader("🏁 Data Collection for Preliminary Prediction (Pre-Race Analysis)")

	# Create the closed expander for all technical reports
	# Translate Expander Title
	with st.expander("🛠️ Show Historical and Seasonal Data Loading Details (Diagnostics)", expanded=False):
		expander_placeholder = st.container() # Placeholder to pass inside functions

		with expander_placeholder:
			# Translate Info Message
			st.info(f"🔮 Analyzing track dominance: Loading race data for {event} from {previous_year}...")

			# 1. Load Historical Data (Previous Year on the Same Track)
			context_data_prev, session_name_prev = load_and_process_data(previous_year, event, 'R')
			if context_data_prev:
				# Translate Success Message
				st.success(f"✅ Race data for {event} {previous_year} loaded successfully.")
			else:
				# Translate Warning Message
				st.warning(f"⚠️ Warning: No complete historical data found for {event} {previous_year}. ({session_name_prev})")

			st.markdown("---")

		# 2. Load Seasonal Data (Last 3 Completed Races)
		race_reports_current, status_msg = find_last_three_races_data(current_year, event, expander_placeholder)

		# Display FastF1 status only if a critical failure/warning was returned
		if "❌" in status_msg or "Error" in status_msg or "No" in status_msg:
			st.error(status_msg)
		elif "No complete seasonal data found" in status_msg or "Seasonal skip" in status_msg:
			st.warning(status_msg)

	# 3. Data Check and Report Unification (Outside the Expander)

	based_on_text = ""
	report_current = f"--- Seasonal Pace Report (No Seasonal Data Available) ---\n"

	if context_data_prev:
		# Translate Report Header and Description
		report_prev = (
			f"--- Pace Report: {event} Race {previous_year} (Historical Track Context) ---\n"
			f"The report describes the drivers' performance on the specific track {event} in the previous year. Compare Average Pace and Var:\n"
			f"{context_data_prev}\n"
		)
		based_on_text += f"{event} {previous_year} Race Data"
	else:
		# Translate Report Header
		report_prev = f"--- Pace Report: {event} Race {previous_year} (No Historical Track Data Available) ---\n"

	# Ensure race_reports_current is a non-empty list
	if race_reports_current and isinstance(race_reports_current, list):
		report_current = "\n" + "\n".join(race_reports_current)
		num_races = len(race_reports_current)

		if based_on_text:
			based_on_text += " & "
		based_on_text += f"Analysis of the Last {num_races} Races of {current_year}."
	else:
		# If no seasonal data
		if not based_on_text:
			based_on_text = f"No Current Season Context or Historical Data Available."
		else:
			based_on_text += " Only (No Current Season Context)."


	# If there is absolutely no data (neither historical nor seasonal), stop
	if not context_data_prev and not race_reports_current:
		# Translate Error Message
		st.error("❌ No historical or seasonal data available. Cannot perform analysis.")
		return None


	# 4. Build the prompt combining all reports

	full_data_prompt = report_prev + report_current

	prompt = f"""
You are a Senior F1 Analyst. Analyze the following combined data to provide a Preliminary (Pre-Race) Prediction Report for **{event} {current_year} Race**.

{full_data_prompt}

--- Analysis Guidelines (V47 - Weight 65/35, Implicit Weather) ---
1. **Immediate Prediction (Executive Summary):** Select one winner and present the main justification (average pace, consistency, or seasonal trend) in a single sentence, **in English only**. (Mandatory)
2. **Past Performance Analysis:** Analyze the historical report (previous year on this track). Explain who was dominant in terms of pace and consistency on this track.
3. **Current Season Trend Analysis:** Analyze the seasonal race reports. **Provide a brief summary of the trend in the balance of power between the leading teams (Red Bull, Ferrari, Mercedes) in the last 3 races.** Who is improving and who is declining?
4. **Strategic Conclusions and Winner Justification:** Justify the winner choice based on a combination of **current seasonal capability (65% weight)** and **previous track dominance (35% weight)**. The analysis must reflect this bias.
5. **Weather & Tire Degradation (Implicit):** Analyze the data and provide a recommended **tire strategy** for the upcoming race (e.g., Hard-Medium-Hard) and an **Pit-Stop Window** analysis. **Assume dry and normal weather conditions,** unless high Var data clearly indicates rain/wet conditions (then state this explicitly).
6. **Confidence Score Table (D5):** Provide a Confidence Score table (in Markdown format) containing the top 5 candidates with a confidence percentage (total percentage must be 100%). **Ensure the table format appears correctly in Markdown**.

--- Mandatory Output Format (Markdown, English for the entire report) ---
🔮 Pre-Race Strategy Report: {event} {current_year}

Based on: {based_on_text}

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
| ... | :--- |
| ... | :--- |
| ... | :--- |
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
	"""Main function that runs the Streamlit application."""

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

	# Translate Description & Center Text
	st.markdown("<div style='text-align: center; font-size: 3.5em; font-weight: bold;'>Who's on Pole?</div>", unsafe_allow_html=True)
	st.markdown("---")

	# API Key Check
	try:
		api_key_check = st.secrets.get("GEMINI_API_KEY")
		if not api_key_check:
			# Translate Error Message
			st.error("❌ Error: Gemini API Key is not configured in Streamlit Secrets. Please set it.")
		if not api_key_check:
			# Translate Warning Message
			st.warning("⚠️ Note: API Key is missing. The analysis will fail when attempting to connect to Gemini.")

	except Exception:
		# Translate Error Message
		st.error("❌ Error: Failed to read API key. Ensure you have configured it correctly in Secrets.")

	st.markdown("---")

	# Parameter Selection
	col1, col2, col3 = st.columns(3)

	with col1:
		# Translate Label
		selected_year = st.selectbox("Year:", YEARS, index=2, key="select_year")
	with col2:
		# Translate Label
		selected_event = st.selectbox("Track:", TRACKS, index=5, key="select_event")
	with col3:
		# Translate Label
		selected_session = st.selectbox("Session:", SESSIONS, index=5, key="select_session")

	st.markdown("---")

	# 1. Current Data Analysis Button
	# Translate Button Text
	if st.button("🏎️ Predict Winner (Current Session Data)", use_container_width=True, type="primary"):

		# Translate Subheader
		st.subheader(f"🔄 Starting Analysis: {selected_event} {selected_year} ({selected_session})")

		status_placeholder = st.empty()
		# Translate Info Message
		status_placeholder.info("...Loading and processing data from FastF1...")

		# Load and process data
		context_data, status_msg = load_and_process_data(selected_year, selected_event, selected_session)

		if context_data is None:
			# Translate Error Message
			status_placeholder.error(f"❌ Error: {status_msg}")
			return

		# Translate Success Message
		status_placeholder.success("✅ Data processed successfully. Sending to AI for analysis...")

		# Create prompt and get prediction
		try:
			prompt = create_prediction_prompt(context_data, selected_year, selected_event, selected_session)

			prediction_report = get_gemini_prediction(prompt)

			# Translate Success Message
			status_placeholder.success("🏆 Analysis completed successfully!")
			st.markdown("---")

			# Display Report
			st.markdown(prediction_report)

		except APIError as e:
			# Translate Error Message
			status_placeholder.error(f"❌ Gemini API Error: Failed to receive response. Details: {e}")
		except ValueError as e: # Catch API Key errors from get_gemini_prediction
			# Translate Error Message
			status_placeholder.error(f"❌ Critical Error: {e}")
		except Exception as e:
			# Translate Error Message
			status_placeholder.error(f"❌ Unexpected Error: {e}")

	st.markdown("---")

	# 2. Pre-Race Prediction Button
	# Translate Button Text
	if st.button("🔮 Preliminary Prediction (Past & Seasonal Context)", use_container_width=True, type="secondary"):
		# Translate Subheader
		st.subheader(f"🔮 Starting Preliminary Prediction: {selected_event} {selected_year}")

		prelim_report = get_preliminary_prediction(selected_year, selected_event)

		if prelim_report:
			st.markdown("---")
			st.markdown(prelim_report)


if __name__ == "__main__":
	main()
