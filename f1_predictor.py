import streamlit as st
import fastf1
import pandas as pd
import logging
from google import genai
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential

# --- הגדרות ראשוניות ---
pd.options.mode.chained_assignment = None
logging.getLogger('fastf1').setLevel(logging.ERROR)

# **כיבוי מוחלט של FastF1 Cache מקומי (פתרון לבעיות רשת/סביבה ב-Streamlit Cloud)**
try:
    # מונע כשלים הקשורים לקאשינג בסביבת Streamlit Cloud
    fastf1.set_cache_path(None)
except Exception:
    pass

# --- קבועים ---
TRACKS = ["Bahrain", "Saudi Arabia", "Australia", "Imola", "Miami", "Monaco", 
          "Spain", "Canada", "Austria", "Great Britain", "Hungary", "Belgium", 
          "Netherlands", "Monza", "Singapore", "Japan", "Qatar", "United States", 
          "Mexico", "Brazil", "Las Vegas", "Abu Dhabi", "China", "Turkey", 
          "France"]
SESSIONS = ["FP1", "FP2", "FP3", "Q", "S", "R"]
YEARS = [2025, 2024, 2023, 2022, 2021, 2020]
MODEL_NAME = "gemini-2.5-flash"


# --- פונקציות עזר לטיפול בנתונים ---

# שימוש ב-st.cache_data כדי לשפר יציבות בטעינת נתונים
@st.cache_data(ttl=3600, show_spinner="טוען נתוני F1 (מכבה FastF1 Cache מקומי)...")
def load_and_process_data(year, event, session_key):
    """טוען נתונים מ-FastF1 ומבצע עיבוד ראשוני, עם Caching של Streamlit."""
    
    try:
        # **תיקון קריטי: נסה קודם לאמת את קיום האירוע**
        event_data = fastf1.get_event(year, event)
        if event_data is None:
             return None, f"שגיאה: האירוע '{event}' לשנת {year} לא נמצא בלוח הזמנים של FastF1."

        # 1. טוען את הסשן (רק אם האירוע קיים)
        session = fastf1.get_session(year, event, session_key)
        
        # 2. ניסיון טעינת ההקפות
        session.load_laps(with_telemetry=False)
        
        # 3. בדיקה: אם אין הקפות, זה כנראה אירוע חסר נתונים
        if session.laps is None or session.laps.empty:
            return None, f"שגיאה: האירוע {year} {event} {session_key} טרם התקיים, או שלא נמצאו נתונים תקינים עבורו."
            
    except Exception as e:
        # טיפול בשגיאות FastF1 נפוצות וכלליות
        error_message = str(e)
        
        if "load_laps" in error_message or "schedule data" in error_message or "not found" in error_message:
             return None, f"שגיאה בטעינת FastF1: נתונים חסרים עבור {year} {event} {session_key}. ייתכן שמדובר בבעיית רשת/חיבור."
        
        if "'Session' object has no attribute 'load_laps'" in error_message:
            return None, f"שגיאה: האירוע {year} {event} {session_key} לא מכיל נתוני הקפות (FastF1 'load_laps' error). נסה סשן אחר."

        # ודא שכל שגיאה אחרת חוזרת כהודעה כללית
        return None, f"שגיאת FastF1 כללית בטעינה: {error_message}"

    laps = session.laps.reset_index(drop=True)
    
    # סינון הקפות נדרש
    laps_filtered = laps.loc[
        (laps['IsAccurate'] == True) & 
        (laps['LapTime'].notna()) & 
        (laps['Driver'] != 'OUT') & 
        (laps['Team'].notna()) &
        (laps['Time'].notna()) &
        (laps['Sector1SessionTime'].notna())
    ].copy()

    # 4. חישוב נתונים סטטיסטיים
    driver_stats = laps_filtered.groupby('Driver').agg(
        Best_Time=('LapTime', 'min'),
        Avg_Time=('LapTime', 'mean'),
        Var=('LapTime', 'var'),
        Laps=('LapTime', 'count')
    ).reset_index()

    # המרת זמנים לשניות לצורך חישובים
    driver_stats['Best_Time_s'] = driver_stats['Best_Time'].dt.total_seconds()
    driver_stats['Avg_Time_s'] = driver_stats['Avg_Time'].dt.total_seconds()
    
    # סינון נהגים עם פחות מ-5 הקפות לניתוח סטטיסטי
    driver_stats = driver_stats[driver_stats['Laps'] >= 5]
    
    if driver_stats.empty:
        return None, "לא נמצאו נתונים מספקים (פחות מ-5 הקפות לנהג) לניתוח סטטיסטי. נסה סשן אחר."

    # עיבוד נתונים לפורמט טקסט (Top 10)
    data_lines = []
    driver_stats = driver_stats.sort_values(by='Avg_Time_s', ascending=True).head(10)
    
    for index, row in driver_stats.iterrows():
        # טיפול בפורמט datetime של LapTime
        best_time_str = str(row['Best_Time']).split('0 days ')[-1][:10] if row['Best_Time'] is not pd.NaT else 'N/A'
        avg_time_str = str(row['Avg_Time']).split('0 days ')[-1][:10] if row['Best_Time'] is not pd.NaT else 'N/A'
        
        data_lines.append(
            f"DRIVER: {row['Driver']} | Best: {best_time_str} | Avg: {avg_time_str} | Var: {row['Var']:.3f} | Laps: {int(row
