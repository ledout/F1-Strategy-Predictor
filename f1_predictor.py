import streamlit as st
import fastf1
import pandas as pd
import logging
import re
from google import genai
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential

# --- הגדרות ראשוניות ---
# הסרת הגדרות Cache כדי למנוע שגיאות ב-Streamlit Cloud.
pd.options.mode.chained_assignment = None
logging.getLogger('fastf1').setLevel(logging.ERROR)

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

def load_and_process_data(year, event, session_key):
    """טוען נתונים מ-FastF1 ומבצע עיבוד ראשוני."""
    try:
        session = fastf1.get_session(year, event, session_key)
        # FastF1 יוריד נתונים בכל פעם מכיוון שה-Cache כבוי, וזה תקין ב-Streamlit Cloud.
        session.load_laps(with_telemetry=False) 
    except Exception as e:
        # זה יכול לקרות אם אין נתונים זמינים (למשל, מרוץ עתידי מדי)
        return None, f"שגיאת FastF1 בטעינה: לא נמצאו נתונים עבור {year} {event} {session_key}. פרטי שגיאה: {e}"

    laps = session.laps.reset_index(drop=True)
    
    # סינון הקפות נדרש (V33)
    laps_filtered = laps.loc[
        (laps['IsAccurate'] == True) & 
        (laps['LapTime'].notna()) & 
        (laps['Driver'] != 'OUT') & 
        (laps['Team'].notna()) &
        (laps['Time'].notna()) &
        (laps['Sector1SessionTime'].notna())
    ].copy()

    # 3. חישוב נתונים סטטיסטיים
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
        return None, "לא נמצאו נתונים מספקים (פחות מ-5 הקפות לנהג) לניתוח סטטיסטי."

    # עיבוד נתונים לפורמט טקסט (Top 10)
    data_lines = []
    driver_stats = driver_stats.sort_values(by='Avg_Time_s', ascending=True).head(10)
    
    for index, row in driver_stats.iterrows():
        # טיפול בפורמט datetime של LapTime
        best_time_str = str(row['Best_Time']).split('0 days ')[-1][:10] if row['Best_Time'] is not pd.NaT else 'N/A'
        avg_time_str = str(row['Avg_Time']).split('0 days ')[-1][:10] if row['Avg_Time'] is not pd.NaT else 'N/A'
        
        data_lines.append(
            f"DRIVER: {row['Driver']} | Best: {best_time_str} | Avg: {avg_time_str} | Var: {row['Var']:.3f} | Laps: {int(row['Laps'])}"
        )

    # יצירת טקסט קונטקסט ל-Gemini
    context_data = "\n".join(data_lines)

    return context_data, session.name

def create_prediction_prompt(context_data, year, event, session_name):
    """בניית הפרומפט המלא למודל Gemini."""
    
    prompt_data = f"--- נתונים גולמיים לניתוח (Top 10 Drivers, Race/Session Laps) ---\n{context_data}"

    # 2. בניית הפרומפט המלא (בהתבסס על V33)
    prompt = f"""
    אתה אנליסט אסטרטגיה בכיר של פורמולה 1. משימתך היא לנתח את הנתונים הסטטיסטיים של הקפות המרוץ ({session_name}, {event} {year}) ולספק דוח אסטרטגי מלא ותחזית מנצח.
    
    {prompt_data}

    --- הנחיות לניתוח (V33 - ניתוח משולב R/Q/S וקונטקסט) ---
    1. **Immediate Prediction (Executive Summary):** בחר מנצח אחד והצג את הנימוק העיקרי (קצב ממוצע או קונסיסטנטיות) בשורה אחת, **באנגלית בלבד**. (חובה)
    2. **Overall Performance Summary:** נתח את הקצב הממוצע (Avg Time) והעקביות (Var). Var < 1.0 נחשב לעקביות מעולה. Var > 5.0 עשוי להצביע על חוסר קונסיסטנטיות או הפרעות במרוץ (כגון תאונה או דגל אדום).
    3. **Tire and Strategy Deep Dive:** נתח את הנתונים ביחס למסלול (למשל, מקסיקו=גובה רב, מונזה=מהירות גבוהה). הסבר איזה סוג הגדרה (High Downforce/Low Downforce) משתקף בנתונים, בהנחה שנתון ה-Max Speed של הנהגים המובילים זמין בניתוח שלך.
    4. **Weather/Track Influence:** הוסף קונטקסט כללי על תנאי המסלול והשפעתם על הצמיגים. הנח תנאים יציבים וחמים אלא אם כן ה-Var הגבוה מעיד על שימוש בצמיגי גשם/אינטר.
    5. **Strategic Conclusions and Winner Justification:** הצג סיכום והצדקה ברורה לבחירת המנצח על בסיס נתונים ושיקולים אסטרטגיים.
    6. **Confidence Score Table (D5):** ספק טבלת Confidence Score (בפורמט Markdown) המכילה את 5 המועמדים המובילים עם אחוז ביטחון (סך כל האחוזים חייב להיות 100%). **תקן את פורמט הטבלה כך שיופיע תקין ב-Markdown**.
       
    --- פורמט פלט חובה (Markdown, עברית למעט הכותרת הראשית) ---
    🏎️ Strategy Report: {event} {year}
    
    Based on: Specific Session Data ({session_name} Combined)
    
    Immediate Prediction (Executive Summary)
    ...
    
    Overall Performance Summary
    ...
    
    Tire and Strategy Deep Dive
    ...

    Weather/Track Influence
    ...
    
    Strategic Conclusions and Winner Justification
    ...

    📊 Confidence Score Table (D5 - Visual Data)
    | Driver | Confidence Score (%) |
    |:--- | :--- |
    ...
    
    """
    return prompt

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def get_gemini_prediction(prompt):
    """שולח את הפרומפט ל-Gemini Flash ומשתמש במפתח מה-Secrets."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        raise ValueError("GEMINI_API_KEY לא נמצא ב-Streamlit Secrets. אנא הגדר אותו.")
        
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text

# --- פונקציה ראשית של Streamlit ---

def main():
    """פונקציה ראשית המריצה את האפליקציה ב-Streamlit."""
    st.set_page_config(page_title="F1 Strategy Predictor V33", layout="centered")

    st.title("🏎️ F1 Strategy Predictor V33")
    st.markdown("---")
    st.markdown("כלי לניתוח אסטרטגיה
