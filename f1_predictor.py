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
        # 1. טוען את הסשן
        session = fastf1.get_session(year, event, session_key)
        
        # 2. ניסיון טעינת הנתונים (שימוש ב-load() לטיפול טוב יותר בנתונים חסרים/בעיות רשת)
        session.load(telemetry=False, weather=False) 
        
        # 3. בדיקה: אם אין הקפות, זה כנראה אירוע חסר נתונים
        if session.laps is None or session.laps.empty:
            return None, f"שגיאה: האירוע {year} {event} {session_key} טרם התקיים, או שלא נמצאו נתונים תקינים עבורו."
            
    except Exception as e:
        # טיפול בשגיאות FastF1 נפוצות וכלליות
        error_message = str(e)
        
        if "Failed to load any schedule data" in error_message or "schedule data" in error_message:
             return None, f"שגיאה בטעינת FastF1: ייתכן שיש בעיית רשת/חיבור או שהשנה/מסלול לא קיימים. פרטי שגיאה: {error_message}"
        
        if "not found" in error_message:
             return None, f"שגיאה: נתונים חסרים עבור {year} {event} {session_key}. ייתכן שמדובר באירוע מבוטל או שטרם התקיים."

        if "'Session' object has no attribute 'load'" in error_message:
            # במקרה נדיר שגרסת FastF1 ישנה, נחזיר שגיאה.
            return None, f"שגיאה: גרסת FastF1 אינה תואמת (load() attribute missing). אנא עדכן לגרסה 2.3.0 ומעלה."


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

    # 5. חישוב נתונים סטטיסטיים
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
        
        # בניית מחרוזת הנתונים (תוקן)
        data_lines.append(
            f"DRIVER: {row['Driver']} | Best: {best_time_str} | Avg: {avg_time_str} | Var: {row['Var']:.3f} | Laps: {int(row['Laps'])}"
        )

    # יצירת טקסט קונטקסט ל-Gemini
    context_data = "\n".join(data_lines)

    return context_data, session.name

def create_prediction_prompt(context_data, year, event, session_name):
    """בניית הפרומפט המלא למודל Gemini."""
    
    prompt_data = f"--- נתונים גולמיים לניתוח (Top 10 Drivers, Race/Session Laps) ---\n{context_data}"

    # 2. בניית הפרומפט המלא 
    prompt = (
        "אתה אנליסט אסטרטגיה בכיר של פורמולה 1. משימתך היא לנתח את הנתונים הסטטיסטיים של הקפות המרוץ "
        f"({session_name}, {event} {year}) ולספק דוח אסטרטגי מלא ותחזית מנצח.\n\n"
        # f-string תוקן 
        f"{prompt_data}\n\n" 
        "--- הנחיות לניתוח (V33 - ניתוח משולב R/Q/S וקונטקסט) ---\n"
        "1. **Immediate Prediction (Executive Summary):** בחר מנצח אחד והצג את הנימוק העיקרי (קצב ממוצע או קונסיסטנטיות) בשורה אחת, **באנגלית בלבד**. (חובה)\n"
        "2. **Overall Performance Summary:** נתח את הקצב הממוצע (Avg Time) והעקביות (Var). Var < 1.0 נחשב לעקביות מעולה. Var > 5.0 עשוי להצביע על חוסר קונסיסטנטיות או הפרעות במרוץ (כגון תאונה או דגל אדום).\n"
        # גרשיים פנימיים תוקנו
        "3. **Tire and Strategy Deep Dive:** נתח את הנתונים ביחס למסלול (למשל, מקסיקו=גובה רב, מונזה=מהירות גבוהה). הסבר איזה סוג הגדרה ('High Downforce'/'Low Downforce') משתקף בנתונים, בהנחה שנתון ה-Max Speed של הנהגים המובילים זמין בניתוח שלך.\n"
        "4. **Weather/Track Influence:** הוסף קונטקסט כללי על תנאי המסלול והשפעתם על הצמיגים. הנח תנאים יציבים וחמים אלא אם כן ה-Var הגבוה מעיד על שימוש בצמיגי גשם/אינטר.\n" 
        "5. **Strategic Conclusions and Winner Justification:** הצג סיכום והצדקה ברורה לבחירת המנצח על בסיס נתונים ושיקולים אסטרטגיים.\n"
        "6. **Confidence Score Table (D5):** ספק טבלת Confidence Score (בפורמט Markdown) המכילה את 5 המועמדים המובילים עם אחוז ביטחון (סך כל האחוזים חייב להיות 100%). **תקן את פורמט הטבלה כך שיופיע תקין ב-Markdown**.\n\n"
        
        "--- פורמט פלט חובה (Markdown, עברית למעט הכותרת הראשית) ---\n"
        f"🏎️ Strategy Report: {event} {year}\n\n"
        f"Based on: Specific Session Data ({session_name} Combined)\n\n"
        "Immediate Prediction (Executive Summary)\n"
        "...\n\n"
        "Overall Performance Summary\n"
        "...\n\n"
        "Tire and Strategy Deep Dive\n"
        "...\n\n"
        "Weather/Track Influence\n"
        "...\n\n"
        "Strategic Conclusions and Winner Justification\n"
        "...\n\n"
        "📊 Confidence Score Table (D5 - Visual Data)\n"
        "| Driver | Confidence Score (%) |\n"
        "|:--- | :--- |\n"
        "...\n"
    )
    return prompt

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def get_gemini_prediction(prompt):
    """שולח את הפרומפט ל-Gemini Flash ומשתמש במפתח מה-Secrets."""
    try: # <--- הנקודתיים (:) נוספו/וודאו כאן, זו הייתה השגיאה הקריטית האחרונה
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        # מעלה שגיאה ברורה אם המפתח לא נמצא ב-Streamlit Secrets
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
    st.markdown("כלי לניתוח אסטרטגיה וחיזוי מנצח מבוסס נתוני FastF1 ו-Gemini AI.")
    
    # בדיקת מפתח API (בשרת Streamlit)
    try:
        if "GEMINI_API_KEY" not in st.
