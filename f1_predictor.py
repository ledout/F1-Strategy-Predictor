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

@st.cache_data(ttl=3600, show_spinner="טוען נתוני F1 (מכבה FastF1 Cache מקומי)...")
def load_and_process_data(year, event, session_key):
    """טוען נתונים מ-FastF1 ומבצע עיבוד ראשוני, עם Caching של Streamlit."""
    
    try:
        session = fastf1.get_session(year, event, session_key)
        session.load(telemetry=False, weather=False, allow_n_attempt=5) 
        
        if session.laps is None or session.laps.empty:
            return None, f"נתונים חסרים עבור {year} {event} {session_key}. ייתכן שמדובר באירוע מבוטל או שטרם התקיים. שגיאה: FastF1 'load_laps' error."
            
    except Exception as e:
        error_message = str(e)
        
        if "Failed to load any schedule data" in error_message or "schedule data" in error_message:
             return None, f"FastF1: Failed to load any schedule data. שגיאה בטעינת FastF1: ייתכן שיש בעיית רשת/חיבור או שהשנה/מסלול לא קיימים."
        
        if "not found" in error_message or "The data you are trying to access has not been loaded yet" in error_message:
             return None, f"נתונים חסרים עבור {year} {event} {session_key}. ייתכן שמדובר באירוע מבוטל או שטרם התקיים. שגיאה: {error_message.split(':', 1)[-1].strip()}"

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

    laps_filtered['LapTime_s'] = laps_filtered['LapTime'].dt.total_seconds()
    
    # 5. חישוב נתונים סטטיסטיים
    driver_stats = laps_filtered.groupby('Driver').agg(
        Best_Time=('LapTime', 'min'),
        Avg_Time=('LapTime', 'mean'),
        Var=('LapTime_s', 'var'), 
        Laps=('LapTime', 'count')
    ).reset_index()

    driver_stats['Best_Time_s'] = driver_stats['Best_Time'].dt.total_seconds()
    driver_stats['Avg_Time_s'] = driver_stats['Avg_Time'].dt.total_seconds()
    
    driver_stats = driver_stats[driver_stats['Laps'] >= 5]
    
    if driver_stats.empty:
        return None, "לא נמצאו נתונים מספקים (פחות מ-5 הקפות לנהג) לניתוח סטטיסטי. נסה סשן אחר."

    # עיבוד נתונים לפורמט טקסט (Top 10)
    data_lines = []
    driver_stats = driver_stats.sort_values(by='Avg_Time_s', ascending=True).head(10)
    
    for index, row in driver_stats.iterrows():
        best_time_str = str(row['Best_Time']).split('0 days ')[-1][:10] if row['Best_Time'] is not pd.NaT else 'N/A'
        avg_time_str = str(row['Avg_Time']).split('0 days ')[-1][:10] if row['Best_Time'] is not pd.NaT else 'N/A'
        
        # **תיקון: ודא שכל הגרשיים וסוגרי ה-f-string נסגרים כראוי**
        data_lines.append(
            f"DRIVER: {row['Driver']} | Best: {best_time_str} | Avg: {avg_time_str} | Var: {row['Var']:.3f} | Laps: {int(row['Laps'])}"
        )

    context_data = "\n".join(data_lines)

    return context_data, session.name

def create_prediction_prompt(context_data, year, event, session_name):
    """בניית הפרומפט המלא למודל Gemini עבור נתונים עכשוויים."""
    
    prompt_data = f"--- נתונים גולמיים לניתוח (Top 10 Drivers, Race/Session Laps) ---\n{context_data}"

    # **תיקון שגיאות 'unterminated string literal' בשורות 125/128**
    # לוודא שאין תווי escape חסרים או גרשיים מיותרים בתוך מחרוזת ה-f-string המשולשת
    prompt = f"""
אתה אנליסט אסטרטגיה בכיר של פורמולה 1. משימתך היא לנתח את הנתונים הסטטיסטיים של הקפות המרוץ 
({session_name}, {event} {year}) ולספק דוח אסטרטגי מלא ותחזית מנצח.

{prompt_data}

--- הנחיות לניתוח (V33 - ניתוח משולב R/Q/S וקונטקסט) ---
1. **Immediate Prediction (Executive Summary):** בחר מנצח אחד והצג את הנימוק העיקרי (קצב ממוצע או קונסיסטנטיות) בשורה אחת, **באנגלית בלבד**. (חובה)
2. **Overall Performance Summary:** נתח את הקצב הממוצע (Avg Time) והעקביות (Var). Var < 1.0 נחשב לעקביות מעולה. Var > 5.0 עשוי להצביע על חוסר קונסיסטנטיות או הפרעות במרוץ (כגון תאונה או דגל אדום).
3. **Tire and Strategy Deep Dive:** נתח את הנתונים ביחס למסלול. הסבר איזה סוג הגדרה ('High Downforce'/'Low Downforce') משתקף בנתונים, בהנחה שנתון ה-Max Speed של הנהגים המובילים זמין בניתוח שלך.
4. **Weather/Track Influence:** הוסף קונטקסט כללי על תנאי המסלול והשפעתם על הצמיגים. הנח תנאים יציבים וחמים אלא אם כן ה-Var הגבוה מעיד על שימוש בצמיגי גשם/אינטר. 
5. **Strategic Conclusions and Winner Justification:** הצג סיכום והצדקה ברורה לבחירת המנצח על בסיס נתונים ושיקולים אסטרטגיים.
6. **Confidence Score Table (D5):** ספק טבלת Confidence Score (בפורמט Markdown) המכילה את 5 המועמדים המובילים עם אחוז ביטחון (סך כל האחוזים חייב להיות 100%). **תקן את פורמט הטבלה כך שיופיע תקין ב-Markdown**.

--- פורמט פלט חובה (Markdown, עברית למעט הכותרת הראשית) ---
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

# **תיקון קריטי: ודא שהסוגריים בדקורטור נסגרים באותה שורה כדי למנוע SyntaxError**
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def get_gemini_prediction(prompt):
    """שולח את הפרומפט ל-Gemini Flash ומשתמש במפתח מה-Secrets."""
    
    # **תיקון שגיאת 'expected :' (line 159) ושיפור הטיפול במפתח API**
    try:
        # שימוש ב-get() בטוח יותר
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
             raise ValueError("GEMINI_API_KEY לא נמצא ב-Streamlit Secrets. אנא הגדר אותו.")
    except Exception as e:
        # מעביר את השגיאה הלאה אם המפתח לא נמצא או אם יש שגיאת סביבה
        raise ValueError(f"שגיאת API Key: {e}")
        
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text

# --- פונקציות לתחזית מוקדמת (Pre-Race) ---

@st.cache_data(ttl=3600, show_spinner="טוען לוח זמנים F1...")
def find_last_three_races_data(current_year, event):
    """מוצא את שלושת המרוצים האחרונים שהתקיימו העונה ומחזיר את נתוני המרוץ שלהם."""
    
    try:
        schedule = fastf1.get_event_schedule(current_year)
    except Exception:
        return [], "שגיאה: לא ניתן לטעון את לוח הזמנים של השנה הנוכחית."
    
    try:
        # מנסה למצוא את האינדקס של האירוע הנוכחי
        event_index = schedule[schedule['EventName'] == event].index[0]
    except IndexError:
        # אם האירוע לא נמצא (למשל, עדיין לא נוסף ללוח הזמנים של FastF1)
        event_index = len(schedule) 
    
    # --- טיפול בשגיאת KeyError: 'EventCompleted' ---
    try:
        if 'EventCompleted' not in schedule.columns or 'EventFormat' not in schedule.columns:
            st.warning(f"⚠️ אזהרה: לוח הזמנים של {current_year} אינו מכיל נתוני השלמה מרוץ ('EventCompleted'). לא ניתן לטעון קונטקסט עונתי.")
            return [], f"אין נתוני סיום מרוץ זמינים עבור {current_year}."

        # 3. מוצא את 3 המרוצים ה'רגילים' האחרונים שהסתיימו לפני המרוץ הנוכחי
        completed_races = schedule.loc[
            (schedule.index < event_index) & 
            (schedule['EventFormat'] == 'conventional') &
            (schedule['EventCompleted'] == True)
        ].sort_index(ascending=False).head(3) 

    except KeyError as e:
        # לכידת שגיאת KeyError ספציפית הנובעת מעמודה חסרה
        st.error(f"❌ שגיאת FastF1: עמודה חסרה ({e}). לא ניתן לבצע ניתוח עונתי. אנא בחר שנה שבה הנתונים מלאים יותר.")
        return [], f"FastF1: עמודה חסרה ({e}). לא ניתן לבצע ניתוח עונתי."
    
    
    if completed_races.empty:
        return [], f"אין מרוצים מלאים שהתקיימו טרם מרוץ {event} {current_year} לצורך השוואה עונתית."
    
    race_reports = []
    
    for _, race in completed_races.iterrows():
        event_name = race['EventName']
        st.info(f"🔮 מנתח קונטקסט עונתי: טוען נתוני מרוץ {event_name} {current_year}...")
        
        context_data, session_name = load_and_process_data(current_year, event_name, 'R')
        
        if context_data:
            report = (
                f"--- דוח קצב: מרוץ {event_name} {current_year} (מרוץ עונתי) ---\n"
                f"{context_data}\n"
            )
            race_reports.append(report)
        else:
            st.warning(f"⚠️ לא ניתן היה לטעון נתוני מרוץ מלאים עבור {event_name}.")

    return race_reports, "נתונים עונתיים נטענו"


def get_preliminary_prediction(current_year, event):
    """משלב נתוני מרוץ מהשנה הקודמת ומשלושת המרוצים האחרונים העונה ליצירת תחזית מוקדמת חזקה יותר."""
    
    previous_year = current_year - 1
    
    st.subheader("🏁 איסוף נתונים לתחזית מוקדמת (Pre-Race Analysis)")
    st.info(f"🔮 מנתח דומיננטיות במסלול: טוען נתוני מרוץ {event} משנה {previous_year}...")
    context_data_prev, session_name_prev = load_and_process_data(previous_year, event, 'R')

    race_reports_current, status_msg = find_last_three_races_data(current_year, event)
