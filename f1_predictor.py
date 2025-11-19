import streamlit as st
import fastf1
import pandas as pd
import logging
import re
from google import genai
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential

# --- הגדרות ראשוניות ---
pd.options.mode.chained_assignment = None
logging.getLogger('fastf1').setLevel(logging.ERROR)
CACHE_DIR = './fastf1_cache_dir'
try:
    fastf1.Cache.enable_cache(CACHE_DIR)
except fastf1.api.NotADirectoryError:
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

def load_and_process_data(year, event, session_key):
    """טוען נתונים מ-FastF1 ומבצע עיבוד ראשוני."""
    try:
        session = fastf1.get_session(year, event, session_key)
        session.load_laps(with_telemetry=False)
    except Exception as e:
        return None, f"שגיאת FastF1 בטעינה: לא נמצאו נתונים עבור {year} {event} {session_key}."

    laps = session.laps.reset_index(drop=True)
    laps_filtered = laps.loc[
        (laps['IsAccurate'] == True) & 
        (laps['LapTime'].notna()) & 
        (laps['Driver'] != 'OUT') & 
        (laps['Team'].notna()) &
        (laps['Time'].notna()) &
        (laps['Sector1SessionTime'].notna())
    ].copy()

    driver_stats = laps_filtered.groupby('Driver').agg(
        Best_Time=('LapTime', 'min'),
        Avg_Time=('LapTime', 'mean'),
        Var=('LapTime', 'var'),
        Laps=('LapTime', 'count')
    ).reset_index()

    driver_stats['Best_Time_s'] = driver_stats['Best_Time'].dt.total_seconds()
    driver_stats['Avg_Time_s'] = driver_stats['Avg_Time'].dt.total_seconds()
    
    driver_stats = driver_stats[driver_stats['Laps'] >= 5]
    
    if driver_stats.empty:
        return None, "לא נמצאו נתונים מספקים (פחות מ-5 הקפות לנהג) לניתוח סטטיסטי."

    data_lines = []
    driver_stats = driver_stats.sort_values(by='Avg_Time_s', ascending=True).head(10)
    
    for index, row in driver_stats.iterrows():
        best_time_str = str(row['Best_Time']).split('0 days ')[-1][:10]
        avg_time_str = str(row['Avg_Time']).split('0 days ')[-1][:10]
        
        data_lines.append(
            f"DRIVER: {row['Driver']} | Best: {best_time_str} | Avg: {avg_time_str} | Var: {row['Var']:.3f} | Laps: {int(row['Laps'])}"
        )

    context_data = "\n".join(data_lines)
    return context_data, session.name

def create_prediction_prompt(context_data, year, event, session_name):
    """בניית הפרומפט המלא למודל Gemini."""
    
    prompt_data = f"--- נתונים גולמיים לניתוח (Top 10 Drivers, Race/Session Laps) ---\n{context_data}"

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
    # **המפתח נקרא ישירות מתוך st.secrets בשרת הפריסה.**
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
    st.set_page_config(page_title="F1 Strategy Predictor V33", layout="centered")

    st.title("🏎️ F1 Strategy Predictor V33")
    st.markdown("---")
    st.markdown("כלי לניתוח אסטרטגיה וחיזוי מנצח מבוסס נתוני FastF1 ו-Gemini AI.")
    
    # בדיקת מפתח API (בשרת Streamlit)
    try:
        if "GEMINI_API_KEY" not in st.secrets or not st.secrets["GEMINI_API_KEY"]:
            st.error("❌ שגיאה: מפתח ה-API של Gemini לא הוגדר ב-Streamlit Secrets. אנא עקוב אחר שלב 3.3.")
            return

    except Exception:
        st.error("❌ שגיאה: כשל בקריאת מפתח API. ודא שהגדרת אותו כראוי.")
        return

    st.markdown("---")

    # בחירת פרמטרים
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_year = st.selectbox("שנה:", YEARS, index=2)
    with col2:
        selected_event = st.selectbox("מסלול:", TRACKS, index=18)
    with col3:
        selected_session = st.selectbox("סשן:", SESSIONS, index=5)
    
    st.markdown("---")
    
    # כפתור הפעלה
    if st.button("🏎️ חזה את המנצח (אוטומטי)", use_container_width=True, type="primary"):
        st.subheader(f"🔄 מתחיל ניתוח: {selected_event} {selected_year} ({selected_session})")
        
        status_placeholder = st.empty()
        status_placeholder.info("...טוען ומעבד נתונים מ-FastF1 (יכול לקחת דקה-שתיים בפעם הראשונה)")
        
        # 1. טעינת ועיבוד הנתונים
        context_data, session_name = load_and_process_data(selected_year, selected_event, selected_session)

        if context_data is None:
            status_placeholder.error(f"❌ שגיאה: {session_name}")
            return
        
        status_placeholder.success("✅ נתונים עובדו בהצלחה. שולח לניתוח AI...")

        # 2. יצירת הפרומפט וקבלת התחזית
        try:
            prompt = create_prediction_prompt(context_data, selected_year, selected_event, selected_session)
            
            prediction_report = get_gemini_prediction(prompt)

            status_placeholder.success("🏆 הניתוח הושלם בהצלחה!")
            st.markdown("---")
            
            # 3. הצגת הדו"ח
            st.markdown(prediction_report)

        except APIError as e:
            status_placeholder.error(f"❌ שגיאת Gemini API: לא הצליח לקבל תגובה. פרטי שגיאה: {e}")
        except Exception as e:
            status_placeholder.error(f"❌ שגיאה בלתי צפויה: {e}")


if __name__ == "__main__":
    main()
