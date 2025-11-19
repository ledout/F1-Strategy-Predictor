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


# --- פונקציות עזר לטיפול בנתונים (ללא שינוי מהותי) ---

@st.cache_data(ttl=3600, show_spinner="טוען נתוני F1 (מכבה FastF1 Cache מקומי)...")
def load_and_process_data(year, event, session_key):
    """טוען נתונים מ-FastF1 ומבצע עיבוד ראשוני, עם Caching של Streamlit."""
    
    try:
        session = fastf1.get_session(year, event, session_key)
        session.load(telemetry=False, weather=False) 
        
        if session.laps is None or session.laps.empty:
            return None, f"נתונים חסרים עבור {year} {event} {session_key}. ייתכן שמדובר באירוע מבוטל או שטרם התקיים. שגיאה: FastF1 'load_laps' error."
            
    except Exception as e:
        error_message = str(e)
        
        if "Failed to load any schedule data" in error_message or "schedule data" in error_message:
             return None, f"FastF1: Failed to load any schedule data. שגיאה בטעינת FastF1: ייתכן שיש בעיית רשת/חיבור או שהשנה/מסלול לא קיימים."
        
        if "not found" in error_message:
             return None, f"נתונים חסרים עבור {year} {event} {session_key}. ייתכן שמדובר באירוע מבוטל או שטרם התקיים."

        return None, f"שגיאת FastF1 כללית בטעינה: {error_message}"

    laps = session.laps.reset_index(drop=True)
    
    laps_filtered = laps.loc[
        (laps['IsAccurate'] == True) & 
        (laps['LapTime'].notna()) & 
        (laps['Driver'] != 'OUT') & 
        (laps['Team'].notna()) &
        (laps['Time'].notna()) &
        (laps['Sector1SessionTime'].notna())
    ].copy()

    laps_filtered['LapTime_s'] = laps_filtered['LapTime'].dt.total_seconds()
    
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

    data_lines = []
    driver_stats = driver_stats.sort_values(by='Avg_Time_s', ascending=True).head(10)
    
    for index, row in driver_stats.iterrows():
        best_time_str = str(row['Best_Time']).split('0 days ')[-1][:10] if row['Best_Time'] is not pd.NaT else 'N/A'
        avg_time_str = str(row['Avg_Time']).split('0 days ')[-1][:10] if row['Best_Time'] is not pd.NaT else 'N/A'
        
        data_lines.append(
            f"DRIVER: {row['Driver']} | Best: {best_time_str} | Avg: {avg_time_str} | Var: {row['Var']:.3f} | Laps: {int(row['Laps'])}"
        )

    context_data = "\n".join(data_lines)

    return context_data, session.name

def create_prediction_prompt(context_data, year, event, session_name):
    """בניית הפרומפט המלא למודל Gemini עבור נתונים עכשוויים."""
    
    prompt_data = f"--- נתונים גולמיים לניתוח (Top 10 Drivers, Race/Session Laps) ---\n{context_data}"

    prompt = (
        "אתה אנליסט אסטרטגיה בכיר של פורמולה 1. משימתך היא לנתח את הנתונים הסטטיסטיים של הקפות המרוץ "
        f"({session_name}, {event} {year}) ולספק דוח אסטרטגי מלא ותחזית מנצח.\n\n"
        f"{prompt_data}\n\n" 
        "--- הנחיות לניתוח (V33 - ניתוח משולב R/Q/S וקונטקסט) ---\n"
        "1. **Immediate Prediction (Executive Summary):** בחר מנצח אחד והצג את הנימוק העיקרי (קצב ממוצע או קונסיסטנטיות) בשורה אחת, **באנגלית בלבד**. (חובה)\n"
        "2. **Overall Performance Summary:** נתח את הקצב הממוצע (Avg Time) והעקביות (Var). Var < 1.0 נחשב לעקביות מעולה. Var > 5.0 עשוי להצביע על חוסר קונסיסטנטיות או הפרעות במרוץ (כגון תאונה או דגל אדום).\n"
        "3. **Tire and Strategy Deep Dive:** נתח את הנתונים ביחס למסלול. הסבר איזה סוג הגדרה ('High Downforce'/'Low Downforce') משתקף בנתונים, בהנחה שנתון ה-Max Speed של הנהגים המובילים זמין בניתוח שלך.\n"
        "4. **Weather/Track Influence:** הוסף קונטקסט כללי על תנאי המסלול והשפעתם על הצמיגים. הנח תנאים יציבים וחמים אלא אם כן ה-Var הגבוה מעיד על שימוש בצמיגי גשם/אינטר.\n" 
        "5. **Strategic Conclusions and Winner Justification:** הצג סיכום והצדקה ברורה לבחירת המנצח על בסיס נתונים ושיקולים אסטרטגיים.\n"
        "6. **Confidence Score Table (D5):** ספק טבלת Confidence Score (בפורמט Markdown) המכילה את 5 המועמדים המובילים עם אחוז ביטחון (סך כל האחוזים חייב להיות 100%). **תקן את פורמט הטבלה כך שיופיע תקין ב-Markdown**.\n\n"
        
        "--- פורמט פלט חובה (Markdown, עברית למעט הכותרת הראשית) ---\n"
        f"🏎️ Strategy Report: {event} {year}\n\n"
        f"Based on: Specific Session Data ({session_name} Combined)\n\n"
        "## Immediate Prediction (Executive Summary)\n"
        "...\n\n"
        "## Overall Performance Summary\n"
        "...\n\n"
        "## Tire and Strategy Deep Dive\n"
        "...\n\n"
        "## Weather/Track Influence\n"
        "...\n\n"
        "## Strategic Conclusions and Winner Justification\n"
        "...\n\n"
        "## 📊 Confidence Score Table (D5 - Visual Data)\n"
        "| Driver | Confidence Score (%) |\n"
        "|:--- | :--- |\n"
        "| ... | ... |\n"
        "| ... | ... |\n"
        "| ... | ... |\n"
        "| ... | ... |\n"
        "| ... | ... |\n"
    )
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

# --- פונקציות לתחזית מוקדמת (Pre-Race) - הקוד המעודכן שאנו רוצים ---

@st.cache_data(ttl=3600, show_spinner="טוען לוח זמנים F1...")
def find_last_three_races_data(current_year, event):
    """מוצא את שלושת המרוצים האחרונים שהתקיימו העונה ומחזיר את נתוני המרוץ שלהם."""
    
    try:
        schedule = fastf1.get_event_schedule(current_year)
    except Exception:
        return [], "שגיאה: לא ניתן לטעון את לוח הזמנים של השנה הנוכחית."
    
    try:
        event_index = schedule[schedule['EventName'] == event].index[0]
    except IndexError:
        event_index = len(schedule) 
    
    # 3. מוצא את 3 המרוצים ה'רגילים' האחרונים שהסתיימו לפני המרוץ הנוכחי
    completed_races = schedule.loc[
        (schedule.index < event_index) & 
        (schedule['EventFormat'] == 'conventional') &
        (schedule['EventCompleted'] == True)
    ].sort_index(ascending=False).head(3) 
    
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
    
    # 1. טעינת נתוני מרוץ מהשנה הקודמת (קונטקסט מסלול)
    st.subheader("🏁 איסוף נתונים לתחזית מוקדמת (Pre-Race Analysis)")
    st.info(f"🔮 מנתח דומיננטיות במסלול: טוען נתוני מרוץ {event} משנה {previous_year}...")
    context_data_prev, session_name_prev = load_and_process_data(previous_year, event, 'R')

    # 2. טעינת נתונים משלושת המרוצים האחרונים (קונטקסט עונתי)
    race_reports_current, status_msg = find_last_three_races_data(current_year, event)

    # 3. בדיקת נתונים ואיחוד דוחות
    
    # דוח 1: דומיננטיות מסלול
    if context_data_prev:
        report_prev = (
            f"--- דוח קצב: {event} מרוץ {previous_year} (קונטקסט מסלול היסטורי) ---\n"
            f"הדוח מתאר את ביצועי הנהגים במסלול הספציפי {event} בשנה הקודמת. השווה קצב ממוצע ו-Var:\n"
            f"{context_data_prev}\n"
        )
    else:
        report_prev = f"--- דוח קצב: {event} מרוץ {previous_year} (אין נתונים היסטוריים זמינים למסלול) ---\n"
        
    # דוח 2: קונטקסט עונתי (שלושה דוחות מאוחדים)
    if race_reports_current:
        report_current = "\n".join(race_reports_current)
        num_races = len(race_reports_current)
        based_on_text = f"{event} {previous_year} Race Data & Analysis of the Last {num_races} Races of {current_year}."
    else:
        report_current = f"--- דוח קצב עונתי (אין נתונים עונתיים זמינים) ---\n"
        based_on_text = f"{event} {previous_year} Race Data Only (No Current Season Context)."


    # 4. בניית פרומפט המשלב את כל הדוחות
    
    full_data_prompt = report_prev + "\n" + report_current
    
    prompt = (
        f"אתה אנליסט בכיר ב-F1. נתח את הנתונים המשולבים הבאים כדי לספק דוח תחזית מוקדמת (Pre-Race) עבור **מרוץ {event} {current_year}**.\n\n"
        f"{full_data_prompt}\n\n"
        "--- הנחיות לניתוח (V33 - שילוב היסטוריה וקונטקסט רחב) ---\n"
        "1. **Immediate Prediction (Executive Summary):** בחר מנצח אחד והצג את הנימוק העיקרי (קצב ממוצע, עקביות או מגמה עונתית) בשורה אחת, **באנגלית בלבד**. (חובה)\n"
        "2. **Past Performance Analysis:** נתח את הדו\"ח ההיסטורי (שנה קודמת במסלול זה). הסבר מי היה דומיננטי מבחינת קצב ועקביות במסלול זה.\n"
        "3. **Current Season Trend Analysis:** נתח את דוחות המרוצים העונתיים. **בצע סיכום קצר של מגמת יחסי הכוחות בין הקבוצות המובילות (Red Bull, Ferrari, Mercedes) ב-3 המרוצים האחרונים.** מי נמצא במגמת שיפור ומי בירידה?\n"
        "4. **Strategic Conclusions and Winner Justification:** הצדק את בחירת המנצח על בסיס שילוב של **דומיננטיות קודמת במסלול** (מ-2024) ו**יכולת עונתית עדכנית** (מגמת 3 המרוצים האחרונים). עדיפות לנהג עם שילוב של חוזק היסטורי ומגמת שיפור עונתית.\n"
        "5. **אסטרטגיה מומלצת:** נתח את הנתונים וספק **אסטרטגיית צמיגים** מומלצת למרוץ הקרוב (לדוגמה: Hard-Medium-Hard) וניתוח **Pit-Stop Window**.\n"
        "6. **Confidence Score Table (D5):** ספק טבלת Confidence Score (בפורמט Markdown) המכילה את 5 המועמדים המובילים עם אחוז ביטחון (סך כל האחוזים חייב להיות 100%). **תקן את פורמט הטבלה כך שיופיע תקין ב-Markdown**.\n\n"
        
        "--- פורמט פלט חובה (Markdown, עברית למעט הכותרת הראשית) ---\n"
        f"🔮 Pre-Race Strategy Report: {event} {current_year}\n\n"
        f"Based on: {based_on_text}\n\n"
        "## Immediate Prediction (Executive Summary)\n"
        "...\n\n"
        "## Past Performance Analysis\n"
        "...\n\n"
        "## Current Season Trend Analysis\n"
        "...\n\n"
        "## Strategic Conclusions and Winner Justification\n"
        "...\n\n"
        "## 🏎️ Recommended Strategy & Pit-Stop Window\n"
        "...\n\n"
        "## 📊 Confidence Score Table (D5 - Visual Data)\n"
        "| Driver | Confidence Score (%) |\n"
        "|:--- | :--- |\n"
        "| ... | ... |\n"
        "| ... | ... |\n"
        "| ... | ... |\n"
        "| ... | ... |\n"
        "| ... | ... |\n"
    )
    
    # 5. שליחה ל-Gemini
    try:
        report = get_gemini_prediction(prompt)
        return report
    except Exception as e:
        st.error(f"❌ שגיאה ב-Gemini API במהלך יצירת תחזית מוקדמת: {e}")
        return None

# --- פונקציה ראשית של Streamlit ---

def main():
    """פונקציה ראשית המריצה את האפליקציה ב-Streamlit."""
    
    st.set_page_config(page_title="F1 P1 Predict", layout="centered")

    st.title("🏎️ F1 P1 Predict")
    st.markdown("---")
    st.markdown("An Online data-based strategy analysis and winning prediction tool")
    
    # בדיקת מפתח API
    try:
        if "GEMINI_API_KEY" not in st.secrets or not st.secrets["GEMINI_API_KEY"]:
            st.error("❌ שגיאה: מפתח ה-API של Gemini לא הוגדר ב-Streamlit Secrets. אנא ודא שהגדרת אותו כראוי.")
            return

    except Exception:
        st.error("❌ שגיאה: כשל בקריאת מפתח API. ודא שהגדרת אותו כראוי ב-Secrets.")
        return

    st.markdown("---")

    # בחירת פרמטרים 
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_year = st.selectbox("שנה:", YEARS, index=1, key="select_year") 
    with col2:
        selected_event = st.selectbox("מסלול:", TRACKS, index=0, key="select_event") 
    with col3:
        selected_session = st.selectbox("סשן:", SESSIONS, index=5, key="select_session")
    
    st.markdown("---")
    
    # 1. כפתור ניתוח נתונים קיימים
    if st.button("🏎️ חזה את המנצח (נתוני סשן נוכחי)", use_container_width=True, type="primary"):
        
        st.subheader(f"🔄 מתחיל ניתוח: {selected_event} {selected_year} ({selected_session})")
        
        status_placeholder = st.empty()
        status_placeholder.info("...טוען ומעבד נתונים מ-FastF1 (מנסה לעקוף בעיות חיבור/קאש)")
        
        context_data, session_name = load_and_process_data(selected_year, selected_event, selected_session)

        if context_data is None:
            status_placeholder.error(f"❌ שגיאה: {session_name}")
            return
        
        status_placeholder.success("✅ נתונים עובדו בהצלחה. שולח לניתוח AI...")

        try:
            prompt = create_prediction_prompt(context_data, selected_year, selected_event, selected_session)
            
            prediction_report = get_gemini_prediction(prompt)

            status_placeholder.success("🏆 הניתוח הושלם בהצלחה!")
            st.markdown("---")
            
            st.markdown(prediction_report)

        except APIError as e:
            status_placeholder.error(f"❌ שגיאת Gemini API: לא הצליח לקבל תגובה. פרטי שגיאה: {e}")
        except Exception as e:
            status_placeholder.error(f"❌ שגיאה בלתי צפויה: {e}")

    st.markdown("---")
    
    # 2. כפתור תחזית מוקדמת (Pre-Race Prediction) - כעת עם קונטקסט עונתי רחב
    if st.button("🔮 תחזית מוקדמת (שילוב עבר וקונטקסט עונתי)", use_container_width=True, type="secondary"):
        st.subheader(f"🔮 מתחיל תחזית מוקדמת: {selected_event} {selected_year}")
        
        prelim_report = get_preliminary_prediction(selected_year, selected_event)
        
        if prelim_report:
            st.markdown("---")
            st.markdown(prelim_report)


if __name__ == "__main__":
    main()
