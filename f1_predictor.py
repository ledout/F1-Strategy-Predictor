import streamlit as st
import fastf1
import pandas as pd
import logging
from google import genai
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential
import io 
from datetime import date # ייבוא חדש לשימוש בבדיקת תאריך

# --- הגדרות ראשוניות ---
pd.options.mode.chained_assignment = None
logging.getLogger('fastf1').setLevel(logging.ERROR)

# **כיבוי מוחלט של FastF1 Cache מקומי**
try:
    # הגדרת Cache Path ל-None מכבה את ה-Cache המקומי של FastF1.
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

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def get_gemini_prediction(prompt):
    """שולח את הפרומפט ל-Gemini Flash ומשתמש במפתח מה-Secrets."""
    
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
             raise ValueError("GEMINI_API_KEY לא נמצא ב-Streamlit Secrets. אנא הגדר אותו.")
    except Exception as e:
        raise ValueError(f"שגיאת API Key: {e}")
        
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text


# ללא Caching של Streamlit
def load_and_process_data(year, event, session_key):
    """טוען נתונים מ-FastF1 ומבצע עיבוד ראשוני, עם טיפול בשגיאות גרסה של session.load()."""
    
    try:
        session = fastf1.get_session(year, event, session_key)
        
        # **תיקון V39/V44: ניסיון Session.load() בסיסי ועמיד לגרסאות FastF1 שונות**
        try:
            # 1. ניסיון טעינה בסיסי (אנו רוצים רק הקפות)
            session.load(laps=True, telemetry=False, weather=False, messages=False, pit_stops=False)
        except TypeError as e:
            # 2. אם נכשל בגלל ארגומנטים לא צפויים, ננסה טעינה ללא ארגומנטים כלל.
            if "unexpected keyword argument" in str(e):
                 # אנו נותנים ל-FastF1 לטעון הכל לבד אם הארגומנטים לא עובדים
                 session.load()
            else:
                 # אם זו שגיאת Type אחרת, זרוק אותה הלאה
                 raise e 
        except Exception as e:
            # שגיאת טעינה כללית - מעבר לדגל מפורש
            error_message = str(e)
            if "not loaded yet" in error_message:
                 # ניסיון טעינה מפורשת אם יש בעיה ב-metadata
                 session.load(telemetry=False, weather=False, messages=False, laps=True, pit_stops=False)
            else:
                 raise e
        
        # **בדיקת עמידות:** ודא ש-session.laps הוא DataFrame תקף
        if session.laps is None or session.laps.empty or not isinstance(session.laps, pd.DataFrame):
            return None, f"נתונים חסרים עבור {year} {event} {session_key}. FastF1 'load_laps' error."
            
    except Exception as e:
        error_message = str(e)
        
        if "Failed to load any schedule data" in error_message or "schedule data" in error_message:
             return None, f"FastF1: Failed to load any schedule data. שגיאה בטעינת FastF1: ייתכן שיש בעיית רשת/חיבור או שהשנה/מסלול לא קיימים."
        
        if "not found" in error_message or "The data you are trying to access has not been loaded yet" in error_message:
             return None, f"נתונים חסרים עבור {year} {event} {session_key}. ייתכן שמדובר באירוע מבוטל או שטרם התקיים. שגיאה: {error_message.split(':', 1)[-1].strip()}"
        
        if "unexpected keyword argument" in error_message:
             return None, f"שגיאת גרסה ב-FastF1: הפונקציה Session.load() קיבלה ארגומנט לא צפוי. (שגיאה: {error_message})"

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
    
    # נתונים סטטיסטיים רק אם בוצעו 5 הקפות ומעלה
    driver_stats = driver_stats[driver_stats['Laps'] >= 5]
    
    if driver_stats.empty:
        return None, "לא נמצאו נתונים מספקים (פחות מ-5 הקפות לנהג) לניתוח סטטיסטי. נסה סשן אחר."

    # עיבוד נתונים לפורמט טקסט (Top 10)
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

# --- פונקציות לתחזית מוקדמת (Pre-Race) ---

def find_last_three_races_data(current_year, event, expander_placeholder):
    """מוצא את שלושת המרוצים ה'רגילים' האחרונים שהיו אמורים להתקיים העונה ומחזיר את נתוני המרוץ שלהם."""
    
    with expander_placeholder.container():
        st.info("🔄 מתחיל איסוף נתונים עונתי (3 מרוצים אחרונים)")
        
        schedule = None
        try:
            schedule = fastf1.get_event_schedule(current_year)
            if schedule.empty:
                return [], "שגיאה: לוח הזמנים של השנה הנוכחית ריק." 

        except Exception as e:
            # אם יש שגיאה בטעינת Schedule (בדרך כלל FastF1), נצא
            return [], f"שגיאה: לא ניתן לטעון את לוח הזמנים של השנה הנוכחית. {e}" 
        
        
        # 1. מצא את האירוע הנוכחי
        current_event = schedule[schedule['EventName'] == event]
        
        
        # V46: טיפול עמיד במקרה שבו האירוע הנוכחי חסר ב-Schedule (הסיבה לכשלים הקודמים).
        
        current_event_date = None
        
        if current_event.empty:
             st.warning(f"⚠️ אזהרה: האירוע הנוכחי ({event}) לא נמצא בלוח הזמנים המלא. מנסה להשתמש בתאריך היום כנקודת ייחוס.")
             
             # אם אין לנו תאריך ייחוס, נשתמש בתאריך היום (ואנו מניחים שאם עברנו את סוף אפריל, יש נתונים)
             current_event_date = pd.to_datetime(date.today())
             
             # V46: אם השנה הנבחרת עתידית (לדוגמה 2025), זה עלול להכשיל.
             if current_year > date.today().year:
                 st.error("❌ לא ניתן לבצע ניתוח עונתי לשנה עתידית ללא תאריך אירוע מוגדר.")
                 return [], "❌ לא ניתן לבצע ניתוח עונתי לשנה עתידית."
             
             # אם לא מצאנו את האירוע, אנחנו לא יכולים לדעת את ה-RoundNumber
             # ולכן נדלג על בדיקת הסיבוב.

        else:
             try:
                 # האירוע נמצא, משתמשים במידע שלו
                 current_event_date = current_event['EventDate'].iloc[0]
                 current_event_round = current_event['RoundNumber'].iloc[0]
                 
                 # 2. בדיקת סיבוב (Round Number) - רק אם מצאנו את האירוע
                 if current_event_round <= 4:
                     st.warning(f"⚠️ אזהרה: האירוע הנוכחי ({event}) הוא אחד מ-4 המרוצים הראשונים של העונה. אין מספיק קונטקסט עונתי. מדלג.")
                     return [], "דילוג עונתי (מרוץ מוקדם מדי בעונה)." 
             except KeyError as e:
                 # V46: אם חסרה עמודה ב-Schedule
                 st.error(f"שגיאה בלוח הזמנים של FastF1: חסרה עמודה ({e}).")
                 return [], "FastF1: עמודה חסרה. לא ניתן לבצע ניתוח עונתי."
             except Exception as e:
                 # V46: שגיאה אחרת ב-Schedule
                 st.error(f"שגיאת Schedule לא צפויה: {e}")
                 return [], "שגיאה ב-FastF1 Schedule."
        
        
        # 3. סינון מרוצים על בסיס התאריך (או תאריך היום אם לא נמצא האירוע)
        try:
            # V46: סינון על פי תאריך האירוע הנוכחי
            potential_races = schedule.loc[
                (schedule['EventFormat'] == 'conventional') &
                (schedule['EventDate'] < current_event_date)
            ].sort_values(by='EventDate', ascending=False).head(3) 
        except KeyError as e:
            # אם אחת העמודות (EventFormat/EventDate) חסרה, נכשל ונחזיר סטטוס
            return [], f"FastF1: עמודה חסרה ({e}). לא ניתן לבצע ניתוח עונתי."
        
        
        if potential_races.empty:
            st.warning(f"אין מרוצים רגילים קודמים בלוח הזמנים של {current_year} טרם מרוץ {event}.")
            return [], f"אין מרוצים קודמים ב-{current_year}." 
        
        race_reports = []
        
        for index, race in potential_races.iterrows():
            event_name = race['EventName']
            st.info(f"🔮 מנסה לטעון נתוני מרוץ: {event_name} {current_year}...")
            
            # ננסה לטעון נתונים (Load)
            context_data, session_name = load_and_process_data(current_year, event_name, 'R')
            
            if context_data:
                report = (
                    f"--- דוח קצב: מרוץ {event_name} {current_year} (מרוץ עונתי) ---\n"
                    f"{context_data}\n"
                )
                race_reports.append(report)
                st.success(f"✅ נתוני מרוץ {event_name} נטענו בהצלחה.")
            else:
                # V46: אם ה-load_and_process_data נכשל
                st.warning(f"⚠️ לא ניתן היה לטעון נתוני מרוץ מלאים עבור {event_name}. ה-AI יתעלם מהמרוץ הזה. (שגיאה: {session_name})") 

        if not race_reports:
            # V46: מחזיר סטטוס כשל עונתי
            return [], f"לא נמצאו נתונים עונתיים מלאים ב-{current_year}." 
        
        st.success("✅ נתונים עונתיים עובדו בהצלחה. ממשיך ל-AI.")
        return race_reports, "נתונים עונתיים נטענו"


# ... (שאר הפונקציות: create_prediction_prompt, get_preliminary_prediction, main נשארות זהות)
# ...
