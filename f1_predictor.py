import streamlit as st
import fastf1
import pandas as pd
import logging
from google import genai
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential
import io 

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

# ללא Caching של Streamlit
def load_and_process_data(year, event, session_key):
    """טוען נתונים מ-FastF1 ומבצע עיבוד ראשוני, עם טיפול בשגיאות גרסה של session.load()."""
    
    try:
        session = fastf1.get_session(year, event, session_key)
        
        # **תיקון V39: ניסיון Session.load() בסיסי ועמיד לגרסאות FastF1 שונות**
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

# --- פונקציות לתחזית מוקדמת (Pre-Race) ---

def find_last_three_races_data(current_year, event, expander_placeholder):
    """מוצא את שלושת המרוצים ה'רגילים' האחרונים שהיו אמורים להתקיים העונה ומחזיר את נתוני המרוץ שלהם."""
    
    with expander_placeholder.container():
        st.info("🔄 מתחיל איסוף נתונים עונתי (3 מרוצים אחרונים)")
        
        try:
            schedule = fastf1.get_event_schedule(current_year)
            if schedule.empty:
                st.error("שגיאה: לוח הזמנים של השנה הנוכחית ריק.")
                return [], "שגיאה בטעינת לוח זמנים."

        except Exception as e:
            st.error(f"שגיאה: לא ניתן לטעון את לוח הזמנים של השנה הנוכחית. {e}")
            return [], "שגיאה בטעינת לוח זמנים."
        
        # 1. מצא את תאריך המרוץ הנוכחי ואת מספר הסיבוב שלו
        try:
            current_event = schedule[schedule['EventName'] == event]
            current_event_date = current_event['EventDate'].iloc[0]
            current_event_round = current_event['RoundNumber'].iloc[0]
        except IndexError:
            # זו השגיאה שראית בצילומים (קנדה 2024, לא נמצא בלוח הזמנים)
            st.error(f"שגיאה: {event} {current_year} לא נמצא בלוח הזמנים. לא ניתן למצוא תאריך יחוס.")
            return [], "אירוע לא נמצא בלוח הזמנים."
        
        # 2. בדיקת סיבוב (Round Number)
        if current_event_round <= 4:
            st.warning(f"⚠️ אזהרה: האירוע הנוכחי ({event}) הוא אחד מ-4 המרוצים הראשונים של העונה. אין מספיק קונטקסט עונתי. מדלג על טעינת 3 המרוצים הקודמים.")
            return [], "דילוג עונתי (מרוץ מוקדם מדי בעונה)."
        
        # 3. סינון מרוצים: רק אירועים שמתכונתם 'conventional' והתאריך שלהם קטן מתאריך המרוץ הנוכחי
        # **הסרה סופית של בדיקת EventCompleted עקב אי זמינותו בלוחות זמנים עתידיים**
        try:
            potential_races = schedule.loc[
                (schedule['EventFormat'] == 'conventional') &
                (schedule['EventDate'] < current_event_date)
            ].sort_values(by='EventDate', ascending=False).head(3) # מיין לפי תאריך יורד וקח את 3 האחרונים
        except KeyError as e:
            st.error(f"שגיאת FastF1: עמודה חסרה ({e}). לא ניתן לבצע ניתוח עונתי.")
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
                # אם ה-load_and_process_data נכשל, מציג אזהרה בתוך האקספנדר
                st.warning(f"⚠️ לא ניתן היה לטעון נתוני מרוץ מלאים עבור {event_name}. ה-AI יתעלם מהמרוץ הזה. (שגיאה: {session_name})") 

        if not race_reports:
            st.error(f"לא נמצאו נתונים מלאים לאף אחד מ-3 המרוצים הקודמים ב-{current_year}. הניתוח יתבסס על היסטוריה בלבד.")
            return [], f"לא נמצאו נתונים עונתיים מלאים ב-{current_year}."
        
        st.success("✅ נתונים עונתיים עובדו בהצלחה. ממשיך ל-AI.")
        return race_reports, "נתונים עונתיים נטענו"


def create_prediction_prompt(context_data, year, event, session_name):
    """בניית הפרומפט המלא למודל Gemini עבור נתונים עכשוויים."""
    
    prompt_data = f"--- נתונים גולמיים לניתוח (Top 10 Drivers, Race/Session Laps) ---\n{context_data}"

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


def get_preliminary_prediction(current_year, event):
    """משלב נתוני מרוץ מהשנה הקודמת ומשלושת המרוצים האחרונים העונה ליצירת תחזית מוקדמת חזקה יותר."""
    
    previous_year = current_year - 1
    
    st.subheader("🏁 איסוף נתונים לתחזית מוקדמת (Pre-Race Analysis)")
    
    # יוצרים כאן את האקספנדר הסגור לכל הדיווחים הטכניים
    with st.expander("🛠️ הצג פרטי טעינת נתונים היסטוריים ועונתיים (דיאגנוסטיקה)", expanded=False):
        expander_placeholder = st.container() # פלייסהולדר להעברת פנימה לפונקציות
        
        with expander_placeholder:
             st.info(f"🔮 מנתח דומיננטיות במסלול: טוען נתוני מרוץ {event} משנה {previous_year}...")
            
             # 1. טעינת נתונים היסטוריים (שנה קודמת באותו מסלול)
             context_data_prev, session_name_prev = load_and_process_data(previous_year, event, 'R')
             if context_data_prev:
                 st.success(f"✅ נתוני מרוץ {event} {previous_year} נטענו בהצלחה.")
             else:
                 st.warning(f"⚠️ אזהרה: לא נמצאו נתונים היסטוריים מלאים עבור {event} {previous_year}. ({session_name_prev})")
             
             st.markdown("---")
        
        # 2. טעינת נתונים עונתיים (3 המרוצים האחרונים שהושלמו)
        race_reports_current, status_msg = find_last_three_races_data(current_year, event, expander_placeholder)

    # 3. בדיקת נתונים ואיחוד דוחות (מחוץ לאקספנדר)
    
    if context_data_prev:
        report_prev = (
            f"--- דוח קצב: {event} מרוץ {previous_year} (קונטקסט מסלול היסטורי) ---\n"
            f"הדוח מתאר את ביצועי הנהגים במסלול הספציפי {event} בשנה הקודמת. השווה קצב ממוצע ו-Var:\n"
            f"{context_data_prev}\n"
        )
    else:
        report_prev = f"--- דוח קצב: {event} מרוץ {previous_year} (אין נתונים היסטוריים זמינים למסלול) ---\n"
        
    if race_reports_current:
        report_current = "\n" + "\n".join(race_reports_current)
        num_races = len(race_reports_current)
        based_on_text = f"{event} {previous_year} Race Data & Analysis of the Last {num_races} Races of {current_year}."
    else:
        report_current = f"--- דוח קצב עונתי (אין נתונים עונתיים זמינים) ---\n"
        based_on_text = f"{event} {previous_year} Race Data Only (No Current Season Context)."


    # 4. בניית פרומפט המשלב את כל הדוחות
    
    full_data_prompt = report_prev + report_current
    
    prompt = f"""
אתה אנליסט בכיר ב-F1. נתח את הנתונים המשולבים הבאים כדי לספק דוח תחזית מוקדמת (Pre-Race) עבור **מרוץ {event} {current_year}**.

{full_data_prompt}

--- הנחיות לניתוח (V33 - שילוב היסטוריה וקונטקסט רחב) ---
1. **Immediate Prediction (Executive Summary):** בחר מנצח אחד והצג את הנימוק העיקרי (קצב ממוצע, עקביות או מגמה עונתית) בשורה אחת, **באנגלית בלבד**. (חובה)
2. **Past Performance Analysis:** נתח את הדו\"ח ההיסטורי (שנה קודמת במסלול זה). הסבר מי היה דומיננטי מבחינת קצב ועקביות במסלול זה.
3. **Current Season Trend Analysis:** נתח את דוחות המרוצים העונתיים. **בצע סיכום קצר של מגמת יחסי הכוחות בין הקבוצות המובילות (Red Bull, Ferrari, Mercedes) ב-3 המרוצים האחרונים.** מי נמצא במגמת שיפור ומי בירידה?
4. **Strategic Conclusions and Winner Justification:** הצדק את בחירת המנצח על בסיס שילוב של **דומיננטיות קודמת במסלול** (מ-2024/3) ו**יכולת עונתית עדכנית** (מגמת 3 המרוצים האחרונים). עדיפות לנהג עם שילוב של חוזק היסטורי ומגמת שיפור עונתית.
5. **אסטרטגיה מומלצת:** נתח את הנתונים וספק **אסטרטגיית צמיגים** מומלצת למרוץ הקרוב (לדוגמה: Hard-Medium-Hard) וניתוח **Pit-Stop Window**.
6. **Confidence Score Table (D5):** ספק טבלת Confidence Score (בפורמט Markdown) המכילה את 5 המועמדים המובילים עם אחוז ביטחון (סך כל האחוזים חייב להיות 100%). **תקן את פורמט הטבלה כך שיופיע תקין ב-Markdown**.

--- פורמט פלט חובה (Markdown, עברית למעט הכותרת הראשית) ---
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
| ... | ... |
| ... | ... |
| ... | ... |
| ... | ... |
| ... | ... |
"""
    
    try:
        report = get_gemini_prediction(prompt)
        return report
    except Exception as e:
        st.error(f"❌ שגיאה ב-Gemini API במהלך יצירת תחזית מוקדמת: {e}")
        return None

# --- פונקציה ראשית של Streamlit ---

def main():
    """פונקציה ראשית המריצה את האפליקציה ב-Streamlit."""
    
    st.set_page_config(page_title="F1 Strategy Predictor", layout="centered")

    st.title("🏎️ F1 P1 Predict")
    st.markdown("An Online data-based strategy analysis and winning prediction tool")
    st.markdown("---")
    
    # בדיקת מפתח API
    try:
        api_key_check = st.secrets.get("GEMINI_API_KEY")
        if not api_key_check:
            st.error("❌ שגיאה: מפתח ה-API של Gemini לא הוגדר ב-Streamlit Secrets. אנא ודא שהגדרת אותו כראוי.")
        if not api_key_check:
             st.warning("⚠️ שימו לב: מפתח ה-API לא נמצא. הניתוח יכשל כאשר ינסה להתחבר ל-Gemini.")

    except Exception:
        st.error("❌ שגיאה: כשל בקריאת מפתח API. ודא שהגדרת אותו כראוי ב-Secrets.")
        
    st.markdown("---")

    # בחירת פרמטרים 
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_year = st.selectbox("שנה:", YEARS, index=2, key="select_year") 
    with col2:
        selected_event = st.selectbox("מסלול:", TRACKS, index=5, key="select_event") 
    with col3:
        selected_session = st.selectbox("סשן:", SESSIONS, index=5, key="select_session")
    
    st.markdown("---")
    
    # 1. כפתור ניתוח נתונים קיימים
    if st.button("🏎️ חזה את המנצח (נתוני סשן נוכחי)", use_container_width=True, type="primary"):
        
        st.subheader(f"🔄 מתחיל ניתוח: {selected_event} {selected_year} ({selected_session})")
        
        status_placeholder = st.empty()
        status_placeholder.info("...טוען ומעבד נתונים מ-FastF1...")
        
        # טעינת ועיבוד הנתונים 
        context_data, status_msg = load_and_process_data(selected_year, selected_event, selected_session)

        if context_data is None:
            status_placeholder.error(f"❌ שגיאה: {status_msg}")
            return
        
        status_placeholder.success("✅ נתונים עובדו בהצלחה. שולח לניתוח AI...")

        # יצירת הפרומפט וקבלת התחזית
        try:
            prompt = create_prediction_prompt(context_data, selected_year, selected_event, selected_session)
            
            prediction_report = get_gemini_prediction(prompt)

            status_placeholder.success("🏆 הניתוח הושלם בהצלחה!")
            st.markdown("---")
            
            # הצגת הדו"ח
            st.markdown(prediction_report)

        except APIError as e:
            status_placeholder.error(f"❌ שגיאת Gemini API: לא הצליח לקבל תגובה. פרטי שגיאה: {e}")
        except ValueError as e: # לכידת שגיאות API Key מ-get_gemini_prediction
            status_placeholder.error(f"❌ שגיאה קריטית: {e}")
        except Exception as e:
            status_placeholder.error(f"❌ שגיאה בלתי צפויה: {e}")

    st.markdown("---")
    
    # 2. כפתור תחזית מוקדמת (Pre-Race Prediction)
    if st.button("🔮 תחזית מוקדמת (שילוב עבר וקונטקסט עונתי)", use_container_width=True, type="secondary"):
        st.subheader(f"🔮 מתחיל תחזית מוקדמת: {selected_event} {selected_year}")
        
        prelim_report = get_preliminary_prediction(selected_year, selected_event)
        
        if prelim_report:
            st.markdown("---")
            st.markdown(prelim_report)


if __name__ == "__main__":
    main()
