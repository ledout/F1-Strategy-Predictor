# ... קוד קודם ...

# --- פונקציה ראשית של Streamlit ---

def main():
    """פונקציה ראשית המריצה את האפליקציה ב-Streamlit."""
    st.set_page_config(page_title="F1 Strategy Predictor V33", layout="centered")

    st.title("🏎️ F1 Strategy Predictor V33")
    st.markdown("---")
    st.markdown("כלי לניתוח אסטרטגיה וחיזוי מנצח מבוסס נתוני FastF1 ו-Gemini AI.")
    
    # בדיקת מפתח API (בשרת Streamlit)
    try:
        if "GEMINI_API_KEY" not in st.secrets or not st.secrets["GEMINI_API_KEY"]:
            st.error("❌ שגיאה: מפתח ה-API של Gemini לא הוגדר ב-Streamlit Secrets. אנא הגדר אותו.")
            return

    except Exception:
        st.error("❌ שגיאה: כשל בקריאת מפתח API. ודא שהגדרת אותו כראוי ב-Secrets.")
        return

    st.markdown("---")

    # בחירת פרמטרים
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_year = st.selectbox("שנה:", YEARS, index=1) # 2024
    with col2:
        selected_event = st.selectbox("מסלול:", TRACKS, index=0) # Bahrain
    with col3:
        selected_session = st.selectbox("סשן:", SESSIONS, index=5)
    
    st.markdown("---")
    
    # כפתור הפעלה
    if st.button("🏎️ חזה את המנצח (אוטומטי)", use_container_width=True, type="primary"):
        
        # *** ודא ששורה זו קיימת ותקינה כפי שהיא: ***
        st.subheader(f"🔄 מתחיל ניתוח: {selected_event} {selected_year} ({selected_session})")
        
        # *** ודא ששורה זו קיימת ותקינה כפי שהיא: ***
        status_placeholder = st.empty()
        status_placeholder.info("...טוען ומעבד נתונים מ-FastF1 (מנסה לעקוף בעיות חיבור)")
        
        # 1. טעינת ועיבוד הנתונים (משתמש ב-st.cache_data)
        context_data, session_name = load_and_process_data(selected_year, selected_event, selected_session)

        if context_data is None:
            # הצגת השגיאה שהוחזרה מ-load_and_process_data
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
