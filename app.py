import streamlit as st
import pandas as pd
import re
import openai
import os  # ✅ Added to support os.getenv()

# ✅ Load key from Render's secure environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("💬 Verbatim Analysis Tool")
st.markdown("""
This tool analyzes only the **`additional_comment`** column in your `.csv` file using **Regex + GPT-4o**.  
It enriches the file and returns all original columns **plus** AI categorization columns.
""")

regex_patterns = {
    "Search/Navigation": r"(?i)finding|to find|problem finding|issue|where.*find",
    "Resource Mention": r"(?i)worksheet|resource|work sheet|activity pack",
    "User Question": r"(?i)\b(what|where|when|why|how|who|which|can|could|should)\b",
    "Translation Mention": r"(?i)\btranslation\b|\btranslated\b|\btranslating\b",
    "User Suggestion": r"(?i)suggestion|should|could|would|suggest|recommend",
    "Pain Point": r"(?i)problem|issue|bug|error|difficult",
    "AI": r"(?i)\bAI\b|artificial intelligence|machine learning",
    "Competitor": r"(?i)competitor|another provider|used to use",
    "Site Error": r"(?i)website error|site down|page missing",
    "Social Media": r"(?i)facebook|meta|instagram|twitter|social media",
    "Curriculum Mention": r"(?i)curriculum|ks1|ks2|key stage|EYFS",
    "Twinkl Mention": r"(?i)twinkl",
    "Download Trouble": r"(?i)can't download|not downloading|download problem",
    "Payment Problem": r"(?i)payment|charge|billing|credit card",
    "Video Mention": r"(?i)\bvideo\b|watch|YouTube",
    "Navigation": r"(?i)hard to find|navigation|menu|confusing",
    "Positive Experience": r"(?i)love|great|excellent|helpful|amazing",
    "Negative Experience": r"(?i)bad|hate|useless|frustrating|annoying",
    "Pricing Feedback": r"(?i)too expensive|pricing|price|cost",
    "Login Issue": r"(?i)login|log in|can't sign in|password",
    "Account Access": r"(?i)account locked|cannot access",
    "Already Cancelled": r"(?i)cancel|canceled|cancelled|already cancelled",
    "Auto-renwal": r"(?i)auto.?renew|automatic renewal",
    "Book Club": r"(?i)\bbook club\b|\bbooks\b",
    "Cancellation difficulty": r"(?i)cancel(l|ing|led)? difficulty|can't cancel",
    "CS General": r"(?i)customer service|support team|agent",
    "CS Negative": r"(?i)(customer service|support).*(bad|unhelpful|rude)",
    "CS Positive": r"(?i)(customer service|support).*(great|helpful|nice)",
    "Negative words": r"(?i)awful|annoying|angry|disappointed",
    "Positive words": r"(?i)amazing|awesome|best|fantastic|love",
    "Support Request": r"(?i)need help|how do i|support",
    "Teacher Reference": r"(?i)i teach|my class|my students",
    "Child Mention": r"(?i)my child|son|daughter|kids",
    "Feedback General": r"(?i)feedback|thoughts|suggestions",
    "Language Mention": r"(?i)\benglish\b|\bspanish\b|\bfrench\b",
    "Error Feedback": r"(?i)wrong|error|typo|fix this",
    "Membership Issue": r"(?i)member(ship)?|sign up|join",
    "Mobile Use": r"(?i)phone|mobile|tablet|app",
    "Subject Mention": r"(?i)maths|science|history|english|geography",
    "Topic Request": r"(?i)do you have|can you make|topic request"
}

def match_categories(text):
    return [label for label, pattern in regex_patterns.items() if re.search(pattern, text, re.IGNORECASE)]

uploaded_file = st.file_uploader("📤 Upload your .csv file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

    if "additional_comment" not in df.columns:
        st.error("❌ Column 'additional_comment' not found in file.")
        st.write("Available columns:", list(df.columns))
        st.stop()

    df["Regex Categories"] = ""
    df["GPT Categories"] = ""
    df["Total Categories Found"] = 0

    for i, row in df.iterrows():
        comment = str(row["additional_comment"])
        regex_cats = match_categories(comment)

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You classify survey comments into a predefined list of categories. Only return the matching category names from this list. If none apply, return nothing (leave blank). Do not explain anything."
                    },
                    {
                        "role": "user",
                        "content": f"Text: '{comment}'\nCategories: {list(regex_patterns.keys())}\nReturn only matching category names (comma-separated). Leave blank if none."
                    }
                ],
                max_tokens=80
            )
            gpt_cats = response.choices[0].message.content.strip()
            if gpt_cats.lower() in ["none", "no match", "nothing applies", "nothing"]:
                gpt_cats = ""

        except:
            gpt_cats = "GPT Error"

        total_found = len(regex_cats) + (len(gpt_cats.split(",")) if gpt_cats else 0)

        df.at[i, "Regex Categories"] = ", ".join(regex_cats)
        df.at[i, "GPT Categories"] = gpt_cats
        df.at[i, "Total Categories Found"] = total_found

    st.markdown("### 🧾 Enriched Results")
    st.dataframe(df)

    st.download_button("📥 Download CSV", df.to_csv(index=False), "verbatim_analysis.csv", "text/csv")
