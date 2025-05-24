import streamlit as st
import pandas as pd
import re
import openai
import os
import matplotlib.pyplot as plt
from openai import OpenAI
from collections import Counter

# ✅ Password gate before anything else
st.set_page_config(page_title="Verbatim Analysis Tool", layout="centered")
st.title("🔐 Enter Password")

password = st.text_input("This tool is password protected. Please enter the password to continue:", type="password")

if password != "5577":
    st.warning("❌ Incorrect or missing password. Please contact james.shamsi@twinkl.co.uk to request access.")
    st.stop()

# ✅ Load OpenAI key securely and validate it's present
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY environment variable not set. Please check your Render settings.")
    st.stop()
client = OpenAI(api_key=api_key)

# 🎨 Tool Info
st.title("💖 Verbatim Analysis Tool")

st.markdown("""
This tool analyzes only the **`additional_comment`** column in your `.csv` file using **Regex**, **GPT-4o Mini**, or both. 
It enriches the file and returns all original columns **plus** AI/Regex categorization columns.

🧠 **Bonus:** At the end, it estimates **OpenAI API token usage and cost** based on character count in your data.

⚠️ If GPT categorization fails, the app will display a descriptive error message to help you troubleshoot.
""")

# User selection for analysis type
analysis_type = st.radio("Choose what kind of analysis you want:", ["Regex only", "AI only", "Combined Regex + AI"])

# File uploader
uploaded_file = st.file_uploader("📤 Upload your .csv file", type=["csv"])

# Regex patterns (40 total)
regex_patterns = {
    "Search/Navigation": r"(?i)finding|to find|problem finding|issue|where.*find",
    "Resource Mention": r"(?i)worksheet|resource|work sheet|activity pack",
    "User Question": r"(?i)\\b(what|where|when|why|how|who|which|can|could|should)\\b",
    "Translation Mention": r"(?i)\\btranslation\\b|\\btranslated\\b|\\btranslating\\b",
    "User Suggestion": r"(?i)suggestion|should|could|would|suggest|recommend",
    "Pain Point": r"(?i)problem|issue|bug|error|difficult",
    "AI": r"(?i)\\bAI\\b|artificial intelligence|machine learning",
    "Competitor": r"(?i)competitor|another provider|used to use",
    "Site Error": r"(?i)website error|site down|page missing",
    "Social Media": r"(?i)facebook|meta|instagram|twitter|social media",
    "Curriculum Mention": r"(?i)curriculum|ks1|ks2|key stage|EYFS",
    "Twinkl Mention": r"(?i)twinkl",
    "Download Trouble": r"(?i)can't download|not downloading|download problem",
    "Payment Problem": r"(?i)payment|charge|billing|credit card",
    "Video Mention": r"(?i)\\bvideo\\b|watch|YouTube",
    "Navigation": r"(?i)hard to find|navigation|menu|confusing",
    "Positive Experience": r"(?i)love|great|excellent|helpful|amazing",
    "Negative Experience": r"(?i)bad|hate|useless|frustrating|annoying",
    "Pricing Feedback": r"(?i)too expensive|pricing|price|cost",
    "Login Issue": r"(?i)login|log in|can't sign in|password",
    "Account Access": r"(?i)account locked|cannot access",
    "Already Cancelled": r"(?i)cancel|canceled|cancelled|already cancelled",
    "Auto-renwal": r"(?i)auto.?renew|automatic renewal",
    "Book Club": r"(?i)\\bbook club\\b|\\bbooks\\b",
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
    "Language Mention": r"(?i)\\benglish\\b|\\bspanish\\b|\\bfrench\\b",
    "Error Feedback": r"(?i)wrong|error|typo|fix this",
    "Membership Issue": r"(?i)member(ship)?|sign up|join",
    "Mobile Use": r"(?i)phone|mobile|tablet|app",
    "Subject Mention": r"(?i)maths|science|history|english|geography",
    "Topic Request": r"(?i)do you have|can you make|topic request"
}

def match_categories(text):
    return [label for label, pattern in regex_patterns.items() if re.search(pattern, text, re.IGNORECASE)]

def unique_combined_count(regex_list, gpt_list):
    all_labels = set(regex_list) | set([label.strip() for label in gpt_list.split(",") if gpt_list])
    return len(all_labels)

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

    total_chars = 0
    category_counter = Counter()

    for i, row in df.iterrows():
        comment = str(row["additional_comment"])
        total_chars += len(comment)
        regex_cats = match_categories(comment) if analysis_type != "AI only" else []

        gpt_cats = ""
        if analysis_type != "Regex only":
            try:
                prompt = f"Text: '{comment}'\nCategories: {list(regex_patterns.keys())}\nReturn matching category names only (comma-separated). Leave blank if none."
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Classify the text using the following list. Return matching category names only. If none match, return nothing."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=80
                )
                gpt_cats = response.choices[0].message.content.strip()
                if gpt_cats.lower() in ["none", "no match", "nothing applies", "nothing"]:
                    gpt_cats = ""
            except Exception as e:
                gpt_cats = f"GPT Error: {str(e)}"

        all_unique = list(set(regex_cats) | set([cat.strip() for cat in gpt_cats.split(",") if gpt_cats]))
        category_counter.update(all_unique)

        df.at[i, "Regex Categories"] = ", ".join(regex_cats)
        df.at[i, "GPT Categories"] = gpt_cats
        df.at[i, "Total Categories Found"] = len(all_unique)

    st.markdown("### 🧾 Enriched Results")
    st.dataframe(df)

    st.download_button("📥 Download CSV", df.to_csv(index=False), "verbatim_analysis.csv", "text/csv")

    # Token and cost estimation based on total_chars and average prompt size
    avg_prompt_chars = 150  # system + user message chars
    input_tokens = (total_chars + len(df) * avg_prompt_chars) / 4
    output_tokens = (len(df) * 80) / 4  # 80 max tokens per output

    input_cost = (input_tokens / 1000) * 0.005
    output_cost = (output_tokens / 1000) * 0.015
    estimated_cost = input_cost + output_cost

    st.markdown("### 💸 Estimated API Cost")
    st.markdown(f"""
    - Approx. input tokens: **{int(input_tokens):,}** → **${input_cost:.2f}**  
    - Approx. output tokens: **{int(output_tokens):,}** → **${output_cost:.2f}**  
    - 💰 **Estimated total cost: ${estimated_cost:.2f} USD**
    """)

    st.markdown("### 📊 Category Match Summary")
    fig, ax = plt.subplots(figsize=(8, 5))
    top_counts = dict(category_counter.most_common(10))
    ax.barh(list(top_counts.keys()), list(top_counts.values()))
    ax.invert_yaxis()
    ax.set_xlabel("Number of Matches")
    ax.set_title("Top 10 Matched Categories")
    st.pyplot(fig)
