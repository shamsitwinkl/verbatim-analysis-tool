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

🌱 If you have any issues, contact james.shamsi@twinkl.co.uk
""")

# User selection for analysis type
analysis_type = st.radio("Choose what kind of analysis you want:", ["Regex only", "AI only", "Combined Regex + AI"])

# File uploader
uploaded_file = st.file_uploader("📤 Upload your .csv file", type=["csv"])

# Regex patterns (truncated here for brevity — keep all 40 in your real code)
regex_patterns = {
    "Search/Navigation": r"(?i)finding|to find|problem finding|issue|where.*find",
    "Resource Mention": r"(?i)worksheet|resource|work sheet|activity pack",
    # ... [include all 40 patterns from your latest code]
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

    if analysis_type != "Regex only":
        st.markdown("### 💸 Estimated API Cost")
        st.markdown(f"""
        These estimates are based on **GPT-4o Mini** pricing:
        - $0.005 per 1K input tokens
        - $0.015 per 1K output tokens

        - Approx. input tokens: **{int(input_tokens):,}** → **${input_cost:.2f}**  
        - Approx. output tokens: **{int(output_tokens):,}** → **${output_cost:.2f}**  
        - 💰 **Estimated total cost: ${estimated_cost:.2f} USD**
        """)
    else:
        st.markdown("### 💸 AI Enhancement Estimate")
        st.info(f"This run used only Regex matching. Based on your data, using **GPT-4o Mini** would cost roughly **${(input_cost + output_cost):.2f} USD** for more accurate AI categorization. 🧠")

    st.markdown("### 📊 Category Match Summary")
    fig, ax = plt.subplots(figsize=(8, 5))
    top_counts = dict(category_counter.most_common(10))
    ax.barh(list(top_counts.keys()), list(top_counts.values()))
    ax.invert_yaxis()
    ax.set_xlabel("Number of Matches")
    ax.set_title("Top 10 Matched Categories")
    st.pyplot(fig)
