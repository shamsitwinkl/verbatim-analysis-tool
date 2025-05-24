import streamlit as st
import pandas as pd
import re
import openai
import os
import matplotlib.pyplot as plt
from openai import OpenAI
from collections import Counter
import time
import io

# ✅ Password gate before anything else
st.set_page_config(page_title="Verbatim Analysis Tool", layout="centered")

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Enter Password")
    password = st.text_input("This tool is password protected. Please enter the password to continue:", type="password")
    
    if st.button("Login"):
        if password == "5577":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password. Please contact james.shamsi@twinkl.co.uk to request access.")
            st.stop()
    else:
        st.stop()

# ✅ Load OpenAI key securely with fallback option
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning("⚠️ OPENAI_API_KEY environment variable not set.")
    api_key = st.text_input("Please enter your OpenAI API key:", type="password")
    if not api_key:
        st.error("❌ OpenAI API key is required for AI analysis.")
        st.stop()

try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error(f"❌ Failed to initialize OpenAI client: {str(e)}")
    st.stop()

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

# Regex patterns (40 total)
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
    """Match regex patterns against text"""
    if pd.isna(text) or text.strip() == "":
        return []
    return [label for label, pattern in regex_patterns.items() if re.search(pattern, str(text), re.IGNORECASE)]

def generate_prompt(comment):
    """Generate prompt for GPT analysis"""
    categories = ", ".join(regex_patterns.keys())
    return f"""Text: '{comment}'
Categories: {categories}
Return a comma-separated list of category names only. Leave blank if none."""

def analyze_with_gpt(comment, client, max_retries=3):
    """Analyze comment with GPT with retry logic"""
    for attempt in range(max_retries):
        try:
            prompt = generate_prompt(comment)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful classifier. Use the category hints in the user's prompt. Only return category names. Do not explain."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=80,
                temperature=0.3
            )
            
            gpt_cats = response.choices[0].message.content.strip()
            
            if gpt_cats.lower() in ["none", "no match", "nothing applies", "nothing", ""]:
                return ""
            
            return gpt_cats
            
        except Exception as e:
            if attempt == max_retries - 1:
                st.warning(f"⚠️ GPT Error after {max_retries} attempts: {str(e)}")
                return ""
            time.sleep(1)  # Wait before retry

def estimate_costs(total_chars, num_requests):
    """Estimate OpenAI API costs"""
    # GPT-4o-mini pricing (approximate)
    input_cost_per_1k = 0.00015  # $0.15 per 1M tokens
    output_cost_per_1k = 0.0006  # $0.60 per 1M tokens
    
    # Rough estimation: 4 chars per token
    input_tokens = total_chars / 4
    output_tokens = num_requests * 20  # Assume ~20 tokens per response
    
    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    
    return input_cost + output_cost, input_tokens, output_tokens

if uploaded_file:
    try:
        # Load and validate CSV
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.replace("﻿", "", regex=False)
        
        if "additional_comment" not in df.columns:
            st.error("❌ Column 'additional_comment' not found in file.")
            st.write("Available columns:", list(df.columns))
            st.stop()
        
        # Show preview
        st.subheader("📊 Data Preview")
        st.write(f"Total rows: {len(df)}")
        st.dataframe(df.head())
        
        # Filter out empty comments
        non_empty_df = df[df["additional_comment"].notna() & (df["additional_comment"].str.strip() != "")]
        st.write(f"Rows with comments to analyze: {len(non_empty_df)}")
        
        if len(non_empty_df) == 0:
            st.warning("⚠️ No non-empty comments found to analyze.")
            st.stop()
        
        # Confirm analysis
        if st.button("🚀 Start Analysis", type="primary"):
            # Initialize new columns
            df["Regex Categories"] = ""
            df["GPT Categories"] = ""
            df["Total Categories Found"] = 0
            
            total_chars = 0
            category_counter = Counter()
            processed_count = 0
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each row
            for i, row in df.iterrows():
                comment = str(row["additional_comment"])
                
                # Skip empty comments
                if pd.isna(row["additional_comment"]) or str(row["additional_comment"]).strip() == "":
                    continue
                
                total_chars += len(comment)
                
                # Regex analysis
                regex_cats = match_categories(comment) if analysis_type != "AI only" else []
                
                # GPT analysis
                gpt_cats = ""
                if analysis_type != "Regex only":
                    gpt_cats = analyze_with_gpt(comment, client)
                
                # Combine results
                gpt_cat_list = [cat.strip() for cat in gpt_cats.split(",") if gpt_cats and cat.strip()]
                all_unique = list(set(regex_cats) | set(gpt_cat_list))
                category_counter.update(all_unique)
                
                # Update dataframe
                df.at[i, "Regex Categories"] = ", ".join(regex_cats)
                df.at[i, "GPT Categories"] = gpt_cats
                df.at[i, "Total Categories Found"] = len(all_unique)
                
                processed_count += 1
                
                # Update progress
                progress = processed_count / len(non_empty_df)
                progress_bar.progress(progress)
                status_text.text(f"Processing row {processed_count}/{len(non_empty_df)}")
            
            progress_bar.progress(1.0)
            status_text.text("✅ Analysis complete!")
            
            # Display results
            st.subheader("📈 Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Comments", len(non_empty_df))
            with col2:
                st.metric("Categories Used", len(category_counter))
            with col3:
                avg_cats = df["Total Categories Found"].mean()
                st.metric("Avg Categories/Comment", f"{avg_cats:.1f}")
            
            # Category distribution
            if category_counter:
                st.subheader("🏷️ Category Distribution")
                most_common = category_counter.most_common(10)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                categories, counts = zip(*most_common)
                ax.barh(categories, counts)
                ax.set_xlabel("Frequency")
                ax.set_title("Top 10 Categories")
                plt.tight_layout()
                st.pyplot(fig)
            
            # Cost estimation
            if analysis_type != "Regex only":
                st.subheader("💰 Cost Estimation")
                estimated_cost, input_tokens, output_tokens = estimate_costs(total_chars, processed_count)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Estimated Cost", f"${estimated_cost:.4f}")
                with col2:
                    st.metric("Input Tokens", f"{input_tokens:,.0f}")
                with col3:
                    st.metric("Output Tokens", f"{output_tokens:,.0f}")
            
            # Download processed file
            st.subheader("📥 Download Results")
            
            # Convert to CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📁 Download Analyzed CSV",
                data=csv_data,
                file_name=f"analyzed_{uploaded_file.name}",
                mime="text/csv"
            )
            
            # Show sample results
            st.subheader("🔍 Sample Results")
            sample_df = df[df["Total Categories Found"] > 0].head(5)
            st.dataframe(sample_df[["additional_comment", "Regex Categories", "GPT Categories", "Total Categories Found"]])
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.write("Please check your file format and try again.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for Twinkl | Contact: james.shamsi@twinkl.co.uk")
