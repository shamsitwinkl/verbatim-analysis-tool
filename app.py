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
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import string
import numpy as np

# ✅ Password gate before anything else
st.set_page_config(page_title="Verbatim Categorizer Tool", layout="centered")

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
st.title("💖 Verbatim Categorizer Tool")

st.markdown("""
This tool analyzes text feedback columns in your `.csv` file using **Regex**, **GPT-4o Mini**, or both approaches.

🔍 **Smart Column Detection**: If your file doesn't have an `additional_comment` column, the tool will automatically detect the best text column for analysis by examining things like average text length and complexity
- Punctuation usage patterns and more.  

🧠 Before processing, it estimates **OpenAI API token usage and cost** based on character count in your data.

🎯 The tool enriches your file and returns all original columns **plus** new AI/Regex categorization columns.

🌱 If you have any issues, contact james.shamsi@twinkl.co.uk
""")

# User selection for analysis type
analysis_type = st.radio("Choose what kind of analysis you want:", ["Regex only", "AI only", "Combined Regex + AI"])

# Performance settings
if analysis_type != "Regex only":
    st.sidebar.header("⚡ Performance Settings")
    batch_size = st.sidebar.slider("Batch Size (higher = faster but more memory)", 5, 50, 20, 
                                   help="Number of comments processed simultaneously. Increase for speed, decrease if you get rate limit errors.")
    max_workers = st.sidebar.slider("Max Workers", 1, 10, 5, 
                                    help="Number of parallel threads. Increase for speed, but be careful of rate limits.")

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

# ----------------------------------------
# SMART COLUMN DETECTION FUNCTIONS
# ----------------------------------------

def detect_freeform_text_column(df, penalty_for_low_uniqueness=0.4):
    """
    Detect the best column for freeform text analysis using heuristics:
    - Average text length
    - Punctuation usage  
    - Response uniqueness
    - Column naming patterns
    """
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    if not text_cols:
        return None, None
    
    scores = {}
    detailed_analysis = {}
    
    for col in text_cols:
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
            
        # Calculate metrics
        avg_len = series.apply(len).mean()
        punct_counts = series.apply(lambda x: sum(1 for char in x if char in string.punctuation))
        avg_punct = punct_counts.mean()
        total = len(series)
        unique_ratio = series.nunique() / total if total else 0
        
        # Store detailed analysis for explanation
        detailed_analysis[col] = {
            'avg_length': round(avg_len, 1),
            'avg_punctuation': round(avg_punct, 1),
            'uniqueness_ratio': round(unique_ratio, 3),
            'total_responses': total
        }
        
        scores[col] = {
            'avg_len': avg_len,
            'avg_punct': avg_punct,
            'unique_ratio': unique_ratio,
        }
    
    if not scores:
        return None, None
    
    # Normalize scores across all columns
    max_len = max(s['avg_len'] for s in scores.values()) or 1e-9
    max_punct = max(s['avg_punct'] for s in scores.values()) or 1e-9
    
    composite_scores = {}
    for col, s in scores.items():
        norm_len = s['avg_len'] / max_len
        norm_punct = s['avg_punct'] / max_punct
        comp_score = (0.4 * norm_len) + (0.3 * norm_punct) + (0.3 * s['unique_ratio'])
        
        # Column name bonuses
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['comment', 'feedback', 'response', 'reason', 'text']):
            comp_score *= 2.0
            detailed_analysis[col]['name_bonus'] = "✅ Contains feedback-related keywords"
        else:
            detailed_analysis[col]['name_bonus'] = "❌ No feedback keywords in name"
            
        # Penalty for low uniqueness
        if s['unique_ratio'] < penalty_for_low_uniqueness:
            comp_score *= 0.6
            detailed_analysis[col]['uniqueness_penalty'] = f"⚠️ Low uniqueness ({s['unique_ratio']:.1%})"
        else:
            detailed_analysis[col]['uniqueness_penalty'] = f"✅ Good uniqueness ({s['unique_ratio']:.1%})"
        
        composite_scores[col] = comp_score
        detailed_analysis[col]['final_score'] = round(comp_score, 3)
    
    best_col = max(composite_scores, key=composite_scores.get)
    return best_col, detailed_analysis

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
            time.sleep(2 ** attempt)  # Exponential backoff

def process_batch_gpt(comments_batch, client):
    """Process a batch of comments with threading"""
    results = []
    
    def process_single(comment):
        return analyze_with_gpt(comment, client)
    
    with ThreadPoolExecutor(max_workers=max_workers if 'max_workers' in locals() else 5) as executor:
        results = list(executor.map(process_single, comments_batch))
    
    return results

def estimate_costs(total_chars, num_requests, avg_response_length=20):
    """Estimate OpenAI API costs with detailed breakdown"""
    # GPT-4o-mini pricing (per 1M tokens)
    input_cost_per_1m = 0.15   # $0.15 per 1M input tokens
    output_cost_per_1m = 0.60  # $0.60 per 1M output tokens
    
    # Token estimation: roughly 4 characters per token
    chars_per_token = 4
    
    # Calculate tokens
    input_tokens = total_chars / chars_per_token
    output_tokens = num_requests * avg_response_length
    
    # Calculate costs
    input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
    total_cost = input_cost + output_cost
    
    return {
        'total_cost': total_cost,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_chars': total_chars,
        'num_requests': num_requests
    }

if uploaded_file:
    try:
        # Load and validate CSV
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.replace("﻿", "", regex=False)
        
        # Smart column detection
        target_column = None
        used_auto_detection = False
        
        if "additional_comment" in df.columns:
            target_column = "additional_comment"
            st.success("✅ Found `additional_comment` column - using this for analysis")
        else:
            st.warning("⚠️ Column `additional_comment` not found. Running smart column detection...")
            
            detected_col, analysis_details = detect_freeform_text_column(df)
            
            if detected_col:
                target_column = detected_col
                used_auto_detection = True
                
                # Show detailed explanation
                st.info(f"🤖 **Auto-detected column**: `{detected_col}`")
                
                with st.expander("🔍 See detection reasoning and column analysis"):
                    st.markdown("### Column Analysis Results")
                    
                    # Create a comparison table
                    comparison_data = []
                    for col, details in analysis_details.items():
                        comparison_data.append({
                            'Column': col,
                            'Avg Length': details['avg_length'],
                            'Avg Punctuation': details['avg_punctuation'],
                            'Uniqueness': f"{details['uniqueness_ratio']:.1%}",
                            'Responses': details['total_responses'],
                            'Final Score': details['final_score'],
                            'Selected': '✅' if col == detected_col else '❌'
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df = comparison_df.sort_values('Final Score', ascending=False)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    st.markdown("### Why this column was selected:")
                    selected_details = analysis_details[detected_col]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**✅ Positive factors:**")
                        st.write(f"• Average length: {selected_details['avg_length']} characters")
                        st.write(f"• Punctuation usage: {selected_details['avg_punctuation']} marks per response")
                        st.write(f"• {selected_details['name_bonus']}")
                        
                    with col2:
                        st.markdown("**⚠️ Assessment:**")
                        st.write(f"• {selected_details['uniqueness_penalty']}")
                        st.write(f"• Total responses: {selected_details['total_responses']}")
                        st.write(f"• Final score: {selected_details['final_score']}")
                    
                    st.markdown("**🧠 How it works:**")
                    st.markdown("""
                    The tool analyzes each text column using these criteria:
                    - **Length** (40%): Longer responses typically contain more detailed feedback
                    - **Punctuation** (30%): More punctuation suggests natural language 
                    - **Uniqueness** (30%): Higher variety indicates genuine responses vs. dropdown values
                    - **Name bonus**: Columns with 'comment', 'feedback', 'response' etc. get priority
                    """)
                    
            else:
                st.error("❌ Could not auto-detect a suitable text column for analysis.")
                st.write("**Available columns:**", list(df.columns))
                st.write("**Suggestion:** Ensure your CSV has a column with text feedback/comments.")
                st.stop()
        
        # Show preview
        st.subheader("📊 Data Preview")
        st.write(f"**Target column for analysis:** `{target_column}`")
        st.write(f"Total rows: {len(df)}")
        st.dataframe(df.head())
        
        # Filter out empty comments
        non_empty_df = df[df[target_column].notna() & (df[target_column].astype(str).str.strip() != "")]
        st.write(f"Rows with text to analyze: {len(non_empty_df)}")
        
        if len(non_empty_df) == 0:
            st.warning("⚠️ No non-empty text found to analyze in the selected column.")
            st.stop()
        
        # Calculate total characters for cost estimation
        total_chars = sum(len(str(comment)) for comment in non_empty_df[target_column])
        processed_count = len(non_empty_df)
        
        # Show cost estimation BEFORE analysis (only for AI analysis)
        if analysis_type != "Regex only":
            st.subheader("💰 OpenAI API Cost Estimation")
            
            cost_details = estimate_costs(total_chars, processed_count)
            
            # Create columns for cost breakdown
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("**Total Estimated Cost**", f"${cost_details['total_cost']:.4f}")
            with col2:
                st.metric("Input Cost", f"${cost_details['input_cost']:.4f}")
            with col3:
                st.metric("Output Cost", f"${cost_details['output_cost']:.4f}")
            
            # Detailed breakdown
            st.markdown("#### 🔍 Calculation Breakdown")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **💰 GPT-4o Mini Token Pricing**
                - **Input**: $0.15 per 1M tokens
                - **Output**: $0.60 per 1M tokens
                """)
                
            with col2:
                st.markdown(f"""
                **📊 Usage Statistics**
                - **Total Characters**: {cost_details['total_chars']:,}
                - **API Requests**: {cost_details['num_requests']:,}
                - **Input Tokens**: {cost_details['input_tokens']:,.0f}
                - **Output Tokens**: {cost_details['output_tokens']:,.0f}
                """)
            
            # Calculation formula
            with st.expander("📝 How We Calculate Costs"):
                st.markdown(f"""
                **Token Estimation:**
                - Characters to tokens: ~4 characters = 1 token
                - Input tokens = {cost_details['total_chars']:,} chars ÷ 4 = {cost_details['input_tokens']:,.0f} tokens
                - Output tokens = {cost_details['num_requests']:,} requests × 20 avg tokens = {cost_details['output_tokens']:,.0f} tokens
                
                **Cost Calculation:**
                - Input cost = ({cost_details['input_tokens']:,.0f} ÷ 1,000,000) × $0.15 = ${cost_details['input_cost']:.4f}
                - Output cost = ({cost_details['output_tokens']:,.0f} ÷ 1,000,000) × $0.60 = ${cost_details['output_cost']:.4f}
                - **Total = ${cost_details['total_cost']:.4f}**
                
                *Note: This is an estimate. Actual costs may vary slightly based on exact tokenization.*
                """)
        
        # Show estimated processing time
        if analysis_type != "Regex only":
            estimated_time = len(non_empty_df) * 0.5  # Rough estimate
            if 'batch_size' in locals():
                estimated_time = estimated_time / (batch_size / 10)  # Batching reduces time
            st.info(f"⏱️ Estimated processing time: {estimated_time/60:.1f} minutes")
        
        # Confirm analysis
        if st.button("🚀 Start Analysis", type="primary"):
            start_time = time.time()
            
            # Initialize new columns
            df["Regex Categories"] = ""
            df["GPT Categories"] = ""
            df["Total Categories Found"] = 0
            
            category_counter = Counter()
            processed_count = 0
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Get comments to process
            comments_to_process = []
            indices_to_process = []
            
            for i, row in df.iterrows():
                if pd.isna(row[target_column]) or str(row[target_column]).strip() == "":
                    continue
                comments_to_process.append(str(row[target_column]))
                indices_to_process.append(i)
            
            # Process in batches for GPT
            if analysis_type != "Regex only" and len(comments_to_process) > 0:
                batch_size = batch_size if 'batch_size' in locals() else 20
                
                for batch_start in range(0, len(comments_to_process), batch_size):
                    batch_end = min(batch_start + batch_size, len(comments_to_process))
                    batch_comments = comments_to_process[batch_start:batch_end]
                    batch_indices = indices_to_process[batch_start:batch_end]
                    
                    status_text.text(f"Processing batch {batch_start//batch_size + 1}/{(len(comments_to_process)-1)//batch_size + 1}")
                    
                    # Process batch with GPT
                    gpt_results = process_batch_gpt(batch_comments, client)
                    
                    # Process each comment in the batch
                    for j, (comment, idx, gpt_result) in enumerate(zip(batch_comments, batch_indices, gpt_results)):
                        # Regex analysis
                        regex_cats = match_categories(comment) if analysis_type != "AI only" else []
                        
                        # Combine results
                        gpt_cat_list = [cat.strip() for cat in gpt_result.split(",") if gpt_result and cat.strip()]
                        all_unique = list(set(regex_cats) | set(gpt_cat_list))
                        category_counter.update(all_unique)
                        
                        # Update dataframe
                        df.at[idx, "Regex Categories"] = ", ".join(regex_cats)
                        df.at[idx, "GPT Categories"] = gpt_result
                        df.at[idx, "Total Categories Found"] = len(all_unique)
                        
                        processed_count += 1
                        
                        # Update progress
                        progress = processed_count / len(comments_to_process)
                        progress_bar.progress(progress)
                    
                    # Small delay between batches to avoid rate limits
                    time.sleep(0.5)
            
            else:
                # Regex only processing
                for comment, idx in zip(comments_to_process, indices_to_process):
                    regex_cats = match_categories(comment)
                    category_counter.update(regex_cats)
                    
                    df.at[idx, "Regex Categories"] = ", ".join(regex_cats)
                    df.at[idx, "Total Categories Found"] = len(regex_cats)
                    
                    processed_count += 1
                    progress = processed_count / len(comments_to_process)
                    progress_bar.progress(progress)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            progress_bar.progress(1.0)
            status_text.text(f"✅ Analysis complete! Processed in {processing_time:.1f} seconds")
            
            # Display results
            st.subheader("📈 Analysis Results")
            
            # Show which column was analyzed
            if used_auto_detection:
                st.info(f"✨ **Analyzed column:** `{target_column}` (auto-detected)")
            else:
                st.info(f"✨ **Analyzed column:** `{target_column}` (found directly)")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Comments", len(comments_to_process))
            with col2:
                st.metric("Categories Used", len(category_counter))
            with col3:
                avg_cats = df["Total Categories Found"].mean()
                st.metric("Avg Categories/Comment", f"{avg_cats:.1f}")
            with col4:
                st.metric("Processing Time", f"{processing_time:.1f}s")
            
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
            display_cols = [target_column, "Regex Categories", "GPT Categories", "Total Categories Found"]
            sample_df = df[df["Total Categories Found"] > 0].head(5)
            if not sample_df.empty:
                st.dataframe(sample_df[display_cols])
            else:
                st.write("No categorized results to show in sample.")
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.write("Please check your file format and try again.")

# Performance tips
if analysis_type != "Regex only":
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### ⚡ Speed Tips
    - **Increase batch size** for faster processing
    - **Reduce max workers** if you hit rate limits
    - **Regex only** is fastest for large datasets
    - Processing ~100 comments takes 1-2 minutes
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for Twinkl | Contact: james.shamsi@twinkl.co.uk")
