# =============================================================================
# AI BULK GRADER - General Purpose Streamlit App
# =============================================================================
# HOW TO RUN:
#   1. pip install streamlit pandas openai openpyxl tenacity
#   2. Save this file as grader_app.py
#   3. In your terminal: python -m streamlit run grader_app.py
# =============================================================================

import json
import os
import uuid
import pandas as pd
import streamlit as st
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai

# --------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# --------------------------------------------------------------------------

PROGRESS_LOG_FILE = "progress_log.json"
CHECKPOINT_EVERY  = 10

# --------------------------------------------------------------------------
# SYSTEM PROMPT — max score is explicitly injected, never left to the AI
# --------------------------------------------------------------------------

def build_system_prompt(rubric: str, max_score: int) -> str:
    return f"""You are a professor's grading assistant. Your job is to grade student responses consistently and fairly.

RUBRIC:
{rubric}

=== SCORING RULES ===
- The MAXIMUM possible score for this assignment is {max_score} points. This is fixed and non-negotiable.
- Every single response must be scored out of {max_score}. No exceptions.
- The "score" field in your output MUST always be formatted as X/{max_score} where X is between 0 and {max_score}.
- Do NOT invent a different maximum. Do NOT split the score into sub-sections. Always return one single score out of {max_score}.

=== GRADING STEPS ===
Step 1 — Read the rubric criteria carefully.
Step 2 — Read the student response.
Step 3 — Determine which rubric criteria are met by the response.
Step 4 — Add up the points earned. The total cannot exceed {max_score}.

=== STRICT OUTPUT FORMAT ===
Return ONLY a JSON object with EXACTLY these three keys — no other text, no markdown fences, no preamble:
{{
  "score": "<points_earned>/{max_score}",
  "points_awarded": "<list the specific rubric criteria or labels the student earned, e.g. 'A, B, C' or '1, 2, 4' — use a blank string if the rubric has no labeled criteria>",
  "justification": "<one concise paragraph explaining which criteria were met and which were not, referencing specific language from the student response>"
}}

REMINDERS:
- "score" must ALWAYS be X/{max_score}. For example: "0/{max_score}", "3/{max_score}", "{max_score}/{max_score}".
- If the student response is blank or gibberish, return "0/{max_score}".
- If the rubric awards points sequentially (must earn A before B, etc.), respect that ordering.
"""

# --------------------------------------------------------------------------
# PRIVACY: Build anonymized mapping — names/IDs never sent to AI
# --------------------------------------------------------------------------

def build_privacy_map(df: pd.DataFrame) -> dict:
    mapping = {}
    for idx, row in df.iterrows():
        temp_id = f"ROW_{idx}_{uuid.uuid4().hex[:6]}"
        mapping[temp_id] = {
            "original_index":   idx,
            "Student_Name":     str(row.get("Student_Name", "")),
            "Student_ID":       str(row.get("Student_ID", "")),
            "Student_Response": str(row.get("Student_Response", "")),
        }
    return mapping

# --------------------------------------------------------------------------
# CHECKPOINT HELPERS
# --------------------------------------------------------------------------

def load_progress_log() -> dict:
    if os.path.exists(PROGRESS_LOG_FILE):
        with open(PROGRESS_LOG_FILE, "r") as f:
            return json.load(f)
    return {}

def save_progress_log(log: dict):
    with open(PROGRESS_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

# --------------------------------------------------------------------------
# API CALL with retry + exponential backoff for rate limits
# --------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type(openai.RateLimitError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
def grade_single_response(client, deployment, system_prompt, response_text, max_score):
    """
    Sends one student response to the AI.
    Returns a dict with score, points_awarded, and justification.
    """
    user_prompt = (
        f"Grade the following student response using the rubric provided. "
        f"Remember: score MUST be out of {max_score}.\n\n"
        f"STUDENT RESPONSE:\n{response_text}\n\n"
        "Return ONLY the JSON object — no preamble, no markdown fences."
    )

    completion = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=500,
    )

    raw = completion.choices[0].message.content.strip()

    # Strip markdown fences if the model accidentally added them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        result = json.loads(raw)
        # Safety net: force correct format even if AI ignored instructions
        score = str(result.get("score", f"ERR/{max_score}"))
        if "/" not in score or not score.endswith(f"/{max_score}"):
            # Try to extract just the numerator and reformat
            try:
                numerator = int(score.split("/")[0].strip())
                numerator = max(0, min(numerator, max_score))  # clamp to valid range
                score = f"{numerator}/{max_score}"
            except Exception:
                score = f"ERR/{max_score}"

        return {
            "score":          score,
            "points_awarded": str(result.get("points_awarded", "")),
            "justification":  str(result.get("justification", "No justification provided.")),
        }
    except json.JSONDecodeError:
        return {
            "score":          f"PARSE_ERROR/{max_score}",
            "points_awarded": "",
            "justification":  f"AI returned non-JSON output: {raw[:300]}",
        }

# --------------------------------------------------------------------------
# MAIN GRADING LOOP
# --------------------------------------------------------------------------

def run_grading(df, rubric, max_score, client, deployment, output_path, progress_bar, status_text):
    """
    Iterates over all rows, skips already-graded ones (resume support),
    calls the AI, and saves results to Excel every CHECKPOINT_EVERY rows.
    """
    privacy_map   = build_privacy_map(df)
    progress_log  = load_progress_log()
    system_prompt = build_system_prompt(rubric, max_score)

    # Add output columns if not present
    for col in ["Score", "Points_Awarded", "Justification"]:
        if col not in df.columns:
            df[col] = ""

    total_rows   = len(privacy_map)
    graded_count = sum(1 for v in progress_log.values() if "score" in v)
    rows_list    = list(privacy_map.items())

    for i, (temp_id, data) in enumerate(rows_list):

        # Skip already-graded rows
        if temp_id in progress_log and "score" in progress_log[temp_id]:
            orig_idx = data["original_index"]
            r = progress_log[temp_id]
            df.at[orig_idx, "Score"]          = r.get("score", "")
            df.at[orig_idx, "Points_Awarded"] = r.get("points_awarded", "")
            df.at[orig_idx, "Justification"]  = r.get("justification", "")
            continue

        rows_remaining = total_rows - graded_count
        status_text.markdown(
            f"⏳ **Grading row {graded_count + 1} of {total_rows}** — {rows_remaining} remaining"
        )
        progress_bar.progress(graded_count / total_rows)

        try:
            result = grade_single_response(
                client, deployment, system_prompt, data["Student_Response"], max_score
            )
        except Exception as e:
            result = {
                "score":          f"API_ERROR/{max_score}",
                "points_awarded": "",
                "justification":  str(e)[:300],
            }

        orig_idx = data["original_index"]
        df.at[orig_idx, "Score"]          = result["score"]
        df.at[orig_idx, "Points_Awarded"] = result["points_awarded"]
        df.at[orig_idx, "Justification"]  = result["justification"]

        progress_log[temp_id] = result
        save_progress_log(progress_log)
        graded_count += 1

        if graded_count % CHECKPOINT_EVERY == 0:
            df.to_excel(output_path, index=False)

    df.to_excel(output_path, index=False)
    progress_bar.progress(1.0)
    status_text.markdown(f"✅ **All {total_rows} rows graded!**")

    if os.path.exists(PROGRESS_LOG_FILE):
        os.remove(PROGRESS_LOG_FILE)

    return df

# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(
    page_title="AI Bulk Grader",
    page_icon="📝",
    layout="wide",
)

st.title("📝 AI Bulk Grader")
st.caption(
    "Upload your student responses and rubric, set the maximum score, "
    "configure your API credentials, then click **Start Grading**."
)
st.divider()

# =============================================================================
# SIDEBAR: API Configuration
# =============================================================================

with st.sidebar:
    st.header("🔑 API Configuration")
    st.caption("Credentials are used only for this session and never stored.")

    api_key     = st.text_input("API Key",         type="password", placeholder="sk-...")
    base_url    = st.text_input("Base URL",        placeholder="https://YOUR-RESOURCE.openai.azure.com/")
    api_version = st.text_input("API Version",     value="2024-02-01")
    deployment  = st.text_input("Deployment Name", placeholder="gpt-4o")

    st.divider()
    st.subheader("📂 Output Settings")
    output_filename = st.text_input("Output File Name", value="graded_results.xlsx")

    st.divider()
    st.markdown(
        "**Output columns added to your Excel:**\n"
        "- `Score` — always X/max (e.g. 7/10)\n"
        "- `Points_Awarded` — rubric criteria met (e.g. A, B, C)\n"
        "- `Justification` — AI reasoning\n\n"
        "**Required columns in your Excel file:**\n"
        "- `Student_Name`\n"
        "- `Student_ID`\n"
        "- `Student_Response`"
    )

# =============================================================================
# MAIN AREA: File upload + Rubric + Max Score
# =============================================================================

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1️⃣  Upload Student Responses")
    uploaded_xlsx = st.file_uploader(
        "Excel file (.xlsx) with Student_Name, Student_ID, Student_Response",
        type=["xlsx"],
    )

    if uploaded_xlsx:
        try:
            preview_df = pd.read_excel(uploaded_xlsx)
            required_cols = {"Student_Name", "Student_ID", "Student_Response"}
            missing = required_cols - set(preview_df.columns)
            if missing:
                st.error(f"❌ Missing columns: {', '.join(missing)}")
            else:
                st.success(f"✅ Loaded **{len(preview_df)}** rows")
                st.dataframe(preview_df.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read file: {e}")

    # Max score input — placed here so it's next to the file upload
    st.subheader("2️⃣  Set the Maximum Score")
    max_score = st.number_input(
        "What is the maximum possible score for this assignment?",
        min_value=1,
        max_value=1000,
        value=10,
        step=1,
        help="Every student will be scored out of this number. E.g. enter 6 for a 6-point rubric, 100 for a 100-point rubric."
    )
    st.caption(f"All scores will be returned as **X/{max_score}**")

with col_right:
    st.subheader("3️⃣  Provide the Rubric")
    rubric_mode = st.radio("Input method", ["Paste text", "Upload .txt file"], horizontal=True)

    rubric_text = ""
    if rubric_mode == "Paste text":
        rubric_text = st.text_area(
            "Paste your rubric here",
            height=260,
            placeholder=(
                "Describe your grading criteria and point values here...\n\n"
                "Example:\n"
                "- Point A (1pt): Student identifies X\n"
                "- Point B (1pt): Student explains Y\n"
                "- Point C (1pt): Student connects X to Y\n"
                "Total: 3 points"
            )
        )
    else:
        rubric_file = st.file_uploader("Upload .txt rubric", type=["txt"])
        if rubric_file:
            rubric_text = rubric_file.read().decode("utf-8")
            st.success("✅ Rubric loaded")
            with st.expander("Preview rubric"):
                st.text(rubric_text[:1000])

# =============================================================================
# START / RESUME BUTTON
# =============================================================================

st.divider()

resume_available = os.path.exists(PROGRESS_LOG_FILE)
btn_label = "▶️  Resume Grading" if resume_available else "🚀  Start Grading"

if resume_available:
    existing_log = load_progress_log()
    st.info(
        f"🔄 A previous run was interrupted. "
        f"**{len(existing_log)} rows** already graded — click **Resume Grading** to continue."
    )

start_btn = st.button(btn_label, type="primary", use_container_width=True)

# =============================================================================
# GRADING EXECUTION
# =============================================================================

if start_btn:

    errors = []
    if not uploaded_xlsx:       errors.append("Please upload an Excel file.")
    if not rubric_text.strip(): errors.append("Please provide a rubric.")
    if not api_key.strip():     errors.append("API Key is required.")
    if not base_url.strip():    errors.append("Base URL is required.")
    if not deployment.strip():  errors.append("Deployment Name is required.")

    if errors:
        for err in errors:
            st.error(err)
        st.stop()

    uploaded_xlsx.seek(0)
    df = pd.read_excel(uploaded_xlsx)

    try:
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=base_url,
            api_version=api_version,
        )
    except Exception as e:
        st.error(f"Failed to create API client: {e}")
        st.stop()

    st.subheader("📊 Grading Progress")
    progress_bar = st.progress(0)
    status_text  = st.empty()
    status_text.markdown("⏳ Starting…")

    output_path = output_filename if output_filename.endswith(".xlsx") else output_filename + ".xlsx"

    try:
        final_df = run_grading(
            df           = df,
            rubric       = rubric_text,
            max_score    = int(max_score),
            client       = client,
            deployment   = deployment,
            output_path  = output_path,
            progress_bar = progress_bar,
            status_text  = status_text,
        )
    except openai.AuthenticationError:
        st.error("❌ Authentication failed. Check your API Key and Base URL.")
        st.stop()
    except openai.RateLimitError:
        st.error("❌ Rate limit exceeded. Wait a few minutes then click Resume.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.stop()

    st.balloons()
    st.success("🎉 Grading complete!")

    # Score summary
    st.subheader("📊 Score Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Score Distribution**")
        st.dataframe(
            final_df["Score"].value_counts().sort_index().rename("Count"),
            use_container_width=True,
        )

    with col2:
        st.markdown("**Class Statistics**")
        try:
            earned = final_df["Score"].str.extract(r"^(\d+(?:\.\d+)?)\/")[0]
            earned = pd.to_numeric(earned, errors="coerce")
            avg_score = earned.mean()
            avg_pct   = (earned / int(max_score) * 100).mean()
            st.metric("Average Score",      f"{avg_score:.1f} / {int(max_score)}")
            st.metric("Average Percentage", f"{avg_pct:.1f}%")
            st.metric("Highest Score",      f"{earned.max():.0f} / {int(max_score)}")
            st.metric("Lowest Score",       f"{earned.min():.0f} / {int(max_score)}")
        except Exception:
            st.caption("Could not compute statistics.")

    st.subheader("📋 Full Results")
    st.dataframe(final_df, use_container_width=True)

    with open(output_path, "rb") as f:
        file_bytes = f.read()

    st.download_button(
        label     = "⬇️  Download Graded Excel",
        data      = file_bytes,
        file_name = output_path,
        mime      = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
