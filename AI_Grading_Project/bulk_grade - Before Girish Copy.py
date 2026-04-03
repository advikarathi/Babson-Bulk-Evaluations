# bulk_grade.py
#
# What this does:
# 1. Reads your rubric docx
# 2. Reads your human graded calibration examples xlsx
# 3. Reads the student response xlsx
# 4. Calls the model once per response
# 5. Writes a new xlsx with per point scores, justifications, totals, and a final rationale
#
# Install:
#   pip install openai pandas openpyxl python-docx python-dotenv tenacity
#
# Run:
#   python bulk_grade.py \
#       --input "Master_without_Identifiers_5.22.25.xlsx" \
#       --rubric "Revised_Rubric_Aug2025.docx" \
#       --examples "SampleRubric_AI_SES.xlsx" \
#       --output "AI_Innovation_Fellowship_Graded_Output.xlsx" \
#       --limit 25
#
# Environment:
#   export AZURE_OPENAI_ENDPOINT="your_endpoint_here"
#   export AZURE_OPENAI_API_KEY="your_key_here"
#   export AZURE_OPENAI_DEPLOYMENT="your_deployment_name"
#   export AZURE_OPENAI_API_VERSION="2024-02-15-preview"

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from docx import Document
from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("grading_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")


@dataclass
class Paths:
    input_xlsx: str
    rubric_docx: str
    examples_xlsx: str
    output_xlsx: str
    checkpoint_xlsx: str


def read_docx_text(path: str) -> str:
    doc = Document(path)
    parts: List[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def read_examples_as_text(path: str) -> str:
    """
    Turns the human graded example workbook into a compact calibration string.
    """
    df = pd.read_excel(path)

    required_like = [
        "Sample Question",
        "Point A", "Rationale",
        "Point B", "Rationale.1",
        "Point C", "Rationale.2",
        "Point D", "Rationale.3",
        "Point E", "Rationale.4",
        "Point F", "Rationale.5",
    ]
    missing = [c for c in required_like if c not in df.columns]
    if missing:
        raise ValueError(f"Examples file is missing expected columns: {missing}")

    blocks = []
    for i, row in df.iterrows():
        try:
            def safe_int(value):
                if pd.isna(value):
                    return 0
                if isinstance(value, str):
                    value = value.strip()
                    if value.lower() in ("1", "yes", "true", "x"):
                        return 1
                    elif value.lower() in ("0", "no", "false", ""):
                        return 0
                    else:
                        try:
                            return int(float(value))
                        except ValueError:
                            return 0
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return 0

            a = safe_int(row["Point A"])
            b = safe_int(row["Point B"])
            c = safe_int(row["Point C"])
            d = safe_int(row["Point D"])
            e = safe_int(row["Point E"])
            f = safe_int(row["Point F"])
        except Exception as exc:
            raise ValueError(f"Examples row {i+1} has non numeric point values.") from exc

        feedback_total = a + b + c + d
        systems_total = e + f

        blocks.append(
            f"""
EXAMPLE {i+1}
Response:
{str(row["Sample Question"]).strip()}

Human Scores:
A={a}, B={b}, C={c}, D={d}, E={e}, F={f}
Feedback_Total={feedback_total}, Systems_Total={systems_total}

Human Rationales:
A: {str(row["Rationale"]).strip()}
B: {str(row["Rationale.1"]).strip()}
C: {str(row["Rationale.2"]).strip()}
D: {str(row["Rationale.3"]).strip()}
E: {str(row["Rationale.4"]).strip()}
F: {str(row["Rationale.5"]).strip()}
""".strip()
        )

    return "\n\n".join(blocks)


def detect_response_column(df: pd.DataFrame) -> str:
    """
    Picks the response column.
    Prefer columns that look like the Q2 prompt or common response names.
    """
    preferred_patterns = [
        r"^Q2",
        r"response",
        r"answer",
        r"text",
        r"essay",
        r"written",
    ]

    cols = list(df.columns)
    for pattern in preferred_patterns:
        for c in cols:
            if re.search(pattern, str(c), flags=re.IGNORECASE):
                return c

    best_col = None
    best_len = -1.0
    for c in cols:
        series = df[c].astype(str)
        avg_len = series.str.len().mean()
        if avg_len > best_len:
            best_len = avg_len
            best_col = c

    if not best_col:
        raise ValueError("Could not detect response column.")
    return best_col


def build_system_prompt(rubric_text: str, examples_text: str) -> str:
    return f"""
You are an academic grading engine for Babson College.

You evaluate systemic reasoning responses using a structured rubric.

You must behave consistently and align with human grading patterns.

You are not allowed to assign points beyond the rubric text.

CALIBRATION PHASE (MANDATORY)

Before grading:
- Read the Examples document
- Identify how A–F were awarded
- Extract patterns for when points are awarded or not
- Calibration must be internal - do NOT output calibration steps

CALIBRATION RULE (CRITICAL)
Examples are authoritative. If a response matches an example structure, assign the SAME score and use the SAME reasoning style.

INFERENCE RULE
Moderate academic inference is allowed ONLY if it matches example reasoning patterns. Do NOT invent logic or assume unstated relationships.

SCORING STRUCTURE
FEEDBACK (0–4 points total): A, B, C, D
SYSTEMS (0–2 points total): E, F

Each element scored independently.

FEEDBACK RULES
A (1 point): Explicit fish–fisher relationship required
B (1 point): Population change required (increase/decrease)
C (1 point): Second distinct relationship required (beyond B)
D (1 point): Requires feedback over time + long-term consequence + directionality. D requires A+B+C.

SYSTEMS RULES
E (1 point): Broader system mentioned (livelihood, economy, environment, etc.)
F (1 point): Cause-effect explanation required. F requires E.

CALIBRATION CORRECTIONS
- Missing directionality → NO D
- Livelihood mention alone → E only, not F
- No long-term outcome → NO D
- Second relationship required for C
- Do NOT over-award F

EVIDENCE RULE
Missing evidence → score 0
Partial but matches examples → award point
Ambiguous → score 0

JUSTIFICATION REQUIREMENTS (CRITICAL)
For EVERY rubric element (A–F):
- Provide justification EVEN if score = 0
- Include a DIRECT QUOTE from the student response
- Explain clearly why awarded OR why NOT awarded
- Format: "Quoted evidence" → Explanation using rubric logic

Rubric text:
{rubric_text}

Calibration examples:
{examples_text}

Return only valid JSON.
""".strip()


def build_user_prompt(response_text: str, response_id: Any) -> str:
    return f"""
Grade this one student response.

Response_ID: {response_id}

Student response:
\"\"\"
{response_text}
\"\"\"

Return JSON with exactly this shape:

{{
  "Feedback_A_Score": 0 or 1,
  "Feedback_A_Justification": "must include a direct quote",
  "Feedback_B_Score": 0 or 1,
  "Feedback_B_Justification": "must include a direct quote",
  "Feedback_C_Score": 0 or 1,
  "Feedback_C_Justification": "must include a direct quote",
  "Feedback_D_Score": 0 or 1,
  "Feedback_D_Justification": "must include a direct quote",
  "Systems_E_Score": 0 or 1,
  "Systems_E_Justification": "must include a direct quote",
  "Systems_F_Score": 0 or 1,
  "Systems_F_Justification": "must include a direct quote",
  "Total_Feedback_Score": 0 to 4,
  "Total_Systems_Score": 0 to 2,
  "Overall_Total": 0 to 6,
  "Grading_Rationale": "2 to 4 sentences summarizing the row"
}}

Rules:
- A score can only be 0 or 1.
- D requires A=1, B=1, C=1.
- F requires E=1.
- Totals must equal sums.
- Every justification must contain a direct quote from the response in quotation marks.
- Do not omit any field.
- Return JSON only, no markdown.
""".strip()


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_result(result: Dict[str, Any], response_text: str) -> Dict[str, Any]:
    required = [
        "Feedback_A_Score", "Feedback_A_Justification",
        "Feedback_B_Score", "Feedback_B_Justification",
        "Feedback_C_Score", "Feedback_C_Justification",
        "Feedback_D_Score", "Feedback_D_Justification",
        "Systems_E_Score", "Systems_E_Justification",
        "Systems_F_Score", "Systems_F_Justification",
        "Total_Feedback_Score", "Total_Systems_Score", "Overall_Total",
        "Grading_Rationale",
    ]
    missing = [k for k in required if k not in result]
    if missing:
        raise ValueError(f"Missing keys from model output: {missing}")

    for k in [
        "Feedback_A_Score", "Feedback_B_Score", "Feedback_C_Score", "Feedback_D_Score",
        "Systems_E_Score", "Systems_F_Score",
    ]:
        if result[k] not in (0, 1):
            raise ValueError(f"{k} must be 0 or 1, got {result[k]}")

    if result["Feedback_D_Score"] == 1:
        if not (
            result["Feedback_A_Score"] == 1 and
            result["Feedback_B_Score"] == 1 and
            result["Feedback_C_Score"] == 1
        ):
            raise ValueError("D was awarded without A, B, and C.")

    if result["Systems_F_Score"] == 1 and result["Systems_E_Score"] != 1:
        raise ValueError("F was awarded without E.")

    feedback_total = (
        result["Feedback_A_Score"] +
        result["Feedback_B_Score"] +
        result["Feedback_C_Score"] +
        result["Feedback_D_Score"]
    )
    systems_total = result["Systems_E_Score"] + result["Systems_F_Score"]
    overall_total = feedback_total + systems_total

    if result["Total_Feedback_Score"] != feedback_total:
        raise ValueError("Feedback total mismatch.")
    if result["Total_Systems_Score"] != systems_total:
        raise ValueError("Systems total mismatch.")
    if result["Overall_Total"] != overall_total:
        raise ValueError("Overall total mismatch.")

    just_cols = [k for k in result if k.endswith("_Justification")]
    for k in just_cols:
        text = str(result[k]).strip()
        if not text:
            raise ValueError(f"{k} is empty.")
        if '"' not in text and "“" not in text and "”" not in text:
            raise ValueError(f"{k} does not contain a quoted snippet.")

    response_lower = response_text.lower()
    for k in just_cols:
        quotes = re.findall(r'"([^"]+)"', str(result[k]))
        for q in quotes[:2]:
            q = q.strip().lower()
            if len(q) >= 8 and q not in response_lower:
                result[k] += " [quote should be manually spot checked]"

    return result


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
def grade_one(
    client: AzureOpenAI,
    model: str,
    system_prompt: str,
    response_text: str,
    response_id: Any,
) -> Dict[str, Any]:
    user_prompt = build_user_prompt(response_text=response_text, response_id=response_id)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2000,
            temperature=0.1,
        )

        data = extract_json(resp.choices[0].message.content)
        data = validate_result(data, response_text=response_text)
        return data

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed for Response_ID {response_id}: {e}")
        raise ValueError(f"AI returned invalid JSON format: {e}")
    except Exception as e:
        logger.error(f"Grading failed for Response_ID {response_id}: {e}")
        raise


def save_checkpoint(df: pd.DataFrame, path: str) -> None:
    df.to_excel(path, index=False, engine="openpyxl")


def main() -> None:
    print("🤖 Babson College AI Grading Assistant")
    print("=" * 50)
    print(f"📊 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    default_input = "Master_without_Identifiers_5.22.25.xlsx"
    default_rubric = "Revised_Rubric_Aug2025.docx"
    default_examples = "SampleRubric_AI_SES.xlsx"
    default_output = "AI_Innovation_Fellowship_Graded_Output.xlsx"

    print(f"📁 Input file: {default_input}")
    print(f"📋 Rubric file: {default_rubric}")
    print(f"📚 Examples file: {default_examples}")
    print(f"💾 Output file: {default_output}")
    print()

    class Args:
        def __init__(self):
            self.input = default_input
            self.rubric = default_rubric
            self.examples = default_examples
            self.output = default_output
            self.limit = None
            self.start_row = 0
            self.model = DEFAULT_MODEL

    args = Args()

    logger.info(
        f"Starting automated grading session with input: {args.input}, rubric: {args.rubric}, "
        f"examples: {args.examples}, output: {args.output}"
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.input = os.path.join(script_dir, args.input)
    args.rubric = os.path.join(script_dir, args.rubric)
    args.examples = os.path.join(script_dir, args.examples)
    args.output = os.path.join(script_dir, args.output)

    print("Starting bulk grading script...")
    print(f"Script directory: {script_dir}")
    print(f"Input file: {args.input}")
    print(f"Rubric file: {args.rubric}")
    print(f"Examples file: {args.examples}")
    print(f"Output file: {args.output}")

    if not os.path.exists(args.input):
        print(f"ERROR: Input file '{args.input}' does not exist!")
        return
    if not os.path.exists(args.rubric):
        print(f"ERROR: Rubric file '{args.rubric}' does not exist!")
        return
    if not os.path.exists(args.examples):
        print(f"ERROR: Examples file '{args.examples}' does not exist!")
        return

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not api_key:
        raise EnvironmentError("AZURE_OPENAI_API_KEY is not set.")
    if not endpoint:
        raise EnvironmentError("AZURE_OPENAI_ENDPOINT is not set.")
    if not deployment:
        raise EnvironmentError("AZURE_OPENAI_DEPLOYMENT is not set.")
    if not api_version:
        raise EnvironmentError("AZURE_OPENAI_API_VERSION is not set.")

    paths = Paths(
        input_xlsx=args.input,
        rubric_docx=args.rubric,
        examples_xlsx=args.examples,
        output_xlsx=args.output,
        checkpoint_xlsx=os.path.splitext(args.output)[0] + "_checkpoint.xlsx",
    )

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )

    rubric_text = read_docx_text(paths.rubric_docx)
    examples_text = read_examples_as_text(paths.examples_xlsx)
    system_prompt = build_system_prompt(rubric_text, examples_text)

    df = pd.read_excel(paths.input_xlsx)
    response_col = detect_response_column(df)

    if "Response_ID" not in df.columns:
        df["Response_ID"] = range(1, len(df) + 1)

    grading_cols = [
        "Feedback_A_Score", "Feedback_A_Justification",
        "Feedback_B_Score", "Feedback_B_Justification",
        "Feedback_C_Score", "Feedback_C_Justification",
        "Feedback_D_Score", "Feedback_D_Justification",
        "Systems_E_Score", "Systems_E_Justification",
        "Systems_F_Score", "Systems_F_Justification",
        "Total_Feedback_Score", "Total_Systems_Score", "Overall_Total",
        "Grading_Rationale",
    ]
    for col in grading_cols:
        if col not in df.columns:
            df[col] = None

    end_row = len(df) if args.limit is None else min(len(df), args.start_row + args.limit)

    print(f"📊 Total responses to grade: {end_row - args.start_row}")
    print(f"🎯 Using response column: '{response_col}'")
    print("🚀 Starting grading process...")
    print("-" * 50)

    successful_gradings = 0
    failed_gradings = 0
    start_time = time.time()

    for i in range(args.start_row, end_row):
        response_id = df.at[i, "Response_ID"]
        response_text = str(df.at[i, response_col]).strip()

        if not response_text or response_text.lower() == "nan":
            logger.warning(f"Skipping row {i} (Response_ID={response_id}): empty response")
            print(f"⏭️  Skipping row {i}: empty response")
            continue

        try:
            print(f"🔄 Grading row {i+1}/{end_row} | Response_ID={response_id}")
            logger.info(f"Starting grading for row {i}, Response_ID={response_id}")

            result = grade_one(
                client=client,
                model=deployment,
                system_prompt=system_prompt,
                response_text=response_text,
                response_id=response_id,
            )

            for k, v in result.items():
                df.at[i, k] = v

            successful_gradings += 1
            print(f"✅ Completed row {i+1}/{end_row}")

        except Exception as e:
            failed_gradings += 1
            error_msg = f"Failed to grade row {i} (Response_ID={response_id}): {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            df.at[i, "Grading_Rationale"] = f"ERROR: {str(e)}"
            continue

        if (i + 1) % 5 == 0:
            save_checkpoint(df, paths.checkpoint_xlsx)
            elapsed = time.time() - start_time
            rate = (i + 1 - args.start_row) / elapsed if elapsed > 0 else 0
            eta_minutes = ((end_row - (i + 1)) / rate) / 60 if rate > 0 else 0
            print(
                f"💾 Checkpoint saved at response {i+1}/{end_row} | "
                f"Rate: {rate:.1f} resp/min | ETA: {eta_minutes:.1f} min"
            )

        time.sleep(0.3)

    save_checkpoint(df, paths.checkpoint_xlsx)

    total_time = time.time() - start_time
    total_processed = successful_gradings + failed_gradings

    print("\n" + "=" * 60)
    print("🎉 GRADING COMPLETED!")
    print("=" * 60)
    print(f"📊 Total responses processed: {total_processed}")
    print(f"✅ Successfully graded: {successful_gradings}")
    print(f"❌ Failed to grade: {failed_gradings}")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    if total_time > 0:
        print(f"📈 Average rate: {total_processed / total_time:.2f} responses/second")
    print(f"💾 Checkpoint saved: {paths.checkpoint_xlsx}")
    print(f"📄 Final output: {paths.output_xlsx}")

    print("\n🔍 Running quality checks...")
    validation_errors = []

    for i in range(args.start_row, end_row):
        if pd.isna(df.at[i, "Overall_Total"]) and not str(df.at[i, "Grading_Rationale"]).startswith("ERROR"):
            validation_errors.append(f"Row {i}: Missing grades")

        if not pd.isna(df.at[i, "Overall_Total"]):
            try:
                ft = (
                    int(df.at[i, "Feedback_A_Score"]) +
                    int(df.at[i, "Feedback_B_Score"]) +
                    int(df.at[i, "Feedback_C_Score"]) +
                    int(df.at[i, "Feedback_D_Score"])
                )
                st = int(df.at[i, "Systems_E_Score"]) + int(df.at[i, "Systems_F_Score"])
                ot = ft + st

                if int(df.at[i, "Total_Feedback_Score"]) != ft:
                    validation_errors.append(f"Row {i}: Feedback total mismatch")
                if int(df.at[i, "Total_Systems_Score"]) != st:
                    validation_errors.append(f"Row {i}: Systems total mismatch")
                if int(df.at[i, "Overall_Total"]) != ot:
                    validation_errors.append(f"Row {i}: Overall total mismatch")
            except (ValueError, TypeError):
                validation_errors.append(f"Row {i}: Invalid score format")

    if validation_errors:
        print(f"⚠️  Found {len(validation_errors)} validation issues:")
        for error in validation_errors[:5]:
            print(f"   - {error}")
        if len(validation_errors) > 5:
            print(f"   ... and {len(validation_errors) - 5} more")
        logger.warning(f"Validation issues found: {validation_errors}")
    else:
        print("✅ All quality checks passed!")

    df.to_excel(paths.output_xlsx, index=False, engine="openpyxl")
    logger.info(f"Final output saved to {paths.output_xlsx}")

    print(f"\n💾 Final results saved to: {paths.output_xlsx}")
    print("🎊 Grading session complete!")
    print(f"Checkpoint saved at {paths.checkpoint_xlsx}")
if __name__ == "__main__":
    main()