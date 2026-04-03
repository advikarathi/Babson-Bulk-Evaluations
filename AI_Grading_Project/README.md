# Babson College AI Grading Assistant

An intelligent grading system that uses AI to evaluate student responses according to detailed rubrics, providing consistent scoring with detailed rationales for each criterion.

## Features

- 🤖 **AI-Powered Grading**: Uses Azure OpenAI GPT-4 for consistent, rubric-based evaluation
- 📊 **Detailed Feedback**: Provides scores and justifications for each rubric criterion
- 🔄 **Resume Capability**: Automatic checkpointing allows resuming interrupted grading sessions
- 📈 **Progress Tracking**: Real-time progress updates with performance metrics
- ✅ **Quality Assurance**: Built-in validation checks for scoring consistency
- 📝 **Comprehensive Logging**: Detailed logs for troubleshooting and audit trails
- 🎯 **Flexible Input**: Handles various Excel formats and automatically detects response columns

## Quick Start

1. **Double-click `run_grading.bat`** in the project folder
2. **Or run:** `python bulk_grade.py`

The system will guide you through the process interactively!

## File Requirements

- **Input Excel**: Student responses (`.xlsx`)
- **Rubric Document**: Grading criteria (`.docx`)
- **Examples Excel**: Human-graded calibration examples (`.xlsx`)
- **Environment**: `.env` file with Azure OpenAI credentials

## Output

The system generates a comprehensive Excel file with:
- Individual scores for each rubric criterion (A-F)
- Detailed justifications with direct quotes from student responses
- Subtotal and total scores
- Overall grading rationale
- Quality validation flags

## Configuration

Set your Azure OpenAI credentials in `.env`:
```
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-01
```

## Advanced Usage

```bash
# Grade first 10 responses
python bulk_grade.py --limit 10

# Use custom model
python bulk_grade.py --model gpt-4-turbo

# Start from specific row
python bulk_grade.py --start-row 50
```

## Error Handling

The system includes robust error handling:
- Automatic retries for API failures
- Graceful handling of malformed responses
- Checkpoint recovery for interrupted sessions
- Detailed error logging in `grading_log.txt`

## Quality Assurance

- Validates all scores sum correctly
- Ensures justifications include direct quotes
- Checks for required rubric criteria coverage
- Flags potential inconsistencies for manual review

## Performance

- Processes ~2-3 responses per minute
- Automatic rate limiting to respect API limits
- Memory-efficient processing of large datasets
- Progress checkpoints every 5 responses