# Harbor Tasks Automated Review System

Automated review system for Harbor Tasks verification that evaluates tasks against 17 checklist criteria and performs agent trajectory root cause analysis.

## Overview

This system automatically evaluates Harbor Tasks using LLM-based evaluation following the proven prompt structure from `sr-code-swe-internal` Step 3. It processes tasks in parallel and outputs results to CSV format with `input_` prefixed columns.

## Architecture

The system uses the proven architecture from `sr-code-swe-internal/phase1/step3_llm_evaluator/` with adaptations for Harbor Tasks:

- **Parallelized LLM calls** - Processes multiple tasks and criteria concurrently
- **One prompt per criterion** - 19 prompts total (17 checklist + 2 root cause)
- **Configurable LLM** - Defaults to Gemini 2.5, configurable via `configs/llm_caller.yaml`
- **CSV output** - Results include all input fields with `input_` prefix

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export GOOGLE_API_KEY="your-api-key"  # For Gemini
# Or set ANTHROPIC_API_KEY / OPENAI_API_KEY for other providers
```

## Usage

Run the evaluation:

```bash
python -m harbor_reviewer.harbor_runner \
    --tasks-harbor main/tasks-harbor \
    --agent-outputs main/11-12-2025-HarborTasks_batch_116_outputs/11-12-2025-HarborTasks \
    --output results.csv \
    --max-parallel 10
```

### Arguments

- `--tasks-harbor`: Path to tasks-harbor folder
- `--agent-outputs`: Path to agent outputs folder
- `--output`: Output CSV file path (default: `harbor_evaluation_results_TIMESTAMP.csv`)
- `--criteria`: List of criteria to evaluate (default: all 19)
- `--max-parallel`: Maximum parallel evaluations (default: 10)
- `--config-dir`: Configuration directory (default: `configs`)
- `--run-id`: Custom run identifier
- `--max-results`: Maximum number of tasks to evaluate

## Evaluation Criteria

### Prompt Quality (4)
1. `prompt_is_clear_and_unambiguous`
2. `prompt_defines_all_success_criteria`
3. `prompt_has_no_solution_leakage`
4. `prompt_does_not_require_runtime_internet`

### Prompt-Tests Alignment (3)
5. `tests_cover_all_prompt_requirements`
6. `tests_do_not_add_hidden_requirements`
7. `tests_validate_only_explicit_success_criteria`

### Test Quality & Robustness (5)
8. `tests_validate_final_state_not_method`
9. `tests_verify_real_correctness_not_superficial_checks`
10. `tests_use_stable_ground_truth`
11. `tests_are_not_easily_bypassable`
12. `tests_are_well_scoped_and_non_redundant`

### Environment & Dependencies (3)
13. `dependencies_are_pinned`
14. `dependencies_cover_all_prompt_requirements`
15. `mock_data_cover_all_cases_to_be_tested`

### Solution (2)
16. `solution_does_not_depend_on_runtime_internet`
17. `solution_passes_all_tests`

### Root Cause Analysis (2)
18. `root_cause_category`
19. `root_cause_summary`

## Output Format

The CSV output includes:
- All input fields prefixed with `input_` (e.g., `input_task_name`, `input_instruction`)
- Each criterion with `_result` and `_justification` columns
- `root_cause_category` and `root_cause_summary`
- `manual_review_notes` (auto-generated if any check fails)
- Evaluation metadata (model_used, timestamp, tokens, cost, etc.)

## Configuration

### LLM Configuration (`configs/llm_caller.yaml`)

Configure the LLM provider and model. Default is Gemini 2.5:

```yaml
llm:
  primary_model: "gemini-2.5-pro"
  models:
    gemini-2.5-pro:
      provider: "google"
      api_key_env: "GOOGLE_API_KEY"
      temperature: 0.1
      max_tokens: 8192
```

### Prompts (`configs/prompts_templates.yaml`)

All 19 prompts follow the proven Step 3 structure with:
- `system_instruction`: Clear guidance with CRITICAL scoring rules
- `task_description`: What to evaluate
- `context_format`: Template with placeholders
- `response_format`: JSON structure

### Criteria (`configs/criteria.yaml`)

Defines rubrics and scoring for each criterion.

## Project Structure

```
.
├── configs/
│   ├── llm_caller.yaml          # LLM configuration
│   ├── prompts_templates.yaml   # 19 evaluation prompts
│   └── criteria.yaml            # Criteria definitions
├── harbor_reviewer/
│   ├── harbor_input_loader.py   # Load Harbor task data
│   ├── harbor_response_parser.py # Parse pass/fail responses
│   ├── harbor_result_formatter.py # Format CSV output
│   ├── harbor_runner.py         # CLI entry point
│   └── [copied architecture files from Step 3]
├── shared/                       # Shared utilities
└── requirements.txt
```

## Notes

- The system uses the proven prompt structure from Step 3, adapted for Harbor criteria
- All prompts follow the same pattern for consistency and high success rate
- Parallelization follows the Step 3 `evaluate_batch` pattern
- Results are formatted with `input_` prefix to match Step 3 output format

