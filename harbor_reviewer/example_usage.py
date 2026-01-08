"""
Example usage of the LLM Caller
----------------------------------
This script demonstrates how to use the evaluator with sample data.
"""

import asyncio
import json

from .evaluation_engine import evaluate_multiple_records, evaluate_single_record
from .llm_caller import LLMCaller


async def example_single_evaluation():
    """Example of evaluating a single criterion"""
    print("=== Single Criterion Evaluation Example ===")

    # Sample context data
    context = {
        "repo_name": "microsoft/vscode",
        "repo_stars": 150000,
        "repo_language": "TypeScript",
        "repo_description": "Visual Studio Code is a code editor redefined and optimized for building and debugging modern web and cloud applications.",
        "pr_title": "Fix memory leak in extension host",
        "pr_body": "This PR addresses a memory leak in the extension host that was causing the editor to consume excessive memory over time.",
        "pr_changed_files": 5,
        "pr_additions": 150,
        "pr_deletions": 50,
    }

    # Initialize evaluator
    evaluator = LLMCaller("configs")

    try:
        # Evaluate a single criterion
        response = await evaluator.evaluate_single_criterion(
            criterion="repo_realism", context=context, record_id="example_001"
        )

        print(f"Criterion: {response.criterion}")
        print(f"Model used: {response.model_used}")
        print(f"Latency: {response.latency_ms:.2f}ms")
        print(f"Raw response: {response.raw_response}")

        if response.parsed_response:
            print(f"Parsed response: {json.dumps(response.parsed_response, indent=2)}")

        if response.error:
            print(f"Error: {response.error}")

    finally:
        await evaluator.close()


async def example_all_criteria():
    """Example of evaluating all criteria for a single record"""
    print("\n=== All Criteria Evaluation Example ===")

    # Sample context data
    context = {
        "repo_name": "facebook/react",
        "repo_stars": 200000,
        "repo_language": "JavaScript",
        "repo_description": "The library for web and native user interfaces",
        "pr_title": "Optimize component re-rendering performance",
        "pr_body": "This PR implements a new optimization algorithm that reduces unnecessary re-renders by 40% in large component trees.",
        "issue_title": "Performance regression in component updates",
        "issue_body": "Users are reporting significant performance degradation when updating components with large prop changes.",
        "pr_url": "https://github.com/facebook/react/pull/12345",
        "pr_diff_url": "https://github.com/facebook/react/pull/12345.diff",
        "pr_patch_url": "https://github.com/facebook/react/pull/12345.patch",
        "pr_changed_files": 8,
        "pr_additions": 300,
        "pr_deletions": 120,
        "pr_test_file_count": 3,
    }

    # Evaluate all criteria
    results = await evaluate_single_record(
        record_data=context, record_id="example_002", config_dir="configs"
    )

    print(f"Evaluated {len(results.criterion_results)} criteria:")
    for criterion, response in results.criterion_results.items():
        print(f"\n{criterion}:")
        print(f"  Score: {response.score}")
        print(f"  Justification: {response.justification[:100]}...")
        if not response.is_valid():
            print(f"  Error: {response.parsing_errors}")


async def example_batch_evaluation():
    """Example of evaluating multiple records in batch"""
    print("\n=== Batch Evaluation Example ===")

    # Sample dataset
    dataset = [
        {
            "record_id": "batch_001",
            "repo_name": "tensorflow/tensorflow",
            "repo_stars": 180000,
            "repo_language": "Python",
            "pr_title": "Fix gradient computation in custom layers",
            "pr_body": "This PR fixes a bug in gradient computation for custom layers that was causing training to fail.",
            "pr_changed_files": 3,
            "pr_additions": 80,
            "pr_deletions": 20,
        },
        {
            "record_id": "batch_002",
            "repo_name": "kubernetes/kubernetes",
            "repo_stars": 100000,
            "repo_language": "Go",
            "pr_title": "Add support for custom resource validation",
            "pr_body": "This PR adds comprehensive validation support for custom resources in the API server.",
            "pr_changed_files": 12,
            "pr_additions": 500,
            "pr_deletions": 100,
        },
    ]

    # Evaluate dataset
    results = await evaluate_multiple_records(
        records=dataset,
        criteria=[
            "repo_realism",
            "problem_clarity",
        ],  # Only evaluate 2 criteria for speed
        config_dir="configs",
    )

    print(f"Evaluated {len(results)} records:")
    for i, record_results in enumerate(results):
        print(f"\nRecord {i+1} ({record_results.record_id}):")
        for criterion, response in record_results.criterion_results.items():
            print(f"  {criterion}:")
            print(f"    Score: {response.score}")
            print(f"    Justification: {response.justification[:100]}...")
            if not response.is_valid():
                print(f"    Error: {response.parsing_errors}")


async def example_error_handling():
    """Example of error handling and monitoring"""
    print("\n=== Error Handling Example ===")

    evaluator = LLMCaller("configs")

    try:
        # Perform evaluation with potential for errors

        # Use invalid model to demonstrate error handling
        response = await evaluator.call_llm_single(
            prompt={"system": "You are an evaluator", "user": "Evaluate repo_realism"},
            model_name="invalid-model",
            request_id="error_demo",
        )

        if response.error:
            print(f"Error occurred: {response.error}")
            print(f"Error code: {response.error_code}")
            print(f"Retry count: {response.retry_count}")
            if response.error_details:
                print(f"Error details: {response.error_details}")
        else:
            print(f"Response: {response.raw_response}")
            print(f"Latency: {response.latency_ms:.2f}ms")

    finally:
        await evaluator.close()


async def main():
    """Run all examples"""
    print("LLM Caller Examples")
    print("=" * 50)

    try:
        await example_single_evaluation()
        await example_all_criteria()
        await example_batch_evaluation()
        await example_error_handling()

    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set the required environment variables:")
        print("- GOOGLE_API_KEY (for Gemini)")
        print("- OPENAI_API_KEY (for GPT-4)")
        print("- ANTHROPIC_API_KEY (for Claude)")


if __name__ == "__main__":
    asyncio.run(main())
