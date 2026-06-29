import sys
import math
import json
from typing import Tuple, List

from pydantic import BaseModel, Field
from litellm import completion
from dotenv import load_dotenv

from evaluation.test import TestQuestion, load_tests
from implementation.answer import answer_question, fetch_context


# Load environment variables
load_dotenv(override=True)

# Model Configuration
MODEL = "ollama/llama3.2"
DB_NAME = "vector_db"


# =========================
# Retrieval Evaluation Model
# =========================
class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr: float = Field(
        description="Mean Reciprocal Rank - average across all keywords"
    )

    ndcg: float = Field(
        description="Normalized Discounted Cumulative Gain (binary relevance)"
    )

    keywords_found: int = Field(
        description="Number of keywords found in top-k results"
    )

    total_keywords: int = Field(
        description="Total number of keywords to find"
    )

    keyword_coverage: float = Field(
        description="Percentage of keywords found"
    )


# =========================
# Answer Evaluation Model
# =========================
class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(
        description=(
            "Concise feedback on the answer quality, "
            "comparing it to the reference answer "
            "and evaluating based on the retrieved context"
        )
    )

    accuracy: float = Field(
        description=(
            "How factually correct is the answer compared "
            "to the reference answer? "
            "1 (wrong) to 5 (perfect)"
        )
    )

    completeness: float = Field(
        description=(
            "How complete is the answer in addressing "
            "all aspects of the question?"
        )
    )

    relevance: float = Field(
        description=(
            "How relevant is the answer to the specific question?"
        )
    )


# =========================
# Retrieval Metrics
# =========================
def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    """
    Calculate reciprocal rank for a single keyword.
    Case-insensitive search.
    """

    keyword_lower = keyword.lower()

    for rank, doc in enumerate(retrieved_docs, start=1):

        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank

    return 0.0


def calculate_dcg(relevances: List[int], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain.
    """

    dcg = 0.0

    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)

    return dcg


def calculate_ndcg(
    keyword: str,
    retrieved_docs: list,
    k: int = 10
) -> float:
    """
    Calculate nDCG using binary relevance.
    """

    keyword_lower = keyword.lower()

    relevances = [
        1 if keyword_lower in doc.page_content.lower() else 0
        for doc in retrieved_docs[:k]
    ]

    dcg = calculate_dcg(relevances, k)

    ideal_relevances = sorted(relevances, reverse=True)

    idcg = calculate_dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


# =========================
# Retrieval Evaluation
# =========================
def evaluate_retrieval(
    test: TestQuestion,
    k: int = 10
) -> RetrievalEval:
    """
    Evaluate retrieval performance.
    """

    retrieved_docs = fetch_context(test.question)

    # MRR
    mrr_scores = [
        calculate_mrr(keyword, retrieved_docs)
        for keyword in test.keywords
    ]

    avg_mrr = (
        sum(mrr_scores) / len(mrr_scores)
        if mrr_scores else 0.0
    )

    # nDCG
    ndcg_scores = [
        calculate_ndcg(keyword, retrieved_docs, k)
        for keyword in test.keywords
    ]

    avg_ndcg = (
        sum(ndcg_scores) / len(ndcg_scores)
        if ndcg_scores else 0.0
    )

    # Keyword Coverage
    keywords_found = sum(
        1 for score in mrr_scores if score > 0
    )

    total_keywords = len(test.keywords)

    keyword_coverage = (
        (keywords_found / total_keywords) * 100
        if total_keywords > 0 else 0.0
    )

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


# =========================
# Answer Evaluation
# =========================
def evaluate_answer(
    test: TestQuestion
) -> Tuple[AnswerEval, str, list]:
    """
    Evaluate generated answer using LLM-as-a-judge.
    """

    # Generate Answer
    generated_answer, retrieved_docs = answer_question(test.question)

    # Judge Prompt
    judge_messages = [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator assessing "
                "the quality of answers.\n"
                "Return ONLY valid JSON."
            ),
        },
        {
            "role": "user",
            "content": f"""
Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Evaluate the generated answer.

Scoring:
- Accuracy: 1 to 5
- Completeness: 1 to 5
- Relevance: 1 to 5

Rules:
- If the answer is factually wrong, accuracy MUST be 1.
- Only give 5 if the answer is perfect.

Return JSON in this format:

{{
    "feedback": "...",
    "accuracy": 4,
    "completeness": 4,
    "relevance": 5
}}
"""
        },
    ]

    # Call LLM
    judge_response = completion(
        model=MODEL,
        messages=judge_messages,
    )

    # Parse JSON Output
    content = judge_response.choices[0].message.content

    try:
        parsed_json = json.loads(content)

    except json.JSONDecodeError:
        print("\nERROR: Invalid JSON returned by model\n")
        print(content)
        sys.exit(1)

    answer_eval = AnswerEval.model_validate(parsed_json)

    return answer_eval, generated_answer, retrieved_docs


# =========================
# Evaluate All Retrieval Tests
# =========================
def evaluate_all_retrieval():
    """
    Evaluate all retrieval tests.
    """

    tests = load_tests()

    total_tests = len(tests)

    for index, test in enumerate(tests):

        result = evaluate_retrieval(test)

        progress = (index + 1) / total_tests

        yield test, result, progress


# =========================
# Evaluate All Answer Tests
# =========================
def evaluate_all_answers():
    """
    Evaluate all answer tests.
    """

    tests = load_tests()

    total_tests = len(tests)

    for index, test in enumerate(tests):

        result = evaluate_answer(test)[0]

        progress = (index + 1) / total_tests

        yield test, result, progress


# =========================
# CLI Evaluation
# =========================
def run_cli_evaluation(test_number: int):
    """
    Run evaluation for a specific test.
    """

    tests = load_tests()

    # Validate Test Number
    if test_number < 0 or test_number >= len(tests):

        print(
            f"Error: test_row_number must be between "
            f"0 and {len(tests) - 1}"
        )

        sys.exit(1)

    test = tests[test_number]

    # =========================
    # Print Test Info
    # =========================
    print("\n" + "=" * 80)
    print(f"Test #{test_number}")
    print("=" * 80)

    print(f"Question:\n{test.question}\n")

    print(f"Keywords:\n{test.keywords}\n")

    print(f"Category:\n{test.category}\n")

    print(f"Reference Answer:\n{test.reference_answer}\n")

    # =========================
    # Retrieval Evaluation
    # =========================
    print("=" * 80)
    print("Retrieval Evaluation")
    print("=" * 80)

    retrieval_result = evaluate_retrieval(test)

    print(f"MRR: {retrieval_result.mrr:.4f}")

    print(f"nDCG: {retrieval_result.ndcg:.4f}")

    print(
        f"Keywords Found: "
        f"{retrieval_result.keywords_found}/"
        f"{retrieval_result.total_keywords}"
    )

    print(
        f"Keyword Coverage: "
        f"{retrieval_result.keyword_coverage:.1f}%"
    )

    # =========================
    # Answer Evaluation
    # =========================
    print("\n" + "=" * 80)
    print("Answer Evaluation")
    print("=" * 80)

    answer_result, generated_answer, retrieved_docs = evaluate_answer(test)

    print("\nGenerated Answer:\n")
    print(generated_answer)

    print("\nFeedback:\n")
    print(answer_result.feedback)

    print("\nScores:")

    print(f"Accuracy: {answer_result.accuracy:.2f}/5")

    print(f"Completeness: {answer_result.completeness:.2f}/5")

    print(f"Relevance: {answer_result.relevance:.2f}/5")

    print("\n" + "=" * 80 + "\n")


# =========================
# Main
# =========================
def main():
    """
    Main CLI Entry.
    """

    if len(sys.argv) != 2:

        print("Usage:")
        print("uv run eval.py <test_row_number>")

        sys.exit(1)

    try:
        test_number = int(sys.argv[1])

    except ValueError:

        print("Error: test_row_number must be an integer")

        sys.exit(1)

    run_cli_evaluation(test_number)


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    main()