"""Comprehensive QA test — simulates a real user session."""
import os, sys, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
logging.basicConfig(level=logging.WARNING)

import yaml
from src.qa_engine.engine import DocumentQAEngine

PDF = "/Users/chinmaymhatre/Desktop/Nlp/NASDAQ_TLF_2024.pdf"

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

print("=" * 70)
print("INITIALIZING ENGINE...")
engine = DocumentQAEngine(config)
info = engine.load_document(PDF)
print(f"Loaded: {info['total_pages']} pages, {info['total_chunks']} chunks")
print(f"Doc overview snippet: {engine._doc_overview[:300]}")
print("=" * 70)

# Test questions organized by category
TESTS = {
    "Identity / Document": [
        ("What is this document about?", None, "conversational"),
        ("Which company's annual report is this?", None, "conversational"),
        ("What type of document is this?", None, "conversational"),
    ],
    "Simple Facts": [
        ("What was total revenue?", "74,391", "table"),
        ("What was net income?", "827", "table"),
        ("What was total revenue in 2023?", "76,229", "table"),
        ("What was net income in 2023?", "3,768", "table"),
        ("What were total assets?", None, "table"),
    ],
    "Comparison / Difference": [
        ("Compare revenue between 2023 and 2024", None, None),
        ("What is the difference in net income between 2023 and 2024?", None, None),
    ],
    "Trends": [
        ("How did revenue change over the years?", None, None),
        ("When did net income increase?", None, None),
    ],
    "Conversational / Qualitative": [
        ("What are the main risk factors?", None, "conversational"),
        ("Is the company profitable?", None, "conversational"),
        ("Summarize the financial performance", None, "conversational"),
        ("What are the main business segments?", None, "conversational"),
        ("How does the company generate revenue?", None, "conversational"),
        ("Does the company have debt?", None, "conversational"),
    ],
    "Percentage / Ratio": [
        ("What is the gross margin percentage?", None, None),
    ],
    "Typo Handling": [
        ("What was total reveneu?", "74,391", "table"),
        ("What was the net incme?", "827", "table"),
    ],
    "Unanswerable": [
        ("What will revenue be in 2025?", "UNANSWERABLE", None),
        ("What is the stock price?", "UNANSWERABLE", None),
    ],
    "Follow-up Conversation": [
        ("What was net income?", "827", None),  # First question
        # Follow-up will be tested separately
    ],
}

results = []
total = 0
passed = 0
failed = []

for category, questions in TESTS.items():
    print(f"\n{'─' * 60}")
    print(f"  {category}")
    print(f"{'─' * 60}")

    for q_tuple in questions:
        question, expected_fragment, expected_method = q_tuple
        total += 1

        start = time.time()
        result = engine.ask(question, top_k=8)
        elapsed = time.time() - start

        answer = result.answer[:200]
        method = result.method
        conf = result.confidence
        followups = result.follow_ups

        # Check if answer is acceptable
        ok = True
        issues = []

        if expected_fragment:
            if expected_fragment == "UNANSWERABLE":
                if not result.is_unanswerable:
                    ok = False
                    issues.append(f"Should be unanswerable")
            elif expected_fragment not in result.answer.replace(",", ","):
                ok = False
                issues.append(f"Missing expected '{expected_fragment}'")

        if expected_method and method != expected_method:
            issues.append(f"Expected method={expected_method}, got {method}")
            # Don't fail just for method mismatch if answer is correct

        # Check for obviously bad answers
        if not result.is_unanswerable:
            if len(result.answer.strip()) < 3:
                ok = False
                issues.append("Answer too short")
            if result.answer.strip().startswith("UNANSWERABLE") and expected_fragment != "UNANSWERABLE":
                ok = False
                issues.append("Incorrectly marked unanswerable")

        status = "✅" if ok else "❌"
        if ok:
            passed += 1
        else:
            failed.append((question, answer, issues))

        print(f"\n  {status} Q: {question}")
        print(f"     A: {answer}{'...' if len(result.answer) > 200 else ''}")
        print(f"     Method: {method} | Confidence: {conf:.0%} | Time: {elapsed:.1f}s")
        if followups:
            print(f"     Follow-ups: {followups}")
        if issues:
            print(f"     ⚠ Issues: {', '.join(issues)}")

# Test follow-up conversation
print(f"\n{'─' * 60}")
print(f"  Follow-up Conversation Test")
print(f"{'─' * 60}")

history = []
followup_qs = [
    ("What was net income?", "827"),
    ("How much was it in 2023?", None),
    ("What about revenue?", None),
]

for q, expected in followup_qs:
    total += 1
    start = time.time()
    result = engine.ask(q, top_k=8, history=history)
    elapsed = time.time() - start

    answer = result.answer[:200]
    ok = True
    issues = []

    if expected and expected not in result.answer:
        ok = False
        issues.append(f"Missing '{expected}'")

    if result.is_unanswerable:
        ok = False
        issues.append("Incorrectly unanswerable")

    status = "✅" if ok else "❌"
    if ok:
        passed += 1
    else:
        failed.append((q, answer, issues))

    print(f"\n  {status} Q: {q}")
    print(f"     A: {answer}{'...' if len(result.answer) > 200 else ''}")
    print(f"     Method: {result.method} | Time: {elapsed:.1f}s")
    if result.follow_ups:
        print(f"     Follow-ups: {result.follow_ups}")
    if issues:
        print(f"     ⚠ Issues: {', '.join(issues)}")

    # Build history
    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": result.answer})

# Summary
print(f"\n{'=' * 70}")
print(f"  RESULTS: {passed}/{total} passed ({passed/total*100:.0f}%)")
print(f"{'=' * 70}")

if failed:
    print(f"\n  FAILURES:")
    for q, a, issues in failed:
        print(f"    ❌ {q}")
        print(f"       Answer: {a[:100]}")
        print(f"       Issues: {', '.join(issues)}")
