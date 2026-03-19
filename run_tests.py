"""Automated test runner for QA engine — 50 questions."""
import requests
import json
import time

BASE = "http://localhost:8000/api"
PDF_PATH = "/Users/chinmaymhatre/Desktop/Nlp/Financial-report-RAG/NASDAQ_AAPL_2024.pdf"

# Each test: question, expected substrings (any match = pass), flags
TESTS = [
    # --- Financial Data (1-8) ---
    (1,  "What was Apple's total net sales in 2024?", ["391,035"]),
    (2,  "What was iPhone revenue in 2024?", ["201,183"]),
    (3,  "How much revenue did Services generate in 2024?", ["96,169"]),
    (4,  "What was the total cost of sales in 2024?", ["210,352"]),
    (5,  "What was Apple's net income for fiscal year 2024?", ["93,736"]),
    (6,  "What was the gross margin for 2024?", ["180,683"]),
    (7,  "How much did Apple spend on research and development in 2024?", ["31,370"]),
    (8,  "What were total operating expenses in 2024?", ["57,467", "56,165"]),

    # --- Comparison (9-12) ---
    (9,  "How did iPhone revenue change from 2023 to 2024?", ["200,583", "201,183"]),
    (10, "Which product category had the highest revenue in 2024?", ["iPhone"]),
    (11, "Did Services revenue increase or decrease compared to 2023?", ["increase"]),
    (12, "What was the change in total net sales from 2022 to 2024?", ["394,328", "391,035"]),

    # --- Facts (13-17) ---
    (13, "What is Apple's fiscal year end date?", ["September", "september"]),
    (14, "How many employees does Apple have?", ["164,000"]),
    (15, "Who is Apple's CEO?", ["Cook"]),
    (16, "Where is Apple headquartered?", ["Cupertino"]),
    (17, "What are Apple's reportable segments?", ["Americas"]),

    # --- Complex (18-22) ---
    (18, "What percentage of total revenue came from iPhone in 2024?", ["51"]),
    (19, "What was the effective tax rate in 2024?", ["14", "15", "16", "24"]),
    (20, "What was earnings per diluted share in 2024?", ["6.08"]),
    (21, "How much cash and cash equivalents did Apple have at the end of 2024?", ["29,943"]),
    (22, "What was total shareholders' equity in 2024?", ["56,950"]),

    # --- Unanswerable (23-25) ---
    (23, "What will Apple's revenue be in 2025?", ["UNANSWERABLE"]),
    (24, "How many iPhones were sold in 2024?", ["UNANSWERABLE"]),
    (25, "What is Apple's stock price?", ["UNANSWERABLE"]),

    # --- Conversational / Open-ended (26-30) ---
    (26, "What is this document about?", ["Apple"]),
    (27, "What are the key financial highlights?", ["391,035", "revenue", "net sales"]),
    (28, "Tell me about Apple's business overview.", ["Apple"]),
    (29, "What are the main risks Apple faces?", ["risk", "Risk"]),
    (30, "Summarize this document for me.", ["Apple"]),

    # --- Difference (36-38) ---
    (36, "What is the difference between 2024 and 2023 net sales?", ["7,750"]),
    (37, "Tell me the difference between 2023 and 2022 revenue", ["11,043"]),
    (38, "Difference between 2022 and 2024 iPhone revenue", ["201,183", "205,489"]),

    # --- Trend (39-40) ---
    (39, "When did the revenue increase?", ["391,035", "383,285", "394,328"]),
    (40, "When did net income grow?", ["93,736"]),

    # --- Alternate phrasing (41-50) ---
    (41, "How much money did Apple make in 2024?", ["93,736", "391,035"]),
    (42, "What is Apple's revenue?", ["391,035"]),
    (43, "Whats the revenue in 2024?", ["391,035"]),
    (44, "How much is the gross margin?", ["180,683"]),
    (45, "What was Mac revenue in 2024?", ["29,984"]),
    (46, "What was iPad revenue in 2024?", ["26,694"]),
    (47, "How much did Wearables generate in 2024?", ["37,005"]),
    (48, "What was the total revenue in 2023?", ["383,285"]),
    (49, "What was net income in 2023?", ["96,995"]),
    (50, "What was iPhone revenue in 2022?", ["205,489"]),
]

# Follow-up tests (need conversation history)
FOLLOWUP_TESTS = [
    (31, "What was the revenue in 2024?", "What about Services?", ["96,169"]),
    (32, "What was iPhone revenue in 2024?", "Compare it with 2023", ["200,583", "201,183"]),
    (33, "What was net income in 2024?", "Now compare it with 2022", ["93,736", "99,803"]),
    (34, "What was total net sales in 2024?", "Difference between 2023 and 2022", ["11,043"]),
    (35, "What was the revenue in 2024?", "When did the revenue increase?", ["391,035", "383,285", "394,328"]),
]


def check(answer, expects):
    a_lower = answer.lower()
    for e in expects:
        if e.lower() in a_lower:
            return True
    return False


def ask(question, history=None, timeout=120):
    payload = {"question": question, "top_k": 5}
    if history:
        payload["history"] = history
    resp = requests.post(f"{BASE}/ask", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def main():
    # Upload PDF
    print("=" * 70)
    print("UPLOADING Apple 10-K PDF...")
    with open(PDF_PATH, "rb") as f:
        resp = requests.post(f"{BASE}/upload", files={"file": ("NASDAQ_AAPL_2024.pdf", f, "application/pdf")}, timeout=300)
    if resp.status_code != 200:
        print(f"UPLOAD FAILED: {resp.status_code} {resp.text}")
        return
    info = resp.json()
    print(f"OK: {info['total_pages']} pages, {info['total_chunks']} chunks")
    print("=" * 70)

    passed = 0
    failed = 0
    failures = []

    # Single questions
    for qnum, question, expects in TESTS:
        try:
            t0 = time.time()
            data = ask(question)
            elapsed = time.time() - t0
            answer = data["answer"]
            is_unans = data.get("is_unanswerable", False)

            # Unanswerable check
            if "UNANSWERABLE" in expects:
                ok = is_unans or "unanswerable" in answer.lower()
            else:
                ok = check(answer, expects)

            tag = "PASS ✅" if ok else "FAIL ❌"
            print(f"Q{qnum:>2}: {tag} ({elapsed:.1f}s) | {answer[:90]}")
            if ok:
                passed += 1
            else:
                failed += 1
                failures.append((qnum, question, answer[:150], expects))
        except Exception as e:
            print(f"Q{qnum:>2}: ERROR ❌ | {e}")
            failed += 1
            failures.append((qnum, question, str(e)[:150], expects))

    # Follow-up questions
    print("\n--- Follow-up Tests (with history) ---")
    for qnum, first_q, followup_q, expects in FOLLOWUP_TESTS:
        try:
            t0 = time.time()
            # Ask first question
            data1 = ask(first_q)
            ans1 = data1["answer"]
            # Build history
            history = [
                {"role": "user", "content": first_q},
                {"role": "assistant", "content": ans1},
            ]
            # Ask follow-up
            data2 = ask(followup_q, history=history)
            elapsed = time.time() - t0
            answer = data2["answer"]

            ok = check(answer, expects)
            tag = "PASS ✅" if ok else "FAIL ❌"
            print(f"Q{qnum:>2}: {tag} ({elapsed:.1f}s) | \"{first_q}\" → \"{followup_q}\" → {answer[:80]}")
            if ok:
                passed += 1
            else:
                failed += 1
                failures.append((qnum, f"{first_q} → {followup_q}", answer[:150], expects))
        except Exception as e:
            print(f"Q{qnum:>2}: ERROR ❌ | {e}")
            failed += 1
            failures.append((qnum, f"{first_q} → {followup_q}", str(e)[:150], expects))

    # Summary
    total = passed + failed
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{total} passed ({100*passed/total:.0f}%)")
    print("=" * 70)

    if failures:
        print(f"\n❌ {len(failures)} FAILURE(S):")
        for qnum, q, got, exp in failures:
            print(f"\n  Q{qnum}: {q}")
            print(f"    Got:      {got}")
            print(f"    Expected: {exp}")


if __name__ == "__main__":
    main()
