"""Generic test runner for any company's annual report (10-K).

Tests core QA capabilities without hardcoded company-specific values.
"""
import requests
import time
import sys

BASE = "http://localhost:8000/api"


def ask(question, history=None, timeout=120):
    payload = {"question": question, "top_k": 5}
    if history:
        payload["history"] = history
    resp = requests.post(f"{BASE}/ask", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def has_number(answer):
    """Check if answer contains a meaningful number (not just a year)."""
    import re
    nums = re.findall(r'[\$]?\s*[\d,]{3,}', answer)
    for n in nums:
        val = int(n.replace("$", "").replace(",", "").strip())
        if val >= 100 and not (2000 <= val <= 2099):
            return True
    return False


def has_any(answer, keywords):
    a = answer.lower()
    return any(k.lower() in a for k in keywords)


def test_document(pdf_path):
    print(f"\n{'='*70}")
    print(f"TESTING: {pdf_path.split('/')[-1]}")
    print(f"{'='*70}")

    # Upload
    print("Uploading...")
    with open(pdf_path, "rb") as f:
        resp = requests.post(
            f"{BASE}/upload",
            files={"file": (pdf_path.split("/")[-1], f, "application/pdf")},
            timeout=300,
        )
    if resp.status_code != 200:
        print(f"UPLOAD FAILED: {resp.status_code} {resp.text}")
        return 0, 0
    info = resp.json()
    print(f"OK: {info['total_pages']} pages, {info['total_chunks']} chunks\n")

    passed = 0
    failed = 0
    failures = []

    tests = [
        # --- Financial data (should return numbers) ---
        (1, "What was the total revenue?",
         lambda a: has_number(a), "should contain a number"),

        (2, "What was the net income?",
         lambda a: has_number(a), "should contain a number"),

        (3, "What was the gross margin or gross profit?",
         lambda a: has_number(a), "should contain a number"),

        (4, "What were the total operating expenses?",
         lambda a: has_number(a), "should contain a number"),

        (5, "What was the total cost of sales or cost of revenue?",
         lambda a: has_number(a), "should contain a number"),

        # --- Factual questions ---
        (6, "What is this document about?",
         lambda a: len(a) > 20, "should give a meaningful description"),

        (7, "How many employees does the company have?",
         lambda a: has_number(a) or "unanswerable" in a.lower(),
         "should have a number or say unanswerable"),

        # --- Unanswerable ---
        (8, "What will the company's revenue be in 2030?",
         lambda a: "unanswerable" in a.lower(),
         "should say UNANSWERABLE for future prediction"),

        (9, "What is the company's stock price?",
         lambda a: "unanswerable" in a.lower(),
         "should say UNANSWERABLE for stock price"),

        (10, "How many units were sold?",
         lambda a: "unanswerable" in a.lower(),
         "should say UNANSWERABLE for unit sales"),

        # --- Conversational ---
        (11, "Summarize this document for me.",
         lambda a: len(a) > 50, "should give a substantial summary"),

        (12, "What are the main risks the company faces?",
         lambda a: has_any(a, ["risk", "Risk"]) and len(a) > 30,
         "should mention risks"),

        # --- Comparison / Trend ---
        (13, "How did revenue change from the previous year?",
         lambda a: has_number(a) or has_any(a, ["increase", "decrease", "grew", "declined", "change"]),
         "should describe revenue change"),

        # --- Follow-up ---
        (14, None, None, None),  # placeholder — handled specially below
    ]

    for qnum, question, check_fn, fail_msg in tests:
        if qnum == 14:
            # Follow-up test
            try:
                t0 = time.time()
                data1 = ask("What was the total revenue?")
                ans1 = data1["answer"]
                history = [
                    {"role": "user", "content": "What was the total revenue?"},
                    {"role": "assistant", "content": ans1},
                ]
                data2 = ask("How does it compare to the previous year?", history=history)
                elapsed = time.time() - t0
                answer = data2["answer"]
                ok = has_number(answer) or has_any(answer, ["increase", "decrease", "grew", "declined", "higher", "lower", "change", "compare"])
                tag = "PASS ✅" if ok else "FAIL ❌"
                print(f"Q{qnum:>2}: {tag} ({elapsed:.1f}s) | Follow-up: {answer[:90]}")
                if ok:
                    passed += 1
                else:
                    failed += 1
                    failures.append((qnum, "Follow-up: revenue comparison", answer[:150], "should compare revenue"))
            except Exception as e:
                print(f"Q{qnum:>2}: ERROR ❌ | {e}")
                failed += 1
            continue

        try:
            t0 = time.time()
            data = ask(question)
            elapsed = time.time() - t0
            answer = data["answer"]

            ok = check_fn(answer)
            tag = "PASS ✅" if ok else "FAIL ❌"
            print(f"Q{qnum:>2}: {tag} ({elapsed:.1f}s) | {answer[:90]}")
            if ok:
                passed += 1
            else:
                failed += 1
                failures.append((qnum, question, answer[:150], fail_msg))
        except Exception as e:
            print(f"Q{qnum:>2}: ERROR ❌ | {e}")
            failed += 1
            failures.append((qnum, question, str(e)[:150], fail_msg))

    total = passed + failed
    print(f"\n{'='*70}")
    print(f"RESULTS: {passed}/{total} passed ({100*passed/total:.0f}%)")
    print(f"{'='*70}")

    if failures:
        print(f"\n❌ {len(failures)} FAILURE(S):")
        for qnum, q, got, exp in failures:
            print(f"\n  Q{qnum}: {q}")
            print(f"    Got:      {got}")
            print(f"    Expected: {exp}")

    return passed, total


if __name__ == "__main__":
    pdfs = [
        "/Users/chinmaymhatre/Desktop/Nlp/Financial-report-RAG/NYSE_AAN_2023.pdf",
        "/Users/chinmaymhatre/Desktop/Nlp/Financial-report-RAG/NYSE_MTRN_2024.pdf",
    ]

    # Allow specifying PDFs on command line
    if len(sys.argv) > 1:
        pdfs = sys.argv[1:]

    grand_passed = 0
    grand_total = 0
    for pdf in pdfs:
        p, t = test_document(pdf)
        grand_passed += p
        grand_total += t

    if len(pdfs) > 1:
        print(f"\n{'='*70}")
        print(f"GRAND TOTAL: {grand_passed}/{grand_total} passed ({100*grand_passed/grand_total:.0f}%)")
        print(f"{'='*70}")
