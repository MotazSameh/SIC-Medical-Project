# src/scripts/test_router.py
"""
Quick manual test for src/router.py — run this after saving router.py
to src/router.py, from the project root with the venv activated:

    python -m src.scripts.test_router
"""

from src.router import classify_query, translate_to_english


TEST_QUERIES = [
    # Expected: arabic
    "دكتور أنا عندي صداع ودوخة أعمل إيه؟",

    # Expected: guidelines
    "What does NICE recommend for hypertension?",

    # Expected: both
    "هل metformin مناسب للحالة دي وما هي التوصيات الرسمية لاستخدامه؟",

    # Extra edge cases worth checking:
    # Arabic question but clearly about an official guideline/drug label
    "ما هي الجرعة القصوى المسموح بها من الباراسيتامول حسب FDA؟",

    # English but sounds like a patient-style personal question
    "I have a headache and dizziness, what should I do?",

    # Arabic, general consultation style, no drug/guideline mention
    "عندي تعب شديد وفقدان شهية من فترة",
]


def main():
    print("=" * 70)
    print("ROUTER CLASSIFICATION TEST")
    print("=" * 70)

    for query in TEST_QUERIES:
        result = classify_query(query)
        print(f"\nQuery: {query}")
        print(f"  -> source={result['source']}  language={result['language']}")

    print("\n" + "=" * 70)
    print("TRANSLATION TEST")
    print("=" * 70)

    translation_cases = [
        "هل metformin مناسب للحالة دي وما هي التوصيات الرسمية لاستخدامه؟",
        "ما هي الجرعة القصوى المسموح بها من الباراسيتامول حسب FDA؟",
        "ما هي أعراض ارتفاع ضغط الدم؟",
    ]

    for query in translation_cases:
        translated = translate_to_english(query)
        print(f"\nOriginal:   {query}")
        print(f"Translated: {translated}")


if __name__ == "__main__":
    main()