import sys
import traceback

try:
    from src.guidelines.Chatbot import get_answer, KNOWN_DRUGS
except Exception as e:
    print("❌ فشل استيراد Chatbot.py — تأكد إن هذا الملف في جذر المشروع (نفس المستوى اللي فيه src/)،")
    print("   وإن الـ ChromaDB، وملف bm25_index.pkl، والـ .env كلهم موجودين.")
    print(f"   خطأ: {e}")
    sys.exit(1)


def run_case(title, query, chat_history=None, expect="answer"):
    chat_history = chat_history or []
    print("=" * 70)
    print(f"🧪 {title}")
    print(f"   السؤال: {query}")
    print("-" * 70)
    try:
        answer, updated_history, sources = get_answer(query, chat_history)
    except Exception:
        print("❌ اتكسر الكود أثناء التنفيذ:")
        traceback.print_exc()
        return False, [], []

    print(f"الإجابة:\n{answer}\n")
    print(f"عدد المصادر المرجعة: {len(sources)}")
    generation_failed = "تعذّر توليد إجابة كاملة" in answer
    rejected = (
        ("غير متوفر" in answer)or ("غير كافٍ" in answer)
        or ("غير مذكور في المصادر" in answer)  or (len(sources) == 0))

    if generation_failed:
        print("❌ فشل توليد الإجابة فعليًا (رجعت رسالة fallback) — مش نجاح حتى لو مفيش رفض")
        passed = False
    elif expect == "reject" and rejected:
        print("✅ النتيجة متوقعة (رفض صحيح)")
        passed = True
    elif expect == "answer" and not rejected:
        print("✅ النتيجة متوقعة (إجابة مع مصدر)")
        passed = True
    else:
        print(f"⚠️ النتيجة غير متوقعة (كنا نتوقع: {expect})")
        passed = False

    return passed, updated_history, sources


def main():
    print(f"\n📚 الأدوية المغطاة في الـ corpus حاليًا ({len(KNOWN_DRUGS)}):")
    print(f"   {sorted(KNOWN_DRUGS)}\n")

    results = []

    known_drug_query = "ما هي الجرعة الآمنة من الميتفورمين؟"
    passed, hist, _ = run_case("1) دواء موجود في الـ corpus",known_drug_query,expect="answer",)
    results.append(passed)

    if hist:
        passed, _, _ = run_case("2) سؤال متابعة (follow-up) على نفس الموضوع",
                                "وهل له آثار جانبية على الكلى؟",chat_history=hist,expect="answer",)
        results.append(passed)

    passed, _, _ = run_case("3) دواء مش موجود في الـ corpus (aspirin)",
                            "aspirin contraindications",expect="reject",)
    results.append(passed)

    passed, _, _ = run_case("4) سؤال عام غامض جدًا بعيد عن أي دواء محدد","ازاي أنحف بسرعة؟",expect="reject",)
    results.append(passed)

    passed, _, _ = run_case("5) نفس سؤال الجرعة، لكن بالإنجليزي (control test)",
                            "What is the safe dosage of metformin?",expect="answer",)
    results.append(passed)

    passed, _, _ = run_case("6) سؤال إنجليزي عن آثار جانبية لدواء تاني في الـ corpus",
                            "What are the side effects of atorvastatin?",expect="answer",)
    results.append(passed)

    print("=" * 70)
    print(f"النتيجة النهائية: {sum(results)}/{len(results)} حالات نجحت")
    print("=" * 70)


if __name__ == "__main__":
    main()