import os
import json
from groq import Groq
from dotenv import load_dotenv
from src.common.documents import build_document_link
from src.guidelines.vector_store import VectorStore
from src.guidelines.bm25_index import BM25Index
from src.guidelines.hybrid_retriever import HybridRetriever

load_dotenv()

COLLECTION_NAME = "drug_chunks"
PERSIST_DIRECTORY = "data/vector_db/chromadb"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
CHUNKS_JSON_PATH = "data/processed/chunks.json"
BM25_INDEX_PATH = "data/bm25_index.pkl"
LLM_MODEL = "qwen/qwen3.8-27b"
MIN_CONFIDENCE_SCORE = 0.7
VECTOR_WEIGHT = 0.7
CONTEXT_DROPOFF_RATIO = 0.5

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

vector_store = VectorStore(
    persist_directory=PERSIST_DIRECTORY,
    embedding_model_name=EMBEDDING_MODEL_NAME,
    collection_name=COLLECTION_NAME,)

vector_store.load_chunks(CHUNKS_JSON_PATH)
vector_store.create_or_load_collection()

bm25_index = BM25Index.load_from_disk(BM25_INDEX_PATH)

hybrid = HybridRetriever(vector_store, bm25_index)


def _build_known_drugs(chunks) -> set:
    drugs = set()
    for c in chunks:
        for n in c.drug_names:
            drugs.add(n.strip().lower())
    return drugs


KNOWN_DRUGS = _build_known_drugs(vector_store.chunks_loaded)
print(f"[INIT] Corpus covers {len(KNOWN_DRUGS)} drug(s): {sorted(KNOWN_DRUGS)}")


def strip_thinking(text: str) -> str:
    import re
    if "<think>" not in text:
        return text.strip()

    if "</think>" in text:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    return text.split("<think>")[0].strip()

def build_retrieval_query(query: str, chat_history: list) -> str:
    if not chat_history:
        return query

    history = chat_history[-6:]

    messages = [
        {
            "role": "system",
            "content": """
أنت مسئول فقط عن إعادة كتابة السؤال لاستخدامه في البحث داخل قاعدة بيانات طبية (إرشادات أدوية).

القواعد:
1- إذا كان السؤال يعتمد على المحادثة السابقة:
    - أعد كتابة السؤال ليصبح كاملاً ومستقلاً.
    - أضف المعلومات الناقصة من الـ history.
    - لا تضف أي معلومات غير موجودة.
2- إذا كان السؤال جديد ولا يعتمد على الـ history:
    - أعد السؤال كما هو بدون أي تعديل.
3- لا تجب على السؤال. لا تشرح. لا تضف أي كلام آخر. أخرج السؤال فقط.
"""
        }
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": query})

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=200,
            reasoning_effort="none",     
            reasoning_format="hidden",
        )
        retrieval_query = strip_thinking(response.choices[0].message.content.strip())
        if not retrieval_query:
            return query
        print(f"\n[Retrieval Query] {retrieval_query}\n")
        return retrieval_query
    except Exception:
        return query


def extract_query_info(query: str) -> dict:
    messages = [
        {
            "role": "system",
            "content": """
حلل السؤال الطبي التالي واستخرج:
1) اسم الدواء المذكور صراحة إن وُجد (اسم علمي واحد بالإنجليزية، أو NONE لو مفيش اسم دواء محدد)
2) الكلمات المفتاحية الطبية الجوهرية (أعراض، جرعة، حالة مرضية...) بدون كلمات السؤال زي "ما هي" أو "هل"

أخرج الناتج بالضبط بهذا الشكل (سطرين فقط، من غير أي شرح إضافي):
DRUG: <اسم الدواء أو NONE>
KEYWORDS: <الكلمات مفصولة بمسافة>

مثال:
السؤال: هل الأسبرين ممنوع في الحمل؟
DRUG: aspirin
KEYWORDS: حمل ممنوع

السؤال: ما هي أعراض جرعة الميتفورمين الزائدة؟
DRUG: metformin
KEYWORDS: أعراض جرعة زائدة
"""
        },
        {"role": "user", "content": query}
    ]

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=150,
            reasoning_effort="none",     
            reasoning_format="hidden",
        )
        raw = strip_thinking(response.choices[0].message.content.strip())

        drug = None
        keywords = ""
        for line in raw.splitlines():
            if line.upper().startswith("DRUG:"):
                value = line.split(":", 1)[1].strip()
                drug = None if value.upper() == "NONE" else value
            elif line.upper().startswith("KEYWORDS:"):
                keywords = line.split(":", 1)[1].strip()

        return {"drug": drug, "keywords": keywords}
    except Exception:
        return {"drug": None, "keywords": ""}

def check_drug_coverage(drug: str) -> bool:
    if not drug:
        return True

    drug_lower = drug.strip().lower()
    return any(drug_lower in known or known in drug_lower for known in KNOWN_DRUGS)


def not_covered_message(drug: str) -> str:
    covered = ", ".join(sorted(KNOWN_DRUGS))
    return (
        f"غير متوفر لدي معلومات موثوقة عن **{drug}** في قاعدة البيانات الحالية.\n\n"
        f"الأدوية المتاحة حاليًا: {covered}.\n\n"
        f"هذا النظام يقدّم معلومات مبنية فقط على المصادر المتاحة له، "
        f"ولا يُغني عن استشارة طبيب أو صيدلي."
    )


def low_confidence_message() -> str:
    return (
        "لم أجد في المصادر المتاحة لدي معلومات كافية وموثوقة للإجابة على هذا السؤال بدقة.\n\n"
        "برجاء إعادة صياغة السؤال بشكل أكثر تحديدًا، أو استشارة طبيب/صيدلي."
    )

def hybrid_retrieve(query, top_k=5, bm25_query=None, vector_weight=VECTOR_WEIGHT):
    if not query or not query.strip():
        return [], 0.0

    dense_hits = hybrid.retrieve_vector(query, top_k=top_k)
    dense_score_map = {c.chunk_id: c.score for c in dense_hits}
    top_dense_score = max(dense_score_map.values(), default=0.0)

    merged = hybrid.retrieve_hybrid(query,top_k=top_k,vector_weight=vector_weight,bm25_query=bm25_query,)

    hits = [{"chunk": c, "dense_score": dense_score_map.get(c.chunk_id, 0.0)} for c in merged]
    return hits, top_dense_score

def select_context_chunks(hits: list, min_chunks: int = 3, max_chunks: int = 5) -> list:
    if len(hits) <= min_chunks:
        return hits

    top_score = hits[0]["dense_score"]
    if top_score <= 0:
        return hits[:min_chunks]

    cutoff_score = hits[min_chunks - 1]["dense_score"]
    ratio = cutoff_score / top_score

    if ratio >= CONTEXT_DROPOFF_RATIO:
        return hits[:min_chunks]   

    return hits[:max_chunks] 


SYSTEM_PROMPT = """أنت مساعد طبي متخصص في معلومات الأدوية، مبني على إرشادات NICE ونشرات FDA الرسمية.

قواعد صارمة:
- أجب فقط من النص المرجعي المقدم لك، ولا تخترع أي معلومة غير موجودة فيه
- إذا الإجابة غير موجودة في النص المتاح: قل بوضوح "غير مذكور في المصادر المتاحة"
- اذكر دائمًا مصدر المعلومة (الجهة، اسم المستند، رقم الصفحة) في نهاية إجابتك
- إذا كان السؤال يتعلق بجرعة أو قرار علاجي، ذكّر المستخدم أن القرار النهائي يعود للطبيب المعالج
- لا تقدّم تشخيصًا طبيًا لحالة فردية، فقط معلومات مرجعية عامة من المصدر

طريقة الإجابة:
- استخرج المعلومة الطبية بدقة (جرعة، تحذير، مؤشر...) كما وردت حرفيًا في النص
- اجعل الإجابة مختصرة وواضحة بدون تكرار
- في نهاية الإجابة أضف سطر "المصدر:" يوضح الجهة والمستند والصفحة
"""

def get_answer(query: str, chat_history: list, response_language: str = "ar") -> tuple:

    info = extract_query_info(query)
    drug = info["drug"]
    keywords = info["keywords"]
    print(f"[DEBUG] drug='{drug}' keywords='{keywords}'")

    if drug and not check_drug_coverage(drug):
        answer = not_covered_message(drug)
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})
        return answer, chat_history[-10:], []

    retrieval_query = build_retrieval_query(query, chat_history)
    if keywords and keywords.strip() not in retrieval_query:
        retrieval_query = f"{retrieval_query} {keywords}"

    hits, top_dense_score = hybrid_retrieve(retrieval_query, top_k=5, bm25_query=keywords if keywords else None)

    print("\n=== Retrieved Chunks ===")
    for i, hit in enumerate(hits):
        c = hit["chunk"]
        print(f"[{i+1}] {c.authority_family} | {c.document_id} "
              f"| p.{c.start_page} | score={c.score:.4f} | dense={hit['dense_score']:.4f}")
        print(f"    {c.text[:150]}")
    print(f"top_dense_score={top_dense_score:.4f}\n")

    if not hits or top_dense_score < MIN_CONFIDENCE_SCORE:
        answer = low_confidence_message()
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})
        return answer, chat_history[-10:], []

    selected_hits = select_context_chunks(hits, min_chunks=3, max_chunks=5)

    context = ""
    sources = []
    for hit in selected_hits:
        c = hit["chunk"]
        context += (f"\n[{c.authority_family or '؟'} - {c.document_id or '؟'} "
                    f"- صفحة {c.start_page if c.start_page is not None else '؟'}]\n{c.text}\n")
        sources.append({
            "authority_family": c.authority_family,
            "document_id": c.document_id,
            "start_page": c.start_page,
            "end_page": c.end_page,
            "document_link": build_document_link(c.document_id, c.authority_family),
            "score": c.score,
        })

    language_instruction = (
        "أجب بالعربية الفصحى الواضحة."
        if response_language == "ar"
        else "Answer in clear English.")

    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + language_instruction}]    
    messages += chat_history
    user_label = "النص الطبي المرجعي" if response_language == "ar" else "Reference medical text"
    question_label = "السؤال" if response_language == "ar" else "Question"
    messages.append({
        "role": "user",
        "content": f"{user_label}:\n{context}\n\n{question_label}: {query}"
    })
    
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=3000,
        frequency_penalty=0.3,
        reasoning_effort="default",  
        reasoning_format="hidden",
    )

    answer = strip_thinking(response.choices[0].message.content)
    if not answer:
        answer = "تعذّر توليد إجابة كاملة لهذا السؤال، برجاء إعادة المحاولة."

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]

    return answer, chat_history, sources