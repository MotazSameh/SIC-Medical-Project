import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv


load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")

CHROMA_DIR = r"E:\MNU\3th_2\NLP\RAG Medical Project\src\arabic\vector_db\chromadb"

COLLECTION_NAME = "arabic_medical_qa"

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = chroma_client.get_collection(name=COLLECTION_NAME)

MIN_CONFIDENCE_SCORE = 0.70

print(f"Arabic ChromaDB loaded: "f"{collection.count():,} chunks")

def dense_retrieve(query, top_k=5):

    query_embedding = embedding_model.encode("query: " + query,normalize_embeddings=True).tolist()

    results = collection.query(query_embeddings=[query_embedding],n_results=top_k,
                                include=["documents","metadatas","distances"])

    hits = []

    for i in range(len(results["ids"][0])):
        similarity = 1 - results["distances"][0][i]
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "meta": results["metadatas"][0][i],
            "score": similarity,
            "rank": i
        })

    return hits


# ============================================================
# Chat History
# ============================================================

def build_retrieval_query(query,chat_history):
    if not chat_history:
        return query

    history = chat_history[-6:]

    messages = [
        {
            "role": "system",
            "content": """
أنت مسئول فقط عن إعادة كتابة السؤال
لاستخدامه في البحث داخل قاعدة بيانات طبية عربية.

القواعد:

1. إذا كان السؤال يعتمد على المحادثة السابقة:
   أعد كتابة السؤال ليصبح مستقلاً وواضحاً.

2. أضف فقط المعلومات الموجودة في المحادثة السابقة.

3. لا تضف أي معلومات طبية جديدة.

4. إذا كان السؤال جديداً:
   أعد السؤال كما هو.

5. لا تجب على السؤال.

6. لا تشرح.

7. أخرج السؤال فقط.
"""
        }
    ]

    messages.extend(history)

    messages.append({ "role": "user","content": query})

    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            include_reasoning=False,
            messages=messages,
            temperature=0,
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()

    except Exception:
        return query


SYSTEM_PROMPT = """
أنت مساعد طبي عربي يعمل بنظام RAG.

قواعد صارمة جدًا:

1. أجب فقط باستخدام المعلومات الموجودة في النص المرجعي المقدم لك.
2. ممنوع إضافة أي معلومة طبية غير موجودة في النص المرجعي.
3. ممنوع التخمين أو التشخيص من عندك.
4. ممنوع تحويل المعلومات الموجودة في النص إلى نصائح جديدة غير مذكورة فيه.
5. إذا كانت المعلومات الموجودة في النص غير كافية للإجابة:
   قل: "المعلومات المتاحة في المصادر غير كافية للإجابة بدقة."
6. لا تذكر "المصدر 1" أو "المصدر 2".
7. لا تقل "حسب المصدر".
8. لا تضف تحذيرات أو نصائح طبية من عندك.
9. حافظ على معنى النص الأصلي بدون اختراع معلومات.
10. أجب بالعربية وبشكل مختصر وواضح.

النص المرجعي هو المصدر الوحيد للإجابة.
"""

def get_answer(query,chat_history=None,response_language: str = "ar"):

    if chat_history is None:
        chat_history = []

    retrieval_query = build_retrieval_query(query,chat_history)


    print( "\n========== Retrieval Query ==========")
    print(retrieval_query)
    print("=====================================")

    hits = dense_retrieve(retrieval_query,top_k=5)

    print("\n========== RETRIEVED HITS ==========")

    for i, hit in enumerate(hits, 1):
        print(f"\nRank {i}")
        print(f"ID: {hit['id']}")
        print(f"Score: {hit['score']:.5f}")
        print(f"Category: {hit['meta'].get('category', '')}")
        print(f"Text:\n{hit['text'][:1000]}")

    print("=====================================")

    top_score = hits[0]["score"] if hits else 0.0

    if not hits or top_score < MIN_CONFIDENCE_SCORE:
        answer = low_confidence_message()
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]
        return answer, chat_history, []
    
    context = ""

    for i, hit in enumerate(hits):
        category = hit["meta"].get("category","")
        context += f"""[المصدر {i + 1}] التصنيف: {category}{hit["text"]}"""

    language_instruction = (
        "أجب بالعربية الفصحى الواضحة."
        if response_language == "ar"
        else "Answer in clear English, even though the reference text below is in Arabic."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + language_instruction}
    ]

    messages.extend(chat_history)


    messages.append({
        "role": "user",
        "content": f"""
النصوص الطبية المرجعية:

{context}

السؤال:
{query}

أجب عن السؤال اعتمادًا فقط على النصوص المرجعية أعلاه.

إذا لم تكن الإجابة موجودة أو كانت المعلومات غير كافية، قل:

"المعلومات المتاحة في المصادر غير كافية للإجابة بدقة."
"""
    })

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0,
        max_tokens=500,
        include_reasoning=False,
        frequency_penalty=0.3)


    answer = response.choices[0].message.content

    chat_history.append({ "role": "user","content": query})
    chat_history.append({"role": "assistant", "content": answer})

    if len(chat_history) > 10:
        chat_history = chat_history[-10:]

    sources = []
    for hit in hits:
        sources.append({
            "chunk_id": hit["id"],
            "category": hit["meta"].get("category",""),
            "score": hit["score"]})
    return answer, chat_history, sources

def low_confidence_message() -> str:
    return (
        "المعلومات المتاحة في المصادر غير كافية للإجابة بدقة على هذا السؤال.\n\n"
        "برجاء إعادة صياغة السؤال بشكل أكثر تحديدًا، أو استشارة طبيب."
    )

if __name__ == "__main__":
    print("=" * 60)
    print("ARABIC MEDICAL CHATBOT")
    print("=" * 60)


    history = []
    while True:
        query = input("\nاكتب سؤالك (exit للخروج):\n> ")
        if query.lower() == "exit":
            break
        if not query.strip():
            continue

        answer, history, sources = get_answer( query,history)

        print("\n" + "=" * 60)
        print("ANSWER")
        print( "=" * 60)
        print(answer)

        print("\n" + "=" * 60)
        print("SOURCES")
        print("=" * 60)

        for source in sources:
            print(
                f"{source['chunk_id']} | "
                f"{source['category']} | "
                f"Dense={source['score']:.5f}"
            )