import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

ROUTER_MODEL = "qwen/qwen3.8-27b"


def _strip_thinking(text: str) -> str:
    import re
    if "<think>" not in text:
        return text.strip()
    if "</think>" in text:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text.split("<think>")[0].strip()

def classify_query(query: str, chat_history: list = None) -> dict:

    system_prompt = """
أنت مسئول فقط عن تصنيف سؤال طبي، بدون الإجابة عليه.

صنّف السؤال إلى مصدرين محتملين:
- "arabic": سؤال استشاري عام شبيه بأسئلة المرضى (أعراض، حالة شخصية، استشارة عامة)
- "guidelines": سؤال عن إرشادات رسمية أو أدوية معتمدة (NICE, FDA) أو جرعات دوائية دقيقة
- "both": لو السؤال يحتاج الاثنين معًا (مثال: سؤال عن حالة شخصية + توصيات رسمية لنفس الدواء)

حدد أيضًا لغة السؤال: "ar" أو "en"

مهم جدًا: لو السؤال الحالي قصير أو غامض ويعتمد على المحادثة السابقة
(مثل "more details"، "وضح أكتر"، "وايه كمان؟")، لا تصنّفه بمعزل عن
السياق. استخدم المحادثة السابقة لتحديد الموضوع الفعلي الذي يشير إليه
السؤال، وصنّف على أساس ذلك الموضوع نفسه (نفس المصدر المستخدم في آخر
سؤال ذي صلة)، وليس على أساس نص السؤال القصير وحده.

أخرج الناتج بالضبط بهذا الشكل (سطرين فقط، بدون أي شرح إضافي):
SOURCE: <arabic|guidelines|both>
LANGUAGE: <ar|en>

أمثلة:
السؤال: دكتور أنا عندي صداع ودوخة أعمل إيه؟
SOURCE: arabic
LANGUAGE: ar

السؤال: What does NICE recommend for hypertension?
SOURCE: guidelines
LANGUAGE: en

السؤال: هل metformin مناسب للحالة دي وما هي التوصيات الرسمية لاستخدامه؟
SOURCE: both
LANGUAGE: ar

مثال على سؤال متابعة يعتمد على السياق:
المحادثة السابقة كانت عن جرعة NICE الرسمية لدواء معين، ثم سأل المستخدم
"more details":
SOURCE: guidelines
LANGUAGE: en
"""

    messages = [{"role": "system", "content": system_prompt}]

    if chat_history:
        messages.extend(chat_history[-6:])

    messages.append({"role": "user", "content": query})

    try:
        response = groq_client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=50,
            reasoning_effort="none",    
            reasoning_format="hidden",
        )
        raw = _strip_thinking(response.choices[0].message.content.strip())

        source = "guidelines"  
        language = "ar"

        for line in raw.splitlines():
            if line.upper().startswith("SOURCE:"):
                value = line.split(":", 1)[1].strip().lower()
                if value in ("arabic", "guidelines", "both"):
                    source = value
            elif line.upper().startswith("LANGUAGE:"):
                value = line.split(":", 1)[1].strip().lower()
                if value in ("ar", "en"):
                    language = value

        return {"source": source, "language": language}

    except Exception as e:
        print("Router API Error:", repr(e))
        raise


def translate_to_english(query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Translate the following medical question into English. "
                "Preserve drug names and medical terms accurately. "
                "Output ONLY the translated question, nothing else."
            )
        },
        {"role": "user", "content": query}
    ]

    try:
        response = groq_client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=150,
            reasoning_effort="none",
            reasoning_format="hidden",
        )
        translated = _strip_thinking(response.choices[0].message.content.strip())
        return translated if translated else query
    except Exception as e:
        print("Translation API Error:", repr(e))
        raise


def translate_to_arabic(query: str) -> str:

    messages = [
        {
            "role": "system",
            "content": (
                "Translate the following medical question into Modern Standard "
                "Arabic. Preserve drug names and medical terms accurately. "
                "Output ONLY the translated question, nothing else."
            )
        },
        {"role": "user", "content": query}
    ]

    try:
        response = groq_client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=150,
            reasoning_effort="none",
            reasoning_format="hidden",
        )
        translated = _strip_thinking(response.choices[0].message.content.strip())
        return translated if translated else query
    except Exception as e:
        print("Translation API Error:", repr(e))
        raise