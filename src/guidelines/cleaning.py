import re
from collections import Counter
from typing import List


def _is_standalone_page_number(line: str) -> bool:
    stripped = line.strip()
    return bool(re.fullmatch(r'-?\s*\d{1,4}\s*-?', stripped))


def detect_boilerplate_lines(page_texts: List[str], min_frequency: float = 0.4) -> set:
    line_counts = Counter()
    for page in page_texts:
        seen_this_page = set(l.strip() for l in page.split('\n') if l.strip())
        for line in seen_this_page:
            line_counts[line] += 1

    threshold = max(2, int(len(page_texts) * min_frequency))
    return {line for line, count in line_counts.items() if count >= threshold}


def clean_page_text(text: str, boilerplate_lines: set) -> str:
    cleaned_lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append('')
            continue
        if stripped in boilerplate_lines:
            continue
        if _is_standalone_page_number(stripped):
            continue
        cleaned_lines.append(line)

    cleaned = '\n'.join(cleaned_lines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def clean_document_pages(page_texts: List[str]) -> List[str]:
    boilerplate = detect_boilerplate_lines(page_texts)
    return [clean_page_text(page, boilerplate) for page in page_texts]