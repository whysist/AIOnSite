import re
from typing import List
from .schemas import Evidence


def rerank_evidence(query: str, candidates: List[Evidence]) -> List[Evidence]:
    """Optional local reranking and metadata scoring:
    - Boosts authoritative limits when query asks for limits/standards.
    - Boosts observation/inspection documents when query asks for current readings.
    - Higher score for exact keyword occurrences.
    - Higher weight for latest revisions.
    """
    if not candidates:
        return []

    q_lower = query.lower()
    is_limit_query = any(w in q_lower for w in ("limit", "max", "normal", "range", "design", "allowed", "sop"))
    is_reading_query = any(w in q_lower for w in ("observed", "reading", "inspection", "current", "measured", "finding"))
    query_tokens = [t for t in re.findall(r'\w+', q_lower) if len(t) > 2]

    reranked = []
    for ev in candidates:
        score = ev.score
        txt = ev.text.lower()

        # Keyword density boost
        match_count = sum(1 for t in query_tokens if t in txt)
        score += (match_count * 0.05)

        # Authority alignment boost
        if is_limit_query and ev.authority in ("authoritative_limit", "sop"):
            score += 0.20
        elif is_reading_query and ev.authority == "observation":
            score += 0.20

        # Revision boost (Rev B preferred over Rev A if present)
        if ev.revision == "B":
            score += 0.05

        ev_copy = ev.model_copy()
        ev_copy.score = min(1.0, score)
        reranked.append(ev_copy)

    reranked.sort(key=lambda x: x.score, reverse=True)
    return reranked
