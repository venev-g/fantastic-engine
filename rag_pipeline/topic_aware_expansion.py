from typing import List

def expand_query_with_topics(query: str, topics: List[str]) -> str:
    expanded_query = f"{query} Related topics: {', '.join(topics)}"
    return expanded_query