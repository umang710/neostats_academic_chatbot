try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

def search_web(query, max_results=3):
    results_text = []

    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)

            for r in results:
                results_text.append(r["body"])

    except Exception as e:
        print("Web search error:", e)

    return "\n".join(results_text)