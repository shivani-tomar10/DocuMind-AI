
# --------------------------------
# Build Context from Retrieved Chunks
# --------------------------------

def build_context(results):

    context = ""

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    for i, document in enumerate(documents):

        source = metadatas[i]["source"]
        page = metadatas[i]["page_number"]

        context += f"""
Source: {source}
Page: {page}

Content:
{document}

-------------------------
"""

    return context
