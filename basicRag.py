import cohere
import wikipedia
#!pip install wikipedia --quiet
#!pip install -qU langchain-text-splitters --quiet

co = cohere.Client(api_key="WY4EUt2yyQj8TJ6ZQTnuAx3EElbU5ojLJdBhySka")

article = wikipedia.page('Dune Part Two')
text = article.content
print(f"The text has roughly {len(text.split())} words.")
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

chunks_ = text_splitter.create_documents([text])
chunks = [c.page_content for c in chunks_]
print(f"The text has been broken down in {len(chunks)} chunks.")

model="embed-english-v3.0"
response = co.embed(
    texts= chunks,
    model=model,
    input_type="search_document",
    embedding_types=['float']
)
embeddings = response.embeddings.float
print(f"We just computed {len(embeddings)} embeddings.")

import numpy as np
vector_database = {i: np.array(embedding) for i, embedding in enumerate(embeddings)}

query = "Name everyone involved in writing the script, directing, and producing 'Dune: Part Two'?"

response = co.embed(
    texts=[query],
    model=model,
    input_type="search_query",
    embedding_types=['float']
)
query_embedding = response.embeddings.float[0]
print("query_embedding: ", query_embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = [cosine_similarity(query_embedding, chunk) for chunk in embeddings]
print("similarity scores: ", similarities)

sorted_indices = np.argsort(similarities)[::-1]

top_indices = sorted_indices[:10]
print("Here are the indices of the top 10 chunks after retrieval: ", top_indices)

top_chunks_after_retrieval = [chunks[i] for i in top_indices]
print("Here are the top 10 chunks after retrieval: ")
for t in top_chunks_after_retrieval:
    print("== " + t)

response = co.rerank(
    query=query,
    documents=top_chunks_after_retrieval,
    top_n=3,
    model="rerank-english-v2.0",
)

preamble = """
## Task & Context
You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.
"""


documents = [
    {"title": "chunk 0", "snippet": top_chunks_after_retrieval[0]},
    {"title": "chunk 1", "snippet": top_chunks_after_retrieval[1]},
    {"title": "chunk 2", "snippet": top_chunks_after_retrieval[2]},
  ]

response = co.chat(
  message=query,
  documents=documents,
  preamble=preamble,
  model="command-r",
  temperature=0.3
)

print("Final answer:")
print(response.text)

def insert_citations_in_order(text, citations):
    """
    A helper function to pretty print citations.
    """
    offset = 0
    document_id_to_number = {}
    citation_number = 0
    modified_citations = []

    # Process citations, assigning numbers based on unique document_ids
    for citation in citations:
        citation_numbers = []
        for document_id in sorted(citation.document_ids):
            if document_id not in document_id_to_number:
                citation_number += 1  # Increment for a new document_id
                document_id_to_number[document_id] = citation_number
            citation_numbers.append(document_id_to_number[document_id])

        # Adjust start/end with offset
        start, end = citation.start + offset, citation.end + offset
        placeholder = ''.join([f'[{number}]' for number in citation_numbers])
        # Bold the cited text and append the placeholder
        modification = f'**{text[start:end]}**{placeholder}'
        # Replace the cited text with its bolded version + placeholder
        text = text[:start] + modification + text[end:]
        # Update the offset for subsequent replacements
        offset += len(modification) - (end - start)

    # Prepare citations for listing at the bottom, ensuring unique document_ids are listed once
    unique_citations = {number: doc_id for doc_id, number in document_id_to_number.items()}
    citation_list = '\n'.join([f'[{doc_id}] source: "{documents[doc_id - 1]["snippet"]}"' for doc_id, number in sorted(unique_citations.items(), key=lambda item: item[1])])
    text_with_citations = f'{text}\n\n{citation_list}'

    return text_with_citations


print(insert_citations_in_order(response.text, response.citations))
