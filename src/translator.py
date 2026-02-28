from langchain_core.prompts import ChatPromptTemplate
# Change 'langchain' to 'langchain_classic'
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

def get_tos_chain(llm, vector_db):
    """Creates a RAG chain for ToS analysis."""
    
    # System prompt forces the AI to be concise and highlight risks
    system_prompt = (
        "You are a legal expert specializing in consumer rights. "
        "Use the provided context from a Terms of Service document to "
        "answer the user's question in plain, simple English. "
        "If there is a hidden risk or something unfair, highlight it with a ⚠️. "
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create the retrieval chain
    # k=3 means it will look at the 3 most relevant paragraphs
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever, combine_docs_chain)