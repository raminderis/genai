from langchain.text_splitter import RecursiveCharacterTextSplitter

with open("D:\\tech-learning\\AI-ML\\Visionworks\\chatagent\\ping_text.txt", "r", encoding="utf8") as file:
    icmp_text = file.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", "", "."]
)

chunks = text_splitter.split_text(icmp_text)


from langchain.schema import Document

chunk_docs = [
    Document(page_content=chunk, metadata={"section": f"chunk-{i}"})
    for i, chunk in enumerate(chunks)
]

# Generate embeddings
from langchain_openai import OpenAIEmbeddings
embedding_model = OpenAIEmbeddings(
    openai_api_key="sk-proj",
)

embedded_vectors = embedding_model.embed_documents([doc.page_content for doc in chunk_docs])

from langchain_neo4j import Neo4jGraph
graph = Neo4jGraph(
    url="bolt://3.83.330.60:7687",
    username="neo4j",
    password="crosses--tags"
)

for i, (doc, vector) in enumerate(zip(chunk_docs, embedded_vectors)):
    graph.query(
        """
        MATCH (parent:StandardDoc {title: $title})
        CREATE (c:StandardDocChunk {
            content: $content,
            section: $section,
            embedding: $embedding
        })-[:PART_OF]->(parent)
        """,
        {
            "title": "ICMP Protocol Reference",
            "content": doc.page_content,
            "section": doc.metadata["section"],
            "embedding": vector,
        }
    )
