from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid


# In-memory Qdrant instance
client = QdrantClient(":memory:")

collection_name = "nav_knowledge"


# Create collection if it doesn't exist
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)


def store_vector(label, vector):

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={"label": label}
    )

    client.upsert(
        collection_name=collection_name,
        points=[point]
    )


def search_vectors(query_vector):

    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=2
    )

    return [r.payload["label"] for r in results]
