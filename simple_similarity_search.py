import redis
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_openai import OpenAIEmbeddings

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Step 1: Set up the tracer provider and the OTLP exporter
print("Set up the tracer provider and the OTLP exporter...")
resource = Resource(attributes={"service.name": "redis-llm-sample"})
provider = TracerProvider(resource=resource)

# Step 2: Configure OTLP exporter
print("Configure OTLP exporter...")
otlp_exporter = OTLPSpanExporter(
    endpoint="localhost:4317",  # Default OTLP endpoint for gRPC
    insecure=True,  # Set to True if no TLS/SSL is used
)

# Step 3: Add a BatchSpanProcessor to send spans to OTLP endpoint
span_processor = BatchSpanProcessor(otlp_exporter)
provider.add_span_processor(span_processor)

# Step 4: Set the global tracer provider
trace.set_tracer_provider(provider)

# Step 5: Instrument Redis with OpenTelemetry
print("Instrument Redis with OpenTelemetry...")
RedisInstrumentor().instrument()


######################
# Vector Store related
######################

print(f"Using OpenAI to create the embeddings")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

redis_client = redis.StrictRedis(host="localhost", port=6379)
redis_client.ping()

config = RedisConfig(
    client=redis_client,
    index_name="newsgroups",
    metadata_schema=[
        {"name": "category", "type": "tag"},
    ],
)


vector_store = RedisVectorStore(embeddings, config=config)
# Query directly

query = "Tell me about space exploration"
results = vector_store.similarity_search(query, k=2)

print("Simple Similarity Search Results:")
for doc in results:
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
    print()

# Similarity search with score and filter
scored_results = vector_store.similarity_search_with_score(query, k=2)

print("Similarity Search with Score Results:")
for doc, score in scored_results:
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
    print(f"Score: {score}")
    print()
