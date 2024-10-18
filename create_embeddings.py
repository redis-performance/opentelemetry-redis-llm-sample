from langchain_redis import RedisConfig, RedisVectorStore
import redis
from redis.exceptions import ResponseError
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from sklearn.datasets import fetch_20newsgroups

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

categories = ["alt.atheism", "sci.space"]
print("Fetching docs...")
newsgroups = fetch_20newsgroups(
    subset="train", categories=categories, shuffle=True, random_state=42
)

# Use only the first 250 documents
texts = newsgroups.data[:250]
metadata = [
    {"category": newsgroups.target_names[target]} for target in newsgroups.target[:250]
]

print(f"There are a total of {len(texts)} docs")

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

print("Add items to vector store")
ids = vector_store.add_texts(texts, metadata)

print(ids[0:10])
