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

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel


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


# Make this look better in the docs.
class Question(BaseModel):
    __root__: str


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
    storage_type="hash",
)


vector_store = RedisVectorStore(embeddings, config=config)

# TODO allow user to change parameters
retriever = vector_store.as_retriever(search_type="mmr")


# Define our prompt
template = """
Use the following pieces of context from The 20 newsgroups dataset which 
comprises around 18000 newsgroups posts on 20 topics. 
We're using a subset for this demonstration and focus on two categories: 
'alt.atheism' and 'sci.space'. Use that dataset to answer the question. 
Do not make up an answer if there is no context provided to help answer it. 
Include the 'source' and 'start_index'
from the metadata included in the context you used to answer the question

Context:
---------
{context}

---------
Question: {question}
---------

Answer:
"""


prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(model="gpt-3.5-turbo-16k")


# RAG Chain
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
).with_types(input_type=Question)


tracer = trace.get_tracer("my.tracer")
with tracer.start_as_current_span("RAG Chain") as span:
    reply = chain.invoke(
        "Tell me about space exploration. Do you have any important person to mention?"
    )
    print(reply)
