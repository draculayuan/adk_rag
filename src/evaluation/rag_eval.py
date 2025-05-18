from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
)

# Initialize the agent engine
agent_engine = vertexai.agent_engines.get(
    "projects/163097687798/locations/us-central1/reasoningEngines/8537074470983041024"
)
session = agent_engine.create_session(user_id="test_user")


def call_agent(query):
    response_text = ""
    contexts = []
    for event in agent_engine.stream_query(
        user_id="test_user", session_id=session["id"], message=query
    ):
        text = event["content"]["parts"][0].get("text", None)
        function_response = event["content"]["parts"][0].get("function_response", None)
        role = event["content"]["role"]
        if (text is not None) and (role == "model"):
            response_text += text
        elif function_response is not None:
            contexts.extend(
                [r["text"] for r in function_response["response"]["result"]]
            )
    return response_text, contexts


def run_rag(q: str):
    while True:  # handle rate limit
        try:
            answer, contexts = call_agent(q)
            break
        except:
            pass
    return answer, contexts


test_data = pd.read_csv("/home/jupyter/code_test/adk_rag/data/vpost_faqs.csv")
TEST_SET = [
    {"question": row["question"], "ground_truth": row["answer"]}
    for _, row in test_data.iterrows()
]

records = []
for sample in TEST_SET:
    ans, ctxs = run_rag(sample["question"])
    records.append(
        {
            "question": sample["question"],
            "answer": ans,
            "contexts": ctxs,
            "ground_truth": sample["ground_truth"],
        }
    )

eval_ds = Dataset.from_list(records)

chat_llm = ChatVertexAI(
    model_name="gemini-2.5-pro-preview-05-06",  # any Gemini family model you’ve enabled
    project="yuan-449301",
    location="us-central1",
    temperature=0,
)
llm = LangchainLLMWrapper(chat_llm)

lc_embed = VertexAIEmbeddings(
    model_name="text-embedding-005"  # fastest/cheapest text emb model today
)
embedder = LangchainEmbeddingsWrapper(lc_embed)

results = evaluate(
    eval_ds,
    metrics=[
        faithfulness,
        answer_correctness,
        answer_relevancy,
        context_recall,  # how retrieved content aligns with ground truth answer
        context_precision,
    ],
    embeddings=embedder,  # "mistral-embed"
    llm=llm,
)

print(results)
