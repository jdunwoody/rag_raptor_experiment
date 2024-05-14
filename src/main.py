import shutil
import re
from pathlib import Path

from langchain import hub
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import pickle
from clustering import recursive_embed_cluster_summarize
from splitting import load_text_splits
from timer import Timer

# https://github.com/langchain-ai/langchain/blob/master/cookbook/RAPTOR.ipynb


def load_model_and_embeddings():
    ollama_model_name = "llama3"
    model = Ollama(model=ollama_model_name)
    embeddings = OllamaEmbeddings(model=ollama_model_name)

    return model, embeddings

    # azure_config = AzureConfig.load(AZURE_CONFIG_PATH)
    #
    # model = AzureChatOpenAI(
    #     model=azure_config.engine,
    #     deployment_name=azure_config.engine,
    #     api_key=azure_config.api_key,
    #     azure_endpoint=azure_config.api_endpoint,
    #     api_version=azure_config.api_version,
    # )

    # embeddings = AzureOpenAIEmbeddings(
    #     azure_endpoint=azure_config.api_endpoint,
    #     model=azure_config.embedding_model,
    #     openai_api_key=azure_config.api_key,
    #     openai_api_version=azure_config.api_version,
    # )


def create_retriever(leaf_texts, results, embd):
    all_texts = leaf_texts.copy()

    for level in sorted(results.keys()):
        summaries = results[level][1]["summaries"].tolist()

        all_texts.extend(summaries)

    vectorstore = Chroma.from_texts(texts=all_texts, embedding=embd)
    retriever = vectorstore.as_retriever()

    return retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    cache_dir = Path(__file__).parents[1] / ".cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    print("Loading models and embeddings")
    model, embeddings = load_model_and_embeddings()

    print("Loading text")

    urls = [
        "https://www.bloombergmedia.com/blog/",
        "https://docs.haystack.deepset.ai/docs/intro",  # "https://openai.com/news/"
    ]

    leaf_texts, texts_split = load_text_splits(cache_dir=cache_dir, urls=urls)

    print("Clustering")
    cluster_results_cache_path = cache_dir / "cluster_results.pkl"
    if cluster_results_cache_path.exists():
        with open(cluster_results_cache_path, "rb") as fp:
            cluster_summary_results = pickle.load(fp)
        # cluster_results_cache_path.
    else:
        cluster_summary_results = recursive_embed_cluster_summarize(
            cache_dir=cache_dir,
            model=model,
            texts=leaf_texts,
            level=1,
            n_levels=5,
            embd=embeddings,
        )
        with open(cluster_results_cache_path, "wb") as fp:
            pickle.dump(cluster_summary_results, fp)

    prompt = hub.pull("rlm/rag-prompt")

    print("Populating vectordb")
    retriever = create_retriever(
        leaf_texts=leaf_texts, results=cluster_summary_results, embd=embeddings
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    print("RAG call")
    rag_result = rag_chain.invoke(
        "How to define a RAG chain? Give me a specific code example."
    )

    print("Output")
    results = ""

    leaf_texts = "\n--------------------------\n".join(leaf_texts)
    leaf_texts = re.sub("\n+", "\n", leaf_texts)

    # results+=leaf_texts
    cluster_results = {}
    final_results = {
        "input_leaf_docs": leaf_texts,
        "texts_split": texts_split,
        "final_result": rag_result,
        "cluster_results": cluster_results,
    }
    # for level, (cluster_result, summary_result) in cluster_summary_results.items():
    for level, (cluster_result, summary_result) in cluster_summary_results.items():
        cluster_results[level] = {
            "cluster": cluster_result.to_json(),
            "summary": summary_result.to_json(),
        }

    import json

    clean_results = results
    # clean_results = "".join(results.splitlines())
    output_path = Path(__file__).parents[1] / "output" / "output.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(clean_results)
    # with open("final_output.txt", mode="w") as f:
    # json.dump(final_results, f)

    return result


if __name__ == "__main__":

    main()
