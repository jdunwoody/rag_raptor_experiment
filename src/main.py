from pathlib import Path

from langchain import hub
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from clustering import load_text_splits, recursive_embed_cluster_summarize
from timer import Timer
from splitting import load_text_splits

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


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    cache_dir = Path(__file__).parents[1] / ".cache"

    model, embeddings = load_model_and_embeddings()

    leaf_texts, texts_split = load_text_splits(cache_dir)

    results = recursive_embed_cluster_summarize(
        cache_dir=cache_dir,
        model=model,
        texts=leaf_texts,
        level=1,
        n_levels=2,
        embd=embeddings,
    )

    prompt = hub.pull("rlm/rag-prompt")

    retriever = create_retriever(
        leaf_texts=leaf_texts, results=results, embd=embeddings
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    result = rag_chain.invoke(
        "How to define a RAG chain? Give me a specific code example."
    )

    return result


if __name__ == "__main__":

    main()
