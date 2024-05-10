import json
from hashlib import sha256

import tiktoken
from bs4 import BeautifulSoup as Soup
from langchain.schema import Document
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from timer import Timer


def load_text_splits(cache_dir):
    print("Loading text splits")
    docs_texts_path = cache_dir / "docs_texts.json"
    splits_path = cache_dir / "splits.json"

    if docs_texts_path.exists() and splits_path.exists():
        docs_texts = json.loads(docs_texts_path.read_text())
        texts_split = json.loads(splits_path.read_text())
    else:
        if not splits_path.exists():
            splits_path.parent.mkdir(parents=True, exist_ok=True)

        docs_texts, texts_split = do_load_text_splits(cache_dir)

        splits_path.write_text(json.dumps(texts_split))
        docs_texts_path.write_text(json.dumps(docs_texts))

    return docs_texts, texts_split


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def do_load_text_splits(cache_dir):
    with Timer(text="load_docs: {:.2f}s") as t:
        docs = load_docs(
            cache_dir=cache_dir,
            url="https://github.com/tnicolas/blpapipd",
        )

    docs_texts = [d.page_content.strip() for d in docs]

    # Calculate the number of tokens for each document
    counts = [num_tokens_from_string(d, "cl100k_base") for d in docs_texts]

    # Doc texts concat
    d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
    d_reversed = list(reversed(d_sorted))
    concatenated_content = "\n\n\n --- \n\n\n".join(
        [doc.page_content for doc in d_reversed]
    )
    print(
        "Num tokens in all context: %s"
        % num_tokens_from_string(concatenated_content, "cl100k_base")
    )

    # Doc texts split

    chunk_size_tok = 2000
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size_tok, chunk_overlap=0
    )
    texts_split = text_splitter.split_text(concatenated_content)

    return docs_texts, texts_split


def load_docs(cache_dir, url):
    print("Loading docs")
    char_limit = 13
    url_hash = sha256(url.encode()).hexdigest()[:char_limit]
    url_dir = cache_dir / url_hash

    if url_dir.exists():
        docs = tqdm(
            [
                Document(**json.loads(file.read_text(encoding="utf-8")))
                for file in url_dir.glob("*.json")
            ]
        )
    else:
        if not url_dir.exists():
            url_dir.mkdir(parents=True, exist_ok=True)

        loader = RecursiveUrlLoader(
            url=url,
            timeout=20,
            max_depth=4,
            extractor=lambda x: Soup(x, "html.parser").text,
        )
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc_url = url_dir / f"{i}.json"
            doc_url.write_text(doc.json(), encoding="utf-8")

    return docs
