# Simple RAG sample document

Simple RAG is a small local Retrieval-Augmented Generation app. You upload Markdown files,
the backend splits them into chunks, turns each chunk into a vector with an embedding API,
and stores the vectors in Milvus and the original text in SQLite.

When you ask a question, the backend embeds the question, searches Milvus for the Top-K
closest chunks, and sends those chunks together with the question to a local LLM served by
Ollama. It also asks the same LLM the question without any context, so you can compare the
two answers side by side.

## 中文说明

Simple RAG 是一个可在本地运行的检索增强生成程序。上传 Markdown 文档后，后端会把文本切块、
调用 Embedding 接口生成向量，向量存进 Milvus，原文存进 SQLite。

提问时，后端先把问题向量化，在 Milvus 中检索最相近的 Top-K 个文本块，再把这些文本块和问题
一起交给 Ollama 上的本地大模型生成答案；同时还会让同一个模型在没有上下文的情况下直接回答，
方便对比两种答案。
