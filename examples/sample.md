# Simple RAG sample document

This file was written for the Simple RAG project and is MIT licensed like the rest of the code.
Upload it in the web UI to try the app, then ask one of the questions at the end.

## What Simple RAG does

Simple RAG is a small local Retrieval-Augmented Generation app. You upload Markdown files,
the backend splits them into chunks, turns each chunk into a vector with an embedding API,
and stores the vectors in Milvus and the original text in SQLite.

When you ask a question, the backend embeds the question, searches Milvus for the Top-K
closest chunks, and sends those chunks together with the question to a local LLM served by
Ollama. It also asks the same LLM the question without any context, so you can compare the
two answers side by side.

## Why compare two answers

A language model only knows what was in its training data. Retrieval adds facts from your own
documents to the prompt. Showing the answer with context next to the answer without context
makes it easy to see what the retrieved chunks changed. The bar chart shows how similar each
answer is to each source chunk: an answer that used a chunk is usually closer to it.

## How chunking works

Text is split by character count. `CHUNK_SIZE` sets the length of each chunk and
`CHUNK_OVERLAP` sets how many characters two neighbouring chunks share. Overlap keeps a
sentence that falls on a boundary readable in at least one chunk.

## What an upload replaces

Every upload replaces the previous index. The backend first drops the Milvus collection and
then deletes all chunk rows from SQLite, so the two stores always describe the same documents.

## Questions to try

- Where does Simple RAG store vectors, and where does it store the original text?
- Why does the app show an answer without context?
- What does `CHUNK_OVERLAP` do?
