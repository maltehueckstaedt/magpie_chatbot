# *Sparklehorse* <img src="img/sparklehorse_logo.svg" align="right" height="218" alt="ggplot2 website" />

![LangChain](https://img.shields.io/badge/Powered%20by-LangChain-blue)
![Ollama](https://img.shields.io/badge/Ollama-LLM%20Integration-orange)
![Python](https://img.shields.io/badge/Developed%20with-Python-yellow)
![DuckDB](https://img.shields.io/badge/Fast%20Queries-DuckDB-green)

## Was ist das *Sparklehorse*?

*Sparklehorse* ist ein in Entwicklung befindlicher Chatbot der Datenbank Magpie. Um den Chatbot Lokal hosten zu können, wird die LLM-Distribution [Ollama](https://ollama.com/) verwendet. Der Chatbot verwendet derzeit die API von OpenAI. Als Agent wird `gpt-4o` verwendet, für die Embeddings `text-embedding-3-large`.

## Entwicklung

### Aktuelles Modell: Retrieval-Augmented Generation (RAG) mit Langchain, DuckDB

*Sparklehorse* kann derzeit erste Fragen zu den Daten des Daten-Navigators des Stifterverbandes beantworten. *Sparklehorse* arbeitet dabei derzeit wie folgt:

 <img src="img/curent_model_v1.png" height="710" />