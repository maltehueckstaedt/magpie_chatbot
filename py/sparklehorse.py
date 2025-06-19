# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# LOAD PACKAGES --------------------------------------------
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


import os
import ast
import re
import sys
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain import hub
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.chains import LLMChain
from langchain_community.vectorstores.faiss import FAISS


# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# DEFINE WORKSPACE/ LOAD ENV -------------------------------
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


# Arbeitsverzeichnis
os.chdir("c:/Users/mhu/Documents/gitHub/magpie_chatbot")
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")
 
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# CONNECT TO MAGPIE-MVIEW ----------------------------------
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

db = SQLDatabase.from_uri("duckdb:///data/view_magpie.db")

# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# GENERATE AGENT-TOOLS -------------------------------------
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# `load_or_create_vectorstore` lädt einen lokal gespeicherten
# FAISS-Vektorstore mit Embeddings der Variablenbeschreibungen
# oder erstellt ihn neu, falls noch nicht vorhanden.
#
# Ablauf:
# 1. Prüft, ob der Vektorstore im angegebenen Pfad existiert:
#    - Wenn ja: Lädt den bestehenden Vektorstore.
#    - Wenn nein: Lädt die `variable_beschr`-Daten aus der
#      Datenbank, erzeugt Embeddings mit OpenAI,
#      baut den Vektorstore auf und speichert ihn lokal.
#
# Danach wird ein Retriever aus dem Vektorstore erzeugt, der als
# Tool `rt_beschr_variable` verwendet wird, um anhand von
# Nutzeranfragen ähnliche Variablennamen zu finden.
#
# Zweck:
# - Effiziente Suche nach passenden Variablennamen via
#   Text-Embedding-Ähnlichkeit.
# - Vermeidet Neuberechnung der Embeddings durch
#   Zwischenspeicherung.
#
# Voraussetzungen:
# - `db` ist die Datenbankverbindung.
# - `create_retriever_tool` ist eine Hilfsfunktion zur
#   Tool-Erstellung in LangChain.
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


def load_or_create_vectorstore(db, path="data/vectorstore_beschr_variable"):
    if os.path.exists(path):
        return FAISS.load_local(path, OpenAIEmbeddings(model="text-embedding-3-large"))

    res = db.run("SELECT variable_beschr FROM view_daten_reichweite_menge")
    beschr_variable = list(set(
        string.strip()
        for sub in ast.literal_eval(res)
        for string in sub if string
    ))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_texts(beschr_variable, embeddings)
    vector_store.save_local(path)
    return vector_store

vector_store = load_or_create_vectorstore(db)
retriever = vector_store.as_retriever(search_kwargs={"k": 10})
rt_beschr_variable = create_retriever_tool(
    retriever,
    name="rt_beschr_variable",
    description="..."
)

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Tool `variable_beschr` nutzt ein LLM und Embeddings, um aus
# einer Nutzerfrage die passendste Variable zu bestimmen und
# gibt die exakte Variable aus der Datenbank zurück.
#
# Ablauf:
# - Sucht mit dem Retriever nach relevanten Dokumenten zu
#   Nutzerfrage.
# - Falls keine Treffer, wird eine Fehlermeldung zurückgegeben.
# - Listet Kandidaten-Variablen auf und gibt sie aus.
# - Nutzt ein Prompt, das exakt eine passende Variable
#   auswählen soll, ohne Synonyme oder Oberkategorien.
# - Fragt das LLM nach der besten Übereinstimmung zur Frage.
# - Führt eine Datenbankabfrage mit der besten Variable aus.
# - Gibt das Ergebnis zurück oder fordert bei Misserfolg
#   zur genaueren Eingabe auf.
#
# Zweck:
# - Präzise Zuordnung von Nutzereingaben zu Datenbankvariablen
#   mittels LLM-gestützter Auswahl.
#
# Voraussetzungen:
# - `retriever_beschr_variable` ist ein konfigurierter Retriever.
# - `llm` ist das verwendete Sprachmodell.
# - `db` ist die Datenbankverbindung.
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

@tool
def variable_beschr(user_question: str) -> str:
    """
    Nutzt ein LLM und Embeddings, um aus der Frage eine passende Variable zu bestimmen
    und gibt dann die exakte Variable aus der Datenbank zurück.
    """
    docs = retriever_beschr_variable.get_relevant_documents(user_question)
    if not docs:
        return "Error: Keine passende Variable gefunden."

    kandidaten = "\n".join(f"- {doc.page_content.strip()}" for doc in docs)
    print(kandidaten)

    auswahl_prompt = PromptTemplate(
        input_variables=["frage", "kandidaten"],
        template="""
    Wähle exakt **eine** der folgenden Variablen, die am besten zur Frage passt.
    Wähle **nur dann** eine Variable aus, wenn sie **exakt** zur Frage passt.
    Nutze **keine verwandten Begriffe**, Oberkategorien oder Synonyme.
    Gib den Text **genau so** zurück, wie er bei den Kandidaten steht.

    Frage: {frage}

    Kandidaten:
    {kandidaten}

    Beste Variable:
    """
    )
    auswahl_chain = auswahl_prompt | llm
    best_match = auswahl_chain.invoke({
        "frage": user_question,
        "kandidaten": kandidaten
    }).content.strip()

    query = f"""
        SELECT variable_beschr 
        FROM view_daten_reichweite_menge 
        WHERE variable_beschr = '{best_match}' 
        LIMIT 1;
    """
    result = db.run_no_throw(query)
    if result:
        return result
    else:
        return "[USER_CLARIFICATION_NEEDED] Ich konnte keine passende Variable finden. Bitte geben Sie die gewünschte Variable genauer an."