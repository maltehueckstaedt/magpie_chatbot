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
# GENERATE STANDARD-AGENT-TOOLS ----------------------------
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# GENERATE CUSTOMIZED AGENT-TOOLS NR.1 variable_beschr -----
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

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

def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [string.strip() for string in res]
    return list(set(res))



def load_or_create_vectorstore(db, path="data/vectorstore_beschr_variable"):
    if os.path.exists(path):
        return FAISS.load_local(
        path,
        OpenAIEmbeddings(model="text-embedding-3-large"),
        allow_dangerous_deserialization=True
    )

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
retriever_beschr_variable  = vector_store.as_retriever(search_kwargs={"k": 10})
rt_beschr_variable  = create_retriever_tool(
    retriever_beschr_variable ,
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

# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# GENERATE CUSTOMIZED AGENT-TOOLS NR.2 variable_beschr -----
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# `get_reichweite_beschr_list` ermittelt eine passende Reichweite
#  basierend auf der Nutzerfrage und der zugehörigen Variable.
#
# Ablauf:
# - Nutzt das `variable_beschr` Tool, um die passende Variable 
#   aus der Frage zu ermitteln.
# - Prüft, ob eine valide Variable extrahiert wurde, sonst Abbruch mit Fehlermeldung.
# - Fragt die Datenbank nach allen gültigen Reichweiten für diese Variable ab.
# - Falls keine Reichweiten gefunden werden, fordert der Bot zur Präzisierung auf.
# - Baut einen InMemory-Vektorstore aus gültigen Reichweiten mit OpenAI-Embeddings auf.
# - Ruft mit Retriever die 5 besten Kandidaten für die Nutzerfrage ab.
# - Nutzt ein FewShot-Prompt mit Beispielen, um mittels LLM den besten Reichweiten-Wert zu
#   bestimmen. Das FewShot-Prompt stellt dem LLM konkrete Beispiel-Frage-Variable-Reichweite-
#   Paare zur Verfügung, um den Kontext und die richtige Auswahlmethode zu vermitteln (z.B. dann
#   wenn trotz semantisch ähnlicherer Reichweite "Deutschland" die richtige Reichweite ist). Dadurch
#   lernt das LLM anhand der Beispiele, die richtige Reichweite für die neue Frage präzise zu
#   bestimmen, auch wenn die Nutzerfrage leicht variiert.
# - Validiert, ob der vom LLM vorgeschlagene Wert unter den gültigen Reichweiten ist.
# - Gibt den validen Wert zurück oder fordert den Nutzer zur Konkretisierung auf.
#
# Zweck:
# - Präzise und kontextbasierte Ermittlung von Reichweiten aus freier Nutzereingabe.
# - Verknüpft Embeddings-gestützte Suche mit FewShot-Lernen für bessere Ergebnisqualität.
#
# Voraussetzungen:
# - `variable_beschr` Tool zur Variablenbestimmung muss vorhanden sein.
# - `db` ist die Datenbankverbindung mit der Tabelle `view_daten_reichweite_menge`.
# - `OpenAIEmbeddings`, `InMemoryVectorStore` und `llm` sind korrekt initialisiert.
# - `reichweite_prompt` ist ein FewShotPromptTemplate mit passenden Beispielen.
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

from langchain.prompts import FewShotPromptTemplate, PromptTemplate

reichweiten_beispiele = [
    {"frage": "Wie viele Absolventen für Berufliche Schulen gab es?", "variable_beschr": "Anzahl der Absolventen für Berufliche Schulen", "reichweite_beschr_list": "Deutschland"},
    {"frage": "Wie hoch war die Studierquote bildungsferner Schichten?", "variable_beschr": "Studierquote bildungsferne Schichten", "reichweite_beschr_list": "Deutschland"},
    {"frage": "Wie viele dauerhaft eingestellte Lehrkräfte (inkl. Seiteneinsteigern, ohne Referendare) gab es?", "variable_beschr": "Anzahl dauerhaft eingestellte Lehrkräfte (inkl. Seiteneinsteigern, ohne Referendare)", "reichweite_beschr_list": "Deutschland"},
    {"frage": "Wie hoch war der Handlungsfeldindex: Lehrer Bildung?", "variable_beschr": "Handlungsfeldindex: Lehrer Bildung", "reichweite_beschr_list": "Deutschland"},
    {"frage": "Wie viele Universitätsschulverbünde gab es?", "variable_beschr": "Anzahl Universitätsschulverbünde", "reichweite_beschr_list": "Deutschland"},
    {"frage": "Wie hoch war der Anteil berufsbegleitender Master?", "variable_beschr": "Anteil berufsbegleitender Master", "reichweite_beschr_list": "Deutschland"},
    {"frage": "Wie viele Studienabsolventen T gab es?", "variable_beschr": "Studienabsolventen T", "reichweite_beschr_list": "Deutschland"},
    {"frage": "Wie hoch waren die internen FuE-Aufwendungen?", "variable_beschr": "Interne FuE-Aufwendungen", "reichweite_beschr_list": "Deutschland"},
    {"frage": "Wie hoch war der Anteil der männlichen Grundschullehramtsstudierenden?", "variable_beschr": "Anteil der männlichen Grundschullehramtsstudierende", "reichweite_beschr_list": "Deutschland"},
    {"frage": "Wie viele Studienabsolventen im Weiterbildungsstudium gab es?", "variable_beschr": "Studienabsolventen im Weiterbildungsstudium", "reichweite_beschr_list": "Deutschland"},
    {"frage": "Wie hoch waren die Drittmittel vom Bund 2021 in Deutschland?", "variable_beschr": "Drittmittel vom Bund", "reichweite_beschr_list": "Deutschland"}
]

example_prompt = PromptTemplate(
    input_variables=["frage", "variable_beschr", "reichweite_beschr_list"],
    template="Frage: {frage}\nVariable: {variable_beschr}\n→ Reichweite: {reichweite_beschr_list}"
)

reichweite_prompt = FewShotPromptTemplate(
    examples=reichweiten_beispiele,
    example_prompt=example_prompt,
    prefix="Wähle aus den möglichen Reichweiten die beste. Nutze 'Deutschland', wenn keine Region, Organisation o. Ä. genannt wird.",
    suffix="Frage: {frage}\nVariable: {variable_beschr}\nKandidaten:\n{kandidaten}\n→ Reichweite:",
    input_variables=["frage", "variable_beschr", "kandidaten"]
)

@tool
def get_reichweite_beschr_list(user_question: str) -> str:
    """
    Ermittelt eine passende Reichweite (z. B. Region, Organisation, etc.), basierend auf der
    zur Frage gehörigen Variable und den verfügbaren Einträgen in der Datenbank.
    """
    print("[DEBUG] Eingabe-Frage:", user_question)

    raw_variable = variable_beschr.run(user_question)
    print("[DEBUG] raw_variable:", raw_variable)

    match = re.search(r"'([^']+)'", str(raw_variable))
    if not match:
        print("[DEBUG] Abbruch: Keine gültige Variable extrahiert")
        return "Fehler: Konnte keine gültige Variable bestimmen."

    variable = match.group(1)
    print("[DEBUG] bereinigte variable:", variable)

    if "Error" in variable:
        return "Fehler: Konnte keine gültige Variable bestimmen."

    escaped_variable = variable.replace("'", "''")
    print("[DEBUG] escaped_variable:", escaped_variable)

    query = f"""
        SELECT DISTINCT reichweite_beschr_list 
        FROM view_daten_reichweite_menge 
        WHERE variable_beschr = '{escaped_variable}'
    """
    print("[DEBUG] SQL-Abfrage gültige_reichweiten:", query)
    gültige_reichweiten = query_as_list(db, query)
    print("[DEBUG] gültige_reichweiten:", gültige_reichweiten)

    if not gültige_reichweiten:
        return "[USER_CLARIFICATION_NEEDED] Ich konnte keine passende Reichweite ermitteln. Bitte präzisieren Sie, welche Region oder Organisation gemeint ist."


    vector_store = InMemoryVectorStore(OpenAIEmbeddings(model="text-embedding-3-large"))
    _ = vector_store.add_texts(gültige_reichweiten)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    top_matches = retriever.get_relevant_documents(user_question)
    reichweiten_kandidaten = [doc.page_content for doc in top_matches]
    print("[DEBUG] Top 5 Reichweiten-Kandidaten:", reichweiten_kandidaten)

    kandidaten_text = "\n".join(reichweiten_kandidaten)

    llm_chain = reichweite_prompt | llm
    best_match = llm_chain.invoke({
        "frage": user_question,
        "variable_beschr": variable,
        "kandidaten": kandidaten_text
    }).content.strip()

    print("[DEBUG] LLM-best_match:", best_match)

    # Validierung: nur erlaubte Rückgabe
    if best_match not in gültige_reichweiten:
        print(f"[DEBUG] LLM-Match ungültig ('{best_match}'), Rückfrage erforderlich")
        return "[USER_CLARIFICATION_NEEDED] Ich konnte keine passende Reichweite ermitteln. Bitte konkretisieren Sie Ihre Anfrage."
        
    query = f"""
        SELECT reichweite_beschr_list 
        FROM view_daten_reichweite_menge 
        WHERE reichweite_beschr_list = '{best_match}' 
        LIMIT 1;
    """
    print("[DEBUG] SQL-Abfrage finale Auswahl:", query)
    result = db.run_no_throw(query)
    print("[DEBUG] Ergebnis:", result)

    return result if result else "Error: Keine passende Reichweite gefunden."

tools.extend([variable_beschr, get_reichweite_beschr_list])

# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# GENERATE ReAct-AGENT -------------------------------------
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Wir laden eine vorgefertigtes Prompt-Template aus dem LangChain Hub.
#
# Ablauf:
# - Das Prompt-Template wird mit `hub.pull` geladen, speziell für einen SQL-Agenten.
# - Es wird geprüft, dass genau eine Nachricht im Template vorhanden ist.
# - Anschließend wird der bestehende Text der Systemnachricht erweitert um
#   einen Hinweis, dass der Chatbot "Sparklehorse" für Stifterverband ist und
#   sich auf Fragen zur Magpie-Datenbank spezialisiert hat.
# - Danach wird das Template formatiert, indem Platzhalter wie SQL-Dialekt und
#   die Anzahl der Top-K Ergebnisse gesetzt werden.
# - Am Ende wird die fertige Systemnachricht ausgegeben.
#
# Zweck:
# - Anpassung eines generischen SQL-Agenten-Prompts für spezifische Anforderungen.
# - Ermöglicht die Erstellung eines spezialisierten Chatbots mit Kontextwissen.
#
# Voraussetzungen:
# - `langchain` und der `hub` müssen installiert sein.
# - `db` muss eine Datenbankverbindung mit `dialect`-Attribut sein.
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

from langchain import hub

prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")

assert len(prompt_template.messages) == 1, "Die Anzahl der Nachrichten im Template ist nicht 1!"
# Bearbeite die bestehende Nachricht, indem du Text hinzufügst
prompt_template.messages[0].prompt.template += (
    "\nYou are Sparklehorse, a chatbot for the Stifterverband organization. "
    "Your primary task is to answer questions related to the Magpie database."
)

system_message = prompt_template.format(
    dialect=db.dialect, 
    top_k=5
) 

# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# GENERATE AGENT -------------------------------------
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Systemnachricht mit extra Anweisungen
suffix = (
    "1. Nutze das Tool `variable_beschr`, um die korrekte Variable aus der Nutzerfrage zu bestimmen. Verwende ausschließlich den exakten Rückgabewert dieses Tools für `variable_beschr` in der SQL-Abfrage.\n"
    "2. Nutze das Tool `get_reichweite_beschr_list`, um die passende Reichweite zu ermitteln. Verwende ausschließlich den Rückgabewert dieses Tools für `reichweite_beschr_list` in der SQL-Abfrage.\n"
    "3. Verwende **niemals** andere Felder wie `tag_list` oder `LIKE`-Abfragen. Nutze **immer exakte Vergleiche** mit `=`.\n"
    "4. Verwende ausschließlich die Tabelle `view_daten_reichweite_menge` für alle Abfragen.\n"
    "5. Falls ein Jahr in der Frage genannt wird, filtere mit `date_part('year', zeit_start) = <Jahr>`.\n"
    "6. Berücksichtige die Spalte `wert_einheit`, z. B. 'in Tsd. Euro', 'Anzahl', 'Prozent', 'VZÄ', 'Mitarbeiter'.\n"
    "7. Gib immer die finale SQL-Abfrage vollständig aus und erkläre sie. Rate niemals IDs oder Werte.\n"
    "8. Falls keine passende Variable oder Reichweite gefunden wurde, rate nicht irgendwelche Werte.\n"
    "9. Stelle sicher, dass Antworttext und SQL-Abfrage immer auf den gleichen `variable_beschr`- und `reichweite_beschr_list`-Werten basieren, um Konsistenz zu gewährleisten.\n"
    "10. Verwende in deiner Antwort exakt die Begriffe, die du in der SQL-Abfrage benutzt hast. Nutze insbesondere den Wert aus `reichweite_beschr_list` vollständig im Antwortsatz. Beispiel: Wenn `reichweite_beschr_list = 'Wirtschaftssektor | Deutschland'`, schreibe: 'im Wirtschaftssektor in Deutschland'."
)

system = f"{system_message}\n\n{suffix}"

# Neuen ReAct-Agent erstellen mit den vollständigen Tools
agent_executor = create_react_agent(llm, tools, state_modifier=system)

def stream_agent_with_check(question: str):
    stream = agent_executor.stream({"messages": [HumanMessage(content=question)]}, stream_mode="values")
    for step in stream:
        msg = step["messages"][-1]
        if "[USER_CLARIFICATION_NEEDED]" in msg.content:
            rückfrage = msg.content.replace("[USER_CLARIFICATION_NEEDED]", "").strip()
            print(f"⚠️ Rückfrage: {rückfrage}")
            break
        else:
            msg.pretty_print()

if __name__ == "__main__":
    while True:
        user_input = input("Frage an Sparklehorse (oder 'exit'): ")
        if user_input.lower() == "exit":
            break
        stream_agent_with_check(user_input)