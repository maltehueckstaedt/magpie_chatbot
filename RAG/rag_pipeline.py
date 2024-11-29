# %%
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import ollama
import os 
import re  # Regular Expressions importieren

print(os.getcwd())  # Zeigt den aktuellen Arbeitsverzeichnis-Pfad an
os.chdir('c:/Users/Hueck/OneDrive/Dokumente/GitHub/magpie_langchain')
# Lade die Datenbank
data = pd.read_excel("data/mview_daten_beschr.xlsx")
# Modell ID
MODEL_ID = "llama3.1:8b-instruct-q4_0"
ollama.pull(MODEL_ID)

# Funktion zur Abfrage relevanter Daten aus der Excel-Datenbank
def retrieve_relevant_data(data, query):
    # Extrahiere das Jahr aus der Frage
    year_match = re.search(r'\b(19|20)\d{2}\b', query)
    year = year_match.group() if year_match else None

    # Suche nach relevanten Daten basierend auf Variable und Jahr
    relevant_data = []
    for _, row in data.iterrows():
        if (year and year in str(row['zeit_start'])) and ('intern' in query.lower() or 'fué' in query.lower()):
            relevant_data.append(
                f"Von {row['zeit_start']} bis {row['zeit_ende']}: {row['variable']} betrug {row['wert']} {row['wert_einheit']} in {row['reichweite']}."
            )
    return " ".join(relevant_data) if relevant_data else "Keine relevanten Daten gefunden."

# System Prompt für das Modell
def create_system_prompt(context):
    return f"""
    Du bist ein KI-Assistent, der Fragen ausschließlich zu den folgenden Daten beantwortet:
    {context}
    Du darfst keine Informationen aus externem Wissen verwenden. Ignoriere alle anderen Datenquellen.
    Wenn die benötigte Information nicht in den Daten enthalten ist, antworte: 'Diese Information ist in den bereitgestellten Daten nicht verfügbar.' 
    Antworte kurz und präzise auf Deutsch. Verwende Emojis für Benutzerfreundlichkeit.
    """

# Funktion zur Beantwortung der Frage mit Ollama
def ask_question_with_ollama(query, context):
    system_prompt = create_system_prompt(context)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    response = ollama.chat(
        model=MODEL_ID,
        messages=messages,
        options={'temperature': 0.1, 'num_predict': 256, "top_p": 0.9}
    )
    return response['message']['content']

# Hauptprogramm
if __name__ == "__main__":
    print("Willkommen zur RAG-Pipeline! Stelle Fragen zu deinen Daten. Tippe 'exit', um zu beenden.")
    while True:
        user_question = input("Frage: ")
        if user_question.lower() == "exit":
            print("Bis bald! 👋")
            break
        
        # Abrufen relevanter Daten
        relevant_context = retrieve_relevant_data(data, user_question)
        
        # Beantwortung der Frage mit Ollama
        if relevant_context == "Keine relevanten Daten gefunden.":
            print("Antwort: Diese Information ist in den bereitgestellten Daten nicht verfügbar. ❌")
        else:
            answer = ask_question_with_ollama(user_question, relevant_context)
            print(f"Antwort: {answer}")