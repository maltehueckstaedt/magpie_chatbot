# %%
import pandas as pd
import re  # Regular Expressions importieren
import ollama
import os

# Debug: Arbeitsverzeichnis prüfen
print(os.getcwd())  # Zeigt den aktuellen Arbeitsverzeichnis-Pfad an
os.chdir('c:/Users/Hueck/OneDrive/Dokumente/GitHub/magpie_langchain')

# Lade die Datenbank
data = pd.read_excel("data/mview_daten_beschr.xlsx")

# Modell ID
MODEL_ID = "llama3.1:8b-instruct-q4_0"
ollama.pull(MODEL_ID)

# Funktion zur Abfrage relevanter Daten aus der Excel-Datenbank
def retrieve_relevant_data(data, query):
    year_match = re.search(r'\b(19|20)\d{2}\b', query)
    year = year_match.group() if year_match else None

    relevant_data = []
    for _, row in data.iterrows():
        if (
            year and year in str(row['zeit_start']) and year in str(row['zeit_ende']) and
            any(term in str(row['variable']).lower() for term in ['fu&e', 'ausgaben', 'aufwendungen'])
        ):
            relevant_data.append(
                f"Von {row['zeit_start']} bis {row['zeit_ende']}: {row['variable']} betrug {row['wert']} {row['wert_einheit']} in {row['reichweite']}."
            )

    # Debug-Ausgabe: Zeige die gefundenen Daten an
    print("Gefundene relevante Daten:", relevant_data)

    # Begrenze die Anzahl der Ergebnisse auf maximal 3
    if len(relevant_data) > 3:
        relevant_data = relevant_data[:3]
        print("Zu viele Daten, auf 3 Einträge begrenzt.")

    return " ".join(relevant_data) if relevant_data else "Keine relevanten Daten gefunden."

# System Prompt für das Modell
def create_system_prompt(context):
    return f"""
    Du bist ein KI-Assistent. Nutze ausschließlich die folgenden Daten, um Fragen zu beantworten:
    {context}
    Antworte präzise auf Basis dieser Daten. Ignoriere alles andere Wissen und liefere keine allgemeinen Informationen.
    Falls die benötigte Information nicht in den Daten enthalten ist, antworte: 'Diese Information ist nicht verfügbar.'
    Antworte kurz und auf Deutsch. Verwende Emojis!
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

        # Debugging: Kontext und Daten prüfen
        if relevant_context == "Keine relevanten Daten gefunden.":
            print("Antwort: Diese Information ist in den bereitgestellten Daten nicht verfügbar. ❌")
            print("Debugging: Keine relevanten Daten gefunden.")
            print("Datenstruktur prüfen:")
            print(data.head())  # Zeige die ersten Zeilen der Datenbank
            print(data.columns)  # Zeige alle Spaltennamen
        else:
            print("Debugging: Kontext für Modell:")
            print(relevant_context)

            # Beantwortung der Frage mit Ollama
            answer = ask_question_with_ollama(user_question, relevant_context)
            print(f"Antwort: {answer}")