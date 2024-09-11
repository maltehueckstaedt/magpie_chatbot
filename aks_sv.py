import streamlit as st
import ollama

# Modell ID
MODEL_ID = "llama3.1:8b-instruct-q4_0"
ollama.pull(MODEL_ID)

# Der feste Text, auf den sich die Fragen beziehen
text = """
2011 wurden in Deutschland 1.296.349 Euro für Forschung und Entwicklung in Architektur- und Ingenieurbüros sowie für technische Untersuchungen ausgegeben. 
1987 gab es im gesamten Wirtschaftssektor 295.332 Vollzeitäquivalente (VZÄ) im Bereich Forschung und Entwicklung. 
2016 wurden im Sektor der freiberuflichen, wissenschaftlichen und technischen Dienstleistungen 47.551 VZÄ gezählt. 
Im Bergbau und in der Steingewinnung wurden im selben Jahr 21.318 Tausend Euro für interne Forschungs- und Entwicklungsarbeiten ausgegeben. 
Im Maschinenbau wurden 2017 49.323 VZÄ im Bereich Forschung und Entwicklung verzeichnet. 
2004 betrugen die internen Forschungs- und Entwicklungsausgaben im gesamten Wirtschaftssektor 38.363.000 Tausend Euro.

2014 wurden in der Architektur und verwandten Bereichen 84.855 Tausend Euro für externe Forschungs- und Entwicklungsaufwendungen ausgegeben. 
2016 wurden im Luft- und Raumfahrzeugbau 1.732.000 Tausend Euro für interne Forschung und Entwicklung aufgewendet. 
In den Finanz- und Versicherungsdienstleistungen wurden 2014 318.000 Tausend Euro und 2010 in der Herstellung von Glas, Keramik sowie in der Verarbeitung von Steinen 285.334 Tausend Euro für interne Forschung und Entwicklung verzeichnet.
"""

# System Prompt für das Modell
system_prompt = f"Du bist ein Deutsch sprechender AI assistent der Nutzern Fragen über folgenden Inhalt beantwortet: \n{text}\n Antworte kurz und immer auf Deutsch!."

# Initialisiere den Session State für den Chat-Verlauf
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": system_prompt}]

st.title("Olama Chatbot - Dialog mit Verlauf")

# Textfeld für Benutzereingabe (Frage)
user_question = st.text_input("Stelle eine Frage zum obigen Text:", "")

# Wenn der Benutzer eine Frage stellt und auf den Button klickt
if st.button("Frage stellen"):
    if user_question.strip():
        # Füge die Benutzerfrage dem Verlauf hinzu
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Die Frage an das Modell senden und den bisherigen Verlauf übergeben
        response = ollama.chat(
            model=MODEL_ID, 
            messages=st.session_state.chat_history, 
            options={'temperature': 0.1, 'num_predict': 256, "top_p": 0.9}
        )

        # Füge die Antwort des Modells zum Verlauf hinzu
        st.session_state.chat_history.append({"role": "assistant", "content": response['message']['content']})

        # Zeige die Antwort an
        st.write(f"Antwort: {response['message']['content']}")
    else:
        st.write("Bitte gib eine Frage ein.")

# Zeige den gesamten Dialogverlauf an
if st.session_state.chat_history:
    st.write("### Chat-Verlauf")
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.write(f"**Du**: {message['content']}")
        elif message['role'] == 'assistant':
            st.write(f"**Bot**: {message['content']}")
