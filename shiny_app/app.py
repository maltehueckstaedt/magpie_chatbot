from shiny import App, reactive, render, ui
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
system_prompt = f"Du bist ein Deutsch sprechender AI Assistent der Nutzern Fragen über folgenden Inhalt beantwortet: \n{text}\n Antworte kurz und immer auf Deutsch!."

# Shiny UI Layout
app_ui = ui.page_fluid(
    # Ein Container für den Chat-Verlauf
    ui.div(
        ui.output_ui("chat_output"),  # Ändere output_text zu output_ui
        style="height: 80vh; overflow-y: auto; padding: 10px; background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 5px;"
    ),
    # Eingabefeld und Senden-Button unten fixieren
    ui.div(
        ui.input_text("user_question", "Deine Nachricht:", placeholder="Sende eine Nachricht an SV OLLAMA..."),
        ui.input_action_button("senden", "Senden"),
        style="position: fixed; bottom: 10px; left: 10px; right: 10px; background-color: #fff; padding: 10px; border-top: 1px solid #ddd;"
    )
)

# Server-Logik
def server(input, output, session):
    
    # Initialisiere den Chatverlauf im Reactive-State
    chat_state = reactive.Value([{"role": "system", "content": system_prompt}])
    
    # Nachricht senden und Verlauf aktualisieren
    @reactive.Effect
    @reactive.event(input.senden)
    def send_message():
        if input.user_question().strip():
            # Füge die Benutzerfrage zum Verlauf hinzu
            chat_state.set(chat_state.get() + [{"role": "user", "content": input.user_question()}])

            # Sende die Frage an das Modell und erhalte die Antwort
            messages = chat_state.get()
            response = ollama.chat(
                model=MODEL_ID, 
                messages=messages, 
                options={'temperature': 0.1, 'num_predict': 256, "top_p": 0.9}
            )

            # Füge die Antwort des Modells zum Verlauf hinzu
            chat_state.set(chat_state.get() + [{"role": "assistant", "content": response['message']['content']}])
    
    # Den gesamten Verlauf formatieren und anzeigen
    @output
    @render.ui
    def chat_output():
        messages = chat_state.get()
        history_text = ""
        for message in messages:
            if message['role'] == 'user':
                history_text += f"<b>Du</b>: {message['content']}<br><br>"
            elif message['role'] == 'assistant':
                history_text += f"<b>Bot</b>: {message['content']}<br><br>"
        return ui.HTML(history_text)  # Verwende ui.HTML, um HTML-Inhalte korrekt zu rendern

# Shiny App-Objekt
app = App(app_ui, server)
