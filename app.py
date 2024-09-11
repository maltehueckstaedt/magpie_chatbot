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
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_text("user_question", "Stelle eine Frage:", placeholder="Frage eingeben..."),
            ui.input_action_button("senden", "Senden")
        ),
        ui.output_text("chat_output")
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
    @render.text
    def chat_output():
        messages = chat_state.get()
        history_text = ""
        for message in messages:
            if message['role'] == 'user':
                history_text += f"**Du**: {message['content']}\n\n"
            elif message['role'] == 'assistant':
                history_text += f"**Bot**: {message['content']}\n\n"
        return history_text

# Shiny App-Objekt
app = App(app_ui, server)
