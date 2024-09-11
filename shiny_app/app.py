from shiny import App, reactive, render, ui
import ollama
from pathlib import Path


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

Mit dem Higher Education Explorer (HEX) baut der Stifterverband eine neue Datenbank auf. In ihr werden erstmals Inhalte der Vorlesungsverzeichnisse von Hochschulen in Deutschland gesammelt.

Vorlesungsverzeichnisse sind ein Spiegel der Lehre an Hochschulen: Inhalte der Lehrveranstaltungen, Formate, Sprachen und vieles mehr werden dort Semester für Semester festgehalten. Bislang wurde dieser Datenfundus jedoch nicht systematisch erfasst - nun soll er für die Hochschulforschung und Hochschulentwicklung nutzbar gemacht werden.

Das Projekt wird unterstützt von der Heinz-Nixdorf-Stiftung
Der Higher Education Explorer ist im September 2024 in einer Beta-Version an den Start gegangen. Die kontinuierlich wachsende Datenbank enthält bereits jetzt mehr als zwei Millionen Daten zu Lehrveranstaltungen sowie relevante Begleitdaten der Hochschulstatistik. Die Daten stammen von 22 deutschen Universitäten, darunter 15 der größten Universitäten. Damit bildet der HEX das Studienangebot für rund 23 Prozent der Studierenden ab.
"""

# System Prompt für das Modell
system_prompt = f"Du bist ein Deutsch sprechender AI Assistent der Nutzern Fragen über folgenden Inhalt beantwortet: \n{text}\n Antworte kurz und immer auf Deutsch!."

# Shiny UI Layout
app_ui = ui.page_fluid(
    
    ui.include_css(Path(__file__).parent / "www" / "styles.css"),  # CSS-Datei einbinden
    ui.div(
        ui.output_ui("chat_output"),
        class_="chat-container"
    ),
    ui.div(
        ui.div(
            ui.input_text("user_question", "", placeholder="Sende eine Nachricht..."),
            ui.input_action_button("senden", ">", class_="send-button"),
            class_="input-row"
        ),
        class_="input-container"
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
        history_html = ""
        for message in messages:
            if message['role'] == 'user':
                history_html += f"""
                <div style='text-align: right; margin: 10px 0;'>
                    <span style='background-color: #d1e7dd; padding: 10px; border-radius: 10px; display: inline-block; max-width: 80%;'>
                        {message['content']}
                    </span>
                </div>
                """
            elif message['role'] == 'assistant':
                history_html += f"""
                <div style='text-align: left; margin: 10px 0;'>
                    <span style='background-color: #f8d7da; padding: 10px; border-radius: 10px; display: inline-block; max-width: 80%;'>
                        {message['content']}
                    </span>
                </div>
                """
        return ui.HTML(history_html)

# Shiny App-Objekt
app = App(app_ui, server)
