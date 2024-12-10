from shiny import App, reactive, render, ui
import ollama
from pathlib import Path


# Modell ID
MODEL_ID = "llama3.1:8b-instruct-q4_0"
ollama.pull(MODEL_ID)

# Der feste Text, auf den sich die Fragen beziehen
text = """
Wie muss sich die Hochschulbildung in Deutschland entwickeln, um Nachwuchskräfte mit den für die Zukunft nötigen Kompetenzen zu versorgen? Wie viel investieren deutsche Unternehmen in die eigene Forschung und reicht das, um den Innovationsstandort Deutschland zu sichern? Diese und weitere Fragen analysiert der Stifterverband regelmäßig in Studien und Erhebungen. Dazu nutzt er Daten und wissenschaftliche Ergebnisse, die er selbst erhebt, sowie Untersuchungen, die er in Auftrag gibt oder von Dritten bereitgestellt werden. Sie zeigen klar auf, wo Veränderungen im Bildungs-, Wissenschafts- und Innovationssystem nötig sind.

Auf diese Weise identifiziert der Stifterverband Stärken und Schwächen, macht Handlungsbedarfe sichtbar und verdeutlicht, welche Themen verstärkt in den Fokus politischer Debatten rücken müssen. Das Ziel: evidenzbasierte Entscheidungen in Politik und Wissenschaft zu ermöglichen - mit konkreten Handlungsempfehlungen und Zukunftsszenarien.

Das bietet der Daten-Navigator:

Daten suchen und analysieren
Vielfältige Filter- und Suchfunktionen erlauben es, genau die Daten zu finden, die für Fragen und Projekte relevant sind. Alle verfügbaren Daten des Stifterverbandes können hier erkundet und analysiert werden.
 
Monitoring
Der Daten-Navigator zeigt auf, wie sich ausgewählte Indikatoren in den beiden Handlungsfeldern des Stifterverbandes entwickeln.
 
Datenanalysen aus dem Stifterverband
Kuratierte Insights zu verschiedenen Themen aus den Bereichen "Bildung & Kompetenzen" sowie "Kollaborative Forschung & Innovation". In einzelnen (Blog-)Artikeln finden sich tiefgehende Analysen und anschauliche Darstellungen sowie Einordnungen und Handlungsempfehlungen.
 
Studienprojekte aus dem Stifterverband
Auf interaktiven Datenseiten lassen sich Analysen mit den Datensätzen aus Untersuchungen, die der Stifterverband durchgeführt hat, in Echtzeit und auf verschiedenen Ebenen durchführen.


"""

# System Prompt für das Modell
system_prompt = f"Du bist der Chatbot des Daten-Navigators des Stifterverbandes, ein Deutsch sprechender AI Assistent der Nutzern Fragen über folgenden Inhalt beantwortet: \n{text}\n Antworte kurz und immer auf Deutsch. Kommentiere die Fragen, die Dir gestellt werden nicht."

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
    chat_state = reactive.Value([
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "Hallo 😊, Ich bin der AI-Assistent des Daten-Navigators des Stifterverbandes. Wie kann ich Ihnen behilflich sein?"}
    ])
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

            # Lösche den Inhalt des Eingabefelds nach dem Senden der Nachricht
            ui.update_text("user_question", value="")

    
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
                    <span style='background-color: #b6ebae; padding: 10px; border-radius: 10px; display: inline-block; max-width: 80%;'>
                        {message['content']}
                    </span>
                </div>
                """
            elif message['role'] == 'assistant':
                history_html += f"""
                <div style='text-align: left; margin: 10px 0;'>
                    <span style='background-color: #aedfeb; padding: 10px; border-radius: 10px; display: inline-block; max-width: 80%;'>
                        {message['content']}
                    </span>
                </div>
                """
        return ui.HTML(history_html)

# Shiny App-Objekt
app = App(app_ui, server)
