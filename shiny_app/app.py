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

Was ist HEX?
Die Abkürzung HEX steht für Higher Education Explorer. HEX ist eine Datenbank, in der die Daten aus Vorlesungsverzeichnissen von Hochschulen vereinheitlicht und zusammengefasst werden. Anhand dieses Datenpools lassen sich vielfältige Recherchen und Analysen durchführen und Entwicklungen rund um die Hochschullehre entdecken und aufzeigen. HEX wurde bisher als Beta-Version veröffentlicht. HEX ist ein Projekt des Stifterverbandes und wird unterstützt von der Heinz Nixdorf Stiftung.

Welches Problem löst HEX?
Der Higher Education Explorer schafft Transparenz über die Inhalte und Struktur der deutsche Hochschullehre sowie relevante Trends und Veränderungen, indem er große Datenmengen bis auf die Ebene des einzelnen Hochschulkurses durchsucht und auswertet. Bisher war dies nur auf Studiengangsebene möglich. HEX ersetzt aufwändige Auszählungen einzelner Vorlesungsverzeichnisse.

Ist HEX ein Unikat oder gibt es vergleichbare Datenbasen und Werkzeuge?
Für die Analyse von Veranstaltungen gibt es unseres Wissens kein vergleichbares Projekt. Zur Vereinfachung von Modulanrechnungen existiert das Projekt PIM und für Literaturverwendung Open Syllabus aus den USA.

Für wen ist HEX wichtig?
HEX ist wichtig für die Beteiligten der Hochschulleitung, Hochschulentwicklung, und Hochschulforschung und die an diesen Themen interessierte Fachöffentlichkeit. Die Universitäten in der Datenbank erhalten individuelle Auswertungen, um schnell einen umfassenden Blick auf ihr Lehrangebot zu bekommen und datenbasierte Entscheidungen für Lehre und Studium treffen zu können. Mit dem HEX haben sie die Möglichkeit, die eigene Strategie zu prüfen, Standortvorteile zu erkennen und hervorzuheben oder frühzeitig Entwicklungen in der Lehre anzustoßen. Gleichzeitig werden Analysen über die allgemeine Entwicklung des Lehrangebots in Deutschland erstellt und für die Allgemeinheit zur Verfügung gestellt.  

Auf welche Daten und welche Datenmengen greift HEX aktuell zu?
Auf welche kann er perspektivisch zugreifen? HEX ist eine kontinuierlich wachsende Datenbank, die momentan mehr als zwei Millionen Daten zu Lehrveranstaltungen sowie relevante Begleitdaten der Hochschulstatistik enthält. Die Daten stammen von 22 deutschen Universitäten, darunter 15 der größten Universitäten. Damit bildet der HEX das Studienangebot für rund 23 Prozent der Studierenden ab. Ob sich das Konzept auch auf HAWs/Fachhochschulen, private Hochschulen und/oder internationale Hochschulen übertragen lässt, wird in einer späteren Phase geprüft.

Welche Themen bzw. Fragen kann HEX untersuchen und in welcher Form präsentiert er die Ergebnisse?
HEX bietet eine Vielzahl von Analysemöglichkeiten, in denen die Grunddaten und Kennzahlen zu Kursen, Lehrformaten, Lehrsprachen, Studierende pro Kurs, wissenschaftliches Personal pro Kurs und Professorin bzw. Professor bis auf die Ebene der Studienbereiche miteinander verknüpft werden können. Darüber hinaus entstehen im HEX-Forschungsprojekt inhaltliche Analysen zu Trends in Lehrthemen, Lehrprofile der Hochschule oder Future Skills. Die methodische Vielfalt reicht dabei von Stichwortauszählungen bis zum maschinellen Lernen.

Wie lange würde eine HEX-Analyse dauern?
Einfache Veranstaltungsrecherchen lassen sich schnell umsetzen. Umfassende wissenschaftliche Analysen, bei denen auch andere Daten und Literatur einbezogen, die Limitationen so weit wie möglich eingrenzt und die Ergebnisse eingeordnet werden, sind aufwändiger – die genaue Dauer ist dabei individuell abzuschätzen.  

Wer kann auf HEX zugreifen?
Der Stifterverband hat bereits mit Partnerstiftungen erste Analysen erarbeitet und wird sukzessive weitere Studienergebnisse veröffentlichen. Wir möchten gerne mit weiteren Partnern aus der Hochschul- und Wissenschaftsforschung zusammenarbeiten. Kommen Sie mit Studienideen gerne auf uns zu!

An wen kann ich mich wenden?
Schreiben Sie uns eine Mail an hex@stifterverband.de und wir gehen ins Gespräch. 
"""

# System Prompt für das Modell
system_prompt = f"Du bist Shiny SV, ein Deutsch sprechender AI Assistent der Nutzern Fragen über folgenden Inhalt beantwortet: \n{text}\n Antworte kurz und immer auf Deutsch!."

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
