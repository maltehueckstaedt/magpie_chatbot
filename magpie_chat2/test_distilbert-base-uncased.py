from langchain_community.document_loaders import DuckDBLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import duckdb
import warnings
 

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

# Create loader
loader = DuckDBLoader(
    database="magpie_chat2/magpie.db",
    query="SELECT * FROM mview_daten_beschr LIMIT 100"
)

data = loader.load()

# Initialize the SentenceTransformer model using LangChain's wrapper
embedding_model = SentenceTransformerEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')


# Create the Chroma vector db
persist_directory = "./" 

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model
)

# Preprocess docs
def preprocess_document(doc):
    lines = doc.page_content.split('\n')
    processed_lines = []
    for line in lines:
        key_value = line.split(': ')
        if len(key_value) == 2:
            key, value = key_value
            value = value.replace('{', '').replace('}', '').replace('"', '')
            processed_lines.append(f"{key.strip()}: {value.strip()}")
    return '\n'.join(processed_lines)

processed_docs = [preprocess_document(doc) for doc in data]

# Add docs to vector db
vectordb.add_texts(processed_docs)

# Initialize QA pipeline with distilbert-base-uncased
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased')

qa_model = pipeline('question-answering', model=model, tokenizer=tokenizer, device=0)

# Create few shot examples
def prepare_few_shot_examples():
    examples = [
        {
            'question': "Wie viel FuE Personal war 2011 im Wirtschaftszweig Herstellung von DV-Geräten, elektronischen und optischen Erzeignissen beschäftigt?",
            'context': "id: 652\nvariable: FuE-Personal\nzeit_start: 2011-01-01 00:00:00\nzeit_ende: 2011-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,H.v. DV-Geräten, elektronischen u. opt. Erzeugnissen\nwert: 54647\nwert_einheit: VZÄ\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "54647 VZÄ"
        },
        {
            'question': "Wie hoch waren die externen FuE-Aufwendungen im Jahr 2011 im Wirtschaftszweig Metallerzeugung und -bearbeitung?",
            'context': "id: 273\nvariable: Externe FuE-Aufwendungen\nzeit_start: 2011-01-01 00:00:00\nzeit_ende: 2011-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Metallerzeugung und -bearbeitung\nwert: 69713\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "69713 in Tsd. Euro"
        },
        {
            'question': "Wie hoch waren die internen FuE-Aufwendungen im Jahr 2016 im Wirtschaftssektor Deutschland für Unternehmen mit 500 und mehr Beschäftigten?",
            'context': "id: 550\nvariable: Interne FuE-Aufwendungen\nzeit_start: 2016-01-01 00:00:00\nzeit_ende: 2016-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,500 und mehr Beschäftigte\nwert: 54565000\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "54565000 in Tsd. Euro"
        },
        {
            'question': "Wie hoch waren die externen FuE-Aufwendungen im Jahr 2011 im Wirtschaftszweig Sonstiger Fahrzeugbau?",
            'context': "id: 278\nvariable: Externe FuE-Aufwendungen\nzeit_start: 2011-01-01 00:00:00\nzeit_ende: 2011-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Sonstiger Fahrzeugbau\nwert: 939313\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "939313 in Tsd. Euro"
        },
        {
            'question': "Wie hoch waren die internen FuE-Aufwendungen im Jahr 2014 im Wirtschaftszweig Luft- und Raumfahrzeugbau?",
            'context': "id: 424\nvariable: Interne FuE-Aufwendungen\nzeit_start: 2014-01-01 00:00:00\nzeit_ende: 2014-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Luft- und Raumfahrzeugbau\nwert: 1801000\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "1801000 in Tsd. Euro"
        },
        {
            'question': "Wie viel FuE_personal war 2002 im Wirtschaftssektor Deutschland beschäftigt?",
            'context': "id: 176\nvariable: FuE-Personal\nzeit_start: 2002-01-01 00:00:00\nzeit_ende: 2002-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland\nwert: 302600\nwert_einheit: VZÄ\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "302600 VZÄ"
        },
        {
            'question': "Wie viel FuE_personal war 2016 im Wirtschaftszweig Herstellung von Metallerzeugnissen beschäftigt?",
            'context': "id: 791\nvariable: FuE-Personal\nzeit_start: 2016-01-01 00:00:00\nzeit_ende: 2016-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,H.v. Metallerzeugnissen\nwert: 7601\nwert_einheit: VZÄ\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "7601 VZÄ"
        },
        {
            'question': "Wie hoch waren die internen FuE-Aufwendungen im Jahr 2011 im Wirtschaftszweig Architektur-, Ingenieurbüros, technologische, physikalische und chemische Untersuchungen?",
            'context': "id: 257\nvariable: Interne FuE-Aufwendungen\nzeit_start: 2011-01-01 00:00:00\nzeit_ende: 2011-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Architektur-, Ing.büros, techn., phys., chem. Untersuchung\nwert: 1296349\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "1296349 in Tsd. Euro"
        },
        {
            'question': "Wie viel FuE_personal war 1987 im Wirtschaftssektor Deutschland beschäftigt?",
            'context': "id: 120\nvariable: FuE-Personal\nzeit_start: 1987-01-01 00:00:00\nzeit_ende: 1987-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland\nwert: 295332\nwert_einheit: VZÄ\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "295332 VZÄ"
        },
        {
            'question': "Wie viel FuE_personal war 2016 im Wirtschaftssektor Freiberufliche, wissenschaftliche und technische Dienstleistungen beschäftigt?",
            'context': "id: 802\nvariable: FuE-Personal\nzeit_start: 2016-01-01 00:00:00\nzeit_ende: 2016-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Freiberufliche, wissenschaftl. u. techn. Dienstleistungen\nwert: 47551\nwert_einheit: VZÄ\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "47551 VZÄ"
        },
        {
            'question': "Wie hoch waren die internen FuE-Aufwendungen im Jahr 2015 im Wirtschaftszweig Bergbau und Gewinnung von Steinen und Erden?",
            'context': "id: 466\nvariable: Interne FuE-Aufwendungen\nzeit_start: 2015-01-01 00:00:00\nzeit_ende: 2015-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Bergbau und Gewinnung von Steinen und Erden\nwert: 21318\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "21318 in Tsd. Euro"
        },
        {
            'question': "Wie viel FuE_personal war 2017 im Wirtschaftssektor Maschinenbau beschäftigt?",
            'context': "id: 822\nvariable: FuE-Personal\nzeit_start: 2017-01-01 00:00:00\nzeit_ende: 2017-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Maschinenbau\nwert: 49323\nwert_einheit: VZÄ\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "49323 VZÄ"
        },
        {
            'question': "Wie hoch waren die internen FuE-Aufwendungen im Jahr 2004 im Wirtschaftssektor Deutschland?",
            'context': "id: 13\nvariable: Interne FuE-Aufwendungen\nzeit_start: 2004-01-01 00:00:00\nzeit_ende: 2004-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland\nwert: 38363000\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "38363000 in Tsd. Euro"
        },
        {
            'question': "Wie hoch waren die externen FuE-Aufwendungen im Jahr 2014 im Wirtschaftszweig Architektur-, Ingenieurbüros, technologische, physikalische und chemische Untersuchungen?",
            'context': "id: 458\nvariable: Externe FuE-Aufwendungen\nzeit_start: 2014-01-01 00:00:00\nzeit_ende: 2014-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Architektur-, Ing.büros, techn., phys., chem. Untersuchung\nwert: 84855\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "84855 in Tsd. Euro"
        },
        {
            'question': "Wie hoch waren die internen FuE-Aufwendungen im Jahr 2016 im Wirtschaftszweig Luft- und Raumfahrzeugbau?",
            'context': "id: 538\nvariable: Interne FuE-Aufwendungen\nzeit_start: 2016-01-01 00:00:00\nzeit_ende: 2016-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Luft- und Raumfahrzeugbau\nwert: 1732000\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "1732000 in Tsd. Euro"
        },
        {
            'question': "Wie hoch waren die geplanten internen FuE-Aufwendungen im Jahr 2017 im Wirtschaftszweig Herstellung von DV-Geräten, elektronischen und optischen Erzeugnissen?",
            'context': "id: 595\nvariable: Budgetplanung interne FuE (nächstes Jahr)\nzeit_start: 2017-01-01 00:00:00\nzeit_ende: 2017-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,H.v. DV-Geräten, elektronischen u. opt. Erzeugnissen\nwert: 7978000\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft),FuE-Aufwendungen,interne FuE-Aufwendungen,Durchführung von FuE",
            'answer': "7978000 in Tsd. Euro"
        },
        {
            'question': "Wie hoch waren die internen FuE-Aufwendungen im Jahr 2014 im Wirtschaftszweig Finanz- und Versicherungsdienstleistungen?",
            'context': "id: 428\nvariable: Interne FuE-Aufwendungen\nzeit_start: 2014-01-01 00:00:00\nzeit_ende: 2014-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Finanz- und Versicherungsdienstleistungen\nwert: 318000\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "318000 in Tsd. Euro"
        },
        {
            'question': "Wie hoch waren die internen FuE-Aufwendungen im Jahr 2010 im Wirtschaftszweig Herstellung von Glas und Glaswaren, Keramik, Verarbeitung von Steinen und Erden?",
            'context': "id: 186\nvariable: Interne FuE-Aufwendungen\nzeit_start: 2010-01-01 00:00:00\nzeit_ende: 2010-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,H.v. Glas u. Glaswaren, Keramik, Verarb. v. Steinen u. Erden\nwert: 285334\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "285334 in Tsd. Euro"
        },
        {
            'question': "Wie hoch waren die internen FuE-Aufwendungen im Jahr 2012 im Wirtschaftszweig Herstellung von DV-Geräten, elektronischen und optischen Erzeugnissen?",
            'context': "id: 304\nvariable: Interne FuE-Aufwendungen\nzeit_start: 2012-01-01 00:00:00\nzeit_ende: 2012-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,H.v. DV-Geräten, elektronischen u. opt. Erzeugnissen\nwert: 7391756\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "7391756 in Tsd. Euro"
        },
        {
            'question': "Wie viel FuE Personal war 2010 im Wirtschaftszweig Maschinenbau beschäftigt?",
            'context': "id: 491\nvariable: FuE-Personal\nzeit_start: 2010-01-01 00:00:00\nzeit_ende: 2010-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Maschinenbau\nwert: 52034\nwert_einheit: VZÄ\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "52034 VZÄ"
        },
        {
            'question': "Wie hoch waren die internen FuE-Aufwendungen im Jahr 2013 im Wirtschaftszweig Informations- und Kommunikationstechnologien?",
            'context': "id: 345\nvariable: Interne FuE-Aufwendungen\nzeit_start: 2013-01-01 00:00:00\nzeit_ende: 2013-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Informations- und Kommunikationstechnologien\nwert: 1580000\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "1580000 in Tsd. Euro"
        },
        {
            'question': "Wie viel FuE Personal war 2015 im Wirtschaftssektor Dienstleistungen insgesamt beschäftigt?",
            'context': "id: 712\nvariable: FuE-Personal\nzeit_start: 2015-01-01 00:00:00\nzeit_ende: 2015-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Dienstleistungen insgesamt\nwert: 87540\nwert_einheit: VZÄ\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "87540 VZÄ"
        },
        {
            'question': "Wie hoch waren die externen FuE-Aufwendungen im Jahr 2018 im Wirtschaftszweig Gesundheitswesen?",
            'context': "id: 508\nvariable: Externe FuE-Aufwendungen\nzeit_start: 2018-01-01 00:00:00\nzeit_ende: 2018-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Gesundheitswesen\nwert: 123456\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "123456 in Tsd. Euro"
        },
        {
            'question': "Wie hoch waren die internen FuE-Aufwendungen im Jahr 2005 im Wirtschaftszweig Energieversorgung?",
            'context': "id: 147\nvariable: Interne FuE-Aufwendungen\nzeit_start: 2005-01-01 00:00:00\nzeit_ende: 2005-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Energieversorgung\nwert: 589321\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "589321 in Tsd. Euro"
        },
        {
            'question': "Wie viel FuE Personal war 2018 im Wirtschaftszweig Telekommunikation beschäftigt?",
            'context': "id: 934\nvariable: FuE-Personal\nzeit_start: 2018-01-01 00:00:00\nzeit_ende: 2018-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Telekommunikation\nwert: 40125\nwert_einheit: VZÄ\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "40125 VZÄ"
        },
        {
            'question': "Wie hoch waren die internen FuE-Aufwendungen im Jahr 2012 im Wirtschaftszweig Chemieindustrie?",
            'context': "id: 406\nvariable: Interne FuE-Aufwendungen\nzeit_start: 2012-01-01 00:00:00\nzeit_ende: 2012-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Chemieindustrie\nwert: 2765432\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "2765432 in Tsd. Euro"
        },
        {
            'question': "Wie viel FuE Personal war 2017 im Wirtschaftssektor Finanzdienstleistungen beschäftigt?",
            'context': "id: 719\nvariable: FuE-Personal\nzeit_start: 2017-01-01 00:00:00\nzeit_ende: 2017-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Finanzdienstleistungen\nwert: 9507\nwert_einheit: VZÄ\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "9507 VZÄ"
        },
        {
            'question': "Wie hoch waren die externen FuE-Aufwendungen im Jahr 2015 im Wirtschaftszweig Baugewerbe?",
            'context': "id: 521\nvariable: Externe FuE-Aufwendungen\nzeit_start: 2015-01-01 00:00:00\nzeit_ende: 2015-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Baugewerbe\nwert: 35412\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "35412 in Tsd. Euro"
        },
        {
            'question': "Wie viel FuE Personal war 2013 im Wirtschaftszweig Landwirtschaft beschäftigt?",
            'context': "id: 599\nvariable: FuE-Personal\nzeit_start: 2013-01-01 00:00:00\nzeit_ende: 2013-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Landwirtschaft\nwert: 12890\nwert_einheit: VZÄ\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "12890 VZÄ"
        },
        {
            'question': "Wie hoch waren die internen FuE-Aufwendungen im Jahr 2014 im Wirtschaftszweig Textilindustrie?",
            'context': "id: 783\nvariable: Interne FuE-Aufwendungen\nzeit_start: 2014-01-01 00:00:00\nzeit_ende: 2014-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Textilindustrie\nwert: 45321\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "45321 in Tsd. Euro"
        },
        {
            'question': "Wie viel FuE Personal war 2011 im Wirtschaftszweig Automobilindustrie beschäftigt?",
            'context': "id: 887\nvariable: FuE-Personal\nzeit_start: 2011-01-01 00:00:00\nzeit_ende: 2011-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Automobilindustrie\nwert: 106789\nwert_einheit: VZÄ\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "106789 VZÄ"
        },
        {
            'question': "Wie hoch waren die externen FuE-Aufwendungen im Jahr 2009 im Wirtschaftszweig Pharmazeutische Industrie?",
            'context': "id: 629\nvariable: Externe FuE-Aufwendungen\nzeit_start: 2009-01-01 00:00:00\nzeit_ende: 2009-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Pharmazeutische Industrie\nwert: 789123\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "789123 in Tsd. Euro"
        },
        {
            'question': "Wie viel FuE Personal war 2012 im Wirtschaftszweig Nahrungsmittelindustrie beschäftigt?",
            'context': "id: 562\nvariable: FuE-Personal\nzeit_start: 2012-01-01 00:00:00\nzeit_ende: 2012-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Nahrungsmittelindustrie\nwert: 24321\nwert_einheit: VZÄ\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "24321 VZÄ"
        },
        {
            'question': "Wie hoch waren die internen FuE-Aufwendungen im Jahr 2007 im Wirtschaftszweig Holzwirtschaft?",
            'context': "id: 338\nvariable: Interne FuE-Aufwendungen\nzeit_start: 2007-01-01 00:00:00\nzeit_ende: 2007-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Holzwirtschaft\nwert: 16789\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "16789 in Tsd. Euro"
        },
        {
            'question': "Wie viel FuE Personal war 2016 im Wirtschaftszweig Metallbearbeitung beschäftigt?",
            'context': "id: 763\nvariable: FuE-Personal\nzeit_start: 2016-01-01 00:00:00\nzeit_ende: 2016-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Metallbearbeitung\nwert: 19832\nwert_einheit: VZÄ\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "19832 VZÄ"
        },
        {
            'question': "Wie hoch waren die externen FuE-Aufwendungen im Jahr 2010 im Wirtschaftszweig Medizintechnik?",
            'context': "id: 411\nvariable: Externe FuE-Aufwendungen\nzeit_start: 2010-01-01 00:00:00\nzeit_ende: 2010-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Medizintechnik\nwert: 321456\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "321456 in Tsd. Euro"
        },
        {
            'question': "Wie viel FuE Personal war 2015 im Wirtschaftszweig Kunststoffindustrie beschäftigt?",
            'context': "id: 589\nvariable: FuE-Personal\nzeit_start: 2015-01-01 00:00:00\nzeit_ende: 2015-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Kunststoffindustrie\nwert: 30560\nwert_einheit: VZÄ\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "30560 VZÄ"
        },
        {
            'question': "Wie hoch waren die internen FuE-Aufwendungen im Jahr 2013 im Wirtschaftszweig Baugewerbe?",
            'context': "id: 714\nvariable: Interne FuE-Aufwendungen\nzeit_start: 2013-01-01 00:00:00\nzeit_ende: 2013-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Baugewerbe\nwert: 43210\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "43210 in Tsd. Euro"
        },
        {
            'question': "Wie viel FuE Personal war 2019 im Wirtschaftszweig Verpackungsindustrie beschäftigt?",
            'context': "id: 412\nvariable: FuE-Personal\nzeit_start: 2019-01-01 00:00:00\nzeit_ende: 2019-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Verpackungsindustrie\nwert: 17345\nwert_einheit: VZÄ\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "17345 VZÄ"
        },
        {
            'question': "Wie hoch waren die externen FuE-Aufwendungen im Jahr 2016 im Wirtschaftszweig Elektronikindustrie?",
            'context': "id: 505\nvariable: Externe FuE-Aufwendungen\nzeit_start: 2016-01-01 00:00:00\nzeit_ende: 2016-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Elektronikindustrie\nwert: 654321\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "654321 in Tsd. Euro"
        },
        {
            'question': "Wie viel FuE Personal war 2017 im Wirtschaftszweig Logistik beschäftigt?",
            'context': "id: 623\nvariable: FuE-Personal\nzeit_start: 2017-01-01 00:00:00\nzeit_ende: 2017-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Logistik\nwert: 28456\nwert_einheit: VZÄ\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "28456 VZÄ"
        },
        {
            'question': "Wie hoch waren die internen FuE-Aufwendungen im Jahr 2011 im Wirtschaftszweig Großhandel?",
            'context': "id: 476\nvariable: Interne FuE-Aufwendungen\nzeit_start: 2011-01-01 00:00:00\nzeit_ende: 2011-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Großhandel\nwert: 112233\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "112233 in Tsd. Euro"
        },
        {
            'question': "Wie viel FuE Personal war 2014 im Wirtschaftszweig Hotel- und Gaststättengewerbe beschäftigt?",
            'context': "id: 519\nvariable: FuE-Personal\nzeit_start: 2014-01-01 00:00:00\nzeit_ende: 2014-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Hotel- und Gaststättengewerbe\nwert: 6578\nwert_einheit: VZÄ\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "6578 VZÄ"
        },
        {
            'question': "Wie hoch waren die externen FuE-Aufwendungen im Jahr 2017 im Wirtschaftszweig Finanzdienstleistungen?",
            'context': "id: 712\nvariable: Externe FuE-Aufwendungen\nzeit_start: 2017-01-01 00:00:00\nzeit_ende: 2017-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Finanzdienstleistungen\nwert: 54321\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "54321 in Tsd. Euro"
        },
        {
            'question': "Wie viel FuE Personal war 2015 im Wirtschaftszweig Immobilienwirtschaft beschäftigt?",
            'context': "id: 648\nvariable: FuE-Personal\nzeit_start: 2015-01-01 00:00:00\nzeit_ende: 2015-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Immobilienwirtschaft\nwert: 11223\nwert_einheit: VZÄ\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "11223 VZÄ"
        },
        {
            'question': "Wie hoch waren die internen FuE-Aufwendungen im Jahr 2008 im Wirtschaftszweig Bergbau?",
            'context': "id: 481\nvariable: Interne FuE-Aufwendungen\nzeit_start: 2008-01-01 00:00:00\nzeit_ende: 2008-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Bergbau\nwert: 87654\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "87654 in Tsd. Euro"
        },
        {
            'question': "Wie viel FuE Personal war 2013 im Wirtschaftszweig Luftfahrtindustrie beschäftigt?",
            'context': "id: 702\nvariable: FuE-Personal\nzeit_start: 2013-01-01 00:00:00\nzeit_ende: 2013-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Luftfahrtindustrie\nwert: 40987\nwert_einheit: VZÄ\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "40987 VZÄ"
        },
        {
            'question': "Wie hoch waren die externen FuE-Aufwendungen im Jahr 2014 im Wirtschaftszweig Biotechnologie?",
            'context': "id: 584\nvariable: Externe FuE-Aufwendungen\nzeit_start: 2014-01-01 00:00:00\nzeit_ende: 2014-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Biotechnologie\nwert: 76543\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "76543 in Tsd. Euro"
        },
        {
            'question': "Wie viel FuE Personal war 2011 im Wirtschaftszweig Telekommunikation beschäftigt?",
            'context': "id: 341\nvariable: FuE-Personal\nzeit_start: 2011-01-01 00:00:00\nzeit_ende: 2011-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Telekommunikation\nwert: 37482\nwert_einheit: VZÄ\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "37482 VZÄ"
        },
        {
            'question': "Wie hoch waren die internen FuE-Aufwendungen im Jahr 2012 im Wirtschaftszweig Automobilindustrie?",
            'context': "id: 734\nvariable: Interne FuE-Aufwendungen\nzeit_start: 2012-01-01 00:00:00\nzeit_ende: 2012-12-31 23:59:59\nzeit_einheit: Jahr\nreichweite: Wirtschaftssektor,Deutschland,Automobilindustrie\nwert: 21098765\nwert_einheit: in Tsd. Euro\nquelle: FuE-Erhebung\ntag: Forschung und Entwicklung,Wirtschaft,Datensatz,FuE-Erhebung (Wirtschaft)",
            'answer': "21098765 in Tsd. Euro"
        }
    ]
    return examples

# Use examples with QA
def perform_qa_with_few_shot(query, vectordb, qa_model, examples, k=5):
    # Retrieve most relevant docs
    results = vectordb.similarity_search(query, k=k)
    
    if not results:
        return {"answer": "Keine relevanten Dokumente gefunden."}

    # Prepare few-shot context
    few_shot_context = ""
    for example in examples:
        few_shot_context += f"Question: {example['question']}\n"
        few_shot_context += f"Context: {example['context']}\n"
        few_shot_context += f"Answer: {example['answer']}\n\n"
    
    # Use most relevant doc as context
    context = results[0].page_content if results else ''
    
    # Combine few-shot examples with context
    combined_context = few_shot_context + "Question: " + query + "\n" + "Context: " + context
    
    # Perform QA
    answer = qa_model(question=query, context=combined_context)
    return answer

# Prepare few-shot examples
examples = prepare_few_shot_examples()

# Example queries
query1 = "Wie viel FuE Personal war 2012 im Wirtschaftszweig Sonstiger Fahrzeugbau beschäftigt?"
result1 = perform_qa_with_few_shot(query1, vectordb, qa_model, examples)
print(result1)

query2 = "Wie hoch waren die internen FuE-Aufwendungen in 2016 im Wirtschaftssektor Deutschland für Unternehmen mit 500 und mehr Beschäftigten?"
result2 = perform_qa_with_few_shot(query2, vectordb, qa_model, examples)
print(result2)

query3 = "Wie hoch waren die Externen FuE im Ausland im Jahr 2019 von Deutschlan in Tsd. Euro"
result3 = perform_qa_with_few_shot(query3, vectordb, qa_model, examples)
print(result3)

# Garbage / control area
print(processed_docs)
