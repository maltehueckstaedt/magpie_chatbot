# Beispielabfrage Sparklehorse:
Der Bot führt eine Abfrage in einer SQL-Datenbank durch, um die Drittmittel der Humboldt-Universität zu Berlin für das Jahr 2008 zu ermitteln. Das ist, was passiert:

## 1. Empfang der Benutzeranfrage
Der Benutzer fragt: „Wie hoch waren 2008 die Drittmittel der HU Berlin?“

## 2. Abruf der Tabellennamen
Der Bot ruft eine Liste aller Tabellen in der Datenbank ab, um herauszufinden, wo relevante Daten gespeichert sein könnten.

## 3. Abruf des Tabellenschemas
Der Bot prüft die Struktur der Tabellen `datensatz_drittmittel_hochschule` und `datensatz_drittmittel_aggregiert`, um die Spaltennamen (wie Jahr, Hochschule, Wert, etc.) zu verstehen.

## 4. Erstellung einer SQL-Abfrage
Basierend auf dem Schema formuliert der Bot eine SQL-Abfrage:
```sql
SELECT Jahr, Hochschule, Wert, Einheit 
FROM datensatz_drittmittel_hochschule 
WHERE Jahr = 2008 AND Hochschule = 'HU Berlin'
```

## 5. Überprüfung und Korrektur des Hochschulnamens
Der Bot erkennt, dass „HU Berlin“ eine Abkürzung ist und sucht nach dem vollständigen Namen.  
Er findet: „Humboldt-Universität zu Berlin“, aktualisiert die Abfrage entsprechend und führt sie erneut aus.

## 6. Ausführung der SQL-Abfrage
Die korrigierte Abfrage wird an die Datenbank gesendet:
```sql
SELECT Jahr, Hochschule, Wert, Einheit 
FROM datensatz_drittmittel_hochschule 
WHERE Jahr = 2008 AND Hochschule = 'Humboldt-Universität zu Berlin'
```
## 7. Ergebnisverarbeitung
Die Datenbank liefert mehrere Zeilen mit Drittmittelbeträgen.  
Der Bot summiert diese Beträge (z. B. 61.275.000 Euro in Tausend Euro).

## 8. Antwort an den Benutzer
Schließlich gibt der Bot die Antwort zurück:  
„Im Jahr 2008 betrugen die Drittmittel der Humboldt-Universität zu Berlin insgesamt 61.275.000 Euro (in Tausend Euro).“
