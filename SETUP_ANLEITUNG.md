# Setup-Anleitung — Häusler-Assistent

## Voraussetzungen
- Python 3.11 oder neuer (https://python.org)
- OpenAI API Key (https://platform.openai.com → API Keys)
- Git (für Deployment)

---

## A — LOKAL STARTEN (5 Minuten)

```bash
# 1. In den Ordner wechseln
cd /Users/rinoymanavalan/Desktop/ARTI/prototype

# 2. Virtuelle Umgebung erstellen (empfohlen)
python -m venv venv
source venv/bin/activate        # Mac/Linux
# oder: venv\Scripts\activate   # Windows

# 3. Pakete installieren
pip install -r requirements.txt

# 4. API Key eintragen
# Öffne .streamlit/secrets.toml und ersetze sk-... mit deinem echten Key

# 5. App starten
streamlit run app.py
```

→ Browser öffnet sich automatisch auf http://localhost:8501

**Kosten**: ~$0.002 pro Anfrage (Embedding + GPT-4o-mini). 50 Testanfragen = ca. $0.10.

---

## B — AUF STREAMLIT CLOUD DEPLOYEN (10 Minuten, öffentliche URL)

### Schritt 1: GitHub Repository erstellen
1. Gehe zu https://github.com → «New repository»
2. Name: `hauesler-assistant` (oder beliebig)
3. Visibility: **Private** (wichtig — API Key-Schutz)
4. Repository erstellen

### Schritt 2: Code pushen
```bash
cd /Users/rinoymanavalan/Desktop/ARTI/prototype
git init
git add .
git commit -m "Initial commit: Häusler-Assistent PoC"
git remote add origin https://github.com/DEIN-USERNAME/hauesler-assistant.git
git push -u origin main
```

### Schritt 3: Streamlit Cloud konfigurieren
1. Gehe zu https://share.streamlit.io
2. «New app» → «From existing repo»
3. Repository auswählen: `hauesler-assistant`
4. Main file: `app.py`
5. Klick auf «Advanced settings» → «Secrets»
6. Folgendes eintragen:
   ```
   OPENAI_API_KEY = "sk-dein-echter-key"
   ```
7. «Deploy» klicken

→ Nach 2–3 Minuten erhältst du eine öffentliche URL wie:
  `https://hauesler-assistant-xyz.streamlit.app`

Diese URL kann ohne Installation im Browser geöffnet werden — ideal für die Demo und das Video.

---

## C — STRUKTUR DES PROJEKTS

```
prototype/
├── app.py                  ← Hauptapplikation (Streamlit)
├── requirements.txt        ← Python-Abhängigkeiten
├── .gitignore              ← Schützt secrets.toml vor GitHub
├── .streamlit/
│   └── secrets.toml        ← API Key (NUR lokal, nie pushen!)
└── docs/                   ← Wissensbasis (Häusler-Dokumente)
    ├── hr_reglement.txt
    ├── produktkatalog.txt
    ├── onboarding_guide.txt
    ├── it_richtlinien.txt
    └── angebotsvorlagen.txt
```

---

## D — NEUE DOKUMENTE HINZUFÜGEN

1. Textdatei (`.txt`) in `docs/` ablegen
2. App neu starten (lokal) oder neu deployen (Streamlit Cloud)
3. Die Wissensbasis wird automatisch neu indexiert

PDF-Dokumente: LangChain unterstützt auch PDFs. In `app.py` Zeile mit `glob="**/*.txt"` 
auf `glob="**/*.{txt,pdf}"` ändern und `PyPDFLoader` hinzufügen.

---

## E — HÄUFIGE FEHLER

**Fehler: `ModuleNotFoundError: No module named 'langchain_community'`**
→ `pip install -r requirements.txt` nochmals ausführen

**Fehler: `AuthenticationError: Incorrect API key`**
→ API Key in `.streamlit/secrets.toml` prüfen (beginnt mit `sk-`)

**Fehler: `No documents found`**
→ Sicherstellen, dass `.txt`-Dateien in `docs/` vorhanden sind

**App startet, aber Antworten sind auf Englisch**
→ Prompt in `app.py` (SYSTEM_PROMPT) ist korrekt auf Deutsch — evtl. LangChain-Cache löschen: `st.cache_resource.clear()`
