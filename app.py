"""
Häusler-Assistent — Unternehmensweiter KI-Wissensassistent mit Agent-Skills
Häusler Technologie AG, Winterthur

Stack: Streamlit + LangChain + ChromaDB + OpenAI API
RLS:   Rollenbasierte Zugriffskontrolle auf Retrieval-Ebene (server-seitig)
Agent: Tool-Calling mit 4 Skills (RAG, IT-Ticket, HR-Anfrage, Angebot)
Seminararbeit "Artifacts in IT" | MSc Wirtschaftsinformatik | Gruppe ARTI
"""

import os
import random
import streamlit as st
from datetime import datetime, timedelta

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# ─── BENUTZERVERWALTUNG ──────────────────────────────────────────────────────
DEMO_USERS = {
    "a.berger":  {"name": "Anna Berger",    "password": "Demo2026",  "rolle": "alle_ma"},
    "p.huber":   {"name": "Peter Huber",    "password": "Demo2026",  "rolle": "support"},
    "s.keller":  {"name": "Sarah Keller",   "password": "Demo2026",  "rolle": "kommerziell"},
    "t.bauer":   {"name": "Thomas Bauer",   "password": "Demo2026",  "rolle": "technisch"},
    "admin":     {"name": "Administrator",  "password": "Admin2026", "rolle": "gl"},
}

# ─── ROLLENKONFIGURATION (RLS) ───────────────────────────────────────────────
ROLLEN_CONFIG = {
    "alle_ma": {
        "label": "👤 Mitarbeitende (Allgemein)",
        "abteilungen": ["alle_ma"],
        "beschreibung": "Zugriff auf allgemeine Unternehmensinfos",
        "badge_css": "background:#dbeafe;color:#1d4ed8",
        "skills": ["suche_wissensbasis", "erstelle_hr_anfrage", "erstelle_it_ticket"],
    },
    "support": {
        "label": "🧑‍💼 HR & IT",
        "abteilungen": ["alle_ma", "hr", "it"],
        "beschreibung": "HR-Reglement, IT-Richtlinien, Onboarding",
        "badge_css": "background:#fce7f3;color:#9d174d",
        "skills": ["suche_wissensbasis", "erstelle_hr_anfrage", "erstelle_it_ticket"],
    },
    "kommerziell": {
        "label": "💼 Kommerziell",
        "abteilungen": ["alle_ma", "kommerziell"],
        "beschreibung": "Vertrieb, Einkauf, Finance, Kundendienst",
        "badge_css": "background:#dcfce7;color:#166534",
        "skills": ["suche_wissensbasis", "erstelle_angebot", "erstelle_it_ticket"],
    },
    "technisch": {
        "label": "⚙️ Technisch",
        "abteilungen": ["alle_ma", "technisch"],
        "beschreibung": "Produktion, Qualität, Entwicklung",
        "badge_css": "background:#ffedd5;color:#9a3412",
        "skills": ["suche_wissensbasis", "erstelle_it_ticket", "erstelle_hr_anfrage"],
    },
    "gl": {
        "label": "🏢 Geschäftsleitung",
        "abteilungen": ["alle_ma", "hr", "it", "kommerziell", "technisch", "gl"],
        "beschreibung": "Vollzugriff inkl. Strategie und Finanzkennzahlen",
        "badge_css": "background:#ede9fe;color:#5b21b6",
        "skills": ["suche_wissensbasis", "erstelle_angebot", "erstelle_hr_anfrage", "erstelle_it_ticket"],
    },
}

DOKUMENT_ABTEILUNG = {
    "onboarding_guide.txt":           "alle_ma",
    "produktkatalog.txt":             "alle_ma",
    "it_richtlinien.txt":             "it",
    "hr_reglement.txt":               "alle_ma",
    "angebotsvorlagen.txt":           "kommerziell",
    "finance_budget.txt":             "kommerziell",
    "einkauf_lieferanten.txt":        "kommerziell",
    "qualitaetshandbuch.txt":         "technisch",
    "technische_spezifikationen.txt": "technisch",
    "gl_strategie.txt":               "gl",
}

SKILL_META = {
    "suche_wissensbasis": {"icon": "🔍", "label": "Wissensbasis durchsucht"},
    "erstelle_it_ticket":  {"icon": "🎫", "label": "IT-Ticket erstellt"},
    "erstelle_hr_anfrage": {"icon": "📋", "label": "HR-Anfrage erstellt"},
    "erstelle_angebot":    {"icon": "📄", "label": "Angebot generiert"},
}

SYSTEM_PROMPT = """Du bist der interne Wissens- und Aufgabenassistent der Häusler Technologie AG in Winterthur.
Du hilfst Mitarbeitenden mit Fragen UND kannst konkrete Aktionen ausführen (Skills).

REGELN:
- Antworte immer auf Deutsch, klar und direkt.
- Nutze die verfügbaren Skills intelligent und kombiniere sie wenn sinnvoll.
- Bei Wissensfragen: nutze IMMER suche_wissensbasis zuerst.
- UNSICHERHEITSREGEL: Wenn eine Information nicht in der Wissensbasis liegt, sage es klar.
  Antworte IMMER mit: "⚠️ Diese Information liegt nicht in der internen Wissensbasis vor."
- Bei Angeboten: lies immer zuerst Preise/Konditionen aus der Wissensbasis, dann erstelle das Angebot.
- Fasse dich kurz — maximal 10 Sätze, ausser mehr Detail ist explizit gewünscht."""

# ─── SEITENKONFIGURATION ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Häusler-Assistent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

    .login-wrap { max-width: 420px; margin: 4rem auto; }

    [data-testid="stSidebar"] { background: #0f1b2d !important; border-right: 1px solid #1e3a5f; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] hr { border-color: #1e3a5f !important; }
    [data-testid="stSidebar"] .stButton button {
        background: #1e3a5f !important; color: #e2e8f0 !important;
        border: 1px solid #2d5a8e !important; border-radius: 6px !important;
        font-size: 0.8rem !important; text-align: left !important;
    }
    [data-testid="stSidebar"] .stButton button:hover { background: #2d5a8e !important; }

    .main .block-container { padding-top: 1.5rem; max-width: 900px; }

    .hauesler-header {
        background: linear-gradient(135deg, #0f1b2d 0%, #1e3a5f 100%);
        border-radius: 12px; padding: 20px 28px; margin-bottom: 16px;
        display: flex; align-items: center; justify-content: space-between;
        border: 1px solid #2d5a8e;
    }
    .hauesler-header h2 { color: white; margin: 0; font-size: 1.4rem; font-weight: 700; }
    .hauesler-header p  { color: #94a3b8; margin: 4px 0 0 0; font-size: 0.82rem; }
    .role-badge { display: inline-block; padding: 5px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }

    .skills-bar {
        display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap;
    }
    .skill-chip {
        display: inline-flex; align-items: center; gap: 4px;
        padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;
        background: #0f2744; border: 1px solid #2d5a8e; color: #93c5fd;
    }

    .rls-info {
        background: #0f2744; border: 1px solid #2d5a8e; border-left: 3px solid #4a90d9;
        padding: 10px 14px; border-radius: 8px; font-size: 0.81rem; color: #93c5fd !important;
    }

    [data-testid="stChatMessage"] {
        border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.06) !important;
        padding: 12px !important; margin-bottom: 8px !important;
    }

    .tool-call-box {
        background: #0a1f0a; border: 1px solid #166534; border-left: 3px solid #16a34a;
        padding: 10px 14px; border-radius: 8px; margin: 6px 0;
        font-size: 0.82rem; color: #86efac;
    }
    .tool-call-box strong { color: #4ade80; }

    .offer-box {
        background: #0f1b2d; border: 1px solid #2d5a8e; border-left: 3px solid #4a90d9;
        padding: 14px 18px; border-radius: 8px; margin: 8px 0;
        font-size: 0.88rem; color: #e2e8f0; line-height: 1.7;
    }
    .offer-box .offer-nr { font-size: 1rem; font-weight: 700; color: #93c5fd; }

    .source-box {
        background: #0f2744; border-left: 3px solid #4a90d9;
        padding: 10px 14px; border-radius: 6px; margin: 6px 0;
        font-size: 0.81rem; color: #cbd5e1 !important; line-height: 1.5;
    }
    .source-box strong { color: #93c5fd !important; }

    [data-testid="stChatInput"] textarea {
        border-radius: 10px !important; border: 1.5px solid #2d5a8e !important;
    }
    [data-testid="stExpander"] {
        border: 1px solid #1e3a5f !important; border-radius: 8px !important; background: #0a1628 !important;
    }
    hr { border-color: #1e3a5f !important; }

    .user-card {
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.6rem 0.9rem; border-radius: 8px; margin-bottom: 0.4rem;
        background: #0f1b2d; border: 1px solid #1e3a5f; font-size: 0.82rem;
    }
    .user-badge { display: inline-block; padding: 2px 10px; border-radius: 10px; font-size: 0.7rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ─── API-KEY ─────────────────────────────────────────────────────────────────
def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            st.error("❌ Kein OpenAI API Key. Bitte in `.streamlit/secrets.toml` eintragen.")
            st.stop()
        return key


# ─── LOGIN ───────────────────────────────────────────────────────────────────
def zeige_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🤖 Häusler-Assistent")
        st.caption("Häusler Technologie AG · Bitte anmelden")

        with st.form("login_form"):
            username = st.text_input("Benutzername", placeholder="vorname.nachname")
            password = st.text_input("Passwort", type="password")
            if st.form_submit_button("Anmelden →", use_container_width=True):
                user = DEMO_USERS.get(username.strip().lower())
                if user and user["password"] == password:
                    st.session_state.update({
                        "authenticated": True,
                        "auth_rolle": user["rolle"],
                        "auth_name": user["name"],
                        "messages": [],
                    })
                    st.rerun()
                else:
                    st.error("❌ Ungültige Zugangsdaten.")

        st.markdown("---")
        st.caption("**Demo-Zugänge**")
        for uname, info in DEMO_USERS.items():
            rolle_info = ROLLEN_CONFIG[info["rolle"]]
            st.markdown(
                f'<div class="user-card">'
                f'<span class="user-badge" style="{rolle_info["badge_css"]}">{rolle_info["label"]}</span>'
                f'<span style="color:#64748b;font-family:monospace;font-size:0.75rem">{uname} / {info["password"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─── WISSENSBASIS ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_vectorstore(api_key: str):
    docs_path = os.path.join(os.path.dirname(__file__), "docs")
    loader = DirectoryLoader(
        docs_path, glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}, show_progress=False,
    )
    raw_docs = loader.load()
    for doc in raw_docs:
        filename = os.path.basename(doc.metadata.get("source", ""))
        doc.metadata["abteilung"] = DOKUMENT_ABTEILUNG.get(filename, "alle_ma")
        doc.metadata["filename"] = filename

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(raw_docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore, len(raw_docs), len(chunks)


def get_retriever(vectorstore, rolle: str):
    erlaubte = ROLLEN_CONFIG[rolle]["abteilungen"]
    chroma_filter = (
        {"abteilung": erlaubte[0]} if len(erlaubte) == 1
        else {"abteilung": {"$in": erlaubte}}
    )
    return vectorstore.as_retriever(search_kwargs={"k": 8, "filter": chroma_filter})


# ─── SKILLS (TOOLS) ──────────────────────────────────────────────────────────
def create_tools(vectorstore, rolle: str, api_key: str):

    @tool
    def suche_wissensbasis(frage: str) -> str:
        """Durchsucht die interne Wissensbasis der Häusler AG nach relevanten Informationen.
        Immer aufrufen bei Fragen zu Prozessen, Produkten, Richtlinien oder Mitarbeiterthemen."""
        retriever = get_retriever(vectorstore, rolle)
        docs = retriever.invoke(frage)
        if not docs:
            return "Keine relevanten Informationen in der Wissensbasis gefunden."
        return "\n\n---\n".join(
            [f"[Quelle: {d.metadata.get('filename','?')} | Abt: {d.metadata.get('abteilung','?')}]\n{d.page_content}"
             for d in docs]
        )

    @tool
    def erstelle_it_ticket(beschreibung: str, prioritaet: str = "normal") -> str:
        """Erstellt ein IT-Support-Ticket für technische Probleme oder Störungen.
        prioritaet: 'niedrig', 'normal', 'hoch', 'kritisch'"""
        ticket_nr = f"INC-{random.randint(10000, 99999)}"
        sla = {"niedrig": "2 Werktage", "normal": "4 Stunden", "hoch": "1 Stunde", "kritisch": "15 Minuten"}
        return (
            f"🎫 **IT-Ticket erstellt**\n"
            f"**Ticket-Nr.:** {ticket_nr}\n"
            f"**Beschreibung:** {beschreibung}\n"
            f"**Priorität:** {prioritaet.capitalize()}\n"
            f"**SLA:** {sla.get(prioritaet, '4 Stunden')}\n"
            f"**Erstellt:** {datetime.now().strftime('%d.%m.%Y %H:%M')}\n"
            f"**Status:** Offen — das IT-Team wird sich gemäss SLA melden."
        )

    @tool
    def erstelle_hr_anfrage(art: str, beschreibung: str, datum: str = "") -> str:
        """Erstellt eine HR-Anfrage (z.B. Ferienantrag, Krankheitsmeldung, Lohnbescheinigung).
        art: Art der Anfrage. beschreibung: Details. datum: optionales Datum."""
        anfrage_nr = f"HR-{random.randint(1000, 9999)}"
        datum_str = f"\n**Datum/Zeitraum:** {datum}" if datum else ""
        return (
            f"📋 **HR-Anfrage eingereicht**\n"
            f"**Anfrage-Nr.:** {anfrage_nr}\n"
            f"**Art:** {art}\n"
            f"**Beschreibung:** {beschreibung}"
            f"{datum_str}\n"
            f"**Eingereicht am:** {datetime.now().strftime('%d.%m.%Y %H:%M')}\n"
            f"**Status:** Eingegangen — HR meldet sich innerhalb von 2 Werktagen."
        )

    @tool
    def erstelle_angebot(kunde: str, produkte: str, menge: str = "1", rabatt_prozent: str = "0") -> str:
        """Erstellt ein Kundenangebot für Häusler-Produkte.
        Zuerst immer suche_wissensbasis aufrufen um aktuelle Preise/Konditionen zu lesen.
        kunde: Firmenname des Kunden. produkte: Produktbezeichnung(en). menge: Stückzahl. rabatt_prozent: Rabatt in %."""
        angebots_nr = f"ANG-2026-{random.randint(10000, 99999)}"
        gueltig_bis = (datetime.now() + timedelta(days=30)).strftime("%d.%m.%Y")
        rabatt = float(rabatt_prozent) if rabatt_prozent else 0
        gl_hinweis = ""
        if rabatt > 15:
            gl_hinweis = "\n⚠️ **GL-Genehmigung erforderlich** (Rabatt > 15% — bitte vor Versand freigeben lassen)"
        elif rabatt > 10:
            gl_hinweis = "\n⚠️ **Abteilungsleiter-Freigabe erforderlich** (Rabatt > 10%)"

        return (
            f"📄 **Angebot erstellt**\n"
            f"**Angebots-Nr.:** {angebots_nr}\n"
            f"**Kunde:** {kunde}\n"
            f"**Produkte:** {produkte}\n"
            f"**Menge:** {menge} Einheit(en)\n"
            f"**Rabatt:** {rabatt:.0f}%\n"
            f"**Gültig bis:** {gueltig_bis}\n"
            f"**Erstellt:** {datetime.now().strftime('%d.%m.%Y %H:%M')}\n"
            f"**Status:** Entwurf — Preise basieren auf aktueller Preisliste"
            f"{gl_hinweis}\n\n"
            f"*Angebot zur Kontrolle und finalen Preisanpassung dem Vertriebsleiter weiterleiten.*"
        )

    # Nur freigeschaltete Skills pro Rolle zurückgeben
    all_tools = {
        "suche_wissensbasis": suche_wissensbasis,
        "erstelle_it_ticket":  erstelle_it_ticket,
        "erstelle_hr_anfrage": erstelle_hr_anfrage,
        "erstelle_angebot":    erstelle_angebot,
    }
    erlaubte_skills = ROLLEN_CONFIG[rolle]["skills"]
    return [all_tools[s] for s in erlaubte_skills if s in all_tools]


# ─── AGENT (Tool-Calling Loop) ────────────────────────────────────────────────
def run_agent(frage: str, vectorstore, rolle: str, api_key: str, chat_history: list):
    tools = create_tools(vectorstore, rolle, api_key)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key, max_tokens=1200)
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in chat_history[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=frage))

    used_tools = []
    MAX_ITERATIONS = 6

    for _ in range(MAX_ITERATIONS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            name   = tc["name"]
            args   = tc["args"]
            result = tool_map[name].invoke(args) if name in tool_map else "Tool nicht verfügbar."
            used_tools.append({"name": name, "args": args, "result": result})
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return response.content, used_tools


# ─── HAUPTFLUSS ──────────────────────────────────────────────────────────────
if not st.session_state.get("authenticated"):
    zeige_login()
    st.stop()

aktuelle_rolle = st.session_state["auth_rolle"]
auth_name      = st.session_state["auth_name"]
rolle_info     = ROLLEN_CONFIG[aktuelle_rolle]

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Häusler-Assistent")
    st.caption("Unternehmensweites Wissensmanagement | PoC v1.0")
    st.divider()

    st.markdown("**🔐 Angemeldet als**")
    st.markdown(
        f'<div class="rls-info">'
        f'👤 <strong>{auth_name}</strong><br>'
        f'<span style="font-size:0.78rem">{rolle_info["label"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**🛠️ Verfügbare Skills**")
    for s in rolle_info["skills"]:
        meta = SKILL_META.get(s, {"icon": "🔧", "label": s})
        st.markdown(f'<span class="skill-chip">{meta["icon"]} {meta["label"]}</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**📁 Wissensbasis**")
    for fname, abt in sorted(DOKUMENT_ABTEILUNG.items()):
        erlaubt = abt in rolle_info["abteilungen"]
        st.markdown(f'{"✅" if erlaubt else "🔒"} `{fname.replace(".txt","")}`')

    st.divider()
    st.markdown("**💡 Beispiele**")
    beispiele = {
        "alle_ma":     [
            "Was sind meine Ferienregelungen?",
            "Wie stelle ich einen VPN-Antrag?",
            "Ich bin krank — was muss ich tun und erstelle eine HR-Anfrage",
            "⚠️ T3: Wie viele Mitarbeitende hat die Häusler AG?",
        ],
        "support":     [
            "Wie viele Ferientage habe ich im ersten Jahr?",
            "Erstelle ein IT-Ticket: Mein Laptop startet nicht mehr",
            "Wie melde ich einen neuen Mitarbeitenden an?",
            "⚠️ T3: Was ist das Durchschnittsgehalt im IT?",
        ],
        "kommerziell": [
            "Erstelle ein Angebot für Müller AG, HV-400, 10 Stück, 8% Rabatt",
            "Ab welchem Rabatt brauche ich GL-Genehmigung?",
            "Erstelle ein Angebot für Meier GmbH, HV-200 und HV-300, 5 Stück je, 18% Rabatt",
            "⚠️ T3: Was ist der aktuelle EUR/CHF Kurs?",
        ],
        "technisch":   [
            "Was sind die Druckstufen des HV-400?",
            "Erstelle ein IT-Ticket: Produktionsanlage Linie 3 ausgefallen, kritisch",
            "Wie läuft ein Reklamationsprozess ab?",
            "⚠️ T3: Wie viele Reklamationen hatten wir im letzten Quartal?",
        ],
        "gl":          [
            "Was sind unsere Stossrichtungen bis 2027?",
            "Erstelle ein Angebot für Swiss Pharma AG, HV-Serie komplett, 25 Stück, 20% Rabatt",
            "Was sind die Top-Risiken laut Risikoregister?",
            "⚠️ T3: Was ist unser aktueller Börsenkurs?",
        ],
    }
    for frage in beispiele.get(aktuelle_rolle, []):
        if st.button(frage, key=frage, use_container_width=True):
            st.session_state["pending_question"] = frage

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Leeren", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            for k in ["authenticated", "auth_rolle", "auth_name", "messages", "pending_question"]:
                st.session_state.pop(k, None)
            st.rerun()
    st.caption("⚠️ KI kann irren — Quellenangaben prüfen.\n© Häusler Technologie AG")


# ─── HAUPTBEREICH ────────────────────────────────────────────────────────────
api_key = get_api_key()

st.markdown(
    f"""<div class="hauesler-header">
        <div>
            <h2>🤖 Häusler-Assistent</h2>
            <p>Wissensmanagement &amp; Agent-Skills &nbsp;·&nbsp; Häusler Technologie AG, Winterthur</p>
        </div>
        <span class="role-badge" style="{rolle_info['badge_css']}">{rolle_info['label']}</span>
    </div>""",
    unsafe_allow_html=True,
)

# Skills-Bar
skills_html = " ".join([
    f'<span class="skill-chip">{SKILL_META[s]["icon"]} {SKILL_META[s]["label"]}</span>'
    for s in rolle_info["skills"] if s in SKILL_META
])
st.markdown(f'<div class="skills-bar">{skills_html}</div>', unsafe_allow_html=True)

with st.spinner("⏳ Wissensbasis wird geladen..."):
    vectorstore, n_docs, n_chunks = build_vectorstore(api_key)
st.success(f"✅ {n_docs} Dokumente · {n_chunks} Abschnitte · {len(rolle_info['skills'])} Skills aktiv")

if "messages" not in st.session_state or not st.session_state["messages"]:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": (
            f"Guten Tag, **{auth_name}**! Ich bin der Häusler-Assistent.\n\n"
            f"Als **{rolle_info['label']}** stehen Ihnen folgende Skills zur Verfügung: "
            f"**{', '.join([SKILL_META[s]['icon']+' '+SKILL_META[s]['label'] for s in rolle_info['skills'] if s in SKILL_META])}**.\n\n"
            f"Stellen Sie eine Frage oder geben Sie mir eine Aufgabe."
        ),
        "tools": [],
        "sources": [],
    }]

# Chat anzeigen
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        inhalt = msg["content"]
        ist_leer = any(p in inhalt for p in ["nicht in der internen Wissensbasis", "liegt nicht vor", "wenden Sie sich"])
        if msg["role"] == "assistant" and ist_leer:
            st.warning(inhalt, icon="⚠️")
        else:
            st.markdown(inhalt)

        # Tool-Calls anzeigen
        if msg.get("tools"):
            with st.expander(f"🛠️ {len(msg['tools'])} Skill(s) verwendet"):
                for tc in msg["tools"]:
                    meta = SKILL_META.get(tc["name"], {"icon": "🔧", "label": tc["name"]})
                    st.markdown(
                        f'<div class="tool-call-box">'
                        f'<strong>{meta["icon"]} {meta["label"]}</strong><br>'
                        f'<pre style="color:#86efac;font-size:0.78rem;margin:4px 0 0 0">{tc["result"]}</pre>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


def verarbeite_frage(frage: str):
    st.session_state["messages"].append({"role": "user", "content": frage, "tools": [], "sources": []})
    with st.chat_message("user"):
        st.markdown(frage)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Denke nach und führe Skills aus..."):
            antwort, used_tools = run_agent(frage, vectorstore, aktuelle_rolle, api_key, st.session_state["messages"][:-1])

        ist_leer = any(p in antwort for p in ["nicht in der internen Wissensbasis", "liegt nicht vor", "wenden Sie sich"])
        if ist_leer:
            st.warning(antwort, icon="⚠️")
        else:
            st.markdown(antwort)

        if used_tools:
            with st.expander(f"🛠️ {len(used_tools)} Skill(s) verwendet"):
                for tc in used_tools:
                    meta = SKILL_META.get(tc["name"], {"icon": "🔧", "label": tc["name"]})
                    st.markdown(
                        f'<div class="tool-call-box">'
                        f'<strong>{meta["icon"]} {meta["label"]}</strong><br>'
                        f'<pre style="color:#86efac;font-size:0.78rem;margin:4px 0 0 0;white-space:pre-wrap">{tc["result"]}</pre>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    st.session_state["messages"].append({
        "role": "assistant", "content": antwort, "tools": used_tools, "sources": [],
    })


if "pending_question" in st.session_state:
    verarbeite_frage(st.session_state.pop("pending_question"))

if user_input := st.chat_input("Frage stellen oder Aufgabe beschreiben..."):
    verarbeite_frage(user_input)
