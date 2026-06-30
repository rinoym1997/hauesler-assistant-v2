"""
Häusler-Assistent — Unternehmensweiter KI-Wissensassistent
Häusler Technologie AG, Winterthur

Stack: Streamlit + LangChain + ChromaDB + OpenAI API
RLS: Rollenbasierte Zugriffskontrolle auf Retrieval-Ebene
Seminararbeit "Artifacts in IT" | MSc Wirtschaftsinformatik | Gruppe ARTI
"""

import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ─── ROLLENKONFIGURATION (RLS) ───────────────────────────────────────────────
ROLLEN_CONFIG = {
    "alle_ma": {
        "label": "👤 Mitarbeitende (Allgemein)",
        "abteilungen": ["alle_ma"],
        "beschreibung": "Zugriff auf allgemeine Unternehmensinfos"
    },
    "support": {
        "label": "🧑‍💼 HR & IT",
        "abteilungen": ["alle_ma", "hr", "it"],
        "beschreibung": "HR-Reglement, IT-Richtlinien, Onboarding"
    },
    "kommerziell": {
        "label": "💼 Kommerziell (Vertrieb / Finance / Einkauf)",
        "abteilungen": ["alle_ma", "kommerziell"],
        "beschreibung": "Vertrieb, Einkauf, Finance, Kundendienst"
    },
    "technisch": {
        "label": "⚙️ Technisch (Operations / QM / F&E)",
        "abteilungen": ["alle_ma", "technisch"],
        "beschreibung": "Produktion, Qualität, Entwicklung, Einkauf-Technik"
    },
    "gl": {
        "label": "🏢 Geschäftsleitung",
        "abteilungen": ["alle_ma", "hr", "it", "kommerziell", "technisch", "gl"],
        "beschreibung": "Vollzugriff inkl. Strategie und Finanzkennzahlen"
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

SYSTEM_PROMPT = """Du bist der interne Wissensassistent der Häusler Technologie AG in Winterthur.
Du beantwortest Fragen von Mitarbeitenden präzise und professionell — ausschliesslich
auf Basis der bereitgestellten internen Dokumente.

REGELN:
- Antworte immer auf Deutsch, klar und direkt.
- Stütze dich NUR auf den untenstehenden Kontext. Keine Annahmen. Keine Erfindungen.
- UNSICHERHEITSREGEL (zwingend): Wenn die gesuchte Information NICHT im Kontext steht,
  antworte IMMER mit:
  "⚠️ Diese Information liegt nicht in der internen Wissensbasis vor.
  Bitte wenden Sie sich an die zuständige Abteilung oder Ihren Vorgesetzten."
  Füge optional hinzu, welche Abteilung zuständig sein könnte (HR, IT, GL, etc.).
  Erfinde NIEMALS Antworten aus allgemeinem Wissen.
- Zitiere konkrete Zahlen, Fristen und Prozesse wenn vorhanden.
- Bei abteilungsübergreifenden Fragen: synthesiere Informationen aus mehreren Quellen.
- Maximal 8 Sätze, ausser die Frage erfordert mehr Detail."""

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

    [data-testid="stSidebar"] {
        background: #0f1b2d !important;
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #94a3b8 !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
    [data-testid="stSidebar"] hr { border-color: #1e3a5f !important; }
    [data-testid="stSidebar"] .stButton button {
        background: #1e3a5f !important; color: #e2e8f0 !important;
        border: 1px solid #2d5a8e !important; border-radius: 6px !important;
        font-size: 0.8rem !important; text-align: left !important;
        transition: all 0.15s ease;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: #2d5a8e !important; border-color: #4a90d9 !important;
    }

    .main .block-container { padding-top: 1.5rem; max-width: 900px; }

    .hauesler-header {
        background: linear-gradient(135deg, #0f1b2d 0%, #1e3a5f 100%);
        border-radius: 12px; padding: 20px 28px; margin-bottom: 20px;
        display: flex; align-items: center; justify-content: space-between;
        border: 1px solid #2d5a8e;
    }
    .hauesler-header h2 { color: white; margin: 0; font-size: 1.4rem; font-weight: 700; }
    .hauesler-header p { color: #94a3b8; margin: 4px 0 0 0; font-size: 0.82rem; }

    .role-badge {
        display: inline-block; padding: 5px 14px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.02em;
    }
    .role-alle_ma     { background: #dbeafe; color: #1d4ed8; }
    .role-support     { background: #fce7f3; color: #9d174d; }
    .role-kommerziell { background: #dcfce7; color: #166534; }
    .role-technisch   { background: #ffedd5; color: #9a3412; }
    .role-gl          { background: #ede9fe; color: #5b21b6; }

    .rls-info {
        background: #0f2744; border: 1px solid #2d5a8e;
        border-left: 3px solid #4a90d9;
        padding: 10px 14px; border-radius: 8px; font-size: 0.81rem;
        color: #93c5fd !important;
    }

    [data-testid="stChatMessage"] {
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        padding: 12px !important;
        margin-bottom: 8px !important;
    }

    .source-box {
        background: #0f2744;
        border-left: 3px solid #4a90d9;
        padding: 10px 14px; border-radius: 6px; margin: 6px 0;
        font-size: 0.81rem; color: #cbd5e1 !important;
        line-height: 1.5;
    }
    .source-box strong { color: #93c5fd !important; }
    .source-box em { color: #64748b !important; }

    [data-testid="stChatInput"] textarea {
        border-radius: 10px !important;
        border: 1.5px solid #2d5a8e !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #4a90d9 !important;
        box-shadow: 0 0 0 3px rgba(74,144,217,0.15) !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid #1e3a5f !important;
        border-radius: 8px !important;
        background: #0a1628 !important;
    }
    [data-testid="stExpander"] summary { font-size: 0.83rem !important; }

    hr { border-color: #1e3a5f !important; }
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


# ─── WISSENSBASIS LADEN ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_vectorstore(api_key: str):
    docs_path = os.path.join(os.path.dirname(__file__), "docs")
    loader = DirectoryLoader(
        docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    raw_docs = loader.load()

    for doc in raw_docs:
        filename = os.path.basename(doc.metadata.get("source", ""))
        doc.metadata["abteilung"] = DOKUMENT_ABTEILUNG.get(filename, "alle_ma")
        doc.metadata["filename"] = filename

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore, len(raw_docs), len(chunks)


# ─── RLS-RETRIEVER ───────────────────────────────────────────────────────────
def get_retriever(vectorstore, rolle: str):
    erlaubte = ROLLEN_CONFIG[rolle]["abteilungen"]
    chroma_filter = (
        {"abteilung": erlaubte[0]}
        if len(erlaubte) == 1
        else {"abteilung": {"$in": erlaubte}}
    )
    return vectorstore.as_retriever(
        search_kwargs={"k": 8, "filter": chroma_filter}
    )


# ─── RAG-ABFRAGE (ohne deprecated Chain) ─────────────────────────────────────
def rag_query(frage: str, vectorstore, rolle: str, api_key: str, chat_history: list):
    """
    Direkte RAG-Implementierung ohne ConversationalRetrievalChain:
    1. Kontext per RLS-Retriever abrufen
    2. LLM mit Kontext + Gesprächsverlauf aufrufen
    """
    retriever = get_retriever(vectorstore, rolle)
    quellen = retriever.invoke(frage)

    kontext = "\n\n---\n\n".join([doc.page_content for doc in quellen])

    system_msg = SystemMessage(content=f"{SYSTEM_PROMPT}\n\nKontext aus internen Dokumenten:\n{kontext}")

    messages = [system_msg]
    # Gesprächsverlauf einfügen (max. letzte 6 Nachrichten)
    for msg in chat_history[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=frage))

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=api_key,
        max_tokens=900,
    )
    response = llm.invoke(messages)
    return response.content, quellen


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Häusler-Assistent")
    st.caption("Unternehmensweites Wissensmanagement | PoC v1.0")
    st.divider()

    st.markdown("**🔐 Anmeldung (Rolle simulieren)**")
    st.caption("In der Produktivversion: Microsoft Entra ID SSO")

    rolle_options = {v["label"]: k for k, v in ROLLEN_CONFIG.items()}
    rolle_labels = list(rolle_options.keys())

    # Rolle aus URL-Parameter lesen (gesetzt vom Frontend-Login)
    url_rolle = st.query_params.get("rolle", None)
    default_idx = 0
    if url_rolle and url_rolle in ROLLEN_CONFIG:
        default_label = ROLLEN_CONFIG[url_rolle]["label"]
        if default_label in rolle_labels:
            default_idx = rolle_labels.index(default_label)

    gewaehlte_label = st.selectbox("Ich bin...", options=rolle_labels, index=default_idx, key="rolle_selector")
    aktuelle_rolle = rolle_options[gewaehlte_label]
    rolle_info = ROLLEN_CONFIG[aktuelle_rolle]

    st.markdown(
        f'<div class="rls-info">🔒 Zugriff auf: '
        f'<strong>{", ".join(rolle_info["abteilungen"])}</strong><br>'
        f'{rolle_info["beschreibung"]}</div>',
        unsafe_allow_html=True
    )

    if st.session_state.get("letzte_rolle") != aktuelle_rolle:
        st.session_state["messages"] = []
        st.session_state["letzte_rolle"] = aktuelle_rolle

    st.divider()

    st.markdown("**📁 Wissensbasis**")
    for fname, abt in sorted(DOKUMENT_ABTEILUNG.items()):
        erlaubt = abt in rolle_info["abteilungen"]
        icon = "✅" if erlaubt else "🔒"
        st.markdown(f"{icon} `{fname.replace('.txt','')}`")

    st.divider()

    st.markdown("**💡 Beispielfragen**")
    beispiele = {
        "alle_ma": [
            "Welche Produkte stellen wir her?",
            "Wie melde ich mich für VPN an?",
            "⚠️ T3: Wie viele Mitarbeitende hat die Häusler AG?",
        ],
        "support": [
            "Wie viele Ferientage habe ich im ersten Jahr?",
            "Wie melde ich einen IT-Incident?",
            "Was passiert bei Krankheit länger als 3 Tage?",
            "⚠️ T3: Was ist das Durchschnittsgehalt im IT-Bereich?",
        ],
        "kommerziell": [
            "Ab welchem Rabatt brauche ich GL-Genehmigung?",
            "Was sind die Mindestmargen für die HV-Serie?",
            "Wann brauche ich eine Kreditversicherung für ein Zahlungsziel?",
            "⚠️ T3: Was ist der aktuelle EUR/CHF Wechselkurs?",
        ],
        "technisch": [
            "Was sind die Druckstufen des HV-400?",
            "Wie läuft ein Reklamationsprozess ab?",
            "Wie werden neue Lieferanten freigeschaltet?",
            "⚠️ T3: Wie viele Reklamationen hatten wir im letzten Quartal?",
        ],
        "gl": [
            "Was sind unsere strategischen Stossrichtungen bis 2027?",
            "Wie hoch ist unser geplanter Umsatz 2027?",
            "Was sind die Top-Risiken laut Risikoregister?",
            "⚠️ T3: Was ist unser aktueller Börsenkurs?",
        ],
    }

    for frage in beispiele.get(aktuelle_rolle, beispiele["alle_ma"]):
        if st.button(frage, key=frage, use_container_width=True):
            st.session_state["pending_question"] = frage

    st.divider()
    if st.button("🗑️ Chatverlauf löschen", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    st.caption("⚠️ KI kann irren — Quellenangaben immer prüfen.\n"
               "© Häusler Technologie AG | Interne Nutzung")


# ─── HAUPTBEREICH ────────────────────────────────────────────────────────────
st.markdown(
    f"""<div class="hauesler-header">
        <div>
            <h2>🤖 Häusler-Assistent</h2>
            <p>Unternehmensweites Wissensmanagement-System &nbsp;·&nbsp; Häusler Technologie AG, Winterthur</p>
        </div>
        <span class="role-badge role-{aktuelle_rolle}">{gewaehlte_label}</span>
    </div>""",
    unsafe_allow_html=True
)

api_key = get_api_key()
with st.spinner("⏳ Wissensbasis wird geladen und indexiert..."):
    vectorstore, n_docs, n_chunks = build_vectorstore(api_key)
st.success(f"✅ {n_docs} Dokumente indexiert ({n_chunks} Abschnitte) — "
           f"Ihr Zugriff: **{', '.join(rolle_info['abteilungen'])}**")

if "messages" not in st.session_state or not st.session_state["messages"]:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": (
            f"Guten Tag! Ich bin der Wissensassistent der Häusler Technologie AG.\n\n"
            f"Sie sind als **{gewaehlte_label}** angemeldet und haben Zugriff auf: "
            f"**{', '.join(rolle_info['abteilungen'])}**.\n\n"
            f"Was kann ich für Sie tun?"
        ),
        "sources": [],
    }]

# Chat-Verlauf anzeigen
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        inhalt = msg["content"]
        if msg["role"] == "assistant" and any(p in inhalt for p in [
            "nicht in der internen Wissensbasis", "nicht in der Wissensbasis", "liegt nicht vor",
        ]):
            st.warning(inhalt, icon="⚠️")
        else:
            st.markdown(inhalt)
        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} Quellen"):
                for i, doc in enumerate(msg["sources"], 1):
                    fname = doc.metadata.get("filename", "Unbekannt")
                    abt = doc.metadata.get("abteilung", "—")
                    st.markdown(
                        f'<div class="source-box">'
                        f'<strong>Quelle {i}:</strong> {fname} '
                        f'<em style="color:#888">[{abt}]</em><br>'
                        f'{doc.page_content[:300].strip()}...</div>',
                        unsafe_allow_html=True,
                    )


def verarbeite_frage(frage: str):
    st.session_state["messages"].append({"role": "user", "content": frage, "sources": []})
    with st.chat_message("user"):
        st.markdown(frage)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Suche in der Wissensbasis..."):
            antwort, quellen = rag_query(
                frage, vectorstore, aktuelle_rolle, api_key,
                st.session_state["messages"][:-1]  # ohne die gerade hinzugefügte user-msg
            )

        ist_nicht_beantwortbar = any(p in antwort for p in [
            "nicht in der internen Wissensbasis", "nicht in der Wissensbasis",
            "liegt nicht vor", "wenden Sie sich an",
        ])

        if ist_nicht_beantwortbar:
            st.warning(antwort, icon="⚠️")
        else:
            st.markdown(antwort)

        if quellen and not ist_nicht_beantwortbar:
            with st.expander(f"📎 {len(quellen)} Quellen"):
                for i, doc in enumerate(quellen, 1):
                    fname = doc.metadata.get("filename", "Unbekannt")
                    abt = doc.metadata.get("abteilung", "—")
                    st.markdown(
                        f'<div class="source-box">'
                        f'<strong>Quelle {i}:</strong> {fname} '
                        f'<em style="color:#888">[{abt}]</em><br>'
                        f'{doc.page_content[:300].strip()}...</div>',
                        unsafe_allow_html=True,
                    )
        elif not ist_nicht_beantwortbar:
            st.info("ℹ️ Keine relevanten Quellen in Ihrem Zugriffsbereich gefunden.")

    st.session_state["messages"].append({"role": "assistant", "content": antwort, "sources": quellen})


if "pending_question" in st.session_state:
    verarbeite_frage(st.session_state.pop("pending_question"))

if user_input := st.chat_input("Ihre Frage an den Häusler-Assistenten..."):
    verarbeite_frage(user_input)
