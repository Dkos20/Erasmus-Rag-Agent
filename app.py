import streamlit as st
import os
import openai
from ingest import run_ingestion
from evaluate import run_evaluation

st.set_page_config(
    page_title="Erasmus+ RAG Assistant",
    page_icon="🇪🇺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif;
}

.gradient-text {
    background: linear-gradient(135deg, #4D96FF 0%, #6BCB77 50%, #FFD93D 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.8rem;
    margin-bottom: 0.2rem;
}

.subtitle-text {
    font-size: 1.15rem;
    color: #A5A5A5;
    margin-bottom: 1.5rem;
}

.context-box {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-left: 4px solid #4D96FF;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 10px;
    font-size: 0.9rem;
}

.context-title {
    font-weight: 600;
    color: #4D96FF;
    margin-bottom: 4px;
}

[data-testid="stSidebar"] {
    background-color: #0f1116 !important;
}

.status-badge {
    padding: 8px 12px;
    border-radius: 16px;
    font-weight: 600;
    display: inline-block;
    margin-top: 10px;
}

.status-ready {
    background-color: rgba(107, 203, 119, 0.15);
    color: #6BCB77;
    border: 1px solid rgba(107, 203, 119, 0.3);
}

.status-missing {
    background-color: rgba(255, 107, 107, 0.15);
    color: #FF6B6B;
    border: 1px solid rgba(255, 107, 107, 0.3);
}
</style>
""", unsafe_allow_html=True)

TRANSLATIONS = {
    "HR": {
        "title": "Erasmus+ RAG Pomoćnik",
        "subtitle": "Vaš inteligentni suputnik za pitanja o Erasmus+ programu mobilnosti",
        "lang_label": "Odaberite jezik / Select Language",
        "ingest_header": "Baza Znanja",
        "ingest_btn": "Pokreni vektorizaciju dokumenata",
        "ingest_success": "Uspješno indeksirano! Baza je spremna.",
        "ingest_error": "Greška tijekom indeksiranja! Provjerite datoteke.",
        "ingest_running": "Vektorizacija u tijeku, molimo pričekajte...",
        "no_db_warning": "⚠️ Vektor baza nije učitana. Molimo dodajte dokumente u mapi `./data` i kliknite gumb za vektorizaciju u izborniku.",
        "chat_placeholder": "Upišite svoje pitanje ovdje...",
        "retrieved_context": "Dohvaćeni kontekst iz baze:",
        "clear_chat": "Očisti razgovor",
        "thinking": "Razmišljam...",
        "sidebar_info": "Ovaj chatbot koristi lokalni Llama3 model preko LiteLLM proxyja i LangChain RAG pipelinea.",
        "status_title": "Status Baze Znanja",
        "status_ready": "Spremno za odgovaranje",
        "status_missing": "Potrebna vektorizacija",
        "db_loaded_info": "Vektor baza (FAISS) je uspješno učitana u radnu memoriju.",
        "tab_chat": "💬 Pomoćnik",
        "tab_eval": "📊 Evaluacija sustava",
        "eval_btn": "Pokreni evaluaciju sustava",
        "eval_running": "Evaluacija je u tijeku, molimo pričekajte (ovo može potrajati nekoliko minuta)...",
        "eval_success": "Evaluacija uspješno završena!",
        "eval_desc": "Pokrenite automatsku evaluaciju na 5 testnih pitanja kako biste procijenili točnost pretraživanja i generiranja odgovora.",
        "system_score": "Sustavna ocjena (1-5)",
        "system_comments": "Komentar sustava"
    },
    "EN": {
        "title": "Erasmus+ RAG Assistant",
        "subtitle": "Your intelligent companion for questions about the Erasmus+ mobility program",
        "lang_label": "Odaberite jezik / Select Language",
        "ingest_header": "Knowledge Base",
        "ingest_btn": "Start Document Vectorization",
        "ingest_success": "Successfully indexed! Database is ready.",
        "ingest_error": "Error during indexing! Please verify documents.",
        "ingest_running": "Vectorization in progress, please wait...",
        "no_db_warning": "⚠️ Vector store not found. Please add documents to the `./data` folder and run vectorization in the sidebar.",
        "chat_placeholder": "Ask your question here...",
        "retrieved_context": "Retrieved context from documents:",
        "clear_chat": "Clear chat",
        "thinking": "Thinking...",
        "sidebar_info": "This chatbot runs a local Llama3 model via a LiteLLM proxy and a LangChain RAG pipeline.",
        "status_title": "Database Status",
        "status_ready": "Ready for questions",
        "status_missing": "Needs Indexing",
        "db_loaded_info": "Vector database (FAISS) is loaded into memory.",
        "tab_chat": "💬 Assistant",
        "tab_eval": "📊 System Evaluation",
        "eval_btn": "Run System Evaluation",
        "eval_running": "Evaluation in progress, please wait (this may take a few minutes)...",
        "eval_success": "Evaluation completed successfully!",
        "eval_desc": "Run an automated evaluation on 5 benchmark questions to assess retrieval and generation accuracy.",
        "system_score": "System Score (1-5)",
        "system_comments": "System Comments"
    }
}

if "lang" not in st.session_state:
    st.session_state["lang"] = "HR"

if "eval_results" not in st.session_state:
    st.session_state["eval_results"] = None

with st.sidebar:
    st.markdown("## ⚙️ Postavke / Settings")
    selected_lang = st.selectbox(
        "Language / Jezik",
        options=["HR", "EN"],
        index=0 if st.session_state["lang"] == "HR" else 1,
        label_visibility="collapsed"
    )
    st.session_state["lang"] = selected_lang
    t = TRANSLATIONS[selected_lang]
    
    st.markdown(f"### 📂 {t['ingest_header']}")
    
    db_exists = os.path.exists("./vectorstore/db_faiss/index.faiss")
    
    st.markdown(f"**{t['status_title']}:**")
    if db_exists:
        st.markdown(f'<div class="status-badge status-ready">✓ {t["status_ready"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-badge status-missing">✗ {t["status_missing"]}</div>', unsafe_allow_html=True)
        
    st.write("")
    
    if st.button(t["ingest_btn"], use_container_width=True):
        with st.spinner(t["ingest_running"]):
            try:
                success = run_ingestion()
                if success:
                    st.success(t["ingest_success"])
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error(t["ingest_error"])
            except Exception as e:
                st.error(f"{t['ingest_error']}: {str(e)}")
                
    st.markdown("---")
    st.markdown(f"<small>{t['sidebar_info']}</small>", unsafe_allow_html=True)

st.markdown(f'<div class="gradient-text">{t["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle-text">{t["subtitle"]}</div>', unsafe_allow_html=True)

litellm_url = os.environ.get("LITELLM_PROXY_URL", "http://litellm:4000")
client = openai.OpenAI(
    api_key="sk-dummy",
    base_url=f"{litellm_url}/v1"
)

@st.cache_resource(show_spinner=False)
def load_faiss_db():
    if not os.path.exists("./vectorstore/db_faiss/index.faiss"):
        run_ingestion()
        
    if os.path.exists("./vectorstore/db_faiss/index.faiss"):
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
        return FAISS.load_local("./vectorstore/db_faiss", embeddings, allow_dangerous_deserialization=True)
    return None

db = load_faiss_db()

tab1, tab2 = st.tabs([t["tab_chat"], t["tab_eval"]])

with tab1:
    if db is None:
        st.warning(t["no_db_warning"])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if len(st.session_state.messages) > 0:
        if st.button(t["clear_chat"], type="secondary"):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "context" in message and message["context"]:
                with st.expander(t["retrieved_context"]):
                    for doc, score in message["context"]:
                        source_name = os.path.basename(doc.metadata.get("source", "Document"))
                        st.markdown(f'<div class="context-box"><div class="context-title">{source_name} (Distance: {score:.4f})</div>{doc.page_content}</div>', unsafe_allow_html=True)

    if prompt := st.chat_input(t["chat_placeholder"]):
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            context_docs = []
            retrieved_text = ""
            
            if db is not None:
                docs_with_scores = db.similarity_search_with_score(prompt, k=4)
                for doc, score in docs_with_scores:
                    context_docs.append((doc, score))
                    retrieved_text += f"\n---\nSource: {os.path.basename(doc.metadata.get('source', 'Unknown'))}\nContent: {doc.page_content}\n"
            
            system_instruction = (
                f"You are a local RAG Chatbot assistant for the Erasmus+ program. Answer the user's questions truthfully using only the provided context.\n"
                f"If the information is not present in the context, clearly state that you don't know based on the provided information.\n"
                f"Answer in the selected language: {selected_lang}.\n\n"
                f"Context:\n{retrieved_text}"
            )
            
            if context_docs:
                with st.expander(t["retrieved_context"]):
                    for doc, score in context_docs:
                        source_name = os.path.basename(doc.metadata.get("source", "Document"))
                        st.markdown(f'<div class="context-box"><div class="context-title">{source_name} (Distance: {score:.4f})</div>{doc.page_content}</div>', unsafe_allow_html=True)
            
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                with st.spinner(t["thinking"]):
                    stream = client.chat.completions.create(
                        model="llama3",
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": prompt}
                        ],
                        stream=True
                    )
                    
                    iterator = iter(stream)
                    for chunk in iterator:
                        if chunk.choices and chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            response_placeholder.write(full_response + "▌")
                            break
                
                for chunk in iterator:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        response_placeholder.write(full_response + "▌")
                
                response_placeholder.write(full_response)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "context": context_docs
                })
                
            except Exception as e:
                error_msg = f"Error communicating with model via LiteLLM: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "context": context_docs
                })

with tab2:
    st.markdown(f"### {t['tab_eval']}")
    st.write(t["eval_desc"])
    
    if st.button(t["eval_btn"], use_container_width=True):
        with st.spinner(t["eval_running"]):
            try:
                df_res = run_evaluation()
                st.session_state["eval_results"] = df_res
                st.success(t["eval_success"])
            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")
                
    if st.session_state["eval_results"] is not None:
        df_res = st.session_state["eval_results"]
        eval_df = df_res.copy()
        
        score_col = t["system_score"]
        comment_col = t["system_comments"]
        
        eval_df = eval_df.rename(columns={
            "System Score (1-5)": score_col,
            "System Comments": comment_col
        })
        
        display_cols = [
            "Question",
            "Chatbot Answer",
            "Semantic Cosine Similarity",
            "Hit Rate @ 3",
            "Key Entity Accuracy",
            score_col,
            comment_col
        ]
        
        st.dataframe(eval_df[display_cols], use_container_width=True)
