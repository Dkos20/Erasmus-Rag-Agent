import os
import json
import re
import pandas as pd
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def run_evaluation():
    if not os.path.exists("evaluation_set.json"):
        raise FileNotFoundError("evaluation_set.json not found")
        
    with open("evaluation_set.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)
        
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    db = None
    if os.path.exists("./vectorstore/db_faiss/index.faiss"):
        db = FAISS.load_local("./vectorstore/db_faiss", embeddings, allow_dangerous_deserialization=True)
        
    litellm_url = os.environ.get("LITELLM_PROXY_URL", "http://litellm:4000")
    client = openai.OpenAI(
        api_key="sk-dummy",
        base_url=f"{litellm_url}/v1"
    )
    
    results = []
    
    for item in eval_set:
        question = item["question"]
        ref_answer = item["reference_answer"]
        expected_entities = item["expected_entities"]
        expected_source = item["expected_source"]
        
        retrieved_text = ""
        hit_at_3 = 0.0
        
        if db is not None:
            docs_with_scores = db.similarity_search_with_score(question, k=4)
            for idx, (doc, score) in enumerate(docs_with_scores):
                source_file = os.path.basename(doc.metadata.get("source", ""))
                if idx < 3 and source_file == expected_source:
                    hit_at_3 = 1.0
                retrieved_text += f"\n---\nSource: {source_file}\nContent: {doc.page_content}\n"
                
        system_instruction = (
            f"You are a local RAG Chatbot assistant for the Erasmus+ program. Answer the user's questions truthfully using only the provided context.\n"
            f"If the information is not present in the context, clearly state that you don't know based on the provided information.\n"
            f"Answer in the same language as the question. If the question is in Croatian, your response must be in Croatian. If the question is in English, your response must be in English. Do not mix languages.\n\n"
            f"Context:\n{retrieved_text}"
        )
        
        chatbot_answer = "No vector database available to retrieve context."
        if db is not None:
            try:
                response = client.chat.completions.create(
                    model="llama3",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": question}
                    ],
                    stream=False
                )
                chatbot_answer = response.choices[0].message.content
            except Exception as e:
                chatbot_answer = f"Error: {str(e)}"
                
        emb_ref = model.encode(ref_answer)
        emb_ans = model.encode(chatbot_answer)
        cos_sim = float(np.dot(emb_ref, emb_ans) / (np.linalg.norm(emb_ref) * np.linalg.norm(emb_ans)))
        
        found_entities = 0
        for entity in expected_entities:
            if entity.lower() in chatbot_answer.lower():
                found_entities += 1
        entity_acc = found_entities / len(expected_entities) if expected_entities else 0.0
        
        score_1_to_5 = 1.0 + 4.0 * (0.5 * cos_sim + 0.3 * entity_acc + 0.2 * hit_at_3)
        score_1_to_5 = round(score_1_to_5, 2)
        
        is_croatian = question.startswith(("Koji", "Koliki", "Od", "Koje", "Što", "Tko"))
        if is_croatian:
            if score_1_to_5 >= 4.5:
                comment = "Izvrstan odgovor. Visoka semantička usklađenost i pokrivenost ključnih entiteta."
            elif score_1_to_5 >= 3.5:
                comment = "Dobar odgovor. Pronađen je relevantan kontekst, uz manje razlike u izražavanju."
            elif score_1_to_5 >= 2.5:
                comment = "Zadovoljavajući odgovor. Kontekst je točan, ali nedostaju ključni entiteti."
            else:
                comment = "Loš odgovor. Niska uspješnost dohvaćanja ili nedostaju ključni entiteti."
        else:
            if score_1_to_5 >= 4.5:
                comment = "Excellent response. High semantic alignment and key entity coverage."
            elif score_1_to_5 >= 3.5:
                comment = "Good response. Relevant context found, minor phrasing or entity omissions."
            elif score_1_to_5 >= 2.5:
                comment = "Fair response. Correct context retrieved, but response lacks key entities."
            else:
                comment = "Poor response. Low retrieval success or missing critical context/entities."
                
        results.append({
            "Question": question,
            "Reference Answer": ref_answer,
            "Chatbot Answer": chatbot_answer,
            "Semantic Cosine Similarity": round(cos_sim, 4),
            "Hit Rate @ 3": hit_at_3,
            "Key Entity Accuracy": round(entity_acc, 4),
            "System Score (1-5)": score_1_to_5,
            "System Comments": comment
        })
        
    df = pd.DataFrame(results)
    os.makedirs("./vectorstore", exist_ok=True)
    df.to_csv("./vectorstore/evaluation_results.csv", index=False)
    return df

if __name__ == "__main__":
    df_results = run_evaluation()
    print(df_results.to_string())
