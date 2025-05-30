from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel, HttpUrl
import logging
import os
import httpx, bs4, openai
from readability import Document
from dotenv import load_dotenv
from typing import Optional
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def process_url(item_id: str, url: str):
    """
    Download the HTML, extract the text, and create an embedding.
    Then, use OpenAI to generate a title, description, and tags.
    Finally, update the item in the database.
    """
    try:
        html = httpx.get(url, timeout=20).text

        article_html = Document(html).summary()
        text = bs4.BeautifulSoup(article_html, "lxml").get_text(" ", strip=True)

        embedding_response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8192]
        )
        embedding = embedding_response.data[0].embedding

        metadata_response = client.responses.create(
            model="gpt-4o-mini",
            text={"format": {"type": "json_object"}},
            input=[
                {"role": "system", "content": "Eres un asistente que resume artículos y propone etiquetas."},
                {"role": "user", "content": f"Artículo:\n{text[:6000]}\n\nDevuélveme JSON {{title, description, tags(array)}}"}
            ]
        )
        import json
        try:
            meta = json.loads(metadata_response.output_text)
        except Exception as e:
            logging.error(f"Respuesta OpenAI no es JSON válido: {metadata_response.output_text}")
            raise

        supabase.table("items").update({
            "title": meta["title"],
            "description": meta["description"],
            "tags": meta["tags"],
            "embedding": embedding,
        }).eq("id", item_id).execute()
    except Exception as e:
        logging.error(f"Procesando {url}: {e}")

supabase: Client = create_client(
    os.getenv("SB_URL"), 
    os.getenv("SB_SERVICE_ROLE_KEY")
)

app = FastAPI(
    title="friend-fox",
    version="0.0.1",
    description="friend-fox es el mejor compañero del principito.",
)

origins = [
    os.getenv("LITTLE_PRINCE_URL"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IngestRequest(BaseModel):
    url: HttpUrl

@app.post("/ingest")
async def ingest(
    request: IngestRequest,
    authorization: Optional[str] = Header(default=None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    jwt = authorization.split(" ", 1)[1]

    user_response = supabase.auth.get_user(jwt)
    user = user_response.user
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    user_id = user.id
    logging.info(f"Ingesting URL: {request.url} for user {user_id}")

    data = {
        "user_id": user_id,
        "url": str(request.url),
        "title": None,
        "description": None,
        "tags": [],
        "embedding": None,
    }

    insert_response = supabase.table("items").insert(data).execute()
    if not insert_response.data:
        raise HTTPException(status_code=500, detail="Insert failed or returned no data.")
    
    item = insert_response.data[0]
    
    background_tasks.add_task(process_url, item["id"], str(request.url))
    return {"status": "queued", "item": item}

@app.get("/")
async def main():
    return {"message": "The fox said, 'It is only with the heart that one can see rightly; what is essential is invisible to the eye.'"}