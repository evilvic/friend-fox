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
import mimetypes, tempfile, uuid
from urllib.parse import urljoin, urlparse

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def resolve_absolute_url(base_url: str, relative_url: str) -> str:
    """
    Resolve a relative URL to an absolute URL.
    """
    return relative_url if bool(urlparse(relative_url).netloc) else urljoin(base_url, relative_url)

def get_cover_url(base_url: str, html: str, item_id: str) -> Optional[str]:
    """
    Extract the cover URL from the HTML.
    """
    soup = bs4.BeautifulSoup(html, "lxml")

    for attr in ["property", "name"]:
        tag = soup.find("meta", {attr: ["og:image", "twitter:image"]})
        if tag and tag.get("content"):
            return resolve_absolute_url(base_url, tag["content"])

    with tempfile.TemporaryDirectory() as tmp:
        from playwright.sync_api import sync_playwright
        tmp_path = f"{tmp}/{uuid.uuid4()}.jpg"
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1200, "height": 630})
            page.goto(base_url, timeout=20000)
            page.screenshot(path=tmp_path, full_page=False, quality=85, type="jpeg")
            browser.close()

        bucket = supabase.storage.from_("items-assets")
        file_key = f"{item_id}/cover.jpg"
        bucket.upload(file_key, open(tmp_path, "rb"), upsert=True,
                      content_type=mimetypes.guess_type(tmp_path)[0])

        signed = bucket.create_signed_url(file_key, 60*60*24*7).get("signedUrl")
        return signed

def process_url(item_id: str, url: str):
    """
    Download the HTML, extract the text, and create an embedding.
    Then, use OpenAI to generate a title, description, and tags.
    Finally, update the item in the database.
    """
    try:
        html = httpx.get(url, timeout=20).text

        cover_url = get_cover_url(url, html, item_id)
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
            "cover_url": cover_url,
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