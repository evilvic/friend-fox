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
import time
import re
import bleach  # sanitiza HTML para reader mode

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def resolve_absolute_url(base_url: str, relative_url: str) -> str:
    """
    Resolve a relative URL to an absolute URL.
    """
    return relative_url if bool(urlparse(relative_url).netloc) else urljoin(base_url, relative_url)

def safe_get(url: str, timeout=60, retries=3) -> Optional[str]:
    headers = {"User-Agent": "Mozilla/5.0 (friend-fox bot) Gecko/2025-05-30"}
    logging.info(f"Headers: {headers}")
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=timeout, headers=headers) as c:
                r = c.get(url)
                if r.status_code == 403:
                    logging.warning(f"403 Forbidden for {url}, skipping HTML download.")
                    return ""
                r.raise_for_status()
                return r.text
        except httpx.RequestError as e:
            logging.warning(f"Network error {url}: {e}, retry {attempt+1}/{retries}")
            time.sleep(2 * (attempt + 1))
        except httpx.HTTPStatusError as e:
            logging.warning(f"Bad status {url}: {e.response.status_code}, retry {attempt+1}/{retries}")
            time.sleep(2 * (attempt + 1))
    return None

def get_cover_url(base_url: str, html: str, item_id: str) -> Optional[str]:
    """
    1. Intenta extraer og:image / twitter:image y comprueba que el recurso sea accesible (HEAD 200).
    2. Si no es válido, toma una captura del primer viewport (1200 × 630) con Playwright
       usando wait_until="networkidle" para que cargue la hero.
    3. Devuelve una URL pública y estable (no firmada) en el bucket `items-assets`.
    """
    soup = bs4.BeautifulSoup(html, "lxml")

    candidate = None
    for attr in ("property", "name"):
        tag = soup.find("meta", {attr: ["og:image", "twitter:image"]})
        if tag and tag.get("content"):
            candidate = resolve_absolute_url(base_url, tag["content"])
            break

    # Verificar que la imagen remota es accesible
    if candidate:
        try:
            r = httpx.head(candidate, timeout=5, follow_redirects=True)
            if r.status_code == 200:
                return candidate
        except httpx.RequestError:
            pass  # cae al screenshot

    # ---------- screenshot fallback ----------
    try:
        from playwright.sync_api import sync_playwright

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = f"{tmp}/{uuid.uuid4()}.jpg"
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page(viewport={"width": 1200, "height": 630})
                page.goto(base_url, timeout=30000, wait_until="networkidle")
                page.screenshot(path=tmp_path, full_page=False, quality=85, type="jpeg")
                browser.close()

            bucket = supabase.storage.from_("items-assets")
            file_key = f"{item_id}/cover.jpg"
            mime = mimetypes.guess_type(tmp_path)[0] or "image/jpeg"

            bucket.upload(
                file_key,
                open(tmp_path, "rb"),
                file_options={
                    "content_type": mime,
                    "upsert": "true"
                }
            )
            # URL pública constante (no caduca)
            public_url = bucket.get_public_url(file_key)
            return public_url
    except Exception as e:
        logging.warning(f"Screenshot cover failed for {base_url}: {e}")

    return None

def capture_full_screenshot(base_url: str, item_id: str) -> Optional[str]:
    """
    Captura un screenshot de toda la página (full_page=True) y retorna la URL pública.
    """
    try:
        from playwright.sync_api import sync_playwright

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = f"{tmp}/{uuid.uuid4()}.jpg"
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page(viewport={"width": 1200, "height": 750})
                page.goto(base_url, timeout=30000, wait_until="networkidle")
                page.screenshot(path=tmp_path, full_page=True, quality=80, type="jpeg")
                browser.close()

            bucket = supabase.storage.from_("items-assets")
            file_key = f"{item_id}/screenshot.jpg"
            mime = "image/jpeg"
            bucket.upload(
                file_key,
                open(tmp_path, "rb"),
                file_options={
                    "content_type": mime,
                    "upsert": "true"
                }
            )
            public_url = bucket.get_public_url(file_key)
            return public_url
    except Exception as e:
        logging.warning(f"Full screenshot failed for {base_url}: {e}")
        return None

def guess_content_type(url: str, html: str, reader_text: str) -> str:
    """
    Heurísticamente clasifica el tipo de contenido.
    Posibles resultados: 'article', 'product', 'book', 'website'
    """
    soup = bs4.BeautifulSoup(html, "lxml")
    og_type = soup.find("meta", {"property": "og:type"}) or soup.find("meta", {"name": "og:type"})
    if og_type and og_type.get("content"):
        t = og_type["content"].lower()
        if "article" in t:
            return "article"
        if "product" in t:
            return "product"
        if "book" in t:
            return "book"
    if re.search(r"/(blog|news|article|posts?)/", url, re.I) or re.search(r"/\d{4}/\d{2}/\d{2}/", url):
        return "article"
    if re.search(r"/product/", url, re.I):
        return "product"
    if re.search(r"/book/", url, re.I):
        return "book"
    if soup.find(attrs={"itemprop": "price"}) or soup.find(text=re.compile(r"\\$\\d")):
        return "product"
    if html:
        density = len(reader_text) / max(len(html), 1)
        if density > 0.35:
            return "article"
    return "website"

def process_url(item_id: str, url: str):
    """
    Download the HTML, extract the text, and create an embedding.
    Then, use OpenAI to generate a title, description, and tags.
    Finally, update the item in the database.
    """
    try:
        html = safe_get(url)
        cover_url = get_cover_url(url, html if html else "", item_id)
        screenshot_url = capture_full_screenshot(url, item_id)
        if not html:
            supabase.table("items").update({
                "cover_url": cover_url,
                "screenshot_url": screenshot_url,
                "type": "website"
            }).eq("id", item_id).execute()
            return
        article_html = Document(html).summary()
        text = bs4.BeautifulSoup(article_html, "lxml").get_text(" ", strip=True)
        content_type = guess_content_type(url, html, text)
        # Solo guardar HTML limpio si es artículo
        reader_html_clean = None
        if content_type == "article":
            reader_html_clean = bleach.clean(
                article_html,
                tags=[
                    "a", "p", "blockquote", "code", "pre", "h1", "h2", "h3", "h4", "h5", "h6",
                    "ul", "ol", "li", "strong", "em", "img", "br"
                ],
                attributes={"a": ["href", "title"], "img": ["src", "alt"]},
                strip=True
            )
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
            "screenshot_url": screenshot_url,
            "type": content_type,
            "reader_html": reader_html_clean,
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
    allow_origins=["*"],
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

@app.get("/search")
async def search(
    query: str,
    limit: int = 20,
    authorization: Optional[str] = Header(default=None),
):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    jwt = authorization.split(" ", 1)[1]

    user_response = supabase.auth.get_user(jwt)
    user = user_response.user
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    user_id = user.id

    embedding_response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    embedding = embedding_response.data[0].embedding

    res = supabase.rpc("search_items", {
        "match_count": limit,
        "q": query,
        "query_embedding": embedding,
        "p_user_id": user_id
    }).execute()
    return res.data

@app.get("/")
async def main():
    return {"message": "The fox said, 'It is only with the heart that one can see rightly; what is essential is invisible to the eye.'"}

@app.get("/article/{item_id}")
async def get_reader(item_id: str, authorization: Optional[str] = Header(default=None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    jwt = authorization.split(" ", 1)[1]
    user = supabase.auth.get_user(jwt).user
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    res = supabase.table("items").select("reader_html")\
        .eq("id", item_id).eq("user_id", user.id).single().execute()
    if res.data and res.data["reader_html"]:
        return {"html": res.data["reader_html"]}
    raise HTTPException(status_code=404, detail="Not found")