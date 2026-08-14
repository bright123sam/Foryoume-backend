# foryou - FastAPI visual-search backend for Hugging Face Spaces
#
# Endpoints (POST multipart: image=JPEG bytes, mode=face|exact):
#   /scan-face                  face detection / analysis
#   /engines/web                reverse image search (web)
#   /engines/social             reverse image search (social platforms)
#   /engines/video              reverse image search (video platforms)
#   /engines/opensource         match against the local media_library/ folder
#
# All engines reply with {"results": [ {...}, ... ]}; the Android client
# renders any result object that carries title / description / thumbnail /
# image_url / link / engine / confidence. Empty result arrays are shown as
# "no matches", never as an error.
#
# Free, no-API-key behavior:
#   - reverse image search uploads the query to a temporary public host
#     (0x0.st, then tmpfiles.org) and scrapes Bing + Google results.
#   - the "opensource" engine perceptual-hashes the query against the optional
#     media_library/ folder in this Space and returns local matches.
#   - the "face" engine detects faces with OpenCV (no ML download needed).
# If a step fails, it degrades gracefully to an empty result set.

import html
import io
import json
import os
import re
from urllib.parse import quote, unquote, urlparse

import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

try:
    import imagehash
except Exception:  # pragma: no cover - optional dependency
    imagehash = None

APP_TITLE = "foryou visual search backend"

app = FastAPI(title=APP_TITLE, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

LIBRARY_DIR = os.environ.get("FORYOU_LIBRARY_DIR", "media_library")

if os.path.isdir(LIBRARY_DIR):
    app.mount("/library", StaticFiles(directory=LIBRARY_DIR), name="library")

ENGINES = {"face", "web", "social", "video", "opensource"}

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

SOCIAL_HOSTS = {
    "instagram.com", "facebook.com", "fb.watch", "twitter.com", "x.com",
    "tiktok.com", "threads.net", "pinterest.com", "linkedin.com", "reddit.com",
    "snapchat.com", "tumblr.com",
}

VIDEO_HOSTS = {
    "youtube.com", "youtu.be", "vimeo.com", "dailymotion.com", "twitch.tv",
    "tiktok.com", "kick.com", "rumble.com",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# --------------------------------------------------------------------------
# Result helpers
# --------------------------------------------------------------------------

def result_item(title, description, thumbnail, link, engine, confidence):
    item = {
        "title": title,
        "description": description,
        "engine": engine,
        "confidence": float(confidence),
    }
    if thumbnail:
        item["thumbnail"] = thumbnail
        item["image_url"] = thumbnail
    if link:
        item["link"] = link
    return item


def dedupe(items):
    seen = set()
    out = []
    for item in items:
        key = item.get("link") or item.get("title")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def host_of(url):
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def filter_by_hosts(items, hosts):
    out = []
    for item in items:
        h = host_of(item.get("link") or "")
        if any(h == host or h.endswith("." + host) for host in hosts):
            out.append(item)
    return out


# --------------------------------------------------------------------------
# Image processing
# --------------------------------------------------------------------------

def decode_image(data):
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


def detect_faces(data):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    return faces.tolist()


# --------------------------------------------------------------------------
# Local media library matching (free, no external API)
# --------------------------------------------------------------------------

def library_search(data, base_url):
    if imagehash is None or not os.path.isdir(LIBRARY_DIR):
        return []
    img = decode_image(data)
    if img is None:
        return []
    query_hash = imagehash.phash(img)
    results = []
    for name in sorted(os.listdir(LIBRARY_DIR)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
            continue
        path = os.path.join(LIBRARY_DIR, name)
        try:
            candidate_hash = imagehash.phash(Image.open(path).convert("RGB"))
        except Exception:
            continue
        distance = query_hash - candidate_hash
        if distance <= 12:
            score = round(max(0.0, 1.0 - distance / 12.0), 2)
            media_url = base_url + "library/" + quote(name)
            results.append(result_item(
                name, "Matched from your media library.",
                media_url, media_url, "opensource", score))
    return results


# --------------------------------------------------------------------------
# Free reverse image search (no paid API)
# --------------------------------------------------------------------------

def upload_to_public_host(data):
    files = {"file": ("scan.jpg", data, "image/jpeg")}
    for endpoint in ("https://0x0.st", "https://tmpfiles.org/api/v1/upload"):
        try:
            r = requests.post(endpoint, files=files, timeout=30)
            if r.status_code != 200:
                continue
            text = r.text.strip()
            if "tmpfiles.org" in endpoint:
                try:
                    text = r.json().get("data", {}).get("url", "")
                except Exception:
                    text = ""
            if text.startswith("http"):
                return text
        except Exception:
            continue
    return None


def bing_reverse_search(image_url):
    target = ("https://www.bing.com/images/searchbyimage?cbir=sbi&imgurl="
              + quote(image_url, safe=""))
    try:
        r = requests.get(target, headers=HEADERS, timeout=25)
    except Exception:
        return []
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    seen = set()
    for a in soup.select("a.iusc"):
        raw = a.get("m")
        if not raw:
            continue
        try:
            meta = json.loads(html.unescape(raw))
        except Exception:
            continue
        murl = meta.get("murl", "") or ""
        turl = meta.get("turl", "") or ""
        purl = meta.get("purl", "") or ""
        title = (meta.get("t") or "").strip() or (purl or murl)
        key = purl or murl
        if not key or key in seen:
            continue
        seen.add(key)
        item = result_item(title, purl, turl or murl, purl, "web", 0.8)
        if murl and not turl:
            item["thumbnail"] = murl
        results.append(item)
    return results


def google_reverse_search(image_url):
    target = ("https://www.google.com/searchbyimage?site=search&image_url="
              + quote(image_url, safe="") + "&hl=en")
    try:
        r = requests.get(target, headers=HEADERS, timeout=25)
    except Exception:
        return []
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    seen = set()
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if not href.startswith("/url?q="):
            continue
        q = href[len("/url?q="):]
        if "&" in q:
            q = q.split("&", 1)[0]
        try:
            q = unquote(q)
        except Exception:
            q = ""
        if not q.startswith("http") or q in seen:
            continue
        seen.add(q)
        title = a.get_text(" ", strip=True) or urlparse(q).netloc
        results.append(result_item(title, q, "", q, "web", 0.75))
    return results


def reverse_search(data):
    hosted = upload_to_public_host(data)
    if not hosted:
        return []
    return dedupe(bing_reverse_search(hosted) + google_reverse_search(hosted))


# --------------------------------------------------------------------------
# Engine dispatch
# --------------------------------------------------------------------------

def face_results(data):
    try:
        faces = detect_faces(data)
    except Exception:
        faces = []
    if not faces:
        return []
    return [result_item(
        "Face Detected",
        str(len(faces)) + " face(s) detected in the image. Save it into your "
        "media library to enable identity matching.",
        "", "", "face", 1.0)]


def run_engine(engine, data, base_url):
    if engine == "face":
        return face_results(data)
    if engine == "opensource":
        return library_search(data, base_url)
    results = reverse_search(data)
    if engine == "social":
        social = filter_by_hosts(results, SOCIAL_HOSTS)
        if not social and results:
            social = results[:5]
        results = social
    elif engine == "video":
        results = filter_by_hosts(results, VIDEO_HOSTS)
    for item in results:
        item["engine"] = engine
    return results


def read_upload(image):
    data = image.file.read()
    if not data or len(data) < 32:
        raise HTTPException(status_code=400, detail="Empty or invalid image upload.")
    return data


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": APP_TITLE,
        "endpoints": ["/scan-face", "/engines/web", "/engines/social",
                      "/engines/video", "/engines/opensource",
                      "/library", "/health"],
        "note": "POST multipart fields: image (JPEG), mode (face|exact).",
    }


@app.get("/health")
def health():
    return {"status": "ok", "engines": sorted(ENGINES)}


@app.post("/scan-face")
def scan_face(image: UploadFile = File(...), mode: str = Form("exact")):
    data = read_upload(image)
    return {"results": face_results(data)}


@app.post("/engines/{engine}")
def engine_search(engine: str, request: Request,
                  image: UploadFile = File(...), mode: str = Form("exact")):
    if engine not in ENGINES:
        raise HTTPException(status_code=404, detail="Unknown engine: " + engine)
    data = read_upload(image)
    base_url = str(request.base_url)
    return {"results": run_engine(engine, data, base_url)}
  
