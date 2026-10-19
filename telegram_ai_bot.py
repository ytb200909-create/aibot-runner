#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram AI Controller
----------------------
A Telegram bot that runs on your PC. It lets you command your computer from
your phone through Telegram and use one of several AI backends (local LM Studio
or cloud providers). Only Python's standard library is used.

Commands:
  <text> (optional)     -> replies with help / shortcuts
  /ai <prompt>          -> ask the default AI
  /ai.<provider> ...    -> ask a specific provider, e.g. /ai.openai ...
  /providers            -> list configured providers
  /provider [name]      -> show / set default provider
  /models               -> list models loaded in LM Studio (local only)
  /exec <command>       -> run a command on this PC
  /open <app or URL>    -> open an app / website
  /id                   -> show your Telegram ID
  /help                 -> this help
"""

import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
import unicodedata
import uuid

import pdf_maker


def _norm(t):
    """Strip Vietnamese/Unicode diacritics + lowercase, so 'xoá' == 'xóa'."""
    t = unicodedata.normalize("NFD", t or "")
    return "".join(c for c in t if unicodedata.category(c) != "Mn").lower()

# Force UTF-8 for stdout/stderr regardless of Windows console codepage
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import urllib.error
import urllib.parse
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
POLL_TIMEOUT = 50   # seconds for long polling

_config_mtime = 0
_config = {}

DEFAULT_PROVIDERS = {
    "local": {
        "type": "lmstudio",
        "url": "http://127.0.0.1:1234/v1",
        "model": "",
        "api_key": "",
        "cost_per_1k": 0,
    }
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_cfg():
    global _config, _config_mtime
    try:
        mtime = os.path.getmtime(CONFIG_PATH)
    except OSError:
        return _config
    if mtime != _config_mtime:
        _config = load_config()
        _config_mtime = mtime
    return _config


def provider(name):
    """Return a provider dict with defaults merged in, or None."""
    cfg = get_cfg()
    provs = cfg.get("providers") or DEFAULT_PROVIDERS
    raw = provs.get(name)
    if raw is None:
        return None
    p = dict(raw)
    p.setdefault("type", "lmstudio")
    p.setdefault("url", "http://127.0.0.1:1234/v1")
    p.setdefault("model", "")
    p.setdefault("api_key", "")
    p.setdefault("cost_per_1k", 0)
    return p


def default_provider_name():
    return get_cfg().get("default_provider", "local") or "local"


def provider_list():
    cfg = get_cfg()
    provs = cfg.get("providers") or DEFAULT_PROVIDERS
    return list(provs.keys())


# ---------------------------------------------------------------------------
# Telegram transport (stdlib only)
# ---------------------------------------------------------------------------

def tg_call(method, params=None, token=None):
    if token is None:
        token = get_cfg().get("telegram_bot_token", "")
    url = TELEGRAM_API.format(token=token, method=method)
    data = urllib.parse.urlencode(params).encode("utf-8") if params else None
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=POLL_TIMEOUT + 10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def send_text(chat_id, text, token=None):
    text = (text or "").strip() or "(empty response)"
    if len(text) > 4096:
        text = text[:4000] + "\n...[truncated]"
    try:
        tg_call("sendMessage", {"chat_id": chat_id, "text": text,
                                "parse_mode": "HTML"}, token)
    except Exception:
        try:
            tg_call("sendMessage", {"chat_id": chat_id, "text": text}, token)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# AI backends
# ---------------------------------------------------------------------------

_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
               "AppleWebKit/537.36 (KHTML, like Gecko) "
               "Chrome/120.0.0.0 Safari/537.36")


def _http_json(url, payload, headers, method="POST", timeout=180):
    data = json.dumps(payload).encode("utf-8") if method == "POST" else None
    hdrs = dict(headers or {})
    hdrs.setdefault("User-Agent", _BROWSER_UA)  # avoid Cloudflare 1010 blocks
    req = urllib.request.Request(url, data=data, headers=hdrs,
                                 method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _chat_lmstudio(p, messages, max_tokens):
    model = p.get("model") or pick_local_model()
    if not model:
        return "Không có model local nào được tải. Hãy load model trong LM Studio (lms load ...)."
    base = p.get("url", "http://127.0.0.1:1234/v1").rstrip("/")
    data = _http_json(f"{base}/chat/completions", {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }, {"Content-Type": "application/json"})
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return "Lỗi AI local: " + json.dumps(data)[:500]


def _chat_openai_compat(p, messages, max_tokens):
    base = p.get("url", "https://api.openai.com/v1").rstrip("/")
    headers = {"Content-Type": "application/json"}
    if p.get("api_key"):
        headers["Authorization"] = "Bearer " + p["api_key"]
    data = _http_json(f"{base}/chat/completions", {
        "model": p.get("model") or "gpt-4o-mini",
        "messages": messages,
        "max_tokens": max_tokens,
    }, headers)
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return "Lỗi AI cloud: " + json.dumps(data)[:500]


def ask_vision(p, prompt, image_b64, mime="image/jpeg"):
    """Send an image (base64) + prompt to a vision-capable OpenAI-compatible
    model and return the text reply."""
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {
            "url": f"data:{mime};base64,{image_b64}"}},
    ]}]
    max_tokens = int(get_cfg().get("max_tokens", 400) or 400)
    return _chat_openai_compat(p, messages, max_tokens)


def _chat_anthropic(p, messages, max_tokens):
    # Convert OpenAI-style messages to Anthropic format
    system = "\n".join(m["content"] for m in messages
                       if m["role"] == "system")
    conv = [{"role": "assistant" if m["role"] == "assistant" else "user",
             "content": m["content"]}
            for m in messages if m["role"] != "system"]
    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": p.get("model") or "claude-3-5-haiku-latest",
        "max_tokens": max_tokens or 512,
        "messages": conv,
    }
    if system:
        payload["system"] = system
    data = _http_json(url, payload, {
        "x-api-key": p["api_key"],
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    })
    try:
        return "".join(b.get("text", "") for b in data["content"]).strip()
    except Exception:
        return "Lỗi Anthropic: " + json.dumps(data)[:500]


_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI thân thiện trong app 'AI Controller'."
    "Hãy trò chuyện tự nhiên, gần gũi bằng tiếng Việt như người bạn: "
    "trả lời đúng trọng tâm, ngắn gọn, dễ đọc, có thể dùng chút cảm xúc."
    "Không liệt kê rườm rà. Các lệnh điều khiển (đổi AI, nhớ quy tắc, menu) "
    "do bot xử lý riêng; bạn chỉ cần trò chuyện, giải thích, viết, phân tích.")


def _build_messages(prompt):
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    rules = _load_rules()
    if rules:
        messages.append({"role": "system", "content":
            "Người dùng đặt các quy tắc sau, hãy luôn tuân thủ:\n- " +
            "\n- ".join(rules)})
    messages.append({"role": "user", "content": prompt})
    return messages


def ask_ai(name, prompt, context=None):
    """Ask provider `name`. Returns (reply_text, cost_text)."""
    p = provider(name)
    if not p:
        return f"Provider '{name}' chưa cấu hình. Xem /providers.", ""
    # Safe mode: no local/PC-bound inference, only cloud AI + Bionic info.
    if get_cfg().get("restricted_to_bionic", False) and \
            p.get("type", "lmstudio") in ("lmstudio", "local"):
        return ("🚫 Chế độ an toàn: AI local (chạy trên máy) đã tắt. "
                "Chỉ dùng AI cloud qua bot.", "")
    messages = _build_messages(prompt)
    if context:
        messages.append({"role": "system", "content": context})

    max_tokens = int(get_cfg().get("max_tokens", 400) or 400)
    ptype = p.get("type", "lmstudio").lower()

    if ptype in ("cloudflare_image", "flux_image"):
        return ("🎨 Cloudflare FLUX dùng để <b>tạo ảnh</b>. "
                "Gõ <code>/imagine &lt;mô tả&gt;</code> hoặc nhắn "
                "<code>tạo ảnh &lt;mô tả&gt;</code>.", "")

    if ptype in ("anthropic", "claude"):
        reply = _chat_anthropic(p, messages, max_tokens)
    elif ptype in ("openai", "openai_compat", "groq", "deepseek", "openrouter",
                   "gemini", "cloud"):
        reply = _chat_openai_compat(p, messages, max_tokens)
    else:  # lmstudio / local
        reply = _chat_lmstudio(p, messages, max_tokens)

    # Optional cost estimate only when explicitly enabled in config.
    cost_txt = ""
    if get_cfg().get("show_cost_estimate", False):
        c = float(p.get("cost_per_1k", 0) or 0)
        if c > 0:
            cost_txt = f"\n💰 ~${c * max_tokens / 1000:.4f} (ước lượng)"
    return reply, cost_txt


# ---------------------------------------------------------------------------
# Async AI (background thread so the bot stays responsive while AI answers)
# ---------------------------------------------------------------------------

_ai_semaphore = threading.Semaphore(2)  # max 2 AI requests at once


def _sse_openai(p, messages, max_tokens):
    """Streaming generator for OpenAI-compatible endpoints.
    Yields text deltas as they arrive."""
    base = p.get("url", "https://api.openai.com/v1").rstrip("/")
    headers = {"Content-Type": "application/json"}
    if p.get("api_key"):
        headers["Authorization"] = "Bearer " + p["api_key"]
    headers["User-Agent"] = _BROWSER_UA
    payload = {"model": p.get("model") or "gpt-4o-mini",
               "messages": messages,
               "max_tokens": max_tokens,
               "stream": True}
    req = urllib.request.Request(f"{base}/chat/completions",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                txt = delta.get("content")
                if txt:
                    yield txt


def _multipart(fields, filename, filedata, content_type, file_field="document"):
    """Build a correct multipart/form-data body (raw values, CRLF)."""
    boundary = "----Bionic" + uuid.uuid4().hex
    CRLF = b"\r\n"
    out = bytearray()
    for k, v in fields.items():
        out += b"--" + boundary.encode("utf-8") + CRLF
        out += f'Content-Disposition: form-data; name="{k}"'.encode("utf-8") + CRLF
        out += CRLF
        out += str(v).encode("utf-8") + CRLF
    out += b"--" + boundary.encode("utf-8") + CRLF
    out += (f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"').encode("utf-8") + CRLF
    out += f"Content-Type: {content_type}".encode("utf-8") + CRLF
    out += CRLF
    out += filedata
    out += CRLF
    out += b"--" + boundary.encode("utf-8") + b"--" + CRLF
    return boundary, bytes(out)


def _tg_upload(chat_id, filename, data, content_type, method,
               caption=None, file_field="document"):
    token = get_cfg().get("telegram_bot_token", "")
    fields = {"chat_id": chat_id}
    if caption:
        fields["caption"] = caption
    boundary, body = _multipart(fields, filename, data, content_type,
                                file_field=file_field)
    url = TELEGRAM_API.format(token=token, method=method)
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "User-Agent": _BROWSER_UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def send_document(chat_id, filename, data, caption=None,
                  content_type="application/pdf"):
    return _tg_upload(chat_id, filename, data, content_type, "sendDocument",
                      caption=caption, file_field="document")


def make_pdf_and_send(chat_id, content_request, direct_text=None):
    """Create a PDF (AI-written or direct text) and send it to the chat."""

    def worker():
        _ai_semaphore.acquire()
        try:
            send_text(chat_id, "⏳ Đang tạo PDF...")
            if direct_text is not None:
                text = direct_text
                title = direct_text.strip().splitlines()[0][:60] or "Tài liệu"
            else:
                text, _ = ask_ai(default_provider_name(),
                                 "Viết nội dung ngắn gọn, đủ ý bằng tiếng Việt cho: " + content_request)
                title = content_request[:60] or "Tài liệu"
            data = pdf_maker.build_pdf(text, title)
            send_document(chat_id, "tai-lieu.pdf", data,
                          caption=f"📄 {title}")
            send_text(chat_id, "✅ Đã tạo xong file PDF!")
        except Exception as e:
            send_text(chat_id, f"Lỗi tạo PDF: {e}")
        finally:
            _ai_semaphore.release()

    threading.Thread(target=worker, daemon=True).start()


def cloudflare_generate_image(prompt):
    """Generate an image via Cloudflare Workers AI. Returns image bytes
    (base64-decoded) or raises an exception."""
    cfg = get_cfg().get("image", {}) or {}
    aid = cfg.get("account_id", "").strip()
    token = cfg.get("api_token", "").strip()
    if not aid or not token:
        raise RuntimeError("Chưa cấu hình Cloudflare Workers AI (account_id + api_token).")
    model = cfg.get("model") or "@cf/black-forest-labs/flux-1-schnell"
    url = f"https://api.cloudflare.com/client/v4/accounts/{aid}/ai/run/{model}"
    payload = {"prompt": prompt}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json",
                 "User-Agent": _BROWSER_UA})
    last_err = None
    for attempt in range(3):
        try:
            resp = urllib.request.urlopen(req, timeout=180)
            break
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429 and attempt < 2:   # rate limited -> wait & retry
                time.sleep(10 + attempt * 10)
            else:
                raise
    else:
        raise last_err
    raw = resp.read()
    ctype = resp.headers.get("Content-Type", "") or ""
    if "application/json" in ctype or raw[:1] in (b"{", b"["):
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            data = None
        if data and not data.get("success"):
            raise RuntimeError(json.dumps(data.get("errors", data))[:300])
        img = ((data or {}).get("result") or {}).get("image") or ""
        if isinstance(img, str) and img.startswith("data:"):
            img = img.split(",", 1)[1]
        if img:
            return base64.b64decode(img)
        # JSON but no image field -> fall through to raw
    # Models that return the image bytes directly in the body.
    if raw[:3] == b"\xff\xd8\xff" or raw[:8] == b"\x89PNG\r\n\x1a\n":
        return raw
    raise RuntimeError("Không nhận được ảnh hợp lệ từ Cloudflare.")


def _image_mime(data):
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg", "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png", "png"
    return "image/png", "png"


def send_photo(chat_id, image_bytes, caption=None):
    mime, ext = _image_mime(image_bytes)
    return _tg_upload(chat_id, f"image.{ext}", image_bytes, mime,
                      "sendPhoto", caption=caption, file_field="photo")


def detect_image_instruction(text):
    """Return an image prompt if the message asks to create an image,
    else None."""
    t = _norm(text)
    keys = [_norm(k) for k in ["tạo ảnh", "tao anh", "tạo hình ảnh",
                               "tạo 1 ảnh", "tao 1 anh", "tạo một ảnh",
                               "tao mot anh", "tạo bức ảnh", "ve anh",
                               "imagine", "generate an image",
                               "generate image", "draw"]]
    for k in keys:
        if k in t:
            idx = t.find(k)
            rest = text[idx + len(k):].strip(" ,:.-;")
            return rest or "Một chú mèo dễ thương"
    return None


def enhance_image_prompt(user_desc):
    """Use the chat AI to turn a short (often Vietnamese) description into a
    detailed English image prompt for better accuracy. Falls back to the
    original text on any error."""
    try:
        reply, _ = ask_ai(default_provider_name(),
            "Bạn là chuyên gia viết prompt tạo ảnh. Hãy viết MỘT prompt tiếng Anh "
            "chi tiết, cụ thể (màu sắc, tư thế, bối cảnh, phong cách) để tạo ảnh "
            "ĐÚNG theo ý này. Chỉ in ra prompt, không giải thích:\n\n" + user_desc)
        reply = (reply or "").strip().strip('"\n')
        return reply[:400] or user_desc
    except Exception:
        return user_desc


def make_image_and_send(chat_id, prompt):
    def worker():
        _ai_semaphore.acquire()
        try:
            send_text(chat_id, f"🎨 Đang tạo ảnh: <i>{prompt}</i>...")
            enhanced = enhance_image_prompt(prompt)
            img = cloudflare_generate_image(enhanced)
            send_photo(chat_id, img, caption=f"🖼 {prompt}")
        except Exception as e:
            send_text(chat_id, f"Lỗi tạo ảnh: {e}")
        finally:
            _ai_semaphore.release()
    threading.Thread(target=worker, daemon=True).start()


def detect_public_instruction(text):
    """Understand 'open to friends/everyone' vs 'lock bot' requests.
    Returns 'on', 'off', or None."""
    t = _norm(text)
    on = [_norm(k) for k in ["cho phép bạn bè", "cho phép tất cả",
                             "mở cho mọi người", "cho mọi người dùng",
                             "public mode", "mở chế độ", "mở cho bạn",
                             "cho bạn bè dùng"]]
    off = [_norm(k) for k in ["đóng chế độ", "khóa bot", "chỉ mình tôi",
                              "private mode", "khóa lại", "tắt cho mọi người"]]
    if any(k in t for k in on):
        return "on"
    if any(k in t for k in off):
        return "off"
    return None


def detect_pdf_instruction(text):
    """Return a content request if the message asks to create a PDF,
    else None."""
    t = _norm(text)
    keys = [_norm(k) for k in ["tạo pdf", "tao pdf", "tạo file pdf",
                               "xuất pdf", "xuất ra pdf", "ra pdf",
                               "làm file pdf", "chuyển thành pdf",
                               "make pdf", "create pdf"]]
    for k in keys:
        if k in t:
            # take everything after the matched phrase as the content request
            idx = t.find(k)
            rest = text[idx + len(k):].strip(" ,:.-;")
            return rest or "Viết nội dung cho file PDF"
    return None


def _edit_msg(chat_id, msg_id, text):
    if not msg_id:
        return
    try:
        tg_call("editMessageText", {"chat_id": chat_id,
                                    "message_id": msg_id, "text": text})
    except Exception:
        pass


def reply_ai_async(chat_id, prov, prompt, wait_status=True, context=None):
    """Background thread: ask the AI and show the reply streaming in.
    Returns immediately so the polling loop keeps handling other messages."""

    def worker():
        _ai_semaphore.acquire()
        try:
            # Immediate “typing…” indicator so it never feels idle.
            try:
                tg_call("sendChatAction", {"chat_id": chat_id,
                                            "action": "typing"})
            except Exception:
                pass
            p = provider(prov)
            msgs = _build_messages(prompt)
            if context:
                msgs.append({"role": "system", "content": context})

            msg_id = None
            try:
                m = tg_call("sendMessage", {"chat_id": chat_id, "text": "▍"})
                msg_id = m["result"]["message_id"]
            except Exception:
                msg_id = None

            ptype = (p.get("type", "lmstudio") if p else "").lower()
            full = ""
            last = time.time()
            if ptype in ("openai", "openai_compat", "groq", "deepseek",
                         "openrouter", "gemini", "cloud", "lmstudio",
                         "local") and p:
                max_tokens = int(get_cfg().get("max_tokens", 400) or 400)
                try:
                    for delta in _sse_openai(p, msgs, max_tokens):
                        full += delta
                        if time.time() - last > 0.35 and len(full) >= 3:
                            _edit_msg(chat_id, msg_id, full[:3800])
                            last = time.time()
                    if full.strip():
                        _edit_msg(chat_id, msg_id, full[:4000])
                        return
                except Exception:
                    pass  # fall through to non-streaming

            # Fallback: non-streaming full answer.
            reply, cost = ask_ai(prov, prompt)
            if msg_id:
                _edit_msg(chat_id, msg_id, (reply + cost)[:4000])
            else:
                send_text(chat_id, reply + cost)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:1000]
            send_text(chat_id, f"Lỗi (mã {e.code}):\n{body}")
        except Exception as e:
            send_text(chat_id, f"Lỗi khi gọi AI: {e}")
        finally:
            _ai_semaphore.release()

    threading.Thread(target=worker, daemon=True).start()


def bionic_balance():
    """Read Bionic cloud credit balance from the app's local data files."""
    base = os.path.join(os.path.expanduser("~"), ".lmstudio", "apps",
                        "bionic", ".internal")
    acc_path = os.path.join(base, "cloud-account.json")
    use_path = os.path.join(base, "cloud-account-usage.json")
    out = []
    if not os.path.exists(acc_path):
        return "Không tìm thấy dữ liệu tài khoản Bionic trên máy."
    try:
        with open(acc_path, encoding="utf-8") as f:
            acc = json.load(f).get("payload", {})
        plan = acc.get("plan", {}) or {}
        name = plan.get("name", "Bionic")
        price = float(plan.get("monthlyPriceMicrodollars", 0) or 0) / 1e6
        out.append(f"Gói: {name} (${price:.0f}/tháng)")
        lims = acc.get("limits") or []
        if lims:
            w = lims[0].get("weekly", {}) or {}
            limit = int(w.get("limitMicrocredits", 0) or 0)
            remain = int(w.get("remainingMicrocredits", 0) or 0)
            pct = (remain / limit * 100) if limit else 0
            reset = (w.get("resetsAtIso") or "?")[:10]
            out.append(f"Credits tuần: còn {remain:,} / {limit:,} = {pct:.1f}%")
            out.append(f"Reset: {reset}")
    except Exception as e:
        out.append(f"Lỗi đọc tài khoản: {e}")
    try:
        with open(use_path, encoding="utf-8") as f:
            use = json.load(f).get("payload", {})
        spend = float(use.get("spendMicrodollars", 0) or 0) / 1e6
        out.append(f"30 ngày qua: {use.get('requestCount', 0)} lượt dùng, "
                   f"chi ${spend:.2f}")
    except Exception:
        pass
    return "\n".join(out)


def groq_quota():
    """Report remaining Groq quota (tokens/requests) from API rate-limit
    headers. Returns a display string (or error message)."""
    p = provider("groq")
    if not p or not p.get("api_key"):
        return "Chưa cấu hình Groq (api_key trống)."
    url = p.get("url", "https://api.groq.com/openai/v1").rstrip("/") + "/chat/completions"
    payload = {"model": p.get("model") or "qwen/qwen3.8-27b",
               "messages": [{"role": "user", "content": "hi"}],
               "max_tokens": 1}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + p["api_key"],
                 "User-Agent": _BROWSER_UA})
    try:
        resp = urllib.request.urlopen(req, timeout=40)
        h = resp.headers
        resp.read()
        rows = [
            ("Requests/min", "x-ratelimit-limit-requests",
             "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests"),
            ("Tokens", "x-ratelimit-limit-tokens",
             "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens"),
        ]
        lines = []
        for label, lk, rk, sk in rows:
            if h.get(lk):
                lines.append(f"• {label}: còn {h.get(rk)} / {h.get(lk)} "
                             f"(reset sau {h.get(sk)}s)")
        if not lines:
            return "Không lấy được hạn mức từ Groq (thiếu header rate-limit)."
        return "Hạn mức Groq hiện tại:\n" + "\n".join(lines)
    except urllib.error.HTTPError as e:
        return "Lỗi Groq: " + e.read().decode("utf-8", "replace")[:200]
    except Exception as e:
        return "Lỗi: " + str(e)


def pick_local_model():
    base = get_cfg().get("lmstudio_url", "http://127.0.0.1:1234/v1").rstrip("/")
    try:
        data = _http_json(f"{base}/models", {}, {}, method="GET")
        models = [m["id"] for m in data.get("data", [])]
        if not models:
            return None
        return models[0]["id"]
    except Exception:
        return None


def list_models():
    models = []
    try:
        data = _http_json(
            f"{get_cfg().get('lmstudio_url','http://127.0.0.1:1234/v1').rstrip('/')}/models",
            {}, {}, method="GET")
        models = [m["id"] for m in data.get("data", [])]
    except Exception:
        return "Không kết nối được LM Studio. Hãy bật Developer -> Local Server."
    if not models:
        return "Không có model nào được tải."
    return "Đã tải models:\n" + "\n".join("- " + m for m in models)


# ---------------------------------------------------------------------------
# Local command execution
# ---------------------------------------------------------------------------

def run_shell(command):
    cfg = get_cfg()
    cwd = os.path.expanduser("~")
    use_shell = cfg.get("shell", True)

    def _run(sh):
        return subprocess.run(
            command, shell=sh, capture_output=True, text=True,
            timeout=60, cwd=cwd, errors="replace")

    try:
        try:
            proc = _run(use_shell)
        except (FileNotFoundError, OSError):
            # retry once through cmd.exe (fixes WinError 2 for built-ins)
            proc = _run(True)
        out = (proc.stdout or "") + (proc.stderr or "")
        return (out.strip() or "(no output)") + f"\n[exit code: {proc.returncode}]"
    except subprocess.TimeoutExpired:
        return "(lệnh chạy quá 60 giây, đã hủy)"
    except Exception as e:
        return f"Lỗi thực thi: {e}"


def open_target(target):
    try:
        os.startfile(target.strip())
        return f"Đã mở: {target.strip()}"
    except Exception as e:
        return f"Không mở được: {e}"


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------

def is_allowed(uid):
    cfg = get_cfg()
    if cfg.get("public_mode", False):
        return True
    allowed = cfg.get("allowed_user_ids") or []
    if allowed:
        return uid in allowed
    return uid == cfg.get("telegram_user_id", 0)


def main_menu_markup():
    return {
        "inline_keyboard": [
            [{"text": "💰 Số dư Bionic", "callback_data": "balance"}],
            [{"text": "📊 Hạn mức token (Groq)", "callback_data": "quota"}],
            [{"text": "🤖 Danh sách AI", "callback_data": "providers"}],
            [{"text": "💬 Hỏi AI", "callback_data": "ai"}],
            [{"text": "❓ Trợ giúp / Menu", "callback_data": "help"}],
        ]
    }


def providers_text():
    defn = default_provider_name()
    lines = [f"<b>Các AI đã cấu hình</b> (mặc định: {defn}):"]
    for nm in provider_list():
        p = provider(nm)
        ptype = p.get("type", "lmstudio")
        model = p.get("model") or "(auto)"
        marker = " ⭐" if nm == defn else ""
        lines.append(f"• <code>{nm}</code> ({ptype}) model={model}{marker}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smart AI-provider switching (understand natural-language switch commands)
# ---------------------------------------------------------------------------

PROVIDER_ALIASES = {
    "openai": ["chatgpt", "openai", "gpt"],
    "groq": ["groq"],
    "deepseek": ["deepseek"],
    "anthropic": ["anthropic", "claude"],
    "openrouter": ["openrouter"],
    "gemini": ["gemini"],
    "flux": ["flux"],
    "local": ["local", "lmstudio", "trên máy"],
}

# Words/phrases that suggest the user wants to switch the active AI.
SWITCH_WORDS = [
    "đổi", "chuyển", "switch", "change", "set", "chọn", "đặt",
    "dùng", "sang", "thành", "mặc định", "use", "switch to", "change to",
]


def _load_rules():
    """Return the persisted custom rules (one per line)."""
    rp = os.path.join(BASE_DIR, "custom_rules.txt")
    if not os.path.exists(rp):
        return []
    try:
        with open(rp, encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return []


def _save_rules(rules):
    with open(os.path.join(BASE_DIR, "custom_rules.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(rules) + "\n")


def add_rule(rule):
    rules = _load_rules()
    rules.append(rule.strip())
    _save_rules(rules)


def rules_text():
    rules = _load_rules()
    if not rules:
        return "Chưa có quy tắc nào."
    return "Quy tắc bạn đã đặt:\n" + \
        "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rules))


def clear_rules():
    _save_rules([])
    return "Đã xóa toàn bộ quy tắc."


RULE_TRIGGERS = ["nhớ rằng", "ghi nhớ", "từ giờ", "hãy luôn",
                 "thêm quy tắc", "quy tắc:", "chú ý:", "nhớ giúp",
                 "nhớ là", "xưng hô", "thêm điều", "đừng bao giờ"]


def detect_rule_instruction(text):
    """If the message looks like the user teaching a persistent rule, return
    the rule text. Otherwise return None (ordinary chat)."""
    t = _norm(text)
    for tr in RULE_TRIGGERS:
        idx = t.find(_norm(tr))
        if idx != -1:
            # slice using normalized position mapped back onto original length
            rule = text[idx: idx + len(tr)].strip(" ,:.-;")
            after = text[idx + len(tr):].strip(" ,:.-;")
            rule = after if after else text.strip()
            return rule if rule else text.strip()
    return None


def _persist_config(**updates):
    """Apply updates to config.json on disk."""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg.update(updates)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def detect_button_menu_instruction(text):
    """Understand 'turn off/on the button menu' requests.
    Returns 'off', 'on', or None."""
    t = _norm(text)
    off = [_norm(k) for k in ["xóa menu nút bấm", "tắt menu nút bấm",
                              "gỡ menu nút bấm", "xóa menu", "tắt menu"]]
    on = [_norm(k) for k in ["bật menu nút bấm", "hiện menu nút bấm",
                             "mở menu nút bấm", "bật menu"]]
    if any(k in t for k in off):
        return "off"
    if any(k in t for k in on):
        return "on"
    return None


def detect_forget_instruction(text):
    """Understand 'forget / erase my memory / delete name' requests.
    Returns True if the user wants stored info cleared."""
    t = _norm(text)
    keys = [_norm(k) for k in ["quên tôi", "quên hết", "quên mọi thứ",
                               "quên tên", "quên tên tôi", "quên ký ức",
                               "quên kỉ ức", "xóa hết", "xóa ký ức",
                               "xóa kỉ ức", "xóa thông tin",
                               "xóa dữ liệu của tôi", "xóa tên",
                               "xóa tên tôi", "xóa toàn bộ", "bỏ hết",
                               "delete my memory", "forget"]]
    return any(k in t for k in keys)


def smart_command_result(text):
    """If `text` is a control command (toggle menu / teach rule / switch AI /
    forget / open-to-friends), apply it -> (reply, True). Else (None, False)."""
    pub = detect_public_instruction(text)
    if pub:
        if pub == "on":
            _persist_config(public_mode=True)
            return ("✅ Đã mở chế độ cho bạn bè/mọi người cùng dùng. "
                    "Bạn vẫn là chủ; lệnh nguy hiểm (chạy máy tính) vẫn bị chặn.", True)
        _persist_config(public_mode=False)
        return ("🔒 Đã khóa lại, chỉ mình bạn dùng.", True)
    m = detect_button_menu_instruction(text)
    if m:
        if m == "off":
            _persist_config(show_button_menu=False)
            return ("🗑 Đã xóa menu nút bấm. Bạn vẫn ra lệnh bằng chữ hoặc nút / như thường.", True)
        _persist_config(show_button_menu=True)
        return ("✅ Đã bật lại menu nút bấm. Gõ /menu để xem.", True)
    if detect_forget_instruction(text):
        cleared = len(_load_rules())
        clear_rules()
        return (f"🗑 Đã quên {cleared} ký ức/quy tắc. "
                "Từ giờ mình trả lời như người mới.", True)
    r = detect_rule_instruction(text)
    if m:
        if m == "off":
            _persist_config(show_button_menu=False)
            return ("🗑 Đã xóa menu nút bấm. Bạn vẫn ra lệnh bằng chữ hoặc nút / như thường.", True)
        _persist_config(show_button_menu=True)
        return ("✅ Đã bật lại menu nút bấm. Gõ /menu để xem.", True)
    r = detect_rule_instruction(text)
    if r:
        add_rule(r)
        return (f"🧠 Đã ghi nhớ: <i>{r}</i>\nTừ nay mình sẽ tuân theo trong mọi câu trả lời.", True)
    sw = detect_switch_instruction(text)
    if sw:
        cfg = get_cfg()
        if sw == "flux":
            return ("🎨 FLUX dùng để <b>tạo ảnh</b>. Gõ <code>/imagine &lt;mô tả&gt;</code> "
                    "hoặc nhắn <code>tạo ảnh &lt;mô tả&gt;</code>.", True)
        if sw not in provider_list():
            return (f"Không có AI '{sw}'. Xem /providers", True)
        if cfg.get("restricted_to_bionic", False) and sw == "local":
            return ("🚫 AI local (chạy trên máy) đã tắt.", True)
        if sw == default_provider_name():
            return (f"AI mặc định đã là <code>{sw}</code> rồi.", True)
        _write_default_provider(sw)
        return (f"✅ Đã đổi AI thành <code>{sw}</code>. Nhắn câu hỏi để dùng nhé!", True)
    return None, False


def detect_switch_instruction(text):
    """Try to interpret a natural-language message as an AI-switch request.
    Returns a provider name, or '' if none, or None if it's just a chat."""
    t = _norm(text)
    if not t:
        return None
    norm_switch = [_norm(w) for w in SWITCH_WORDS]
    has_switch_word = any(w in t for w in norm_switch)
    found = None
    for prov, aliases in PROVIDER_ALIASES.items():
        for a in aliases:
            if _norm(a) in t:
                found = prov
                break
        if found:
            break
    if not found:
        return None
    if not has_switch_word:
        return None
    return found


def handle_photo(chat_id, user_id, msg):
    """Handle a photo message: download it and ask a vision-capable AI."""
    if not is_allowed(user_id):
        return
    photos = msg.get("photo") or []
    if not photos:
        return
    fid = photos[-1]["file_id"]
    caption = (msg.get("caption") or "").strip() or \
        "Mô tả ảnh này bằng tiếng Việt, ngắn gọn."

    def worker():
        _ai_semaphore.acquire()
        try:
            try:
                tg_call("sendChatAction", {"chat_id": chat_id,
                                            "action": "typing"})
            except Exception:
                pass
            send_text(chat_id, "⏳ Đang xem ảnh...")
            g = tg_call("getFile", {"file_id": fid})
            fp = g["result"]["file_path"]
            token = get_cfg().get("telegram_bot_token", "")
            url = f"https://api.telegram.org/file/bot{token}/{fp}"
            req = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
            data = urllib.request.urlopen(req, timeout=60).read()
            b64 = base64.b64encode(data).decode("ascii")
            mime = "image/png" if fp.lower().endswith(".png") else "image/jpeg"
            # Prefer Gemini (has vision); fall back to the default provider.
            if "gemini" in provider_list() and provider("gemini").get("api_key"):
                prov = "gemini"
            else:
                prov = default_provider_name()
            p = provider(prov)
            reply = ask_vision(p, caption, b64, mime)
            send_text(chat_id, reply)
        except Exception as e:
            send_text(chat_id, f"Lỗi khi xử lý ảnh: {e}")
        finally:
            _ai_semaphore.release()

    threading.Thread(target=worker, daemon=True).start()


def handle_callback(cq):
    """Handle an inline-button callback query."""
    uid = (cq.get("from") or {}).get("id")
    chat_id = ((cq.get("message") or {}).get("chat") or {}).get("id")
    data = cq.get("data", "")
    if not is_allowed(uid):
        return
    try:
        tg_call("answerCallbackQuery", {"callback_query_id": cq["id"]})
    except Exception:
        pass
    if not chat_id or not data:
        return
    if data == "balance":
        send_text(chat_id, bionic_balance())
    elif data == "quota":
        send_text(chat_id, groq_quota())
    elif data == "providers":
        send_text(chat_id, providers_text())
    elif data == "help":
        send_text(chat_id, help_text())
    elif data == "rules":
        send_text(chat_id, rules_text())
    elif data == "ruleshelp":
        send_text(chat_id,
            "📝 <b>Dạy bot nhớ quy tắc:</b>\n"
            "• Gõ: <code>nhớ rằng &lt;nội dung&gt;</code>\n"
            "• Gõ: <code>từ giờ &lt;nội dung&gt;</code>\n"
            "• Gõ: <code>ghi nhớ &lt;nội dung&gt;</code>\n"
            "• Hoặc /remember &lt;nội dung&gt;\n\n"
            "Quản lý: /rules · /forget &lt;số&gt; · /clearrules")
    elif data == "ai":
        send_text(chat_id, "💬 Gõ câu hỏi của bạn bên dưới, mình sẽ trả lời bằng AI!")
    return (
        "<b>Telegram AI Controller</b>\n\n"
        "• <code>/ai &lt;câu hỏi&gt;</code> -> hỏi AI (mặc định)\n"
        "• <code>/ai.&lt;tên&gt; ...</code> -> đổi AI nhanh (vd /ai.openai)\n"
        "• <code>/providers</code> -> danh sách AI đã cấu hình\n"
        "• <code>/provider</code> -> xem/đổi AI mặc định\n"
        f"• AI hiện có: {provs}\n"
        "• Chat tự nhiên: bot tự hiểu <code>đổi ai thành X</code>, <code>nhớ rằng ...</code>, <code>quên hết</code>\n"
        "• <code>/balance</code> -> số dư credits Bionic còn lại\n"
        "• <code>/quota</code> -> hạn mức token Groq\n"
        "• 🖼️ <code>/imagine &lt;mô tả&gt;</code> hoặc nhắn <code>tạo ảnh ...</code> -> tạo ảnh\n"
        "• 📄 <code>/pdf &lt;nội dung&gt;</code> hoặc nhắn <code>tạo pdf ...</code> -> tạo file PDF\n"
        "• <code>/models</code> -> model local đã tải\n"
        "• <code>/exec / /open</code> -> đang TẮT (chế độ an toàn, chỉ AI + Bionic)\n"
        "• <code>/id</code> -> Telegram ID của bạn\n"
        "• <code>/help</code> -> trợ giúp"
    )


def handle(chat_id, user_id, text):
    cfg = get_cfg()
    text = (text or "").strip()

    # PRIVATE bot: silently ignore anyone not on the allowlist.
    if not is_allowed(user_id):
        return

    cmd = (text.split()[0] if text else "").lower()

    # ---- /menu ----
    if cmd == "/menu":
        if not cfg.get("show_button_menu", True):
            send_text(chat_id,
                "Menu nút bấm đã tắt. Dùng các lệnh (hoặc nút /): "
                "/balance, /quota, /remember, /rules, /ai... (xem /help)")
            return
        tg_call("sendMessage", {"chat_id": chat_id, "text": "📋 Menu:",
                                 "reply_markup": json.dumps(main_menu_markup())})
        return

    # ---- /id ----
    if cmd == "/id":
        send_text(chat_id, f"Telegram user ID: <code>{user_id}</code>")
        return

    # ---- ownership/admin commands ----
    owner_id = cfg.get("telegram_user_id", 0)
    is_owner = user_id == owner_id
    if cmd == "/allow" and is_owner:
        arg = text[len(cmd):].strip()
        try:
            i = int(arg)
        except Exception:
            send_text(chat_id, "Cách dùng: <code>/allow &lt;ID số&gt;</code> (lấy ID của bạn bè qua @userinfobot)")
            return
        ids = set(cfg.get("allowed_user_ids") or [])
        ids.add(i)
        _persist_config(allowed_user_ids=sorted(ids))
        send_text(chat_id, f"✅ Đã cho phép ID <code>{i}</code> dùng bot.")
        return
    if cmd == "/remove" and is_owner:
        arg = text[len(cmd):].strip()
        try:
            i = int(arg)
        except Exception:
            send_text(chat_id, "Cách dùng: <code>/remove &lt;ID số&gt;</code>")
            return
        ids = set(cfg.get("allowed_user_ids") or [])
        ids.discard(i)
        _persist_config(allowed_user_ids=sorted(ids))
        send_text(chat_id, f"Đã bỏ ID <code>{i}</code>.")
        return
    if cmd == "/allowlist":
        ids = cfg.get("allowed_user_ids") or []
        pub = "CÓ" if cfg.get("public_mode", False) else "KHÔNG"
        lst = ", ".join(str(x) for x in ids) if ids else "(trống)"
        send_text(chat_id, f"Chế độ mở cho mọi người: <b>{pub}</b>\nCho phép: {lst}")
        return

    # ---- /help /start ----
    if cmd in ("/help", "/start"):
        send_text(chat_id, help_text())
        if cfg.get("show_button_menu", True):
            try:
                tg_call("sendMessage", {"chat_id": chat_id, "text": "📋 Menu nhanh:",
                                         "reply_markup": json.dumps(main_menu_markup())})
            except Exception:
                pass
        return

    # ---- /models ----
    if cmd == "/models":
        if cfg.get("restricted_to_bionic", False):
            send_text(chat_id, "🚫 Chế độ an toàn: đã tắt xem model local trên máy tính.")
            return
        send_text(chat_id, list_models())
        return

    # ---- /balance (Bionic credits) ----
    if cmd in ("/balance", "/credit", "/usage"):
        send_text(chat_id, bionic_balance())
        return

    # ---- /quota (Groq limit) ----
    if cmd == "/quota":
        send_text(chat_id, groq_quota())
        return

    # ---- /providers ----
    if cmd == "/providers":
        send_text(chat_id, providers_text())
        return

    # ---- /provider [name] ----
    if cmd == "/provider":
        rest = text[len(cmd):].strip()
        if not rest:
            send_text(chat_id,
                      f"Provider mặc định hiện tại: <code>{default_provider_name()}</code>\n"
                      "Đổi bằng: <code>/provider &lt;tên&gt;</code> "
                      f"({', '.join(provider_list())})")
            return
        if rest in provider_list():
            cfg["default_provider"] = rest
            _write_default_provider(rest)
            send_text(chat_id, f"Đã đổi AI mặc định thành <code>{rest}</code>")
        else:
            send_text(chat_id, f"Không có provider '{rest}'. Xem /providers")
        return

    # ---- /ai or /ai.<provider> ----
    if cmd == "/ai" or cmd.startswith("/ai."):
        prov = "local"
        prompt = text[3:].strip()
        if cmd.startswith("/ai."):
            pname = cmd[4:]
            if pname not in provider_list():
                send_text(chat_id, f"Không có provider '{pname}'. Xem /providers")
                return
            prov = pname
            prompt = text[len(cmd):].strip()
        if not prompt:
            send_text(chat_id,
                      f"Cách dùng: <code>/ai &lt;câu hỏi&gt;</code> "
                      f"hoặc <code>/ai.{prov} &lt;câu hỏi&gt;</code>")
            return
        reply_ai_async(chat_id, prov, prompt)
        return

    # ---- /exec ----
    if cmd == "/exec":
        if cfg.get("restricted_to_bionic", False) or not cfg.get("allow_shell", False):
            send_text(chat_id, "🚫 Chế độ an toàn: bot chỉ được dùng AI + Bionic, không được chạy lệnh trên máy tính.")
            return
        command = text[5:].strip()
        if not command:
            send_text(chat_id, "Cách dùng: <code>/exec &lt;lệnh&gt;</code>")
            return
        send_text(chat_id, "⚙️ Đang chạy...")
        send_text(chat_id, run_shell(command))
        return

    # ---- /open ----
    if cmd == "/open":
        if cfg.get("restricted_to_bionic", False) or not cfg.get("allow_open", False):
            send_text(chat_id, "🚫 Chế độ an toàn: bot chỉ được dùng AI + Bionic, không được mở ứng dụng trên máy tính.")
            return
        target = text[5:].strip()
        if not target:
            send_text(chat_id, "Cách dùng: <code>/open notepad</code> hoặc <code>/open https://...</code>")
            return
        send_text(chat_id, open_target(target))
        return

    # ---- /remember /rules /clearrules /forget ----
    if cmd == "/remember":
        rule = text[len(cmd):].strip()
        if not rule:
            send_text(chat_id, "Cách dùng: <code>/remember &lt;quy tắc&gt;</code>, vd: <code>/remember hãy gọi tôi là sếp</code>")
            return
        add_rule(rule)
        send_text(chat_id, f"✅ Đã ghi nhớ: <i>{rule}</i>\nNó sẽ áp dụng cho mọi câu trả lời.")
        return
    # ---- /imagine /image /vẽ : create an image ----
    if cmd in ("/imagine", "/image", "/vẽ", "/ve"):
        prompt = text[len(cmd):].strip() or "Một chú mèo dễ thương"
        make_image_and_send(chat_id, prompt)
        return

    # ---- /pdf <nội dung> : send a PDF directly from the text ----
    if cmd == "/pdf":
        content = text[len(cmd):].strip()
        if not content:
            send_text(chat_id, "Cách dùng: <code>/pdf &lt;nội dung&gt;</code>")
            return
        make_pdf_and_send(chat_id, content, direct_text=content)
        return

    # ---- plain message ----
    # Natural-language image creation ("tạo ảnh ...")
    img_req = detect_image_instruction(text)
    if img_req:
        make_image_and_send(chat_id, img_req)
        return

    # Natural-language PDF creation ("tạo pdf ...")
    pdf_req = detect_pdf_instruction(text)
    if pdf_req:
        make_pdf_and_send(chat_id, pdf_req)
        return

    reply, handled = smart_command_result(text)
    if handled:
        send_text(chat_id, reply)
        return

    if cfg.get("auto_ai_on_message", False) and cfg.get("allow_ai", True):
        reply_ai_async(chat_id, default_provider_name(), text)
    else:
        send_text(chat_id,
                  "Dùng <code>/ai &lt;câu hỏi&gt;</code> để hỏi AI "
                  "(đỡ tốn token).\nGõ <code>/help</code> để xem các lệnh.")


def _write_default_provider(name):
    """Persist default provider choice back into config.json."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["default_provider"] = name
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    cfg = get_cfg()
    token = cfg.get("telegram_bot_token", "")
    if not token or token.startswith("Paste"):
        print("Chưa có bot token! Mở config.json và dán token từ @BotFather.")
        sys.exit(1)

    # Publish the bot command menu (the "/" button beside the chat input).
    commands = [
        {"command": "ai", "description": "Hỏi AI"},
        {"command": "balance", "description": "Số dư credits Bionic"},
        {"command": "quota", "description": "Hạn mức token"},
        {"command": "providers", "description": "Danh sách AI"},
        {"command": "provider", "description": "Đổi AI mặc định"},
        {"command": "imagine", "description": "Tạo ảnh (vd imagine con mèo)"},
        {"command": "pdf", "description": "Tạo file PDF từ nội dung"},
        {"command": "help", "description": "Trợ giúp"},
        {"command": "menu", "description": "Mở menu nút bấm"},
    ]
    try:
        tg_call("setMyCommands", {"commands": json.dumps(commands)})
        print("Đã cài menu lệnh bên cạnh ô chat.", flush=True)
    except Exception as e:
        print("Không cài được menu lệnh:", e, flush=True)

    print("Telegram AI Controller đang chạy. Bot: ", token.split(":")[0])
    print("Nhắn tin cho bot trên điện thoại để bắt đầu. (Ctrl+C để dừng)")
    offset = 0

    while True:
        try:
            res = tg_call("getUpdates", {
                "offset": offset,
                "timeout": POLL_TIMEOUT,
                "allowed_updates": '["message"]',
            })
        except urllib.error.URLError as e:
            print("Lỗi mạng (kiểm tra kết nối Internet):", e)
            time.sleep(5)
            continue
        except Exception as e:
            print("Lỗi Telegram:", e)
            time.sleep(3)
            continue

        for update in res.get("result", []):
            offset = max(offset, update["update_id"] + 1)
            cq = update.get("callback_query")
            if cq:
                try:
                    handle_callback(cq)
                except Exception as e:
                    print("Lỗi callback:", e, flush=True)
                continue
            msg = update.get("message") or {}
            chat_id = msg.get("chat", {}).get("id")
            user_id = msg.get("from", {}).get("id")
            text = msg.get("text")
            chat_type = msg.get("chat", {}).get("type", "private")
            # Private mode: only accept 1-on-1 private chats on the allowlist.
            if chat_type != "private":
                continue
            if not chat_id:
                continue
            # Photo message -> vision AI.
            if msg.get("photo"):
                handle_photo(chat_id, user_id, msg)
                continue
            if not text:
                continue
            print(f"[MSG] user_id={user_id} chat_id={chat_id}: {text}", flush=True)
            if is_allowed(user_id):
                print(f"  -> nguoi dung {user_id} DUOC PHEP", flush=True)
            try:
                handle(chat_id, user_id, text)
            except Exception as e:
                try:
                    send_text(chat_id, f"Đã xảy ra lỗi: {e}")
                except Exception:
                    pass

        time.sleep(0.5)


if __name__ == "__main__":
    main()
