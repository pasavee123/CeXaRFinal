from __future__ import annotations

import os
import logging
from typing import List, Dict, Iterator

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class TransientHTTPError(Exception):
    pass


def _build_prompt(diagnoses: List[Dict], image_meta: Dict) -> str:
    lines = [
        "บริบทภาพถ่ายทรวงอก (Chest X-ray) สำหรับการทบทวนโดยแพทย์:",
        f"- ขนาดภาพ: {image_meta.get('width')}x{image_meta.get('height')}",
        "- ผลจากโมเดล (เรียงตามความน่าจะเป็นมากไปน้อย):",
    ]
    for d in diagnoses[:5]:
        lines.append(f"  • {d['label']}: {d['confidence']:.2f}")
    lines += [
        "",
        "คำสั่งสำหรับการสรุป (ภาษาไทยเชิงวิชาชีพสำหรับแพทย์):",
        "- เขียนเป็นโน้ตทางรังสีวิทยาแบบกระชับ เป็นกลาง ไม่ใช่การวินิจฉัยขั้นสุดท้าย",
        "- ใช้เฉพาะข้อมูลที่ให้เท่านั้น ห้ามสมมุติข้อมูลใหม่",
        "- โครงสร้าง:",
        "  1) Impression/Key findings",
        "  2) Differential diagnosis (+เหตุผลสั้น)",
        "  3) Supporting evidence (อ้างอิง label+prob จากโมเดล)",
        "  4) Limitations",
        "  5) Suggested next steps (เพื่อการพิจารณา ไม่ใช่คำสั่ง)",
    ]
    return "\n".join(lines)


def _api_headers() -> Dict[str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


@retry(reraise=True,
       stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type(TransientHTTPError))
def _call_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise TransientHTTPError("Missing OPENAI_API_KEY")
    url = f"{OPENAI_BASE_URL}/chat/completions"
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "คุณเป็นผู้ช่วยแพทย์รังสีวิทยา ใช้ภาษาไทยเชิงวิชาชีพ กระชับ เป็นกลาง ไม่ฟันธง "
                    "อิงเฉพาะข้อมูลที่ให้เท่านั้น จัดรูปแบบเป็น: "
                    "Impression, Differential (+เหตุผล), Supporting evidence (label+prob), "
                    "Limitations, Suggested next steps (เพื่อการพิจารณา)"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 500,
    }
    try:
        resp = requests.post(url, headers=_api_headers(), json=payload, timeout=30)
        if resp.status_code >= 500:
            raise TransientHTTPError(f"Server error: {resp.status_code}")
        if resp.status_code != 200:
            raise Exception(f"OpenAI API error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.RequestException as e:
        raise TransientHTTPError(str(e))


def _fallback_text(diagnoses: List[Dict], image_meta: Dict) -> str:
    if not diagnoses:
        return (
            "โน้ตสำหรับแพทย์: ไม่มีผลลัพธ์จากโมเดลในขณะนี้ "
            "โปรดพิจารณาบริบททางคลินิกร่วมและข้อมูลเพิ่มเติม (ไม่ใช่การวินิจฉัย)"
        )
    top = diagnoses[0]
    return (
        "โน้ตเชิงเทคนิค (อัตโนมัติ, ไม่ใช่ข้อวินิจฉัย): "
        f"โมเดลให้ความเป็นไปได้ของ '{top['label']}' ที่ {top['confidence']:.2f}. "
        "โปรดใช้เป็นข้อมูลสนับสนุนและพิจารณาร่วมกับคลินิก/การตรวจเพิ่มเติม."
    )


def generate_explanation(diagnoses: List[Dict], image_meta: Dict) -> str:
    prompt = _build_prompt(diagnoses, image_meta)
    try:
        return _call_openai(prompt)
    except Exception:
        logger.exception("LLM call failed; returning fallback explanation.")
        return _fallback_text(diagnoses, image_meta)


@retry(reraise=True,
       stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type(TransientHTTPError))
def _call_openai_chat(messages: List[Dict[str, str]]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise TransientHTTPError("Missing OPENAI_API_KEY")
    url = f"{OPENAI_BASE_URL}/chat/completions"
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 500,
    }
    try:
        resp = requests.post(url, headers=_api_headers(), json=payload, timeout=30)
        if resp.status_code >= 500:
            raise TransientHTTPError(f"Server error: {resp.status_code}")
        if resp.status_code != 200:
            raise Exception(f"OpenAI API error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.RequestException as e:
        raise TransientHTTPError(str(e))


def _augment_with_context(messages: List[Dict[str, str]], clinical_context: Dict) -> List[Dict[str, str]]:
    top = (clinical_context or {}).get("diagnoses", [])[:5]
    ctx_lines = ["สรุปผลโมเดลล่าสุด (top-5):"] + [f"- {d['label']}: {d['confidence']:.2f}" for d in top]
    augmented = messages.copy()
    if augmented and augmented[-1]["role"] == "user":
        augmented[-1] = {"role": "user", "content": augmented[-1]["content"] + "\n\n" + "\n".join(ctx_lines)}
    return augmented


def _system_msg() -> Dict[str, str]:
    return {
        "role": "system",
        "content": (
            "คุณเป็นผู้ช่วยแพทย์รังสีวิทยา ใช้ภาษาไทยเชิงวิชาชีพ เป็นกลาง ไม่ฟันธง "
            "อิงข้อมูลที่ให้เท่านั้น จัดรูปแบบคำตอบเป็น: "
            "Impression, Differential (+เหตุผล), Supporting evidence (label+prob), "
            "Limitations, Suggested next steps (เพื่อการพิจารณา)"
        ),
    }


def chat_reply(messages: List[Dict[str, str]], clinical_context: Dict) -> str:
    augmented = _augment_with_context(messages, clinical_context)
    try:
        return _call_openai_chat([_system_msg()] + augmented)
    except Exception:
        logger.exception("LLM chat failed; returning fallback.")
        return (
            "โน้ตเชิงเทคนิค (อัตโนมัติ): ใช้ผลโมเดลร่วมกับบริบททางคลินิกและการตรวจเพิ่มเติม "
            "ข้อจำกัด: ไม่มีข้อมูลภาพ/บริบทอื่นในระบบสนทนานี้"
        )


def chat_reply_stream(messages: List[Dict[str, str]], clinical_context: Dict) -> Iterator[str]:
    augmented = _augment_with_context(messages, clinical_context)
    # For OpenAI Chat Completions API, basic streaming uses 'stream': True and Server-Sent Events.
    # We implement a simple token/segment streaming by splitting the final text if the endpoint isn't SSE-enabled in env.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback pseudo-stream
        yield "ระบบไม่ได้กำหนด OPENAI_API_KEY จึงไม่สามารถสตรีมคำตอบได้"
        return
    url = f"{OPENAI_BASE_URL}/chat/completions"
    payload = {
        "model": OPENAI_MODEL,
        "messages": [_system_msg()] + augmented,
        "temperature": 0.2,
        "max_tokens": 500,
        "stream": True,
    }
    try:
        with requests.post(url, headers=_api_headers(), json=payload, timeout=60, stream=True) as resp:
            if resp.status_code >= 500:
                raise TransientHTTPError(f"Server error: {resp.status_code}")
            if resp.status_code != 200:
                raise Exception(f"OpenAI API error {resp.status_code}: {resp.text[:200]}")
            # SSE lines prefixed with 'data: '
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[len("data: "):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        import json as _json
                        obj = _json.loads(data)
                        delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        # If parsing fails, skip that chunk
                        continue
    except requests.RequestException as e:
        raise TransientHTTPError(str(e))