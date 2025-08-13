import os
import types

from llm.openai_client import generate_explanation


class DummyResp:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data or {"choices": [{"message": {"content": "สรุปผลทดสอบ"}}]}
        self.text = text

    def json(self):
        return self._json


def test_generate_explanation_success(monkeypatch):
    os.environ['OPENAI_API_KEY'] = 'test'

    def fake_post(url, headers=None, json=None, timeout=30):
        return DummyResp()

    import requests
    monkeypatch.setattr(requests, 'post', fake_post)

    text = generate_explanation([
        {"label": "Pneumonia", "confidence": 0.9, "logit": 2.3}
    ], {"width": 224, "height": 224})
    assert isinstance(text, str)
    assert len(text) > 0


def test_generate_explanation_fallback(monkeypatch):
    os.environ['OPENAI_API_KEY'] = 'test'

    def fake_post(url, headers=None, json=None, timeout=30):
        return DummyResp(status_code=500, text="error")

    import requests
    monkeypatch.setattr(requests, 'post', fake_post)

    text = generate_explanation([
        {"label": "Pneumonia", "confidence": 0.9, "logit": 2.3}
    ], {"width": 224, "height": 224})
    assert isinstance(text, str)
    assert "Pneumonia" in text


