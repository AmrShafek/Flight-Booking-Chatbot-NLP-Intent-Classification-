import pytest
from app.main import app


@pytest.fixture
def client():
    with app.test_client() as c:
        yield c


class TestHome:
    def test_get_home(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"ATIS Intent Classification API" in resp.data


class TestHealth:
    def test_get_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json() == {"status": "healthy"}


class TestPredict:
    def test_missing_text_returns_400(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_empty_text_returns_400(self, client):
        resp = client.post("/predict", json={"text": ""})
        assert resp.status_code == 400

    def test_null_text_returns_400(self, client):
        resp = client.post("/predict", json={"text": None})
        assert resp.status_code == 400

    def test_valid_text_returns_intent_and_confidence(self, client):
        resp = client.post("/predict", json={"text": "flights from boston to denver"})
        assert resp.status_code == 200
        body = resp.get_json()
        assert "intent" in body and "confidence" in body
        assert isinstance(body["intent"], str)
        assert 0.0 <= body["confidence"] <= 1.0

    def test_invalid_json_returns_400(self, client):
        resp = client.post("/predict", data="not json", content_type="application/json")
        assert resp.status_code == 400
