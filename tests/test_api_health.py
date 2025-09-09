import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from server.main import app

def test_health_and_schema():
    with app.test_client() as c:
        r = c.get("/health")
        assert r.status_code == 200
        body = r.get_json()
        assert body.get("status") == "ok"

        r2 = c.get("/schema")
        assert r2.status_code == 200
        body2 = r2.get_json()
        assert isinstance(body2, dict)
        assert len(body2) >= 1  # expects model schemas map
