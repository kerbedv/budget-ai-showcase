import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SERVER_DIR = os.path.join(REPO_ROOT, "server")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SERVER_DIR)

# ---- Stubs for firebase_admin ----
if "firebase_admin" not in sys.modules:
    fb = types.ModuleType("firebase_admin")
    fb._apps = []

    class _DummyCred: ...
    credentials = types.SimpleNamespace(
        Certificate=lambda *a, **k: _DummyCred(),
        ApplicationDefault=lambda *a, **k: _DummyCred(),
    )

    def initialize_app(*a, **k):
        fb._apps.append(object())
        return types.SimpleNamespace(project_id=k.get("projectId"))

    def get_app():
        return types.SimpleNamespace(project_id="demo")

    auth = types.SimpleNamespace(
        verify_id_token=lambda token: {
            "uid": "ci-user",
            "aud": "demo",
            "iss": "https://securetoken.google.com/demo",
            "email_verified": True,
        }
    )

    fb.credentials = credentials
    fb.initialize_app = initialize_app
    fb.get_app = get_app
    fb.auth = auth

    sys.modules["firebase_admin"] = fb
    sys.modules["firebase_admin.credentials"] = credentials
    sys.modules["firebase_admin.auth"] = auth

# ---- Stubs for google.cloud.firestore ----
if "google" not in sys.modules:
    google = types.ModuleType("google")
    sys.modules["google"] = google
else:
    google = sys.modules["google"]

cloud = types.ModuleType("google.cloud")
sys.modules["google.cloud"] = cloud

class _DummyFSClient:
    def collection(self, *a, **k): return self
    def document(self, *a, **k): return self
    def where(self, *a, **k): return self
    def order_by(self, *a, **k): return self
    def stream(self, *a, **k): return []  # no docs for CI

firestore = types.SimpleNamespace(Client=lambda: _DummyFSClient())
sys.modules["google.cloud.firestore"] = firestore

# ---- Set demo env so auth & showcase shortcuts are enabled ----
os.environ.setdefault("SAFE_SHOWCASE", "1")
os.environ.setdefault("BYPASS_AUTH", "1")
os.environ.setdefault("ALLOW_CLIENT_HISTORY", "1")
os.environ.setdefault("LOG_LEVEL", "INFO")

# Now import the Flask app
from server.main import app

def test_health_and_schema():
    with app.test_client() as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.get_json().get("status") == "ok"

        r2 = c.get("/schema")
        assert r2.status_code == 200
        assert isinstance(r2.get_json(), dict)
