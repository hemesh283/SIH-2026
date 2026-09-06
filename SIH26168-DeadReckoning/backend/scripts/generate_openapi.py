"""Write the live FastAPI app's OpenAPI spec to backend/openapi.json.

Run from backend/: `python scripts/generate_openapi.py`
Re-run after any schema/endpoint change and commit the updated openapi.json,
so the Android teammate always has a static, importable copy of the current
contract (Postman/Insomnia/OpenAPI-generator) without needing to run the
server themselves.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import app

if __name__ == "__main__":
    out_path = Path(__file__).resolve().parent.parent / "openapi.json"
    spec = app.openapi()
    out_path.write_text(json.dumps(spec, indent=2))
    print(f"wrote {out_path} ({len(spec.get('paths', {}))} paths)")
