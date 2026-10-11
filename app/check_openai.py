"""Check what your OpenAI key can actually do. Run:  python -m app.check_openai
Reads OPENAI_API_KEY (and OPENAI_BASE_URL if set) from .env. Never prints the key."""
from app.config import get_settings


def main() -> None:
    s = get_settings()
    key = s.openai_api_key.get_secret_value()
    if not key:
        raise SystemExit("OPENAI_API_KEY is empty in .env")
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("Run: pip install -r requirements.txt")
    client = OpenAI(api_key=key, base_url=s.openai_base_url or None, timeout=30, max_retries=0)
    print(f"Key found (ends with ...{key[-4:]}) | base URL: {s.openai_base_url or 'api.openai.com'}\n")

    try:
        names = sorted(m.id for m in client.models.list())
        print(f"Models visible to this key: {len(names)}")
        print("  chat-type     :", [n for n in names if n.startswith(("gpt-", "o1", "o3", "o4"))][:12])
        print("  embedding-type:", [n for n in names if "embedding" in n])
    except Exception as exc:
        print(f"Could not list models ({type(exc).__name__}, HTTP {getattr(exc, 'status_code', '?')}). "
              "Some company keys block listing; the two live tests below still tell the truth.")

    print()
    for label, call in [
        (f"CHAT       {s.openai_chat_model}", lambda: client.chat.completions.create(
            model=s.openai_chat_model, messages=[{"role": "user", "content": "Say OK"}], max_tokens=5)),
        ("EMBEDDING  text-embedding-3-small", lambda: client.embeddings.create(
            model="text-embedding-3-small", input="test")),
    ]:
        try:
            call()
            print(f"[ OK ] {label}")
        except Exception as exc:
            print(f"[FAIL] {label} -> {type(exc).__name__}, HTTP {getattr(exc, 'status_code', '?')}")
    print("\n401 = key rejected | 403 or 404 = this key is not allowed that model | 429 = quota or rate limit")


if __name__ == "__main__":
    main()
