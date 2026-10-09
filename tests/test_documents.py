import os
import uuid

PDF = b"%PDF-1.4\n% fake but valid-looking test content\n"
DOCX = b"PK\x03\x04" + b"0" * 50


def upload(client, name, content, ctype="application/octet-stream"):
    return client.post("/documents/upload", files={"file": (name, content, ctype)})


def test_upload_pdf_creates_metadata_and_file(client, tmp_path):
    r = upload(client, "policy.pdf", PDF)
    assert r.status_code == 201
    body = r.json()
    assert body["original_filename"] == "policy.pdf"
    assert body["status"] == "pending"
    assert body["file_size_bytes"] == len(PDF)
    assert "stored_path" not in body  # never leak server paths
    files = os.listdir(tmp_path / "storage")
    assert len(files) == 1 and files[0].endswith(".pdf")
    assert files[0] != "policy.pdf"  # stored under a generated name


def test_upload_docx_ok(client):
    assert upload(client, "notes.docx", DOCX).status_code == 201


def test_unsupported_extension_rejected(client, tmp_path):
    r = upload(client, "notes.txt", b"hello")
    assert r.status_code == 415
    assert not (tmp_path / "storage").exists() or os.listdir(tmp_path / "storage") == []


def test_fake_pdf_rejected_and_cleaned_up(client, tmp_path):
    r = upload(client, "evil.pdf", b"this is plain text, not a pdf")
    assert r.status_code == 400
    assert os.listdir(tmp_path / "storage") == []


def test_empty_file_rejected(client):
    assert upload(client, "empty.pdf", b"").status_code == 400


def test_oversized_file_rejected_and_cleaned_up(client, tmp_path):
    r = upload(client, "big.pdf", PDF + b"x" * (1024 * 1024))  # limit is 1 MB in tests
    assert r.status_code == 413
    assert os.listdir(tmp_path / "storage") == []


def test_path_traversal_filename_is_neutralised(client, tmp_path):
    r = upload(client, "../../evil.pdf", PDF)
    assert r.status_code == 201
    assert r.json()["original_filename"] == "evil.pdf"
    assert not (tmp_path.parent / "evil.pdf").exists()
    assert len(os.listdir(tmp_path / "storage")) == 1


def test_duplicate_upload_returns_existing_record(client, tmp_path):
    first = upload(client, "a.pdf", PDF)
    second = upload(client, "copy_of_a.pdf", PDF)
    assert first.status_code == 201 and second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    assert len(os.listdir(tmp_path / "storage")) == 1


def test_list_get_and_status(client):
    a = upload(client, "a.pdf", PDF).json()
    upload(client, "b.pdf", PDF + b"different")
    listing = client.get("/documents").json()
    assert listing["total"] == 2 and len(listing["items"]) == 2
    assert client.get("/documents", params={"limit": 1}).json()["items"].__len__() == 1
    assert client.get("/documents", params={"status": "completed"}).json()["total"] == 0
    assert client.get(f"/documents/{a['id']}").json()["original_filename"] == "a.pdf"
    st = client.get(f"/documents/{a['id']}/status").json()
    assert st["status"] == "pending" and st["id"] == a["id"]


def test_unknown_and_invalid_ids(client):
    assert client.get(f"/documents/{uuid.uuid4()}").status_code == 404
    assert client.get("/documents/not-a-uuid").status_code == 422
    assert client.get("/documents", params={"status": "bogus"}).status_code == 422
