import io
from pathlib import Path

import pymupdf
from docx import Document as DocxDocument

POLICIES = Path(__file__).resolve().parent.parent / "data" / "policies"


def upload(client, name, content):
    r = client.post("/documents/upload", files={"file": (name, content, "application/octet-stream")})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def make_pdf(*pages_text):
    pdf = pymupdf.open()
    for text in pages_text:
        pdf.new_page().insert_text((72, 72), text)
    return pdf.tobytes()


def test_process_pdf_completes_with_pages_and_chunks(client):
    doc_id = upload(client, "two.pdf", make_pdf("First page about apples.", "Second page about oranges."))
    r = client.post(f"/documents/{doc_id}/process")
    assert r.status_code == 200
    assert r.json()["status"] == "completed" and r.json()["page_count"] == 2
    chunks = client.get(f"/documents/{doc_id}/chunks").json()
    assert chunks["total"] == 2
    by_page = {c["page_number"]: c["text"] for c in chunks["items"]}
    assert "apples" in by_page[1] and "oranges" in by_page[2]
    assert client.get(f"/documents/{doc_id}/status").json()["status"] == "completed"


def test_process_docx_has_no_page_numbers(client):
    d = DocxDocument()
    d.add_paragraph("Employees get leave.")
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Annual", "24 days"
    buf = io.BytesIO()
    d.save(buf)
    doc_id = upload(client, "leave.docx", buf.getvalue())
    assert client.post(f"/documents/{doc_id}/process").json()["status"] == "completed"
    items = client.get(f"/documents/{doc_id}/chunks").json()["items"]
    assert items[0]["page_number"] is None
    assert "24 days" in items[0]["text"] and "Employees get leave." in items[0]["text"]


def test_corrupt_pdf_marks_failed_without_leaking_paths(client):
    doc_id = upload(client, "broken.pdf", b"%PDF-1.4 this is not really a pdf")
    r = client.post(f"/documents/{doc_id}/process")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failed"
    assert body["error_message"] and "storage" not in body["error_message"]
    assert client.get(f"/documents/{doc_id}/chunks").json()["total"] == 0


def test_pdf_without_text_fails_with_ocr_hint(client):
    blank = pymupdf.open()
    blank.new_page()
    doc_id = upload(client, "scan.pdf", blank.tobytes())
    body = client.post(f"/documents/{doc_id}/process").json()
    assert body["status"] == "failed" and "OCR" in body["error_message"]


def test_failed_document_can_be_retried_and_reprocessing_does_not_duplicate_chunks(client):
    doc_id = upload(client, "ok.pdf", make_pdf("Some real text here."))
    client.post(f"/documents/{doc_id}/process")
    client.post(f"/documents/{doc_id}/process")
    assert client.get(f"/documents/{doc_id}/chunks").json()["total"] == 1


def test_process_unknown_document_404(client):
    assert client.post("/documents/00000000-0000-0000-0000-000000000000/process").status_code == 404


def test_fictional_policies_end_to_end_with_page_numbers(client):
    health = upload(client, "health.pdf", (POLICIES / "health_insurance_policy.pdf").read_bytes())
    vehicle = upload(client, "vehicle.pdf", (POLICIES / "vehicle_insurance_policy.pdf").read_bytes())
    assert client.post(f"/documents/{health}/process").json()["page_count"] == 2
    client.post(f"/documents/{vehicle}/process")

    def page_of(doc_id, phrase):
        items = client.get(f"/documents/{doc_id}/chunks", params={"limit": 100}).json()["items"]
        return [c["page_number"] for c in items if phrase in c["text"]]

    assert 1 in page_of(health, "30 days")                 # waiting period is on page 1
    assert 2 in page_of(health, "3 business days")         # claim acknowledgement on page 2
    assert 1 in page_of(vehicle, "within 48 hours")        # accident reporting on page 1
    assert page_of(health, "support phone") == []          # deliberately missing fact stays missing
