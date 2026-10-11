"""Thumbs up / down on answers, and the satisfaction metrics built from them."""
import uuid
from pathlib import Path

from sqlalchemy import func, select

from app.models import MessageFeedback

POLICIES = Path(__file__).resolve().parent.parent / "data" / "policies"


def ask(client, question="How many annual leave days are provided per year?"):
    """Returns (chat_id, assistant_message_id). Uses the mock provider, so no AI service is called."""
    doc = client.post("/documents/upload", files={"file": ("employee_leave_policy.pdf",
                      (POLICIES / "employee_leave_policy.pdf").read_bytes(), "application/pdf")}).json()["id"]
    client.post(f"/documents/{doc}/process")
    chat = client.post("/chats", json={}).json()["id"]
    answer = client.post(f"/chats/{chat}/messages", json={"question": question}).json()
    return chat, answer["message_id"]


def url(chat, message):
    return f"/chats/{chat}/messages/{message}/feedback"


def test_thumbs_up_is_saved_and_shown_in_the_history(client):
    chat, msg = ask(client)
    res = client.put(url(chat, msg), json={"rating": "up"})
    assert res.status_code == 200 and res.json()["rating"] == "up"
    history = client.get(f"/chats/{chat}/history").json()["messages"]
    assistant = [m for m in history if m["role"] == "assistant"][0]
    user = [m for m in history if m["role"] == "user"][0]
    assert assistant["feedback"] == "up" and user["feedback"] is None


def test_changing_your_mind_replaces_the_rating_instead_of_adding_a_second(client):
    chat, msg = ask(client)
    client.put(url(chat, msg), json={"rating": "up"})
    res = client.put(url(chat, msg), json={"rating": "down", "comment": "  wrong page  "})
    assert res.json() == {"message_id": msg, "rating": "down", "comment": "wrong page"}
    with client.session_factory() as db:
        assert db.scalar(select(func.count()).select_from(MessageFeedback)) == 1


def test_a_rating_can_be_taken_back_and_taking_back_twice_is_harmless(client):
    chat, msg = ask(client)
    client.put(url(chat, msg), json={"rating": "down"})
    assert client.delete(url(chat, msg)).status_code == 204
    assert client.delete(url(chat, msg)).status_code == 204
    history = client.get(f"/chats/{chat}/history").json()["messages"]
    assert all(m["feedback"] is None for m in history)


def test_only_assistant_answers_in_the_right_chat_can_be_rated(client):
    chat, msg = ask(client)
    user_msg = [m for m in client.get(f"/chats/{chat}/history").json()["messages"] if m["role"] == "user"][0]["id"]
    assert client.put(url(chat, user_msg), json={"rating": "up"}).status_code == 400
    assert client.put(url(chat, str(uuid.uuid4())), json={"rating": "up"}).status_code == 404
    assert client.put(url(str(uuid.uuid4()), msg), json={"rating": "up"}).status_code == 404
    other = client.post("/chats", json={}).json()["id"]
    assert client.put(url(other, msg), json={"rating": "up"}).status_code == 404   # a message of another chat


def test_bad_input_is_rejected(client):
    chat, msg = ask(client)
    assert client.put(url(chat, msg), json={"rating": "meh"}).status_code == 422
    assert client.put(url(chat, msg), json={}).status_code == 422
    assert client.put(url(chat, msg), json={"rating": "down", "comment": "x" * 501}).status_code == 422
    blank = client.put(url(chat, msg), json={"rating": "down", "comment": "   "})
    assert blank.status_code == 200 and blank.json()["comment"] is None


def test_feedback_metrics_with_no_ratings_are_zero_not_an_error(client):
    body = client.get("/metrics/feedback").json()
    assert body["rated"] == 0 and body["satisfaction_rate"] is None and body["rated_share"] is None
    assert body["recent_thumbs_down"] == []


def test_satisfaction_rate_and_recent_thumbs_down(client):
    chat, first = ask(client)
    second = client.post(f"/chats/{chat}/messages", json={"question": "What is the sick leave medical certificate rule?"}).json()["message_id"]
    third = client.post(f"/chats/{chat}/messages", json={"question": "Who is the chief executive?"}).json()["message_id"]
    client.put(url(chat, first), json={"rating": "up"})
    client.put(url(chat, second), json={"rating": "up"})
    client.put(url(chat, third), json={"rating": "down", "comment": "should have said not found"})
    body = client.get("/metrics/feedback").json()
    assert (body["answers_total"], body["rated"], body["thumbs_up"], body["thumbs_down"]) == (3, 3, 2, 1)
    assert body["satisfaction_rate"] == round(2 / 3, 4) and body["rated_share"] == 1.0
    bad = body["recent_thumbs_down"][0]
    assert bad["question"] == "Who is the chief executive?" and bad["comment"] == "should have said not found"
    client.delete(url(chat, third))
    assert client.get("/metrics/feedback").json()["thumbs_down"] == 0


# ----------------------------------------------------------------------------- thumbs on document summaries
def processed_doc(client, name="employee_leave_policy.pdf"):
    doc = client.post("/documents/upload", files={"file": (name, (POLICIES / name).read_bytes(), "application/pdf")}).json()["id"]
    client.post(f"/documents/{doc}/process")
    return doc


def test_a_summary_can_be_rated_and_the_rating_comes_back_with_the_insight(client):
    doc = processed_doc(client)
    client.post(f"/documents/{doc}/insights")
    res = client.put(f"/documents/{doc}/insights/feedback", json={"rating": "down", "comment": "  too vague "})
    assert res.status_code == 200 and res.json() == {"document_id": doc, "rating": "down", "comment": "too vague"}
    got = client.get(f"/documents/{doc}/insights").json()
    assert got["feedback"] == "down" and got["feedback_comment"] == "too vague"
    client.put(f"/documents/{doc}/insights/feedback", json={"rating": "up"})      # change of mind replaces
    assert client.get(f"/documents/{doc}/insights").json()["feedback"] == "up"
    assert client.delete(f"/documents/{doc}/insights/feedback").status_code == 204
    assert client.delete(f"/documents/{doc}/insights/feedback").status_code == 204
    assert client.get(f"/documents/{doc}/insights").json()["feedback"] is None


def test_a_summary_cannot_be_rated_before_it_exists(client):
    doc = processed_doc(client)
    assert client.put(f"/documents/{doc}/insights/feedback", json={"rating": "up"}).status_code == 404
    assert client.put(f"/documents/{uuid.uuid4()}/insights/feedback", json={"rating": "up"}).status_code == 404
    client.post(f"/documents/{doc}/insights")
    assert client.put(f"/documents/{doc}/insights/feedback", json={"rating": "meh"}).status_code == 422


def test_regenerating_a_summary_clears_the_old_rating(client):
    doc = processed_doc(client)
    client.post(f"/documents/{doc}/insights")
    client.put(f"/documents/{doc}/insights/feedback", json={"rating": "up"})
    again = client.post(f"/documents/{doc}/insights", json={"length": "short"}).json()
    assert again["feedback"] is None                                              # a new summary has not been rated
    assert client.get("/metrics/feedback").json()["summaries"]["rated"] == 0


def test_summary_ratings_show_in_the_feedback_metrics_separately_from_answers(client):
    first, second = processed_doc(client), processed_doc(client, "health_insurance_policy.pdf")
    for doc in (first, second):
        client.post(f"/documents/{doc}/insights")
    client.put(f"/documents/{first}/insights/feedback", json={"rating": "up"})
    client.put(f"/documents/{second}/insights/feedback", json={"rating": "down"})
    body = client.get("/metrics/feedback").json()
    assert body["summaries"] == {"summaries_total": 2, "rated": 2, "thumbs_up": 1, "thumbs_down": 1, "satisfaction_rate": 0.5}
    assert body["rated"] == 0 and body["satisfaction_rate"] is None               # answers are counted on their own
