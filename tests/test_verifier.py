"""Regression tests for answers that were correct but wrongly refused (found with python -m app.evaluate)."""
import json
import uuid

from app.providers import build_messages, parse_and_verify
from app.retrieval import RetrievedChunk


def chunk(text, page=1):
    return RetrievedChunk(uuid.uuid4(), uuid.uuid4(), "book.pdf", page, 0, text, 1.0, 1.0)


def reply(**kw):
    return json.dumps(kw)


def test_quote_returned_with_glued_words_is_still_found_and_cited_as_written_in_the_source():
    """A model in JSON mode cannot output newlines, so a quote crossing a line break comes back glued."""
    passages = [chunk("a mere change of units cannot change the number\nof significant figures.")]
    raw = reply(supported=True, answer="No.", quotes=[{"passage": 1, "text": "cannot change the numberof significant figures"}])
    result = parse_and_verify(raw, passages, "m")
    assert result is not None
    assert result.evidence[0].sentence == "cannot change the number of significant figures"   # real spacing restored


def test_glued_quote_that_does_not_exist_is_still_rejected():
    passages = [chunk("a mere change of units cannot change the number\nof significant figures.")]
    raw = reply(supported=True, answer="No.", quotes=[{"passage": 1, "text": "cannot change the numberof atoms"}])
    assert parse_and_verify(raw, passages, "m") is None


def test_numbers_repeated_from_the_question_are_allowed():
    passages = [chunk("The final result should, therefore, be rounded off to 663.8 g.")]
    raw = reply(supported=True, answer="The sum of 436.32 g and 227.2 g is 663.8 g.",
                quotes=[{"passage": 1, "text": "rounded off to 663.8 g"}])
    assert parse_and_verify(raw, passages, "m") is None                                      # question not given: 436.32 unknown
    assert parse_and_verify(raw, passages, "m", "What is 436.32 g plus 227.2 g?") is not None


def test_a_number_that_is_neither_in_the_quotes_nor_the_question_is_still_rejected():
    passages = [chunk("The final result should, therefore, be rounded off to 663.8 g.")]
    raw = reply(supported=True, answer="The sum is 664 g.", quotes=[{"passage": 1, "text": "rounded off to 663.8 g"}])
    assert parse_and_verify(raw, passages, "m", "What is the sum?") is None


def test_unit_exponents_are_not_treated_as_figures():
    passages = [chunk("Surface area of the cube = 311.3 m2")]
    raw = reply(supported=True, answer="The area is 311.3 m2 (cm-3, s-2).", quotes=[{"passage": 1, "text": "Surface area of the cube = 311.3 m2"}])
    assert parse_and_verify(raw, passages, "m") is not None


def test_passages_are_sent_to_the_model_on_one_line():
    messages = build_messages("q?", [chunk("first line\nsecond line\n\nnew paragraph")], [])
    assert "first line second line new paragraph" in messages[-1]["content"]


def test_bullets_quote_marks_and_dash_styles_do_not_break_a_genuine_quote():
    source = [chunk("In CGS system they were centimetre, gram and second respectively. · In FPS system they were foot, "
                    "pound and second respectively. · In MKS system they were metre, kilogram and second respectively.")]
    quote = "In CGS system they were centimetre, gram and second respectively. In FPS system they were foot, pound and second respectively."
    raw = reply(supported=True, answer="CGS used centimetre, gram, second.", quotes=[{"passage": 1, "text": quote}])
    result = parse_and_verify(raw, source, "m")
    assert result is not None and "·" in result.evidence[0].sentence      # cited as it really appears in the document


def test_exponent_notation_variants_match_but_a_flipped_sign_does_not():
    source = [chunk("The dimensions of force are [M L T–2].")]
    ok = reply(supported=True, answer="Force is [M L T-2].", quotes=[{"passage": 1, "text": "dimensions of force are [M L T^-2]"}])
    assert parse_and_verify(ok, source, "m") is not None
    flipped = reply(supported=True, answer="Force is [M L T2].", quotes=[{"passage": 1, "text": "dimensions of force are [M L T2]"}])
    assert parse_and_verify(flipped, source, "m") is None                  # T2 is not T-2


def test_superscript_digits_and_curly_quotes_match():
    source = [chunk("Surface area of the cube = 6(7.203)2 m2 = 311.3 m2 and the ‘unit’ is metre")]
    raw = reply(supported=True, answer="311.3 m2", quotes=[{"passage": 1, "text": "Surface area of the cube = 6(7.203)² m² = 311.3 m² and the 'unit' is metre"}])
    assert parse_and_verify(raw, source, "m") is not None


def test_decimal_point_cannot_be_dropped_or_moved():
    source = [chunk("The final result should be rounded off to 663.8 g.")]
    wrong = reply(supported=True, answer="6638 g", quotes=[{"passage": 1, "text": "rounded off to 6638 g"}])
    assert parse_and_verify(wrong, source, "m") is None


def test_a_number_from_the_cited_passage_is_allowed_even_if_the_quote_does_not_repeat_it():
    source = [chunk("Round to 4 significant figures. Surface area of the cube = 311.3 m2 and volume = 373.7 m3.")]
    raw = reply(supported=True, answer="311.3 m2 and 373.7 m3, to 4 significant figures.",
                quotes=[{"passage": 1, "text": "Surface area of the cube = 311.3 m2 and volume = 373.7 m3"}])
    assert parse_and_verify(raw, source, "m") is not None


def test_a_number_that_exists_only_in_another_passage_is_rejected():
    source = [chunk("The waiting period is 30 days."), chunk("The claim limit is 5000 rupees.")]
    raw = reply(supported=True, answer="The waiting period is 5000 days.", quotes=[{"passage": 1, "text": "The waiting period is 30 days"}])
    assert parse_and_verify(raw, source, "m") is None


def test_quote_with_an_ellipsis_is_checked_piece_by_piece_in_order():
    source = [chunk("The SI was developed by the BIPM in 1971. It has been revised many times since. "
                    "The latest revision was made by the General Conference in November 2018.")]
    good = reply(supported=True, answer="Revised in November 2018.",
                 quotes=[{"passage": 1, "text": "developed by the BIPM in 1971 ... the General Conference in November 2018."}])
    result = parse_and_verify(good, source, "m")
    assert result is not None and result.evidence[0].sentence == "developed by the BIPM in 1971 ... the General Conference in November 2018."
    wrong_order = reply(supported=True, answer="x 2018", quotes=[{"passage": 1, "text": "the General Conference in November 2018 ... developed by the BIPM in 1971"}])
    assert parse_and_verify(wrong_order, source, "m") is None
    invented = reply(supported=True, answer="x 1971", quotes=[{"passage": 1, "text": "developed by the BIPM in 1971 ... and approved by the United Nations"}])
    assert parse_and_verify(invented, source, "m") is None


def test_closing_punctuation_of_the_source_is_kept_in_the_citation():
    source = [chunk("Cosmetic procedures are excluded. Other text follows.")]
    raw = reply(supported=True, answer="No.", quotes=[{"passage": 1, "text": "Cosmetic procedures are excluded."}])
    assert parse_and_verify(raw, source, "m").evidence[0].sentence == "Cosmetic procedures are excluded."


def consecutive(text, cut, overlap):
    """Two chunks of one document, cut like the real chunker does: the second repeats `overlap` characters."""
    doc = uuid.uuid4()
    a = RetrievedChunk(uuid.uuid4(), doc, "book.pdf", 1, 0, text[:cut], 1.0, 1.0)
    b = RetrievedChunk(uuid.uuid4(), doc, "book.pdf", 1, 1, text[cut - overlap:], 1.0, 1.0)
    return [a, b]


DOC = ("The SI was developed by the BIPM in 1971 and the scheme was recently revised by the General Conference "
       "on Weights and Measures in November 2018. The scheme is now for international usage in science.")


def test_a_quote_running_over_a_chunk_boundary_is_verified_against_the_two_consecutive_chunks():
    cut = DOC.index("Weights and") + len("Weights and")
    chunks = consecutive(DOC, cut, overlap=30)
    quote = "developed by the BIPM in 1971 and the scheme was recently revised by the General Conference on Weights and Measures in November 2018."
    assert not any(quote in c.text for c in chunks)                       # really split across the boundary
    raw = reply(supported=True, answer="Revised in November 2018.", quotes=[{"passage": 1, "text": quote}])
    result = parse_and_verify(raw, chunks, "m")
    assert result is not None and result.evidence[0].chunk is chunks[0]


def test_an_invented_ending_across_a_chunk_boundary_is_still_rejected():
    cut = DOC.index("Weights and") + len("Weights and")
    chunks = consecutive(DOC, cut, overlap=30)
    quote = "developed by the BIPM in 1971 and the scheme was recently revised by the General Conference on Weights and Balances in May 2020."
    raw = reply(supported=True, answer="Revised in 2020.", quotes=[{"passage": 1, "text": quote}])
    assert parse_and_verify(raw, chunks, "m") is None


def test_chunks_that_are_not_consecutive_are_not_joined():
    doc = uuid.uuid4()
    a = RetrievedChunk(uuid.uuid4(), doc, "book.pdf", 1, 0, "The waiting period is thirty days and then", 1.0, 1.0)
    b = RetrievedChunk(uuid.uuid4(), doc, "book.pdf", 1, 5, "claims are paid within ten days.", 1.0, 1.0)
    raw = reply(supported=True, answer="Thirty days then ten.", quotes=[{"passage": 1, "text": "waiting period is thirty days and then claims are paid within ten days"}])
    assert parse_and_verify(raw, [a, b], "m") is None


def test_figures_on_the_same_page_are_allowed_but_not_figures_from_another_page():
    doc = uuid.uuid4()
    cited = RetrievedChunk(uuid.uuid4(), doc, "b.pdf", 9, 1, "Formulas (a), (c) and (e) are ruled out.", 1.0, 1.0)
    same_page = RetrievedChunk(uuid.uuid4(), doc, "b.pdf", 9, 0, "(b) K = (1/2)mv2 and (d) K = (3/16)mv2", 1.0, 1.0)
    other_page = RetrievedChunk(uuid.uuid4(), doc, "b.pdf", 4, 2, "The limit is 99 units.", 1.0, 1.0)
    ok = reply(supported=True, answer="(a), (c), (e) are ruled out; (b) is (1/2)mv2 and (d) is (3/16)mv2.",
               quotes=[{"passage": 1, "text": "Formulas (a), (c) and (e) are ruled out."}])
    assert parse_and_verify(ok, [cited, same_page, other_page], "m") is not None
    bad = reply(supported=True, answer="(a), (c), (e) are ruled out, with a limit of 99.",
                quotes=[{"passage": 1, "text": "Formulas (a), (c) and (e) are ruled out."}])
    assert parse_and_verify(bad, [cited, same_page, other_page], "m") is None


def test_glued_words_in_the_answer_are_repaired_only_when_both_halves_are_source_words():
    source = [chunk("The least precise measurement is 227.2 g, so the final result is abbreviated as SI. "
                    "Units were metre, kilogram and second. The unit for plane angle is radian.")]
    raw = reply(supported=True, answer="So thefinal result, abbreviated asSI, uses metre,kilogram and radians.",
                quotes=[{"passage": 1, "text": "the final result is abbreviated as SI"}])
    result = parse_and_verify(raw, source, "m")
    assert result.text == "So the final result, abbreviated as SI, uses metre, kilogram and radians."   # 'radians' untouched


def test_unknown_words_that_do_not_split_into_source_words_are_left_alone():
    source = [chunk("The waiting period is 30 days.")]
    raw = reply(supported=True, answer="The waiting period is 30 days (Xyzzyfoo).", quotes=[{"passage": 1, "text": "The waiting period is 30 days"}])
    assert parse_and_verify(raw, source, "m").text == "The waiting period is 30 days (Xyzzyfoo)."


def test_difference_between_two_quoted_figures_is_allowed_but_other_arithmetic_is_not():
    passages = [chunk("The insurer will acknowledge every claim within 2 business days."),
                chunk("The insurer will acknowledge every claim within 3 business days.")]
    quotes = [{"passage": 1, "text": "acknowledge every claim within 2 business days"},
              {"passage": 2, "text": "acknowledge every claim within 3 business days"}]
    ok = reply(supported=True, answer="The first is faster by 1 business day.", quotes=quotes)
    assert parse_and_verify(ok, passages, "m") is not None
    total = reply(supported=True, answer="Together they take 5 business days.", quotes=quotes)
    assert parse_and_verify(total, passages, "m") is None
    invented = reply(supported=True, answer="The first is faster by 4 business days.", quotes=quotes)
    assert parse_and_verify(invented, passages, "m") is None
