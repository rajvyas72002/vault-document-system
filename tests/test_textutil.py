from app.textutil import expects_quantity, has_quantity, split_sentences, stem, tokenize, wants_comparison


def test_stem_makes_related_forms_match():
    assert stem("reported") == stem("reporting") == stem("report")
    assert stem("policies") == "policy"
    assert stem("acknowledgement") == stem("acknowledge")
    assert stem("accidental") == stem("accident")


def test_tokenize_drops_stopwords_and_question_words():
    assert tokenize("What is the return window?") == ["return", "window"]


def test_quantity_detection():
    assert expects_quantity("How soon should it be reported?")
    assert expects_quantity("What is the exact retention period?")
    assert not expects_quantity("Does the policy cover cosmetic surgery?")
    assert has_quantity("within 48 hours") and has_quantity("every quarter")
    assert not has_quantity("Retention periods are defined in a separate schedule.")


def test_sentence_split_and_comparison_cue():
    assert split_sentences("One. Two!\n\nHeading") == ["One.", "Two!", "Heading"]
    assert wants_comparison("Compare the two policies") and not wants_comparison("What is the limit?")
