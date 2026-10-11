import pytest

from app.chunking import chunk_pages, split_text
from app.extraction import PageText

TEXT = " ".join(f"word{i}" for i in range(300))


def test_chunks_respect_max_size_and_cover_all_text():
    pieces = split_text(TEXT, 200, 40)
    assert len(pieces) > 1
    assert all(len(p) <= 200 for p in pieces)
    assert pieces[0].startswith("word0") and pieces[-1].endswith("word299")


def test_chunks_overlap_and_keep_whole_words():
    pieces = split_text(TEXT, 200, 40)
    for a, b in zip(pieces, pieces[1:]):
        assert set(a.split()) & set(b.split()), "neighbouring chunks should share words"
    original = set(TEXT.split())
    assert all(set(p.split()) <= original for p in pieces), "no word may be cut in half"


def test_short_text_is_one_chunk_and_empty_is_none():
    assert split_text("hello world", 200, 40) == ["hello world"]
    assert split_text("   \n  ", 200, 40) == []


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        split_text("abc", 100, 100)
    with pytest.raises(ValueError):
        split_text("abc", 0, 0)


def test_page_numbers_preserved_and_indexes_continuous():
    pages = [PageText(1, "alpha " * 100), PageText(2, "beta " * 100)]
    chunks = chunk_pages(pages, 150, 20)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.page_number for c in chunks} == {1, 2}
    assert all("alpha" in c.text for c in chunks if c.page_number == 1)
    assert all("beta" in c.text for c in chunks if c.page_number == 2)


def test_langchain_splitter_ends_chunks_at_sentence_boundaries():
    pytest.importorskip("langchain_text_splitters")
    from app.chunking import split_text_langchain
    text = " ".join(f"This is sentence number {i} of the document." for i in range(40))
    pieces = split_text_langchain(text, 200, 40)
    assert len(pieces) > 1
    assert all(len(p) <= 200 for p in pieces)
    assert all(p.endswith(".") for p in pieces)          # never cut mid-sentence
    assert all(p.startswith("This") or p[0].isalpha() for p in pieces)


def test_chunk_pages_uses_the_chosen_chunker_and_keeps_page_numbers():
    pytest.importorskip("langchain_text_splitters")
    pages = [PageText(1, "One. Two. Three."), PageText(2, "Four. Five.")]
    chunks = chunk_pages(pages, 200, 40, chunker="langchain")
    assert [c.page_number for c in chunks] == [1, 2]


def test_langchain_splitter_does_not_leak_regex_separators_into_text():
    pytest.importorskip("langchain_text_splitters")
    from app.chunking import split_text_langchain
    text = " ".join(f"Sentence {i} ends here." for i in range(60))
    for piece in split_text_langchain(text, 120, 30):
        assert "(?<=" not in piece and "\\s" not in piece
