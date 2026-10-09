import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional

from fastapi import HTTPException

# extension -> (content type we record, magic bytes the file must start with)
ALLOWED = {
    ".pdf": ("application/pdf", b"%PDF-"),
    ".docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        b"PK\x03\x04",  # .docx is a zip archive
    ),
}
CHUNK_SIZE = 1024 * 1024  # read 1 MB at a time so big files never sit fully in memory


@dataclass
class SavedFile:
    original_filename: str
    stored_path: str
    content_type: str
    size_bytes: int
    sha256: str


def clean_filename(name: Optional[str]) -> str:
    """Display-only name. Never used to build a path on disk."""
    name = (name or "").replace("\\", "/").split("/")[-1]  # drop any directory part
    name = "".join(ch for ch in name if ch.isprintable())  # drop control characters
    name = name.strip().lstrip(".")
    return name[:255] or "unnamed"


def save_upload(fileobj: BinaryIO, filename: Optional[str], storage_dir: str, max_bytes: int) -> SavedFile:
    original = clean_filename(filename)
    ext = Path(original).suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(415, f"Unsupported file type '{ext or 'none'}'. Allowed: .pdf, .docx")
    content_type, magic = ALLOWED[ext]

    dest_dir = Path(storage_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{uuid.uuid4().hex}{ext}"  # WE choose the name; the user's name is ignored

    digest = hashlib.sha256()
    size = 0
    try:
        with open(dest, "wb") as out:
            while chunk := fileobj.read(CHUNK_SIZE):
                if size == 0 and not chunk.startswith(magic):
                    raise HTTPException(400, f"File content does not look like a real {ext} file.")
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"File too large. Limit is {max_bytes // (1024 * 1024)} MB.")
                digest.update(chunk)
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "The uploaded file is empty.")
    except BaseException:
        dest.unlink(missing_ok=True)  # never leave half-written files behind
        raise
    return SavedFile(original, str(dest), content_type, size, digest.hexdigest())
