from sqlalchemy import inspect, text

from app import models  # noqa: F401  (importing registers the tables on Base)
from app.db import Base, engine


def main() -> None:
    with engine.connect() as conn:
        print("Connected to:", conn.execute(text("SELECT version()")).scalar())
    Base.metadata.create_all(engine)
    print("Tables:", inspect(engine).get_table_names())


if __name__ == "__main__":
    main()
