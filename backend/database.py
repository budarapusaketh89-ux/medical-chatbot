from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Change password if needed
DATABASE_URL = "mysql+pymysql://root:saketh%40mysql@localhost/mchatbot"

engine = create_engine(
    DATABASE_URL,
    echo=True
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()