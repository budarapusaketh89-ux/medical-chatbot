from database import SessionLocal
from db_models import User
from auth import hash_password

db = SessionLocal()

try:
    existing = db.query(User).filter(
        User.email == "test@gmail.com"
    ).first()

    if existing:
        print("User already exists")
    else:
        user = User(
            username="testuser",
            email="test@gmail.com",
            password_hash=hash_password("test123")
        )

        db.add(user)
        db.commit()

        print("✅ User inserted successfully")

finally:
    db.close()