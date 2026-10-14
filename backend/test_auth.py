from auth import (
    hash_password,
    verify_password,
    create_access_token
)

password = "saketh123"

hashed = hash_password(password)

print("HASH:", hashed)

print(
    "VERIFY:",
    verify_password(password, hashed)
)

token = create_access_token(
    {"sub": "saketh@gmail.com"}
)

print("TOKEN:", token[:50], "...")