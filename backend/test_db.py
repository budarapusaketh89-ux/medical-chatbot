from database import engine

try:
    conn = engine.connect()
    print("✅ Connected to MySQL successfully!")
    conn.close()
except Exception as e:
    print("❌ Connection failed:")
    print(e)