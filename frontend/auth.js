const API_BASE = "http://localhost:8000";

async function registerUser(username, email, password) {
    const response = await fetch(`${API_BASE}/register`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            username,
            email,
            password
        })
    });

    return await response.json();
}

async function loginUser(login, password) {
    const response = await fetch(`${API_BASE}/login`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            login,
            password
        })
    });

    return await response.json();
}