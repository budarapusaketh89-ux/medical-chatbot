// ─────────────────────────────────────────────────────────────
// CONFIG — change this if your FastAPI server runs elsewhere
// ─────────────────────────────────────────────────────────────

const API_BASE = "http://localhost:8000";
const token =
    localStorage.getItem("token");

if(!token){

    window.location.href =
        "login.html";

}
// ─────────────────────────────────────────────────────────────
// DOM REFS
// ─────────────────────────────────────────────────────────────
const chatWindow  = document.getElementById("chat-window");
const input       = document.getElementById("question-input");
const sendBtn     = document.getElementById("send-btn");
const statusDot   = document.getElementById("status-dot");
const statusLabel = document.getElementById("status-label");

let isLoading = false;
let currentConversationId = null;

const conversationList =
    document.getElementById("conversation-list");

const newChatBtn =
    document.querySelector(".new-chat-btn");
// ─────────────────────────────────────────────────────────────
// API HEALTH CHECK — runs once on load, silent after that
// ─────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, {
      signal: AbortSignal.timeout(5000),
    });
    if (res.ok) {
      statusDot.classList.add("online");
      statusLabel.textContent = "Connected";
      document.querySelector(".status-pill").classList.add("online");
    }
  } catch {
    statusDot.classList.remove("online");
    statusLabel.textContent = "Offline";
    document.querySelector(".status-pill").classList.remove("online");
  }
}

checkHealth();

// ─────────────────────────────────────────────────────────────
// AUTO-RESIZE TEXTAREA
// ─────────────────────────────────────────────────────────────
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 120) + "px";
});

input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});

sendBtn.addEventListener("click", handleSend);

// ─────────────────────────────────────────────────────────────
// SAMPLE QUESTION CHIPS
// ─────────────────────────────────────────────────────────────
function sendSample(text) {
  input.value = text;
  input.style.height = "auto";
  handleSend();
}

// Make chips available globally for onclick handlers
window.sendSample = sendSample;

// ─────────────────────────────────────────────────────────────
// MESSAGE HELPERS
// ─────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function removeWelcome() {
  document.getElementById("welcome-msg")?.remove();
}

function appendUserMsg(text) {
  removeWelcome();
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `
    <div class="avatar">Me</div>
    <div class="bubble">${escHtml(text)}</div>
  `;
  chatWindow.appendChild(el);
  scrollToBottom();
}

function appendTyping() {
  removeTyping(); // prevent duplicates
  const el = document.createElement("div");
  el.className = "msg bot";
  el.id = "typing-indicator";
  el.innerHTML = `
    <div class="avatar">Rx</div>
    <div class="bubble">
      <div class="typing">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  chatWindow.appendChild(el);
  scrollToBottom();
}

function removeTyping() {
  document.getElementById("typing-indicator")?.remove();
}

function appendBotMsg(answer) {
  const el = document.createElement("div");
  el.className = "msg bot";

  const formattedAnswer = escHtml(answer).replace(/\n/g, "<br>");

  el.innerHTML = `
    <div class="avatar">Rx</div>
    <div class="bubble">${formattedAnswer}</div>
  `;

  chatWindow.appendChild(el);
  scrollToBottom();
}

function appendErrorMsg(msg) {
  const el = document.createElement("div");
  el.className = "msg bot";
  el.innerHTML = `
    <div class="avatar">Rx</div>
    <div class="bubble error">${escHtml(msg)}</div>
  `;
  chatWindow.appendChild(el);
  scrollToBottom();
}

// ─────────────────────────────────────────────────────────────
// MAIN SEND LOGIC
// ─────────────────────────────────────────────────────────────
async function handleSend() {
  const question = input.value.trim();
  if (!question || isLoading) return;

  isLoading = true;
  sendBtn.disabled = true;
  input.value = "";
  input.style.height = "auto";

  appendUserMsg(question);
  appendTyping();

  try {
const endpoint =
    reportMode
        ? "/report-chat"
        : "/chat";

const res = await fetch(
    `${API_BASE}${endpoint}`,
{
    method: "POST",
    headers: {
        "Content-Type": "application/json",
        "Authorization":
            "Bearer " +
            localStorage.getItem("token")
    },

    body: JSON.stringify(
        reportMode
        ? {
            question
        }
        : {
            question,
            top_k: 3,
            conversation_id:
                currentConversationId
        }
    )
});

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error (${res.status})`);
    }

    const data = await res.json();
    if (!currentConversationId) {
    currentConversationId =
        data.conversation_id;
}
    removeTyping();
    appendBotMsg(data.answer);
    loadConversations();

  } catch (err) {
    removeTyping();
    if (err.name === "TimeoutError") {
      appendErrorMsg("The request timed out. Please check if the backend is running.");
    } else {
      appendErrorMsg(`Something went wrong: ${err.message}`);
    }
  } finally {
    isLoading = false;
    sendBtn.disabled = false;
    input.focus();
  }
}
newChatBtn.addEventListener(
    "click",
    () => {
        reportMode = false;
        currentConversationId = null;

        chatWindow.innerHTML = `
        <div class="welcome" id="welcome-msg">

            <div class="welcome-icon">
                <svg viewBox="0 0 24 24">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10
                    10-4.48 10-10S17.52 2 12 2z"/>
                </svg>
            </div>

            <h2>Ask a medical question</h2>

            <p>
                Get clear, knowledge-base answers on symptoms,
                treatments, causes, and more.
            </p>

            <div class="sample-chips">

                <button class="chip"
                onclick="sendSample('What are the symptoms of Alzheimer\\'s disease?')">
                    Symptoms of Alzheimer's?
                </button>

                <button class="chip"
                onclick="sendSample('How is type 2 diabetes treated?')">
                    Treating type 2 diabetes
                </button>

                <button class="chip"
                onclick="sendSample('What causes hypertension?')">
                    Causes of hypertension
                </button>

                <button class="chip"
                onclick="sendSample('How can I prevent heart disease?')">
                    Preventing heart disease
                </button>

                <button class="chip"
                onclick="sendSample('What is the prognosis for breast cancer?')">
                    Breast cancer prognosis
                </button>

            </div>

        </div>
        `;

    }
);
async function loadConversations() {

    try {

        const res = await fetch(
            `${API_BASE}/conversations`,
            {
                headers: {
                    Authorization:
                        "Bearer " +
                        localStorage.getItem("token")
                }
            }
        );

        const conversations =
            await res.json();
           
        console.log(conversations);    
        conversationList.innerHTML = "";

        conversations.forEach(conv => {

            const item =
                document.createElement("div");

            item.className =
                "conversation-item";

            item.textContent =
                conv.title;

            item.onclick = () => {

                currentConversationId =
                    conv.id;

                loadMessages(
                    conv.id
                );

            };

            conversationList.appendChild(
                item
            );

        });

    }
    catch(err){

        console.log(err);

    }

}
async function loadMessages(
    conversationId
){

    try{

        const res = await fetch(
            `${API_BASE}/messages/${conversationId}`,
            {
                headers:{
                    Authorization:
                        "Bearer " +
                        localStorage.getItem(
                            "token"
                        )
                }
            }
        );

        const messages =
            await res.json();

        chatWindow.innerHTML = "";

        messages.forEach(msg => {

            if(
                msg.sender ===
                "user"
            ){

                appendUserMsg(
                    msg.message
                );

            }

            else{

                appendBotMsg(
                    msg.message
                );

            }

        });

    }

    catch(err){

        console.log(err);

    }

}
// ─────────────────────────────────────────────
// LOAD REPORTS
// ─────────────────────────────────────────────

async function loadReports(){

    try{

        const res = await fetch(

            `${API_BASE}/reports`,

            {

                headers:{

                    Authorization:
                        "Bearer " +
                        localStorage.getItem(
                            "token"
                        )

                }

            }

        );

        const reports =
            await res.json();

        reportList.innerHTML = "";

        reports.forEach(report => {

            const item =
                document.createElement(
                    "div"
                );

            item.className =
                "conversation-item";

            item.textContent =
                report.filename;
                
            
            item.onclick = async () => {

    await fetch(
        `${API_BASE}/report/${report.id}`,
        {
            headers: {
                Authorization:
                    "Bearer " +
                    localStorage.getItem(
                        "token"
                    )
            }
        }
    );
    reportMode = true;
    appendBotMsg(
        `📄 ${report.filename} is now active. Ask questions about this report.`
    );

};
            reportList.appendChild(
                item
            );

        });

    }

    catch(err){

        console.log(err);

    }

}
loadConversations();

loadReports();
function logout() {
    localStorage.removeItem("token");
    window.location.href = "login.html";
}
// ─────────────────────────────────────────────
// REPORT UPLOAD
// ─────────────────────────────────────────────

const uploadBtn =
    document.getElementById(
        "upload-report-btn"
    );

const reportFile =
    document.getElementById(
        "report-file"
    );

const reportList =
    document.getElementById(
        "report-list"
    );

uploadBtn.addEventListener(
    "click",
    () => {

        reportFile.click();

    }
);

reportFile.addEventListener(
    "change",
    async () => {

        if(
            !reportFile.files.length
        ) return;

        const file =
            reportFile.files[0];

        const formData =
            new FormData();

        formData.append(
            "file",
            file
        );

        try {

            const res = await fetch(
                `${API_BASE}/upload-report`,
                {
                    method: "POST",

                    headers: {

                        Authorization:
                            "Bearer " +
                            localStorage.getItem(
                                "token"
                            )

                    },

                    body: formData
                }
            );

            if(!res.ok){

                throw new Error(
                    "Upload failed"
                );

            }

            const data =
                await res.json();

            appendBotMsg(
                `📄 ${file.name} uploaded successfully. You can now ask questions about this report.`
            );
            loadReports();
            const item =
                document.createElement(
                    "div"
                );

            item.className =
                "conversation-item";

            item.textContent =
                file.name;

            reportList.prepend(
                item
            );

        }

        catch(err){

            appendErrorMsg(
                err.message
            );

        }

    }
);