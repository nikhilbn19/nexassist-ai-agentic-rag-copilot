const chat = document.getElementById("chat");
const form = document.getElementById("chatForm");
const question = document.getElementById("question");
const trace = document.getElementById("trace");
const sourceUsed = document.getElementById("sourceUsed");

// Configure marked.js for GitHub Flavored Markdown
if (typeof marked !== "undefined") {
  marked.setOptions({
    gfm: true,
    breaks: true,
  });
}

function toSafeString(value) {
  if (typeof value === "string") {
    return value;
  }

  if (value === null || value === undefined) {
    return "";
  }

  if (typeof value === "object") {
    try {
      return JSON.stringify(value, null, 2);
    } catch {
      return String(value);
    }
  }

  return String(value);
}

function escapeHtml(value = "") {
  const s = toSafeString(value);

  return s.replace(
    /[&<>'"]/g,
    (c) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "'": "&#39;",
        '"': "&quot;",
      })[c],
  );
}

function renderMarkdown(value = "") {
  const s = toSafeString(value);
  if (!s.trim()) return "";

  if (typeof marked !== "undefined" && typeof marked.parse === "function") {
    try {
      const rawHtml = marked.parse(s);
      if (typeof DOMPurify !== "undefined" && typeof DOMPurify.sanitize === "function") {
        return DOMPurify.sanitize(rawHtml);
      }
      return rawHtml;
    } catch (err) {
      console.warn("Markdown parsing fallback:", err);
    }
  }

  return escapeHtml(s).replace(/\n/g, "<br>");
}

function formatUserText(value = "") {
  const s = toSafeString(value);
  return escapeHtml(s).replace(/\n/g, "<br>");
}

function addMessage(role, text, source = "", citations = []) {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;

  const safeText = toSafeString(text);
  const safeSource = toSafeString(source);
  const isUser = role === "user";
  const bodyHtml = isUser ? formatUserText(safeText) : renderMarkdown(safeText);

  const citationList = Array.isArray(citations) ? citations : [];

  const citeHtml = citationList.length
    ? `<div class="citations">
         <div class="citations-header">
           <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>
           <span>Sources & References</span>
         </div>
         <div class="citations-list">
           ${citationList
             .map((c) => {
               if (typeof c === "string") {
                 return `<span class="citation-badge">${escapeHtml(c)}</span>`;
               }
               if (!c || typeof c !== "object") {
                 return `<span class="citation-badge">${escapeHtml(String(c))}</span>`;
               }
               const url = c.url;
               const title = c.title || c.url || "Document";
               const type = c.type === "private_kb" ? "Private KB" : "Web";
               const typeClass = c.type === "private_kb" ? "kb-type" : "web-type";

               return url
                 ? `<a class="citation-badge with-link" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">
                      <span class="citation-type ${typeClass}">${escapeHtml(type)}</span>
                      <span class="citation-title">${escapeHtml(title)}</span>
                      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
                    </a>`
                 : `<span class="citation-badge">
                      <span class="citation-type ${typeClass}">${escapeHtml(type)}</span>
                      <span class="citation-title">${escapeHtml(title)}</span>
                    </span>`;
             })
             .join("")}
         </div>
       </div>`
    : "";

  wrap.innerHTML = `
        <div class="avatar">${role === "user" ? "You" : "AI"}</div>
        <div class="bubble">
            <div class="markdown-body">
              ${bodyHtml}
            </div>
            ${
              safeSource
                ? `<div class="answer-source">
                       <span class="source-dot"></span>
                       <span>Source: ${escapeHtml(safeSource)}</span>
                   </div>`
                : ""
            }
            ${citeHtml}
        </div>
    `;

  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
}

function renderTrace(items = []) {
  const traceItems = Array.isArray(items) ? items : [items];

  trace.innerHTML = traceItems.length
    ? traceItems
        .map(
          (x) => `
                <div class="trace-item">
                    ${escapeHtml(x)}
                </div>
            `,
        )
        .join("")
    : '<div class="empty">No trace.</div>';
}

async function askAgent(q) {
  addMessage("user", q);

  question.value = "";

  renderTrace(["Running LangGraph workflow..."]);

  sourceUsed.textContent = "Running";

  const btn = form.querySelector("button");
  btn.disabled = true;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: q,
      }),
    });

    const data = await res.json();

    console.log("Full API response:", data);
    console.log("Answer:", data.answer);
    console.log("Answer type:", typeof data.answer);

    if (!res.ok) {
      throw new Error(toSafeString(data.detail) || "Request failed");
    }

    const answer = toSafeString(data.answer);

    const source = typeof data.source_used === "string" ? data.source_used : "";

    const citations = Array.isArray(data.citations) ? data.citations : [];

    addMessage("assistant", answer, source, citations);

    renderTrace(data.trace || []);

    sourceUsed.textContent = source || "Not specified";
  } catch (e) {
    console.error("Chat error:", e);

    addMessage("assistant", `Error: ${e.message}`);

    renderTrace(["Request failed"]);

    sourceUsed.textContent = "Error";
  } finally {
    btn.disabled = false;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();

  const q = question.value.trim();

  if (q) {
    askAgent(q);
  }
});

document.querySelectorAll(".example").forEach((b) => {
  b.addEventListener("click", () => {
    askAgent(b.textContent.trim());
  });
});

const modal = document.getElementById("uploadModal");

document.getElementById("openUpload").onclick = () => {
  modal.classList.remove("hidden");
};

document.getElementById("closeUpload").onclick = () => {
  modal.classList.add("hidden");
};

document.getElementById("uploadBtn").onclick = async () => {
  const file = document.getElementById("fileInput").files[0];

  const key = document.getElementById("adminKey").value;

  const status = document.getElementById("uploadStatus");

  if (!file) {
    status.textContent = "Choose a file first.";
    return;
  }

  status.textContent = "Indexing document...";

  const fd = new FormData();

  fd.append("file", file);

  try {
    const r = await fetch("/api/ingest", {
      method: "POST",
      headers: {
        "X-Admin-Key": key,
      },
      body: fd,
    });

    const d = await r.json();

    if (!r.ok) {
      throw new Error(toSafeString(d.detail) || "Upload failed");
    }

    status.textContent = `Indexed ${toSafeString(d.file)}: ${toSafeString(d.chunks)} chunks.`;
  } catch (e) {
    console.error("Upload error:", e);

    status.textContent = `Error: ${e.message}`;
  }
};
