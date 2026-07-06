// Escape ALL HTML metacharacters. Every server/LLM/document-derived string
// must pass through here before being placed in innerHTML — chunk text and
// filenames ultimately come from remote documents (OCR, mavat DOC_NAME) and
// could carry markup.
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// Escaped text → display HTML: "^" objective separators and newlines become
// <br/>, and https:// URLs become clickable links. Runs on ESCAPED text only,
// so the URL match cannot contain quotes/angle brackets and is attribute-safe.
function formatEscapedText(escaped) {
  let html = escaped
    .replace(/\s*\^\s*/g, '<br/>')
    .replace(/\n/g, '<br/>');
  html = html.replace(
    /https:\/\/[^\s<]+/g,
    match => `<a href="${match}" target="_blank" rel="noopener" style="color:#0066cc;text-decoration:underline;">${match}</a>`
  );
  return html;
}

// Only trust server-provided source paths of the exact shape /files/<name>.
function safeSourcePath(source) {
  return typeof source === 'string' && /^\/files\/[^/\\]+$/.test(source) ? source : null;
}

document.addEventListener('DOMContentLoaded', () => {
  // ── Download handler ──────────────────────────────────────────────────────
  const downloadBtn = document.getElementById('download-btn');
  if (downloadBtn) {
    downloadBtn.addEventListener('click', async () => {
      const planNameEl = document.getElementById('plan-name');
      const statusDiv = document.getElementById('download-status');
      if (!planNameEl || !statusDiv) {
        console.error('Download UI elements not found — please restart the server to reload the HTML.');
        return;
      }
      const planName = planNameEl.value.trim();
      if (!planName) {
        statusDiv.style.display = 'block';
        statusDiv.textContent = 'יש להזין שם תכנית';
        return;
      }
      statusDiv.style.display = 'block';
      statusDiv.textContent = '⏳ מחפש תכניות...';
      downloadBtn.disabled = true;
      try {
        const res = await fetch('/download', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ plan_name: planName })
        });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        const log = (data.log || []).join('\n');
        if (data.downloaded && data.downloaded.length > 0) {
          statusDiv.textContent = log + `\n\n✅ הורדו ${data.downloaded.length} קבצים, אונדקסו ${data.indexed_chunks} קטעים.`;
        } else {
          statusDiv.textContent = log || 'לא נמצאו קבצים חדשים';
        }
      } catch (err) {
        console.error('Download error', err);
        statusDiv.textContent = '❌ שגיאה: ' + err;
      } finally {
        downloadBtn.disabled = false;
      }
    });
  }

  // ── Query handler ─────────────────────────────────────────────────────────
  const sendBtn = document.getElementById('send');
  if (!sendBtn) return;

  sendBtn.addEventListener('click', async () => {
    const q = document.getElementById('question').value.trim();
    const answerBox = document.getElementById('answer-box');
    const resultsDiv = document.getElementById('results');
    const sourcesDetails = document.getElementById('sources-details');
    const sourcesCount = document.getElementById('sources-count');

    if (!q) return;

    answerBox.style.display = 'block';
    answerBox.textContent = '⏳ מחפש ומנתח...';
    if (sourcesDetails) sourcesDetails.style.display = 'none';

    try {
      const res = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q })
      });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();

      // Show generated answer
      if (data.answer) {
        answerBox.innerHTML = '<strong>תשובה:</strong><br/>' + formatEscapedText(escapeHtml(data.answer));
      } else if (!data.results || data.results.length === 0) {
        answerBox.textContent = 'לא נמצאו תוצאות';
      } else {
        // Chunks found but the LLM produced no answer (e.g. model unreachable) —
        // say so instead of silently hiding the box.
        answerBox.textContent = 'לא התקבלה תשובה מהמודל — ניתן לעיין במקורות למטה';
      }

      // Show source chunks in collapsible section
      if (data.results && data.results.length > 0) {
        const html = data.results.map((r) => {
          const text = formatEscapedText(escapeHtml(r.text || ''));
          const source = safeSourcePath(r.source);
          const link = source
            ? `<a href="${escapeHtml(source)}" target="_blank" rel="noopener" style="font-size:0.85em;">[פתח מסמך]</a> `
            : '';
          return `<div style="margin-bottom:10px;padding:8px;background:#fafafa;border-radius:4px;direction:rtl;text-align:right;font-size:0.9em;">${link}${text}</div>`;
        }).join('');
        if (resultsDiv) resultsDiv.innerHTML = html;
        if (sourcesCount) sourcesCount.textContent = data.results.length;
        if (sourcesDetails) sourcesDetails.style.display = 'block';
      }
    } catch (err) {
      console.error('Query error', err);
      answerBox.style.display = 'block';
      answerBox.textContent = '❌ שגיאה: ' + err;
    }
  });
});
