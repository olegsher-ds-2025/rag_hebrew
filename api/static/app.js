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
        answerBox.innerHTML = '<strong>תשובה:</strong>\n' + data.answer.replace(/</g, '&lt;');
      } else if (!data.results || data.results.length === 0) {
        answerBox.textContent = 'לא נמצאו תוצאות';
      } else {
        answerBox.style.display = 'none';
      }

      // Show source chunks in collapsible section
      if (data.results && data.results.length > 0) {
        const uniqueSources = [...new Set(data.results.filter(r => r.source).map(r => r.source))];
        const html = data.results.map((r, i) => {
          const text = (r.text || '').replace(/</g, '&lt;');
          const link = r.source
            ? `<a href="${r.source}" target="_blank" rel="noopener" style="font-size:0.85em;">[פתח מסמך]</a> `
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
