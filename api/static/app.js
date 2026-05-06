document.addEventListener('DOMContentLoaded', () => {
  // ── Download handler ──────────────────────────────────────────────────────
  const downloadBtn = document.getElementById('download-btn');
  if (downloadBtn) {
    downloadBtn.addEventListener('click', async () => {
      const gush = document.getElementById('gush').value.trim();
      const helka = document.getElementById('helka').value.trim();
      const statusDiv = document.getElementById('download-status');
      if (!gush || !helka) {
        statusDiv.textContent = 'יש להזין גוש וחלקה';
        return;
      }
      statusDiv.textContent = 'מוריד ומאנדקס...';
      downloadBtn.disabled = true;
      try {
        const res = await fetch('/download', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ gush, helka })
        });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        if (data.downloaded && data.downloaded.length > 0) {
          statusDiv.innerHTML =
            `הורדו ${data.downloaded.length} קבצים, אונדקסו ${data.indexed_chunks} קטעים.<br>` +
            data.downloaded.map(f => `• ${f}`).join('<br>');
        } else {
          statusDiv.textContent = data.message || 'לא נמצאו קבצים חדשים';
        }
      } catch (err) {
        console.error('Download error', err);
        statusDiv.textContent = 'שגיאה: ' + err;
      } finally {
        downloadBtn.disabled = false;
      }
    });
  }

  // ── Query handler ─────────────────────────────────────────────────────────
  const sendBtn = document.getElementById('send');
  if (!sendBtn) return;

  sendBtn.addEventListener('click', async () => {
    const q = document.getElementById('question').value;
    const respArea = document.getElementById('response');
    const resultsDiv = document.getElementById('results');
    console.log('Sending query:', q);
    if (respArea) respArea.value = 'Loading...';
    if (resultsDiv) resultsDiv.innerHTML = 'Loading...';
    try {
      const res = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q })
      });
      console.log('HTTP status', res.status);
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      console.log('Response data', data);
      if (!data.results || data.results.length === 0) {
        if (respArea) respArea.value = 'No results';
        if (resultsDiv) resultsDiv.innerHTML = '<i>No results</i>';
      } else {
        // render results list with optional download link
        const html = data.results.map((r, i) => {
          const idx = i+1;
          const text = (r.text || '').replace(/</g, '&lt;');
          if (r.source) {
            return `<div style="margin-bottom:12px; direction: rtl; text-align: right;"><strong>${idx}.</strong> <a href="${r.source}" target="_blank" rel="noopener">[open]</a><div>${text}</div></div>`;
          } else {
            return `<div style="margin-bottom:12px; direction: rtl; text-align: right;"><strong>${idx}.</strong> <div>${text}</div></div>`;
          }
        }).join('');
        if (respArea) respArea.value = data.results.map((r,i)=> `${i+1}. ${r.text || r}`).join('\n\n');
        if (resultsDiv) resultsDiv.innerHTML = html;
      }
    } catch (err) {
      console.error('Query error', err);
      respArea.value = 'Error: ' + err;
    }
  });
});
