document.addEventListener('DOMContentLoaded', () => {
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
