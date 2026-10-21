'use strict';
const search = document.querySelector('#source-search');
if (search) search.addEventListener('input', () => {
  const query = search.value.toLowerCase();
  document.querySelectorAll('#repo-rows tr').forEach(row => { row.hidden = !row.textContent.toLowerCase().includes(query); });
});
document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(document.getElementById(button.dataset.copy).textContent); button.textContent = 'Copied'; }
  catch { button.textContent = 'Select the text to copy'; }
}));
