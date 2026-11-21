const fileInput = document.querySelector('#sketch-file');
const dropZone = document.querySelector('#drop-zone');
const uploadPrompt = document.querySelector('#upload-prompt');
const previewWrap = document.querySelector('#preview-wrap');
const previewImage = document.querySelector('#sketch-preview');
const previewLabel = document.querySelector('#preview-label');
const removeFile = document.querySelector('#remove-file');
const styleSelect = document.querySelector('#style-select');
const styleDescription = document.querySelector('#style-description');
const generateButton = document.querySelector('#generate-button');
const statusBox = document.querySelector('#generation-status');
const statusMessage = document.querySelector('#status-message');
const resultState = document.querySelector('#result-state');
const resultEmpty = document.querySelector('#result-empty');
const resultContent = document.querySelector('#result-content');
const resultOriginal = document.querySelector('#result-original');
const resultGenerated = document.querySelector('#result-generated');
const downloadButton = document.querySelector('#download-button');
const anotherButton = document.querySelector('#another-button');

const styleNotes = {
  modern: 'A restrained palette, clean geometry and considered glazing.',
  luxury: 'Natural stone, warm timber and refined contemporary details.',
  indian: 'Sandstone, shaded verandahs and subtle jaali-inspired screens.',
  industrial: 'Exposed concrete, dark steel framing and textured masonry.',
  scandinavian: 'Pale timber, soft stone and calm, natural daylight.'
};
let selectedFile = null;
let previewUrl = null;
let busy = false;

function setStatus(message, state = 'info') {
  statusMessage.textContent = message;
  statusBox.dataset.state = state;
}

function setFile(file) {
  if (!file || busy) return;
  const allowed = ['image/png', 'image/jpeg'];
  if (!allowed.includes(file.type)) {
    setStatus('Please choose a PNG or JPG image.', 'error');
    return;
  }
  if (file.size > 8 * 1024 * 1024) {
    setStatus('That image is larger than 8 MB. Choose a smaller file.', 'error');
    return;
  }
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewLabel.textContent = file.name;
  uploadPrompt.hidden = true;
  previewWrap.hidden = false;
  generateButton.disabled = busy;
  resultEmpty.hidden = false;
  resultContent.hidden = true;
  resultState.textContent = 'SKETCH LOADED';
  setStatus('Sketch ready. Choose a visual direction and generate.', 'ready');
}

function clearFile() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  selectedFile = null;
  fileInput.value = '';
  previewImage.removeAttribute('src');
  uploadPrompt.hidden = false;
  previewWrap.hidden = true;
  generateButton.disabled = true;
  resultEmpty.hidden = false;
  resultContent.hidden = true;
  resultState.textContent = 'AWAITING INPUT';
  setStatus('Choose a PNG or JPG facade sketch to begin.', 'info');
}

async function checkBackend() {
  try {
    const response = await fetch('/api/status');
    const status = await response.json();
    if (status.available) {
      setStatus(status.message || 'Colab inference is connected.', 'ready');
    } else {
      setStatus(status.message || 'Local inference is not available on this machine.', 'error');
    }
  } catch (_error) {
    setStatus('Could not contact the local backend. Start the server and refresh this page.', 'error');
  }
}

async function pollJob(jobId) {
  while (true) {
    await new Promise((resolve) => window.setTimeout(resolve, 1200));
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || 'Could not check generation status.');
    if (job.state === 'queued') {
      setStatus('Request queued. Waiting for the Colab inference worker…', 'running');
      continue;
    }
    if (job.state === 'running') {
      setStatus('The model is generating your facade. This can take a few minutes.', 'running');
      continue;
    }
    if (job.state === 'error') throw new Error(job.message || 'Generation could not be completed.');
    if (job.state === 'complete') return job;
  }
}

async function generate() {
  if (!selectedFile || busy) return;
  busy = true;
  generateButton.disabled = true;
  anotherButton.disabled = true;
  resultState.textContent = 'GENERATION IN PROGRESS';
  setStatus('Submitting your sketch to the local inference worker…', 'running');
  try {
    const form = new FormData();
    form.append('sketch', selectedFile, selectedFile.name);
    form.append('style', styleSelect.value);
    const response = await fetch('/api/generate', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'The request could not be submitted.');
    const job = await pollJob(data.id);
    resultOriginal.src = previewUrl;
    resultGenerated.src = `${job.result_url}?v=${Date.now()}`;
    downloadButton.href = job.download_url;
    resultEmpty.hidden = true;
    resultContent.hidden = false;
    resultState.textContent = 'STUDY COMPLETE';
    setStatus('Generation complete. The new visualization was created from this uploaded sketch.', 'ready');
  } catch (error) {
    resultState.textContent = 'GENERATION UNAVAILABLE';
    setStatus(error.message || 'Generation failed. Check that the backend is running.', 'error');
  } finally {
    busy = false;
    generateButton.disabled = !selectedFile;
    anotherButton.disabled = false;
  }
}

dropZone.addEventListener('click', (event) => {
  if (!event.target.closest('#remove-file')) fileInput.click();
});
dropZone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener('change', () => setFile(fileInput.files?.[0]));
removeFile.addEventListener('click', (event) => { event.stopPropagation(); if (!busy) clearFile(); });
dropZone.addEventListener('dragover', (event) => { event.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (event) => {
  event.preventDefault();
  dropZone.classList.remove('dragover');
  setFile(event.dataTransfer.files?.[0]);
});
styleSelect.addEventListener('change', () => { styleDescription.textContent = styleNotes[styleSelect.value]; });
generateButton.addEventListener('click', generate);
anotherButton.addEventListener('click', generate);
checkBackend();
