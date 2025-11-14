const chatEl = document.getElementById('chat');
const productsEl = document.getElementById('products');
const msgEl = document.getElementById('msg');
const sendBtn = document.getElementById('send');
const attachBtn = document.getElementById('attachBtn');
const fileInput = document.getElementById('fileInput');
const selectedFileInfo = document.getElementById('selectedFileInfo');
const micBtn = document.getElementById('micBtn');

let conversation = [];
let selectedFile = null;
let recognition = null;
let isRecording = false;

function addMsg(role, text){
  const div = document.createElement('div');
  div.className = 'msg ' + (role === 'user' ? 'user' : 'assistant');
  if(role === 'assistant'){
    const label = document.createElement('div');
    label.className = 'label';
    label.textContent = 'BHB Advisor';
    div.appendChild(label);
  }
  const p = document.createElement('div');
  p.textContent = text || '';
  div.appendChild(p);
  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = new Date().toLocaleTimeString();
  div.appendChild(meta);
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function renderProducts(items){
  productsEl.innerHTML = '';
  if(!items || !items.length){
    productsEl.innerHTML = '<p>No products yet.</p>';
    return;
  }
  for(const p of items){
    const card = document.createElement('div');
    card.className = 'card';
    const name = document.createElement('div');
    name.className = 'name';
    name.textContent = `${p.brand} ${p.model_name}`;
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = `${p.category} — ${(p.features && p.features[0]) || p.recommended_for || ''}`;
    const price = document.createElement('div');
    price.className = 'price';
    price.textContent = `RM ${p.price_rm ?? 'N/A'}`;
    const link = document.createElement('a');
    // Prefer direct BHB product link if provided; otherwise fallback to search URL
    let href = '#';
    let label = 'View on BHB Website';
    if(p.bhb_product_url){
      href = p.bhb_product_url;
      label = 'View on BHB Website';
    } else {
      const base = 'https://www.bhb.com.my/search?q=';
      const q = p.website_search_text || `${p.brand} ${p.model_name}`;
      href = base + encodeURIComponent(q);
      label = 'Search on BHB Website';
    }
    link.href = href;
    link.target = '_blank';
    link.textContent = label;
    card.appendChild(name);
    card.appendChild(meta);
    card.appendChild(price);
    card.appendChild(link);
    productsEl.appendChild(card);
  }
}

function updateSelectedFileInfo(){
  if(selectedFile){
    selectedFileInfo.textContent = `Attached: ${selectedFile.name}`;
  } else {
    selectedFileInfo.textContent = '';
  }
}

attachBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  selectedFile = fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;
  updateSelectedFileInfo();
});

function startRecognition(){
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!SR){
    alert('Voice input not supported in this browser.');
    return;
  }
  recognition = new SR();
  recognition.lang = 'en-MY';
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;
  recognition.onresult = (event) => {
    const text = event.results[0][0].transcript;
    msgEl.value = (msgEl.value ? (msgEl.value + ' ') : '') + text;
  };
  recognition.onend = () => {
    isRecording = false;
    micBtn.classList.remove('recording');
  };
  recognition.onerror = () => {
    isRecording = false;
    micBtn.classList.remove('recording');
  };
  recognition.start();
  isRecording = true;
  micBtn.classList.add('recording');
}

function stopRecognition(){
  if(recognition){ recognition.stop(); }
  isRecording = false;
  micBtn.classList.remove('recording');
}

micBtn.addEventListener('click', () => {
  if(isRecording){ stopRecognition(); } else { startRecognition(); }
});

async function handleSend(){
  const text = (msgEl.value || '').trim();
  if(!text && !selectedFile) return;
  // Show the outgoing message
  if(selectedFile){
    addMsg('user', text ? `📷 ${text}` : '📷 Image attached');
  } else {
    addMsg('user', text);
  }
  conversation.push({ role: 'user', content: text || '' });
  msgEl.value = '';

  try{
    let data;
    if(selectedFile){
      const fd = new FormData();
      fd.append('image', selectedFile);
      if(text) fd.append('message', text);
      const resp = await fetch('/api/vision-chat', { method: 'POST', body: fd });
      data = await resp.json();
      // clear selected file after sending
      selectedFile = null; fileInput.value = ''; updateSelectedFileInfo();
    } else {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: conversation })
      });
      data = await resp.json();
    }
    const reply = data.reply || '';
    addMsg('assistant', reply);
    renderProducts(data.suggested_products || []);
    conversation.push({ role: 'assistant', content: reply });
  } catch(err){
    addMsg('assistant', 'Error contacting advisor.');
  }
}

sendBtn.addEventListener('click', handleSend);
msgEl.addEventListener('keydown', (e) => {
  if(e.key === 'Enter'){ handleSend(); }
});

// Show a friendly greeting when the page loads
window.addEventListener('DOMContentLoaded', () => {
  addMsg('assistant', "Hi! I’m BHB Product Advisor. Ask about a TV, fridge, washer, aircond, fan or water heater — or say ‘new house’ for a full-house bundle.");
});