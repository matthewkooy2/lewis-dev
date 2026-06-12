const form = document.getElementById('chat-form');
const log = document.getElementById('chat-log');
const input = document.getElementById('chat-input');
const submit = form.querySelector('button[type="submit"]');
let isPending = false;

function clearEmptyHint() {
  const hint = document.getElementById('chat-empty');
  if (hint) hint.remove();
}

function roleLabel(role) {
  const label = document.createElement('span');
  label.className = 'bubble-role';
  label.textContent = role === 'user' ? 'You' : 'Lewis';
  return label;
}

function bubble(role, text) {
  const node = document.createElement('div');
  node.className = `bubble ${role}`;
  node.appendChild(roleLabel(role));
  const body = document.createElement('span');
  body.className = 'bubble-text';
  body.textContent = text;
  node.appendChild(body);
  log.appendChild(node);
  log.scrollTop = log.scrollHeight;
  return node;
}

function thinkingBubble() {
  const node = document.createElement('div');
  node.className = 'bubble assistant thinking';
  node.setAttribute('aria-live', 'polite');
  node.innerHTML = 'Lewis is thinking<span class="thinking-dots"><span>.</span><span>.</span><span>.</span></span>';
  log.appendChild(node);
  log.scrollTop = log.scrollHeight;
  return node;
}

function memoryContextNote(count) {
  const memoryCount = Number(count) || 0;
  if (memoryCount <= 0) return null;
  const node = document.createElement('div');
  node.className = 'memory-context-note';
  node.textContent = `Using ${memoryCount} approved ${memoryCount === 1 ? 'memory' : 'memories'} for context.`;
  return node;
}

function assistantBubble(text, memoryContextCount, isError) {
  const node = document.createElement('div');
  node.className = isError ? 'bubble assistant error' : 'bubble assistant';
  node.appendChild(roleLabel('assistant'));
  const note = memoryContextNote(memoryContextCount);
  if (note) node.appendChild(note);
  const body = document.createElement('span');
  body.className = 'bubble-text';
  body.textContent = text;
  node.appendChild(body);
  log.appendChild(node);
  log.scrollTop = log.scrollHeight;
  return node;
}

input.addEventListener('keydown', (event) => {
  if (event.key !== 'Enter' || event.shiftKey) return;
  event.preventDefault();
  if (!input.value.trim()) return;
  form.requestSubmit();
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (isPending) return;
  const message = input.value.trim();
  if (!message) return;
  isPending = true;
  clearEmptyHint();
  bubble('user', message);
  const pending = thinkingBubble();
  input.value = '';
  submit.disabled = true;
  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message})
    });
    const data = await response.json();
    pending.remove();
    assistantBubble(data.answer || 'No response.', data.memory_context_count, false);
  } catch (error) {
    pending.remove();
    assistantBubble(`Sorry, I hit an error: ${error.message}`, 0, true);
  } finally {
    isPending = false;
    submit.disabled = false;
    input.focus();
  }
});
