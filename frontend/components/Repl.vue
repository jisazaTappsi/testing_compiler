<script setup lang="ts">
import { ref, nextTick, onMounted } from 'vue'

type HistoryEntry = {
  type: 'input' | 'output' | 'error'
  text: string
}

const history = ref<HistoryEntry[]>([
  { type: 'output', text: 'Welcome to the REPL. Type code and press Enter.' },
  { type: 'output', text: 'Type "clear" to reset.' },
])
const input = ref('')
const loading = ref(false)
const inputEl = ref<HTMLInputElement | null>(null)
const terminalEl = ref<HTMLDivElement | null>(null)

onMounted(() => inputEl.value?.focus())

async function scrollToBottom() {
  await nextTick()
  if (terminalEl.value) {
    terminalEl.value.scrollTop = terminalEl.value.scrollHeight
  }
}

async function submit() {
  const code = input.value.trim()
  if (!code) return

  if (code === 'clear') {
    history.value = []
    input.value = ''
    return
  }

  history.value.push({ type: 'input', text: code })
  input.value = ''
  loading.value = true
  await scrollToBottom()

  try {
    const data = await $fetch<{ result: string; error: string | null }>('/api/interpret', {
      method: 'POST',
      body: { code },
    })

    if (data.error) {
      history.value.push({ type: 'error', text: data.error })
    } else {
      history.value.push({ type: 'output', text: data.result })
    }
  } catch (err: any) {
    history.value.push({ type: 'error', text: err?.message ?? 'Request failed' })
  } finally {
    loading.value = false
    await scrollToBottom()
  }
}

function focusInput() {
  inputEl.value?.focus()
}
</script>

<template>
  <div class="terminal" @click="focusInput" ref="terminalEl">
    <div
      v-for="(entry, i) in history"
      :key="i"
      :class="['line', entry.type]"
    >
      <span v-if="entry.type === 'input'" class="prompt">&gt;&nbsp;</span>
      <span>{{ entry.text }}</span>
    </div>

    <div class="line input-line">
      <span class="prompt">&gt;&nbsp;</span>
      <input
        ref="inputEl"
        v-model="input"
        :disabled="loading"
        @keydown.enter="submit"
        autocomplete="off"
        spellcheck="false"
        class="input-field"
      />
    </div>
  </div>
</template>

<style scoped>
.terminal {
  background: #0d0d0d;
  color: #d4d4d4;
  font-family: 'Courier New', Courier, monospace;
  font-size: 14px;
  line-height: 1.6;
  padding: 16px;
  height: 100%;
  overflow-y: auto;
  cursor: text;
  box-sizing: border-box;
}

.line {
  display: flex;
  align-items: flex-start;
  white-space: pre-wrap;
  word-break: break-all;
  margin-bottom: 2px;
}

.line.input .prompt,
.input-line .prompt {
  color: #4ec9b0;
  flex-shrink: 0;
}

.line.output {
  color: #d4d4d4;
}

.line.error {
  color: #f44747;
}

.input-line {
  display: flex;
  align-items: center;
  margin-top: 4px;
}

.input-field {
  background: transparent;
  border: none;
  outline: none;
  color: #d4d4d4;
  font-family: inherit;
  font-size: inherit;
  flex: 1;
  caret-color: #d4d4d4;
}

.input-field:disabled {
  opacity: 0.5;
}
</style>
