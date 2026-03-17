export default defineEventHandler(async (event) => {
  const body = await readBody(event)
  const code: string = body?.code ?? ''

  const data = await $fetch<{ result: string | null; error: string | null }>(
    'http://localhost:8000/interpret',
    { method: 'POST', body: { code } }
  )

  return data
})
