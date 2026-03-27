export default defineEventHandler(async (event) => {
  const body = await readBody(event)
  const code: string = body?.code ?? ''
  const symbols = body?.symbols ?? null

  const data = await $fetch<{ result: string | null; symbols?: Record<string, unknown> | null; error: string | null }>(
    'http://localhost:9000/interpret',
    { method: 'POST', body: { code, symbols } }
  )

  return data
})
