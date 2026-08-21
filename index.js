export const name = 'ollama-cloud-sync'

export const inject = ['settings', 'credentials']

const SETTINGS_NAMESPACE = 'llm-pi-ai'

const DEFAULTS = Object.freeze({
  route: 'ollama-cloud',
  displayName: 'Ollama Cloud',
  baseURL: 'https://ollama.com/v1',
  api: 'openai-completions',
  apiKeyEnv: 'OLLAMA_API_KEY',

  // Four catalog refreshes per day.
  refreshMs: 6 * 60 * 60 * 1000,

  // llm-pi-ai and this plugin can mount concurrently.
  // Retry briefly until its settings namespace exists.
  startupRetryMs: 1000,
  startupRetryLimit: 30,

  // Per-request timeout for Ollama catalog/detail requests.
  requestTimeoutMs: 30_000,
})

function settingsReady(ctx) {
  return ctx.settings.get(SETTINGS_NAMESPACE) !== undefined
}

function currentProvider(ctx, route) {
  const settings = ctx.settings.get(SETTINGS_NAMESPACE)
  return settings?.providers?.[route]
}

function modelMap(models) {
  const map = new Map()

  if (!Array.isArray(models)) return map

  for (const model of models) {
    if (!model || typeof model.id !== 'string') continue
    map.set(model.id, model)
  }

  return map
}

function validContextWindow(value) {
  return (
    Number.isInteger(value) &&
    value > 0
  )
}

function sameModels(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b)) {
    return false
  }

  if (a.length !== b.length) {
    return false
  }

  for (let i = 0; i < a.length; i += 1) {
    const left = a[i]
    const right = b[i]

    if (left?.id !== right?.id) return false
    if (left?.name !== right?.name) return false
    if (left?.contextWindow !== right?.contextWindow) return false
  }

  return true
}

async function resolveCredential(ctx, ref) {
  const hit = await ctx.credentials.resolve(ref)
  const value = hit?.value

  if (
    typeof value !== 'string' ||
    value.trim().length === 0
  ) {
    throw new Error(
      `Ollama Cloud credential ${ref} is not configured`
    )
  }

  return value.trim()
}

async function fetchJson(
  url,
  {
    apiKey,
    method = 'GET',
    body,
    timeoutMs,
  }
) {
  const response = await fetch(url, {
    method,
    headers: {
      Accept: 'application/json',
      Authorization: `Bearer ${apiKey}`,
      ...(body === undefined
        ? {}
        : { 'Content-Type': 'application/json' }),
    },
    ...(body === undefined
      ? {}
      : { body: JSON.stringify(body) }),
    signal: AbortSignal.timeout(timeoutMs),
  })

  if (!response.ok) {
    throw new Error(
      `${url} returned HTTP ${response.status}`
    )
  }

  return response.json()
}

async function fetchModelIds(
  baseURL,
  apiKey,
  timeoutMs
) {
  const url =
    `${baseURL.replace(/\/+$/, '')}/models`

  const body = await fetchJson(url, {
    apiKey,
    timeoutMs,
  })

  if (!body || !Array.isArray(body.data)) {
    throw new Error(
      'Ollama Cloud /v1/models returned an unexpected response'
    )
  }

  const seen = new Set()
  const ids = []

  for (const entry of body.data) {
    const id = entry?.id

    if (
      typeof id !== 'string' ||
      id.length === 0 ||
      seen.has(id)
    ) {
      continue
    }

    seen.add(id)
    ids.push(id)
  }

  if (ids.length === 0) {
    throw new Error(
      'Ollama Cloud /v1/models returned no usable model IDs'
    )
  }

  return ids
}

function showURL(baseURL) {
  /*
   * baseURL is normally:
   *
   *   https://ollama.com/v1
   *
   * Ollama's model-details endpoint is on its native API:
   *
   *   https://ollama.com/api/show
   *
   * Using URL resolution keeps this correct if baseURL changes
   * to another host later.
   */
  return new URL('/api/show', baseURL).toString()
}

function extractContextWindow(modelInfo) {
  if (
    !modelInfo ||
    typeof modelInfo !== 'object'
  ) {
    return undefined
  }

  /*
   * Ollama names the field according to the model architecture:
   *
   *   deepseek4.context_length
   *   llama.context_length
   *   qwen3.context_length
   *   ...
   *
   * Avoid hardcoding architecture names.
   */
  const matches = Object.entries(modelInfo)
    .filter(([key]) =>
      key.endsWith('.context_length')
    )
    .map(([, value]) => value)
    .filter(validContextWindow)

  if (matches.length === 0) {
    return undefined
  }

  /*
   * Normally there is exactly one context_length field.
   *
   * If Ollama ever returns more than one, use the largest
   * valid declared capacity rather than depending on object
   * key ordering.
   */
  return Math.max(...matches)
}

async function fetchContextWindow(
  baseURL,
  apiKey,
  model,
  timeoutMs
) {
  const body = await fetchJson(
    showURL(baseURL),
    {
      apiKey,
      method: 'POST',
      body: { model },
      timeoutMs,
    }
  )

  const contextWindow =
    extractContextWindow(body?.model_info)

  if (!validContextWindow(contextWindow)) {
    throw new Error(
      `${model}: /api/show returned no valid *.context_length`
    )
  }

  return contextWindow
}

async function buildModels(
  baseURL,
  apiKey,
  ids,
  currentModels,
  timeoutMs
) {
  const previous = modelMap(currentModels)
  const models = []

  let discovered = 0
  let preserved = 0
  let omitted = 0

  /*
   * Intentionally sequential.
   *
   * This runs only every six hours and currently involves roughly
   * twenty requests, so there is little value in fanning them out
   * against Ollama's API.
   */
  for (const id of ids) {
    try {
      const contextWindow =
        await fetchContextWindow(
          baseURL,
          apiKey,
          id,
          timeoutMs
        )

      models.push({
        id,
        name: id,
        contextWindow,
      })

      discovered += 1
    } catch (error) {
      const old = previous.get(id)

      /*
       * A transient /api/show failure must not cause DSH to fall
       * back from a previously-known model capacity to its generic
       * context-window default.
       */
      if (
        old &&
        validContextWindow(old.contextWindow)
      ) {
        models.push({
          id,
          name:
            typeof old.name === 'string'
              ? old.name
              : id,
          contextWindow: old.contextWindow,
        })

        preserved += 1

        console.warn(
          `[ollama-cloud] ${id}: context discovery failed; preserving previous contextWindow=${old.contextWindow}`
        )

        continue
      }

      omitted += 1

      const message =
        error instanceof Error
          ? error.message
          : String(error)

      /*
       * For a brand-new model with unknown capacity, do not expose
       * it to DSH yet. Otherwise DSH would silently assign its
       * generic fallback context window, defeating the purpose of
       * this plugin.
       */
      console.warn(
        `[ollama-cloud] ${id}: omitted because context discovery failed: ${message}`
      )
    }
  }

  if (models.length === 0) {
    throw new Error(
      'Ollama Cloud model discovery produced no models with known context capacity'
    )
  }

  return {
    models,
    discovered,
    preserved,
    omitted,
  }
}

async function sync(ctx, config) {
  if (!settingsReady(ctx)) {
    throw new Error(
      `${SETTINGS_NAMESPACE} settings namespace is not ready`
    )
  }

  /*
   * The secret exists only long enough to authenticate these
   * requests. It is never logged or written to DSH settings.
   */
  const apiKey = await resolveCredential(
    ctx,
    config.apiKeyEnv
  )

  const current = currentProvider(
    ctx,
    config.route
  )

  const ids = await fetchModelIds(
    config.baseURL,
    apiKey,
    config.requestTimeoutMs
  )

  const enriched = await buildModels(
    config.baseURL,
    apiKey,
    ids,
    current?.models,
    config.requestTimeoutMs
  )

  const models = enriched.models

  const metadataChanged =
    current?.displayName !== config.displayName ||
    current?.baseURL !== config.baseURL ||
    current?.api !== config.api ||
    current?.apiKeyEnv !== config.apiKeyEnv

  const modelsChanged =
    !sameModels(current?.models, models)

  if (!metadataChanged && !modelsChanged) {
    return {
      changed: false,
      count: models.length,
      ...enriched,
    }
  }

  const root = [
    'providers',
    config.route,
  ]

  /*
   * Only synchronize fields this plugin owns.
   *
   * Any manual additions such as:
   *
   *   compat
   *   reasoning
   *   headers
   *   retryPolicy
   *   defaultContextWindow
   *
   * remain untouched.
   */
  await ctx.settings.mutate(
    SETTINGS_NAMESPACE,
    [
      {
        op: 'set',
        path: [
          ...root,
          'displayName',
        ],
        value: config.displayName,
      },
      {
        op: 'set',
        path: [
          ...root,
          'baseURL',
        ],
        value: config.baseURL,
      },
      {
        op: 'set',
        path: [
          ...root,
          'api',
        ],
        value: config.api,
      },
      {
        op: 'set',
        path: [
          ...root,
          'apiKeyEnv',
        ],
        value: config.apiKeyEnv,
      },
      {
        op: 'set',
        path: [
          ...root,
          'models',
        ],
        value: models,
      },
    ],
  )

  return {
    changed: true,
    count: models.length,
    ...enriched,
  }
}

export function apply(
  ctx,
  rawConfig = {}
) {
  const config = {
    ...DEFAULTS,
    ...rawConfig,
  }

  let stopped = false
  let running = false

  const refresh = async () => {
    if (stopped || running) {
      return false
    }

    running = true

    try {
      const result =
        await sync(ctx, config)

      const detail = [
        `${result.count} models`,
        `${result.discovered} contexts discovered`,
      ]

      if (result.preserved > 0) {
        detail.push(
          `${result.preserved} preserved`
        )
      }

      if (result.omitted > 0) {
        detail.push(
          `${result.omitted} omitted`
        )
      }

      console.info(
        `[ollama-cloud] catalog refresh succeeded: ${detail.join(', ')}`
        + (
          result.changed
            ? ' (updated)'
            : ' (unchanged)'
        )
      )

      return true
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : String(error)

      console.warn(
        `[ollama-cloud] catalog refresh failed: ${message}`
      )

      return false
    } finally {
      running = false
    }
  }

  ctx.effect(() => {
    let startupTimer
    let interval

    let startupAttempts = 0

    const startInterval = () => {
      interval = setInterval(
        () => {
          void refresh()
        },
        config.refreshMs
      )

      interval.unref?.()
    }

    const startup = async () => {
      if (stopped) return

      /*
       * Do not burn credential/network requests until llm-pi-ai
       * has registered its settings namespace.
       */
      if (!settingsReady(ctx)) {
        startupAttempts += 1

        if (
          startupAttempts >=
          config.startupRetryLimit
        ) {
          console.warn(
            `[ollama-cloud] gave up waiting for ${SETTINGS_NAMESPACE}`
          )

          /*
           * Still schedule normal refreshes.
           * A later cycle may succeed.
           */
          startInterval()
          return
        }

        startupTimer = setTimeout(
          () => void startup(),
          config.startupRetryMs
        )

        startupTimer.unref?.()
        return
      }

      await refresh()

      if (!stopped) {
        startInterval()
      }
    }

    startupTimer = setTimeout(
      () => void startup(),
      0
    )

    startupTimer.unref?.()

    return () => {
      stopped = true

      if (startupTimer) {
        clearTimeout(startupTimer)
      }

      if (interval) {
        clearInterval(interval)
      }
    }
  }, 'ollama-cloud catalog sync')
}

