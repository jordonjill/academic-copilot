import type { ChatArtifacts, ChatRequest, ChatResponseNormalized, ChatResponseRaw } from "../types/api";
import { buildDefaultHeaders, fetchJson, getApiSettings, HttpError } from "./http";

export type ChatStreamEvent = {
  type: string;
  [key: string]: unknown;
};

export type StreamCallbacks = {
  onConnected?: (event: ChatStreamEvent) => void;
  onStatus?: (event: ChatStreamEvent) => void;
  onStep?: (event: ChatStreamEvent) => void;
  onError?: (event: ChatStreamEvent) => void;
  onEvent?: (eventName: string, event: ChatStreamEvent) => void;
  onCompletion?: (response: ChatResponseNormalized, event: ChatStreamEvent) => void;
};

function normalizeArtifacts(data: ChatResponseRaw["data"] | undefined): ChatArtifacts | undefined {
  if (!data || typeof data !== "object") {
    return undefined;
  }
  const maybeArtifacts = (data as { artifacts?: unknown }).artifacts;
  if (maybeArtifacts && typeof maybeArtifacts === "object") {
    return maybeArtifacts as ChatArtifacts;
  }
  return undefined;
}

function collectArtifactKeys(artifacts: ChatArtifacts | undefined, raw: ChatResponseRaw): string[] {
  if (Array.isArray(raw.artifacts_keys)) {
    return raw.artifacts_keys.filter((x): x is string => typeof x === "string");
  }
  if (artifacts && typeof artifacts === "object") {
    return Object.keys(artifacts);
  }
  return [];
}

function normalizeChatResult(raw: ChatResponseRaw, fallbackSessionId: string): ChatResponseNormalized {
  const runtimeFromData =
    raw.data && typeof raw.data === "object"
      ? (raw.data as { runtime?: ChatResponseNormalized["runtime"] }).runtime
      : undefined;
  const artifacts = normalizeArtifacts(raw.data);

  return {
    success: Boolean(raw.success),
    message: typeof raw.message === "string" ? raw.message : "",
    sessionId: typeof raw.session_id === "string" ? raw.session_id : fallbackSessionId,
    timestamp: typeof raw.timestamp === "string" ? raw.timestamp : new Date().toISOString(),
    runtime: raw.runtime ?? runtimeFromData,
    artifacts,
    artifactsKeys: collectArtifactKeys(artifacts, raw)
  };
}

function parseEventBlock(block: string): { eventName: string; payload: ChatStreamEvent } | null {
  const trimmed = block.trim();
  if (!trimmed || trimmed.startsWith(":")) {
    return null;
  }

  let eventName = "message";
  const dataParts: string[] = [];

  for (const line of trimmed.split("\n")) {
    if (line.startsWith(":")) {
      continue;
    }
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim() || "message";
      continue;
    }
    if (line.startsWith("data:")) {
      dataParts.push(line.slice(5).trimStart());
    }
  }

  if (dataParts.length === 0) {
    return null;
  }

  const rawData = dataParts.join("\n");
  let payload: ChatStreamEvent;
  try {
    const parsed = JSON.parse(rawData) as ChatStreamEvent;
    payload = typeof parsed === "object" && parsed ? parsed : { type: eventName, value: parsed };
  } catch {
    payload = { type: eventName, message: rawData };
  }

  if (typeof payload.type !== "string" || !payload.type) {
    payload.type = eventName;
  }

  return { eventName, payload };
}

export async function postChat(payload: ChatRequest): Promise<ChatResponseNormalized> {
  const raw = await fetchJson<ChatResponseRaw>("/chat", {
    method: "POST",
    body: JSON.stringify(payload)
  });
  return normalizeChatResult(raw, payload.session_id);
}

export async function streamChat(
  payload: ChatRequest,
  callbacks?: StreamCallbacks,
): Promise<ChatResponseNormalized> {
  const { baseUrl, accessKey, timeoutMs } = getApiSettings();
  const headers = buildDefaultHeaders(accessKey);
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${baseUrl}/chat/stream`, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!response.ok) {
      const text = await response.text();
      throw new HttpError(response.status, text || response.statusText || "stream request failed");
    }

    if (!response.body) {
      throw new HttpError(502, "Empty stream body");
    }

    const decoder = new TextDecoder();
    const reader = response.body.getReader();
    let buffer = "";
    let completion: ChatResponseNormalized | null = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

      while (true) {
        const sep = buffer.indexOf("\n\n");
        if (sep < 0) {
          break;
        }
        const block = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);

        const parsed = parseEventBlock(block);
        if (!parsed) {
          continue;
        }

        const { eventName, payload: eventPayload } = parsed;
        callbacks?.onEvent?.(eventName, eventPayload);

        if (eventName === "connected") {
          callbacks?.onConnected?.(eventPayload);
          continue;
        }
        if (eventName === "status") {
          callbacks?.onStatus?.(eventPayload);
          continue;
        }
        if (eventName === "step") {
          callbacks?.onStep?.(eventPayload);
          continue;
        }
        if (eventName === "error") {
          callbacks?.onError?.(eventPayload);
          const statusCode = Number(eventPayload.status_code ?? 500);
          const message = String(eventPayload.message ?? "stream error");
          throw new HttpError(statusCode, message);
        }
        if (eventName === "completion") {
          const finalResult = eventPayload.final_result;
          if (finalResult && typeof finalResult === "object") {
            completion = normalizeChatResult(
              {
                ...(finalResult as ChatResponseRaw),
                session_id: String(eventPayload.session_id ?? payload.session_id),
                timestamp: String(eventPayload.timestamp ?? new Date().toISOString()),
              },
              payload.session_id,
            );
            callbacks?.onCompletion?.(completion, eventPayload);
          }
        }
      }
    }

    if (!completion) {
      throw new HttpError(502, "Stream finished without completion event");
    }

    return completion;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new HttpError(408, `Stream timed out after ${Math.round(timeoutMs / 1000)}s`);
    }
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
}
