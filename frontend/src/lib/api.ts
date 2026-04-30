import type { ChatRequest, ChatResponseNormalized, ChatResponseRaw, PublicOutputs, ReportExports } from "../types/api";
import { buildDefaultHeaders, fetchJson, getApiSettings, HttpError } from "./http";

export type ChatStreamEvent = {
  type: string;
  [key: string]: unknown;
};

export type StreamCallbacks = {
  onConnected?: (event: ChatStreamEvent) => void;
  onStatus?: (event: ChatStreamEvent) => void;
  onStep?: (event: ChatStreamEvent) => void;
  onDelta?: (event: ChatStreamEvent) => void;
  onError?: (event: ChatStreamEvent) => void;
  onEvent?: (eventName: string, event: ChatStreamEvent) => void;
  onCompletion?: (response: ChatResponseNormalized, event: ChatStreamEvent) => void;
};

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : undefined;
}

function coerceReportExports(value: unknown): ReportExports | undefined {
  const record = asRecord(value);
  if (!record) {
    return undefined;
  }
  const exports: ReportExports = {};
  if (typeof record.docx_path === "string" && record.docx_path.trim()) {
    exports.docx_path = record.docx_path;
  }
  if (typeof record.pdf_path === "string" && record.pdf_path.trim()) {
    exports.pdf_path = record.pdf_path;
  }
  return Object.keys(exports).length > 0 ? exports : undefined;
}

function normalizeOutputs(data: ChatResponseRaw["data"] | undefined): PublicOutputs | undefined {
  if (!data || typeof data !== "object") {
    return undefined;
  }
  const maybeOutputs = asRecord((data as { outputs?: unknown }).outputs);
  if (maybeOutputs) {
    return maybeOutputs as PublicOutputs;
  }

  const legacyArtifacts = asRecord((data as { artifacts?: unknown }).artifacts);
  const reportExports = coerceReportExports(legacyArtifacts?.report_exports);
  if (reportExports) {
    return { report_exports: reportExports };
  }
  return undefined;
}

function collectOutputKeys(outputs: PublicOutputs | undefined, raw: ChatResponseRaw): string[] {
  if (Array.isArray(raw.outputs_keys)) {
    return raw.outputs_keys.filter((x): x is string => typeof x === "string");
  }
  if (outputs && typeof outputs === "object") {
    return Object.keys(outputs);
  }
  return [];
}

function normalizeChatResult(raw: ChatResponseRaw, fallbackSessionId: string): ChatResponseNormalized {
  const runtimeFromData =
    raw.data && typeof raw.data === "object"
      ? (raw.data as { runtime?: ChatResponseNormalized["runtime"] }).runtime
      : undefined;
  const outputs = normalizeOutputs(raw.data);

  return {
    success: Boolean(raw.success),
    message: typeof raw.message === "string" ? raw.message : "",
    sessionId: typeof raw.session_id === "string" ? raw.session_id : fallbackSessionId,
    timestamp: typeof raw.timestamp === "string" ? raw.timestamp : new Date().toISOString(),
    runtime: raw.runtime ?? runtimeFromData,
    outputs,
    outputKeys: collectOutputKeys(outputs, raw)
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

export async function deleteSession(sessionId: string): Promise<void> {
  await fetchJson<{ success: boolean; message: string }>(`/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
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
        if (eventName === "delta") {
          callbacks?.onDelta?.(eventPayload);
          // Yield once so React can paint incremental text even when multiple
          // delta events arrive in one network chunk.
          await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
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
