export class HttpError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(`HTTP ${status}: ${detail}`);
    this.status = status;
    this.detail = detail;
  }
}

const DEFAULT_TIMEOUT_MS = 650000;

export type ApiSettings = {
  baseUrl: string;
  accessKey: string;
  timeoutMs: number;
};

export function getApiSettings(timeoutOverrideMs?: number): ApiSettings {
  const baseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() || "http://127.0.0.1:8000";
  const accessKey = (import.meta.env.VITE_ACCESS_KEY as string | undefined)?.trim() || "";
  const timeoutMs = timeoutOverrideMs ?? Number(import.meta.env.VITE_REQUEST_TIMEOUT_MS ?? DEFAULT_TIMEOUT_MS);
  return { baseUrl, accessKey, timeoutMs };
}

export function buildDefaultHeaders(accessKey: string): Headers {
  const headers = new Headers();
  headers.set("Content-Type", "application/json");
  if (accessKey) {
    headers.set("Authorization", `Bearer ${accessKey}`);
  }
  return headers;
}

export async function fetchJson<T>(
  path: string,
  init: RequestInit,
  options?: { timeoutMs?: number }
): Promise<T> {
  const { baseUrl, accessKey, timeoutMs } = getApiSettings(options?.timeoutMs);

  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const headers = buildDefaultHeaders(accessKey);
    if (init.headers) {
      const extra = new Headers(init.headers);
      for (const [key, value] of extra.entries()) {
        headers.set(key, value);
      }
    }

    const response = await fetch(`${baseUrl}${path}`, {
      ...init,
      headers,
      signal: controller.signal
    });

    const rawText = await response.text();
    let payload: unknown = null;
    if (rawText.trim()) {
      try {
        payload = JSON.parse(rawText);
      } catch {
        payload = { detail: rawText };
      }
    }

    if (!response.ok) {
      const detailFromObject =
        typeof payload === "object" && payload !== null
          ? String((payload as Record<string, unknown>).detail || (payload as Record<string, unknown>).message || "")
          : "";
      throw new HttpError(response.status, detailFromObject || response.statusText || "Request failed");
    }

    return payload as T;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new HttpError(408, `Request timed out after ${Math.round(timeoutMs / 1000)}s`);
    }
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
}
