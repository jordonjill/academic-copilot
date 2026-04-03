export type SessionItem = {
  id: string;
  title: string;
  createdAt: string;
};

const SESSIONS_KEY = "acp_sessions_v1";

function fallbackSession(): SessionItem {
  const id = `s_${Date.now()}`;
  return {
    id,
    title: "新会话",
    createdAt: new Date().toISOString()
  };
}

export function newSessionId(): string {
  return `s_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

export function loadSessions(): SessionItem[] {
  const raw = localStorage.getItem(SESSIONS_KEY);
  if (!raw) {
    const seed = [fallbackSession()];
    saveSessions(seed);
    return seed;
  }
  try {
    const parsed = JSON.parse(raw) as SessionItem[];
    if (!Array.isArray(parsed) || parsed.length === 0) {
      const seed = [fallbackSession()];
      saveSessions(seed);
      return seed;
    }
    return parsed.filter((s) => s && typeof s.id === "string");
  } catch {
    const seed = [fallbackSession()];
    saveSessions(seed);
    return seed;
  }
}

export function saveSessions(sessions: SessionItem[]): void {
  localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
}
