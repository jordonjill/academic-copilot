import type { SessionItem } from "./sessionStore";

type Props = {
  sessions: SessionItem[];
  activeSessionId: string;
  onSelect: (sessionId: string) => void;
  onCreate: () => void;
};

export function SessionSidebar({ sessions, activeSessionId, onSelect, onCreate }: Props) {
  return (
    <aside className="session-sidebar panel">
      <header>
        <h2>Sessions</h2>
        <button type="button" onClick={onCreate}>
          New
        </button>
      </header>
      <div className="session-list">
        {sessions.map((session) => (
          <button
            key={session.id}
            type="button"
            className={session.id === activeSessionId ? "session-item active" : "session-item"}
            onClick={() => onSelect(session.id)}
          >
            <span className="session-title">{session.title}</span>
            <span className="session-id">{session.id}</span>
          </button>
        ))}
      </div>
    </aside>
  );
}
