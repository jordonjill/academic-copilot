import type { SessionItem } from "./sessionStore";

type Props = {
  sessions: SessionItem[];
  activeSessionId: string;
  onSelect: (sessionId: string) => void;
  onCreate: () => void;
  onDelete: (sessionId: string) => void;
  deleteDisabled?: boolean;
};

export function SessionSidebar({
  sessions,
  activeSessionId,
  onSelect,
  onCreate,
  onDelete,
  deleteDisabled = false,
}: Props) {
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
          <div
            key={session.id}
            className={session.id === activeSessionId ? "session-item active" : "session-item"}
          >
            <button type="button" className="session-select" onClick={() => onSelect(session.id)}>
              <span className="session-title">{session.title}</span>
              <span className="session-id">{session.id}</span>
            </button>
            <button
              type="button"
              className="session-delete"
              aria-label={`Delete session ${session.title}`}
              disabled={deleteDisabled}
              onClick={() => onDelete(session.id)}
            >
              <span aria-hidden="true">x</span>
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}
