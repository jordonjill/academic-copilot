import type { ChatMessage } from "../../types/api";

type Props = {
  messages: ChatMessage[];
  pending: boolean;
  pendingText?: string;
  streamingAssistantText?: string;
};

export function MessageList({ messages, pending, pendingText, streamingAssistantText }: Props) {
  return (
    <div className="message-list">
      {messages.length === 0 ? (
        <div className="empty-state">开始输入问题，或选择 workflow 运行完整流程。</div>
      ) : null}
      {messages.map((message) => (
        <article
          key={message.id}
          className={`message-bubble ${message.role} ${message.isError ? "error" : ""}`.trim()}
        >
          <header>
            <span className="role">{message.role === "user" ? "User" : "Assistant"}</span>
            <time>{new Date(message.timestamp).toLocaleString()}</time>
          </header>
          <pre>{message.text}</pre>
          {message.runtime ? (
            <div className="message-meta">
              mode={message.runtime.mode} step={message.runtime.step_count} loop={message.runtime.loop_count}
              {message.runtime.current_node ? ` node=${message.runtime.current_node}` : ""}
            </div>
          ) : null}
          {message.artifactsKeys && message.artifactsKeys.length > 0 ? (
            <div className="message-meta">artifacts: {message.artifactsKeys.join(", ")}</div>
          ) : null}
        </article>
      ))}
      {pending && streamingAssistantText && streamingAssistantText.trim() ? (
        <article className="message-bubble assistant">
          <header>
            <span className="role">Assistant</span>
            <time>{new Date().toLocaleString()}</time>
          </header>
          <pre>{streamingAssistantText}</pre>
          <div className="message-meta">streaming...</div>
        </article>
      ) : null}
      {pending ? <div className="pending-indicator">{pendingText || "Assistant is processing..."}</div> : null}
    </div>
  );
}
