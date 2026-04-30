import { useEffect, useMemo, useState } from "react";

import { ExportPanel } from "../features/artifacts/ExportPanel";
import { ChatInput } from "../features/chat/ChatInput";
import { MessageList } from "../features/chat/MessageList";
import { RuntimePanel } from "../features/runtime/RuntimePanel";
import { SessionSidebar } from "../features/sessions/SessionSidebar";
import { loadSessions, newSessionId, saveSessions, type SessionItem } from "../features/sessions/sessionStore";
import { WorkflowSelector, type WorkflowMode } from "../features/workflow/WorkflowSelector";
import { deleteSession, streamChat } from "../lib/api";
import { HttpError } from "../lib/http";
import type { ChatArtifacts, ChatMessage, RuntimeInfo, WorkflowId } from "../types/api";

type MessageBucket = Record<string, ChatMessage[]>;
type RuntimeBucket = Record<string, RuntimeInfo | undefined>;
type ArtifactBucket = Record<string, ChatArtifacts | undefined>;
type StreamBucket = Record<string, string | undefined>;
const MESSAGES_KEY = "acp_messages_v1";
const RUNTIME_KEY = "acp_runtime_v1";
const ARTIFACTS_KEY = "acp_artifacts_v1";

function loadJsonObject<T>(key: string, fallback: T): T {
  const raw = localStorage.getItem(key);
  if (!raw) {
    return fallback;
  }
  try {
    const parsed = JSON.parse(raw) as T;
    return parsed;
  } catch {
    return fallback;
  }
}

function createAssistantError(sessionId: string, text: string): ChatMessage {
  return {
    id: `${sessionId}_err_${Date.now()}`,
    role: "assistant",
    text,
    timestamp: new Date().toISOString(),
    isError: true
  };
}

export function WorkspacePage() {
  const [sessions, setSessions] = useState<SessionItem[]>(() => loadSessions());
  const [activeSessionId, setActiveSessionId] = useState<string>(() => loadSessions()[0]?.id ?? newSessionId());
  const [messagesBySession, setMessagesBySession] = useState<MessageBucket>(() => loadJsonObject(MESSAGES_KEY, {}));
  const [runtimeBySession, setRuntimeBySession] = useState<RuntimeBucket>(() => loadJsonObject(RUNTIME_KEY, {}));
  const [artifactsBySession, setArtifactsBySession] = useState<ArtifactBucket>(() => loadJsonObject(ARTIFACTS_KEY, {}));
  const [streamingTextBySession, setStreamingTextBySession] = useState<StreamBucket>({});
  const [pending, setPending] = useState(false);
  const [pendingText, setPendingText] = useState("");
  const [workflowMode, setWorkflowMode] = useState<WorkflowMode>("direct");

  const userId = (import.meta.env.VITE_DEFAULT_USER_ID as string | undefined)?.trim() || "u_demo";

  const activeMessages = useMemo(() => messagesBySession[activeSessionId] ?? [], [messagesBySession, activeSessionId]);
  const canRetry = useMemo(
    () => activeMessages.some((m) => m.role === "user" && m.text.trim().length > 0) && !pending,
    [activeMessages, pending]
  );

  useEffect(() => {
    localStorage.setItem(MESSAGES_KEY, JSON.stringify(messagesBySession));
  }, [messagesBySession]);

  useEffect(() => {
    localStorage.setItem(RUNTIME_KEY, JSON.stringify(runtimeBySession));
  }, [runtimeBySession]);

  useEffect(() => {
    localStorage.setItem(ARTIFACTS_KEY, JSON.stringify(artifactsBySession));
  }, [artifactsBySession]);

  function updateSessions(next: SessionItem[]) {
    setSessions(next);
    saveSessions(next);
  }

  function onCreateSession() {
    const id = newSessionId();
    const next: SessionItem[] = [
      {
        id,
        title: "新会话",
        createdAt: new Date().toISOString()
      },
      ...sessions
    ];
    updateSessions(next);
    setActiveSessionId(id);
  }

  function appendMessage(sessionId: string, message: ChatMessage) {
    setMessagesBySession((prev) => {
      const current = prev[sessionId] ?? [];
      return {
        ...prev,
        [sessionId]: [...current, message]
      };
    });
  }

  function clearSession(sessionId: string) {
    setMessagesBySession((prev) => ({ ...prev, [sessionId]: [] }));
    setRuntimeBySession((prev) => ({ ...prev, [sessionId]: undefined }));
    setArtifactsBySession((prev) => ({ ...prev, [sessionId]: undefined }));
    setStreamingTextBySession((prev) => ({ ...prev, [sessionId]: undefined }));
  }

  function removeSessionBuckets(sessionId: string) {
    setMessagesBySession((prev) => {
      const next = { ...prev };
      delete next[sessionId];
      return next;
    });
    setRuntimeBySession((prev) => {
      const next = { ...prev };
      delete next[sessionId];
      return next;
    });
    setArtifactsBySession((prev) => {
      const next = { ...prev };
      delete next[sessionId];
      return next;
    });
    setStreamingTextBySession((prev) => {
      const next = { ...prev };
      delete next[sessionId];
      return next;
    });
  }

  function onDeleteSession(sessionId: string) {
    if (pending) {
      return;
    }
    const session = sessions.find((item) => item.id === sessionId);
    const confirmed = window.confirm(`Delete session "${session?.title ?? sessionId}"?`);
    if (!confirmed) {
      return;
    }

    const remaining = sessions.filter((item) => item.id !== sessionId);
    const nextSessions =
      remaining.length > 0
        ? remaining
        : [
            {
              id: newSessionId(),
              title: "新会话",
              createdAt: new Date().toISOString(),
            },
          ];
    updateSessions(nextSessions);
    removeSessionBuckets(sessionId);
    if (activeSessionId === sessionId) {
      setActiveSessionId(nextSessions[0].id);
    }
    void deleteSession(sessionId).catch((error) => {
      console.warn("Failed to delete backend session", error);
    });
  }

  async function retryLastUserTurn() {
    const lastUser = [...activeMessages].reverse().find((m) => m.role === "user" && m.text.trim());
    if (!lastUser) {
      return;
    }
    await onSend(lastUser.text);
  }

  function maybeUpdateSessionTitle(sessionId: string, firstUserText: string) {
    const title = firstUserText.slice(0, 24);
    const next = sessions.map((s) => {
      if (s.id !== sessionId) {
        return s;
      }
      if (s.title !== "新会话") {
        return s;
      }
      return {
        ...s,
        title: title || s.title
      };
    });
    updateSessions(next);
  }

  async function onSend(text: string) {
    if (pending) {
      return;
    }
    const sid = activeSessionId;
    maybeUpdateSessionTitle(sid, text);

    appendMessage(sid, {
      id: `${sid}_u_${Date.now()}`,
      role: "user",
      text,
      timestamp: new Date().toISOString()
    });

    setPending(true);
    setPendingText("Connecting stream...");
    setStreamingTextBySession((prev) => ({ ...prev, [sid]: "" }));
    try {
      let streamedText = "";
      const workflowId: WorkflowId | null = workflowMode === "direct" ? null : workflowMode;
      const response = await streamChat(
        {
          message: text,
          user_id: userId,
          session_id: sid,
          workflow_id: workflowId
        },
        {
          onConnected: () => {
            setPendingText("Stream connected. Waiting for model output...");
          },
          onStatus: (event) => {
            const message = String(event.message ?? "");
            setPendingText(message || "Running...");
          },
          onStep: (event) => {
            const stepNumber = Number(event.step_number ?? 0);
            const nodeName = String(event.node_name ?? "");
            const nextNode = String(event.next_node ?? "");
            setRuntimeBySession((prev) => ({
              ...prev,
              [sid]: {
                ...(prev[sid] ?? {
                  mode: workflowId ? "workflow" : "dynamic",
                  workflow_id: workflowId ?? null,
                  current_node: null,
                  step_count: 0,
                  loop_count: 0,
                  status: "running",
                }),
                current_node: nodeName || (prev[sid]?.current_node ?? null),
                step_count: stepNumber > 0 ? stepNumber : prev[sid]?.step_count ?? 0,
                status: "running",
              },
            }));
            if (stepNumber > 0 && nodeName) {
              setPendingText(`Step ${stepNumber}: ${nodeName}${nextNode ? ` -> ${nextNode}` : ""}`);
            }
          },
          onDelta: (event) => {
            const delta = String(event.delta ?? "");
            if (!delta) {
              return;
            }
            streamedText += delta;
            setStreamingTextBySession((prev) => ({ ...prev, [sid]: streamedText }));
          },
        },
      );

      const runtime = response.runtime;
      const artifacts = response.artifacts;
      if (runtime) {
        setRuntimeBySession((prev) => ({ ...prev, [sid]: runtime }));
      }
      if (artifacts) {
        setArtifactsBySession((prev) => ({ ...prev, [sid]: artifacts }));
      }

      appendMessage(sid, {
        id: `${sid}_a_${Date.now()}`,
        role: "assistant",
        text: response.message || streamedText || "(empty response)",
        timestamp: response.timestamp ?? new Date().toISOString(),
        runtime,
        artifacts,
        artifactsKeys: response.artifactsKeys
      });
    } catch (error) {
      if (error instanceof HttpError) {
        const detail =
          error.status === 504
            ? "Workflow/模型调用超时（504）。可直接重试本 session。"
            : error.status === 408
              ? "流式连接超时（408）。可直接重试本 session。"
            : `请求失败 (${error.status}): ${error.detail || "unknown error"}`;
        appendMessage(sid, createAssistantError(sid, detail));
      } else {
        appendMessage(sid, createAssistantError(sid, `请求异常: ${String(error)}`));
      }
    } finally {
      setPending(false);
      setPendingText("");
      setStreamingTextBySession((prev) => ({ ...prev, [sid]: undefined }));
    }
  }

  return (
    <div className="workspace-root">
      <SessionSidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelect={setActiveSessionId}
        onCreate={onCreateSession}
        onDelete={onDeleteSession}
        deleteDisabled={pending}
      />

      <main className="chat-main panel">
        <header className="chat-header">
          <div>
            <h1>Academic Copilot Workspace</h1>
            <p>
              user_id=<code>{userId}</code> session_id=<code>{activeSessionId}</code>
            </p>
          </div>
          <div className="header-controls">
            <WorkflowSelector value={workflowMode} onChange={setWorkflowMode} />
            <div className="action-row">
              <button type="button" className="ghost-btn" onClick={retryLastUserTurn} disabled={!canRetry}>
                Retry Last
              </button>
              <button type="button" className="ghost-btn" onClick={() => clearSession(activeSessionId)} disabled={pending}>
                Clear Session
              </button>
            </div>
          </div>
        </header>

        <MessageList
          messages={activeMessages}
          pending={pending}
          pendingText={pendingText}
          streamingAssistantText={streamingTextBySession[activeSessionId] ?? ""}
        />
        <ChatInput disabled={pending} onSend={onSend} />
      </main>

      <aside className="side-main">
        <RuntimePanel runtime={runtimeBySession[activeSessionId]} />
        <ExportPanel artifacts={artifactsBySession[activeSessionId]} />
      </aside>
    </div>
  );
}
