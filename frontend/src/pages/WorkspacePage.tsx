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
import type { ChatMessage, PublicOutputs, RuntimeInfo, WorkflowId } from "../types/api";

type MessageBucket = Record<string, ChatMessage[]>;
type RuntimeBucket = Record<string, RuntimeInfo | undefined>;
type OutputBucket = Record<string, PublicOutputs | undefined>;
type StreamBucket = Record<string, string | undefined>;
const MESSAGES_KEY = "acp_messages_v1";
const RUNTIME_KEY = "acp_runtime_v1";
const OUTPUTS_KEY = "acp_outputs_v1";
const LEGACY_ARTIFACTS_KEY = "acp_artifacts_v1";

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
  const [outputsBySession, setOutputsBySession] = useState<OutputBucket>(() => loadJsonObject(OUTPUTS_KEY, {}));
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
    localStorage.removeItem(LEGACY_ARTIFACTS_KEY);
  }, []);

  useEffect(() => {
    localStorage.setItem(OUTPUTS_KEY, JSON.stringify(outputsBySession));
  }, [outputsBySession]);

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
    setOutputsBySession((prev) => ({ ...prev, [sessionId]: undefined }));
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
    setOutputsBySession((prev) => {
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
    const workflowId: WorkflowId | null = workflowMode === "direct" ? null : workflowMode;
    maybeUpdateSessionTitle(sid, text);

    setRuntimeBySession((prev) => ({
      ...prev,
      [sid]: {
        mode: workflowId ? "workflow" : "dynamic",
        workflow_id: workflowId,
        current_node: null,
        step_count: 0,
        loop_count: 0,
        status: "running",
      },
    }));
    setOutputsBySession((prev) => ({ ...prev, [sid]: undefined }));

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
            const eventMode = typeof event.mode === "string" ? event.mode : undefined;
            const eventWorkflowId =
              typeof event.workflow_id === "string"
                ? event.workflow_id
                : event.workflow_id === null
                  ? null
                  : undefined;
            const eventCurrentNode =
              typeof event.current_node === "string"
                ? event.current_node
                : event.current_node === null
                  ? null
                  : undefined;
            const eventStatus = typeof event.status === "string" ? event.status : undefined;
            const hasRuntimeUpdate = [
              "mode",
              "workflow_id",
              "current_node",
              "step_count",
              "max_steps",
              "loop_count",
              "max_loops",
              "status",
            ].some((key) => key in event);
            if (hasRuntimeUpdate) {
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
                  mode: eventMode ?? prev[sid]?.mode ?? (workflowId ? "workflow" : "dynamic"),
                  workflow_id:
                    eventWorkflowId !== undefined
                      ? eventWorkflowId
                      : prev[sid]?.workflow_id ?? workflowId ?? null,
                  current_node:
                    eventCurrentNode !== undefined
                      ? eventCurrentNode
                      : prev[sid]?.current_node ?? null,
                  step_count: Number(event.step_count ?? prev[sid]?.step_count ?? 0),
                  max_steps: Number(event.max_steps ?? prev[sid]?.max_steps ?? 0) || undefined,
                  loop_count: Number(event.loop_count ?? prev[sid]?.loop_count ?? 0),
                  max_loops: Number(event.max_loops ?? prev[sid]?.max_loops ?? 0) || undefined,
                  status: eventStatus ?? prev[sid]?.status ?? "running",
                },
              }));
            }
          },
          onStep: (event) => {
            const stepNumber = Number(event.step_number ?? 0);
            const nodeName = String(event.node_name ?? "");
            const nextNode = String(event.next_node ?? "");
            const eventMode = typeof event.mode === "string" ? event.mode : undefined;
            const eventWorkflowId =
              typeof event.workflow_id === "string"
                ? event.workflow_id
                : event.workflow_id === null
                  ? null
                  : undefined;
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
                mode: eventMode ?? prev[sid]?.mode ?? (workflowId ? "workflow" : "dynamic"),
                workflow_id:
                  eventWorkflowId !== undefined
                    ? eventWorkflowId
                    : prev[sid]?.workflow_id ?? workflowId ?? null,
                current_node: nodeName || (prev[sid]?.current_node ?? null),
                step_count: stepNumber > 0 ? stepNumber : prev[sid]?.step_count ?? 0,
                max_steps: Number(event.max_steps ?? prev[sid]?.max_steps ?? 0) || undefined,
                loop_count: Number(event.loop_count ?? prev[sid]?.loop_count ?? 0),
                max_loops: Number(event.max_loops ?? prev[sid]?.max_loops ?? 0) || undefined,
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
      const outputs = response.outputs;
      if (runtime) {
        setRuntimeBySession((prev) => ({ ...prev, [sid]: runtime }));
      }
      if (outputs && Object.keys(outputs).length > 0) {
        setOutputsBySession((prev) => ({ ...prev, [sid]: outputs }));
      } else {
        setOutputsBySession((prev) => ({ ...prev, [sid]: undefined }));
      }

      appendMessage(sid, {
        id: `${sid}_a_${Date.now()}`,
        role: "assistant",
        text: response.message || streamedText || "(empty response)",
        timestamp: response.timestamp ?? new Date().toISOString(),
        runtime,
        outputs,
        outputKeys: response.outputKeys
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
        <ExportPanel outputs={outputsBySession[activeSessionId]} />
      </aside>
    </div>
  );
}
