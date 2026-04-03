export type WorkflowId = "lit_review_v1" | "proposal_v2";

export type ChatRequest = {
  message: string;
  user_id: string;
  session_id: string;
  workflow_id?: WorkflowId | null;
};

export type RuntimeInfo = {
  mode: string;
  workflow_id: string | null;
  current_node: string | null;
  step_count: number;
  loop_count: number;
  status: string;
  tool_budget?: {
    scope: string;
    workflow_id: string | null;
    limits: Record<string, number>;
    counts: Record<string, number>;
  };
};

export type ReportExports = {
  docx_path?: string;
  pdf_path?: string;
};

export type ChatArtifacts = {
  report_exports?: ReportExports;
  [key: string]: unknown;
};

export type ChatResponseRaw = {
  success: boolean;
  type?: string;
  message?: string | null;
  session_id?: string;
  timestamp?: string;
  data?: {
    runtime?: RuntimeInfo;
    artifacts?: ChatArtifacts;
  } | Record<string, unknown> | null;
  runtime?: RuntimeInfo;
  artifacts_keys?: string[];
};

export type ChatResponseNormalized = {
  success: boolean;
  message: string;
  sessionId: string;
  timestamp?: string;
  runtime?: RuntimeInfo;
  artifacts?: ChatArtifacts;
  artifactsKeys: string[];
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  timestamp: string;
  runtime?: RuntimeInfo;
  artifacts?: ChatArtifacts;
  artifactsKeys?: string[];
  isError?: boolean;
};
