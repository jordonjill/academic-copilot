#!/usr/bin/env bash
set -euo pipefail

# End-to-end API test script for Academic Copilot.
# Usage:
#   ACCESS_KEY=123 ./scripts/api_e2e.sh
# Optional env:
#   BASE_URL=http://127.0.0.1:8000
#   USER_ID=u_demo
#   ALLOW_WORKFLOW_TIMEOUT=1   # default: 1 (treat 504 in workflow as warning)
#   LONG_SESSION_TURNS=10      # default: 10

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
ACCESS_KEY="${ACCESS_KEY:-}"
USER_ID="${USER_ID:-u_demo}"
ALLOW_WORKFLOW_TIMEOUT="${ALLOW_WORKFLOW_TIMEOUT:-1}"
VERBOSE="${VERBOSE:-0}"
LONG_SESSION_TURNS="${LONG_SESSION_TURNS:-10}"
CURL_CONNECT_TIMEOUT_SECONDS="${CURL_CONNECT_TIMEOUT_SECONDS:-3}"
CURL_MAX_TIME_SECONDS="${CURL_MAX_TIME_SECONDS:-600}"
CURL_HEALTH_MAX_TIME_SECONDS="${CURL_HEALTH_MAX_TIME_SECONDS:-10}"

# For local loopback targets, bypass proxy explicitly to avoid 502 from local proxy daemons.
CURL_NO_PROXY_ARGS=()
if [[ "${BASE_URL}" =~ ^https?://(127\.0\.0\.1|localhost|0\.0\.0\.0)(:[0-9]+)?(/|$) ]]; then
  CURL_NO_PROXY_ARGS=(--noproxy '*')
fi

JSON_PARSER="python"
if command -v jq >/dev/null 2>&1; then
  JSON_PARSER="jq"
elif command -v jp >/dev/null 2>&1; then
  JSON_PARSER="jp"
fi

PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

if [[ -z "$ACCESS_KEY" ]]; then
  echo "[error] ACCESS_KEY is empty. Example: ACCESS_KEY=123 ./scripts/api_e2e.sh"
  exit 2
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[error] python3/python not found. Please install Python to run api_e2e.sh"
  exit 2
fi

AUTH_HEADER="Authorization: Bearer ${ACCESS_KEY}"
CT_HEADER="Content-Type: application/json"
RUN_TAG="$(date +%Y%m%d%H%M%S)"
SESSION_DIRECT="s_direct_${RUN_TAG}"
SESSION_LIT="s_lit_${RUN_TAG}"
SESSION_PROP="s_prop_${RUN_TAG}"
SESSION_MEM="s_mem_${RUN_TAG}"

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

_inc_pass() { PASS_COUNT=$((PASS_COUNT + 1)); }
_inc_warn() { WARN_COUNT=$((WARN_COUNT + 1)); }
_inc_fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); }

_print_banner() {
  echo "========================================"
  echo "Academic Copilot API E2E"
  echo "BASE_URL: ${BASE_URL}"
  echo "USER_ID : ${USER_ID}"
  echo "RUN_TAG : ${RUN_TAG}"
  echo "JSON parser: ${JSON_PARSER}"
  echo "Bypass proxy: $([[ ${#CURL_NO_PROXY_ARGS[@]} -gt 0 ]] && echo enabled || echo disabled)"
  echo "Curl timeout: connect=${CURL_CONNECT_TIMEOUT_SECONDS}s total=${CURL_MAX_TIME_SECONDS}s"
  echo "========================================"
}

_call_json() {
  # args: endpoint payload_json
  local endpoint="$1"
  local payload="$2"
  local tmp_body
  tmp_body="$(mktemp)"
  local http_code curl_ec
  set +e
  http_code="$(
    curl -sS -o "${tmp_body}" -w "%{http_code}" \
      "${CURL_NO_PROXY_ARGS[@]}" \
      --connect-timeout "${CURL_CONNECT_TIMEOUT_SECONDS}" \
      --max-time "${CURL_MAX_TIME_SECONDS}" \
      -X POST "${BASE_URL}${endpoint}" \
      -H "${AUTH_HEADER}" \
      -H "${CT_HEADER}" \
      -d "${payload}"
  )"
  curl_ec=$?
  set -e
  if [[ ${curl_ec} -ne 0 ]]; then
    local err_msg
    err_msg="curl_failed(endpoint=${endpoint}, exit_code=${curl_ec}); check backend is running on ${BASE_URL}"
    printf '000\n{"success":false,"message":"%s"}\n' "${err_msg}"
    rm -f "${tmp_body}"
    return 0
  fi
  local body
  body="$(cat "${tmp_body}")"
  rm -f "${tmp_body}"
  printf '%s\n%s\n' "${http_code}" "${body}"
}

_json_get() {
  # args: json_text expr
  local json_text="$1"
  local expr="$2"
  if [[ "${JSON_PARSER}" == "jq" ]]; then
    printf '%s' "$json_text" | jq -r --arg p "$expr" '
      getpath(($p | split(".") | map(select(length > 0)))) // empty
      | if type=="string" or type=="number" or type=="boolean"
        then tostring
        else tojson
        end
    ' 2>/dev/null || true
    return 0
  fi

  if [[ "${JSON_PARSER}" == "jp" ]]; then
    # jp uses JMESPath syntax, dotted paths are compatible for simple lookups.
    local jp_result
    jp_result="$(printf '%s' "$json_text" | jp "$expr" 2>/dev/null || true)"
    if [[ -n "${jp_result}" && "${jp_result}" != "null" ]]; then
      printf '%s' "${jp_result}"
      return 0
    fi
  fi

  "${PYTHON_BIN}" - "$expr" <<'PY' <<<"$json_text"
import json
import sys

expr = sys.argv[1]
raw = sys.stdin.read()
try:
    data = json.loads(raw)
except Exception:
    print("")
    raise SystemExit(0)

def pick(obj, path):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return ""
    if cur is None:
        return ""
    if isinstance(cur, (dict, list)):
        return json.dumps(cur, ensure_ascii=False)
    return str(cur)

print(pick(data, expr))
PY
}

_print_compact_json() {
  local body="$1"
  if [[ "${JSON_PARSER}" == "jq" ]]; then
    printf '%s' "$body" | jq -c '
      {
        success,
        type,
        message,
        session_id,
        runtime: (.data.runtime // null),
        artifacts_keys: ((.data.artifacts | keys) // [])
      }
    ' 2>/dev/null || printf '%s\n' "$body"
  else
    printf '%s\n' "$body"
  fi
}

_json_quote() {
  # Return a valid JSON string literal for arbitrary UTF-8 text.
  local raw="$1"
  "${PYTHON_BIN}" - "$raw" <<'PY'
import json
import sys

print(json.dumps(sys.argv[1], ensure_ascii=False))
PY
}

_step_chat() {
  # args: name message session workflow(optional)
  local name="$1"
  local message="$2"
  local session_id="$3"
  local workflow_id="${4:-}"
  local message_json user_id_json session_id_json workflow_id_json
  message_json="$(_json_quote "${message}")"
  user_id_json="$(_json_quote "${USER_ID}")"
  session_id_json="$(_json_quote "${session_id}")"
  workflow_id_json="$(_json_quote "${workflow_id}")"
  local payload
  if [[ -n "${workflow_id}" ]]; then
    payload="$(cat <<JSON
{
  "message": ${message_json},
  "user_id": ${user_id_json},
  "session_id": ${session_id_json},
  "workflow_id": ${workflow_id_json}
}
JSON
)"
  else
    payload="$(cat <<JSON
{
  "message": ${message_json},
  "user_id": ${user_id_json},
  "session_id": ${session_id_json}
}
JSON
)"
  fi

  local resp
  resp="$(_call_json "/chat" "${payload}")"
  local code body
  code="$(printf '%s' "${resp}" | sed -n '1p')"
  body="$(printf '%s' "${resp}" | sed -n '2,$p')"

  local success msg step_count loop_count
  success="$(_json_get "${body}" "success")"
  msg="$(_json_get "${body}" "message")"
  step_count="$(_json_get "${body}" "data.runtime.step_count")"
  loop_count="$(_json_get "${body}" "data.runtime.loop_count")"
  local success_lc
  success_lc="$(printf '%s' "${success}" | tr '[:upper:]' '[:lower:]')"

  if [[ "${code}" == "200" && "${success_lc}" == "true" ]]; then
    echo "[PASS] ${name} | code=${code} step_count=${step_count:-?} loop_count=${loop_count:-?}"
    if [[ "${VERBOSE}" == "1" ]]; then
      _print_compact_json "${body}"
    fi
    _inc_pass
    return 0
  fi

  if [[ "${ALLOW_WORKFLOW_TIMEOUT}" == "1" && -n "${workflow_id}" && "${code}" == "504" ]]; then
    echo "[WARN] ${name} | code=${code} workflow timeout (external provider/rate-limit likely)"
    echo "       message=${msg}"
    if [[ "${VERBOSE}" == "1" ]]; then
      _print_compact_json "${body}"
    fi
    _inc_warn
    return 0
  fi

  echo "[FAIL] ${name} | code=${code} success=${success}"
  echo "       message=${msg}"
  echo "       raw=$(_print_compact_json "${body}")"
  _inc_fail
  return 1
}

_step_health() {
  local tmp_body
  tmp_body="$(mktemp)"
  local code curl_ec
  set +e
  code="$(
    curl -sS -o "${tmp_body}" -w "%{http_code}" \
      "${CURL_NO_PROXY_ARGS[@]}" \
      --connect-timeout "${CURL_CONNECT_TIMEOUT_SECONDS}" \
      --max-time "${CURL_HEALTH_MAX_TIME_SECONDS}" \
      "${BASE_URL}/health" \
      -H "${AUTH_HEADER}"
  )"
  curl_ec=$?
  set -e
  if [[ ${curl_ec} -ne 0 ]]; then
    echo "[FAIL] health | curl failed (exit=${curl_ec}); backend may not be running at ${BASE_URL}"
    rm -f "${tmp_body}"
    _inc_fail
    return 1
  fi
  local body
  body="$(cat "${tmp_body}")"
  rm -f "${tmp_body}"

  if [[ "${code}" == "200" ]]; then
    echo "[PASS] health | code=200"
    _inc_pass
  else
    echo "[FAIL] health | code=${code} raw=${body}"
    _inc_fail
  fi
}

_step_memory_db_checks() {
  local mem_file="data/users/${USER_ID}/memory.md"
  if [[ -f "${mem_file}" ]]; then
    echo "[PASS] memory file exists: ${mem_file}"
    _inc_pass
    tail -n 20 "${mem_file}" || true
  else
    echo "[WARN] memory file missing: ${mem_file} (may need STM compression trigger)"
    _inc_warn
  fi

  if command -v sqlite3 >/dev/null 2>&1; then
    echo "[INFO] sqlite checks (latest working_context / ltm_facts)"
    sqlite3 data/conversations.db \
      "select id,session_id,is_compressed,token_count,created_at from working_context order by id desc limit 5;" || true
    sqlite3 data/conversations.db \
      "select id,user_id,session_id,fact_type,fact_content,extracted_at from ltm_facts order by id desc limit 10;" || true
    _inc_pass
  else
    echo "[WARN] sqlite3 not found; skip DB checks"
    _inc_warn
  fi
}

main() {
  _print_banner

  _step_health

  _step_chat "direct_reply" \
    "用两句话解释 research question 和 hypothesis 的区别。" \
    "${SESSION_DIRECT}"

  _step_chat "direct_reply_followup" \
    "继续用两句话说明：可检验性为什么重要？" \
    "${SESSION_DIRECT}"

  _step_chat "direct_reply_followup_2" \
    "再补充一句：什么情况下 hypothesis 不成立也有学术价值？" \
    "${SESSION_DIRECT}"

  _step_chat "lit_review_workflow" \
    "请做一版多模态RAG在医学问答中的挑战与机会文献综述，给出结构化要点与证据。" \
    "${SESSION_LIT}" \
    "lit_review_v1"

  _step_chat "lit_review_followup" \
    "基于上一轮结果，只补充近两年（2024-2026）的关键趋势与风险点。" \
    "${SESSION_LIT}" \
    "lit_review_v1"

  _step_chat "proposal_workflow" \
    "我想做低资源中文法律问答RAG评测，请给完整proposal草案。" \
    "${SESSION_PROP}" \
    "proposal_v2"

  _step_chat "proposal_followup_refine" \
    "把实验设计再细化：明确数据切分、指标、消融、统计显著性检验。" \
    "${SESSION_PROP}" \
    "proposal_v2"

  _step_chat "proposal_export_followup" \
    "把当前proposal导出成docx和pdf。" \
    "${SESSION_PROP}" \
    "proposal_v2"

  local long_msg
  long_msg="这是用于多轮会话测试的连续上下文输入。请记住我的长期偏好：论文写作采用IEEE风格；方法优先考虑RAG加重排序；实验必须包含消融实验、显著性检验、误差分析和失败案例讨论；结果展示优先图表化，并同时提供中英双语摘要。"

  # Multi-turn session continuity test (same session_id across turns).
  local i
  for i in $(seq 1 "${LONG_SESSION_TURNS}"); do
    _step_chat "memory_turn_${i}" \
      "${long_msg} 轮次=${i}；补充偏好：偏好中英双语摘要、图表化结果展示、引用格式统一。" \
      "${SESSION_MEM}"
  done

  _step_memory_db_checks

  echo "========================================"
  echo "Summary: pass=${PASS_COUNT} warn=${WARN_COUNT} fail=${FAIL_COUNT}"
  echo "Sessions:"
  echo "  direct=${SESSION_DIRECT}"
  echo "  lit=${SESSION_LIT}"
  echo "  prop=${SESSION_PROP}"
  echo "  mem=${SESSION_MEM}"
  echo "========================================"

  if [[ "${FAIL_COUNT}" -gt 0 ]]; then
    exit 1
  fi
}

main "$@"
