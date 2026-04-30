import type { RuntimeInfo } from "../../types/api";

type Props = {
  runtime?: RuntimeInfo;
};

export function RuntimePanel({ runtime }: Props) {
  if (!runtime) {
    return (
      <section className="panel runtime-panel">
        <h3>Runtime</h3>
        <p className="muted">暂无运行信息</p>
      </section>
    );
  }

  const budget = runtime.tool_budget;
  const tokenUsage = runtime.token_usage;

  return (
    <section className="panel runtime-panel">
      <h3>Runtime</h3>
      <dl>
        <dt>mode</dt>
        <dd>{runtime.mode}</dd>
        <dt>workflow</dt>
        <dd>{runtime.workflow_id ?? "(none)"}</dd>
        <dt>status</dt>
        <dd>{runtime.status}</dd>
        <dt>node</dt>
        <dd>{runtime.current_node ?? "(none)"}</dd>
        <dt>steps</dt>
        <dd>
          {runtime.step_count}
          {runtime.max_steps ? `/${runtime.max_steps}` : ""}
        </dd>
        <dt>loops</dt>
        <dd>
          {runtime.loop_count}
          {runtime.max_loops !== undefined ? `/${runtime.max_loops}` : ""}
        </dd>
      </dl>
      {tokenUsage ? (
        <div className="budget-block">
          <h4>Token Usage</h4>
          <ul>
            <li>
              <span>calls</span>
              <span>{tokenUsage.calls}</span>
            </li>
            <li>
              <span>input</span>
              <span>{tokenUsage.input_tokens}</span>
            </li>
            <li>
              <span>output</span>
              <span>{tokenUsage.output_tokens}</span>
            </li>
            <li>
              <span>total</span>
              <span>{tokenUsage.total_tokens}</span>
            </li>
            {tokenUsage.estimated_calls ? (
              <li>
                <span>estimated</span>
                <span>{tokenUsage.estimated_calls}</span>
              </li>
            ) : null}
          </ul>
        </div>
      ) : null}
      {budget ? (
        <div className="budget-block">
          <h4>Tool Budget ({budget.scope})</h4>
          <ul>
            {Object.keys(budget.limits).map((toolId) => {
              const used = budget.counts[toolId] ?? 0;
              const limit = budget.limits[toolId];
              return (
                <li key={toolId}>
                  <span>{toolId}</span>
                  <span>
                    {used}/{limit}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}
    </section>
  );
}
