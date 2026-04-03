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
        <dt>step_count</dt>
        <dd>{runtime.step_count}</dd>
        <dt>loop_count</dt>
        <dd>{runtime.loop_count}</dd>
      </dl>
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
