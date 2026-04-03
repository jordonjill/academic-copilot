import type { WorkflowId } from "../../types/api";

export type WorkflowMode = "direct" | WorkflowId;

type Props = {
  value: WorkflowMode;
  onChange: (value: WorkflowMode) => void;
};

export function WorkflowSelector({ value, onChange }: Props) {
  return (
    <label className="workflow-selector">
      <span>Mode</span>
      <select value={value} onChange={(e) => onChange(e.target.value as WorkflowMode)}>
        <option value="direct">Direct</option>
        <option value="lit_review_v1">Workflow: lit_review_v1</option>
        <option value="proposal_v2">Workflow: proposal_v2</option>
      </select>
    </label>
  );
}
