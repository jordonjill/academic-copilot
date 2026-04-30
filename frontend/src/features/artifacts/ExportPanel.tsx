import type { PublicOutputs, ReportExports } from "../../types/api";

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : undefined;
}

function findReportExports(outputs?: PublicOutputs): ReportExports | undefined {
  if (!outputs) {
    return undefined;
  }
  const direct = asRecord(outputs.report_exports);
  if (direct) {
    return direct as ReportExports;
  }
  return undefined;
}

type Props = {
  outputs?: PublicOutputs;
};

export function ExportPanel({ outputs }: Props) {
  const exports = findReportExports(outputs);

  return (
    <section className="panel export-panel">
      <h3>Exports</h3>
      {exports ? (
        <ul>
          <li>
            <strong>DOCX:</strong> {exports.docx_path ?? "(none)"}
          </li>
          <li>
            <strong>PDF:</strong> {exports.pdf_path ?? "(none)"}
          </li>
        </ul>
      ) : (
        <p className="muted">当前会话还没有导出结果。</p>
      )}
    </section>
  );
}
