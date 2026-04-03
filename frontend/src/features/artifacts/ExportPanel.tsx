import type { ChatArtifacts, ReportExports } from "../../types/api";

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : undefined;
}

function findReportExports(artifacts?: ChatArtifacts): ReportExports | undefined {
  if (!artifacts) {
    return undefined;
  }
  const direct = asRecord(artifacts.report_exports);
  if (direct) {
    return direct as ReportExports;
  }

  const shared = asRecord(artifacts.shared);
  if (!shared) {
    return undefined;
  }

  const reporter = asRecord(shared.report_exporter);
  const parsed = asRecord(reporter?.parsed);
  const parsedArtifacts = asRecord(parsed?.artifacts);
  const fromReporter = asRecord(parsedArtifacts?.report_exports);
  if (fromReporter) {
    return fromReporter as ReportExports;
  }

  for (const item of Object.values(shared)) {
    const record = asRecord(item);
    const p = asRecord(record?.parsed);
    const pa = asRecord(p?.artifacts);
    const re = asRecord(pa?.report_exports);
    if (re) {
      return re as ReportExports;
    }
  }

  return undefined;
}

type Props = {
  artifacts?: ChatArtifacts;
};

export function ExportPanel({ artifacts }: Props) {
  const exports = findReportExports(artifacts);

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
