import fs from "node:fs/promises";
import path from "node:path";

const repoRoot = process.cwd();
const sourceDir = process.argv[2] ?? "/tmp/scprint-op-table";
const outputPath =
  process.argv[3] ??
  path.join(repoRoot, "data/results/openproblems_all_models_transcriptformer_scprint1.csv");

const datasets = ["dkd", "gtex_v9", "hypomap", "mouse_pancreas_atlas"];
const datasetIds = new Map(datasets.map((name) => [`cellxgene_census/${name}`, name]));

function parseSimpleCsv(text) {
  const lines = text.trimEnd().split(/\r?\n/);
  const header = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(header.map((key, index) => [key, values[index] ?? ""]));
  });
}

function csvCell(value) {
  if (value === null || value === undefined || value === "NA") return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

const opResults = JSON.parse(
  await fs.readFile(path.join(repoRoot, "results_batch.json"), "utf8"),
).filter((entry) => datasetIds.has(entry.dataset_id));

const models = [
  ...new Set(
    opResults
      .filter((entry) => entry.dataset_id === "cellxgene_census/dkd")
      .map((entry) => entry.method_id),
  ),
];
const metrics = Object.keys(opResults[0].metric_values);
const values = new Map();

for (const entry of opResults) {
  const dataset = datasetIds.get(entry.dataset_id);
  for (const metric of metrics) {
    values.set(`${dataset}\t${metric}\t${entry.method_id}`, entry.metric_values[metric]);
  }
}

for (const dataset of datasets) {
  const externalFiles = [
    ["TranscriptFormer", `${dataset}_op_scib.csv`],
    ["scPRINT-1", `${dataset}_scprint1.csv`],
  ];
  for (const [model, filename] of externalFiles) {
    const [row] = parseSimpleCsv(await fs.readFile(path.join(sourceDir, filename), "utf8"));
    for (const metric of metrics) values.set(`${dataset}\t${metric}\t${model}`, row[metric]);
  }
}

const columns = [...models, "TranscriptFormer", "scPRINT-1"];
const rows = [["dataset", "metric", ...columns]];
for (const dataset of datasets) {
  for (const metric of metrics) {
    rows.push([
      dataset,
      metric,
      ...columns.map((model) => values.get(`${dataset}\t${metric}\t${model}`) ?? ""),
    ]);
  }
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.writeFile(outputPath, `${rows.map((row) => row.map(csvCell).join(",")).join("\n")}\n`);
console.log(JSON.stringify({ outputPath, datasets: datasets.length, metrics: metrics.length, models: columns.length, rows: rows.length - 1 }));
