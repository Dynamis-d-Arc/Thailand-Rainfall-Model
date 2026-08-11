import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = "C:/Users/Brandon/Documents/Phase 1";
const rawJsonPath = `${projectRoot}/ML_Model_V2/tmd_weatherMinute_station37_20260624_hourly_raw.json`;
const outputPath = `${projectRoot}/outputs/tmd_hourly/tmd_weatherMinute_station37_20260624_hourly.xlsx`;

const rawText = await fs.readFile(rawJsonPath, "utf8");
const raw = JSON.parse(rawText.replace(/^\uFEFF/, ""));
const rows = raw.data.list ?? [];
const station = raw.data.object ?? {};

function n(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseSectime(value) {
  const text = String(value ?? "");
  if (text.length !== 14) return null;
  return new Date(Date.UTC(
    Number(text.slice(0, 4)),
    Number(text.slice(4, 6)) - 1,
    Number(text.slice(6, 8)),
    Number(text.slice(8, 10)),
    Number(text.slice(10, 12)),
    Number(text.slice(12, 14)),
  ));
}

const workbook = Workbook.create();
const dataSheet = workbook.worksheets.add("Hourly Data");
const metaSheet = workbook.worksheets.add("Metadata");
dataSheet.showGridLines = false;
metaSheet.showGridLines = false;

const headers = [
  "UTC Time",
  "Wind Dir Avg (deg)",
  "Max Wind Dir (deg)",
  "Wind Speed Avg (knot)",
  "Max Wind Speed (knot)",
  "Temperature (C)",
  "Precipitation (mm)",
  "Pressure (hPa)",
  "Humidity (%)",
  "Weather Code",
  "Visibility (m)",
];

const values = rows.map((row) => [
  parseSectime(row.sectime),
  n(row.s00a),
  n(row.s00m),
  n(row.s01a),
  n(row.s01m),
  n(row.s02a),
  n(row.r01m),
  n(row.s04a),
  n(row.s05a),
  n(row.s06m),
  n(row.s07a),
]);

dataSheet.getRange("A1:K1").values = [headers];
if (values.length > 0) {
  dataSheet.getRangeByIndexes(1, 0, values.length, headers.length).values = values;
}

const usedRows = Math.max(values.length + 1, 2);
dataSheet.getRange(`A1:K${usedRows}`).format = {
  font: { name: "Aptos", size: 10 },
};
dataSheet.getRange("A1:K1").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF", name: "Aptos", size: 10 },
  wrapText: true,
};
dataSheet.getRange(`A1:K${usedRows}`).format.borders = {
  preset: "inside",
  style: "thin",
  color: "#D9E2F3",
};
dataSheet.getRange(`A2:A${usedRows}`).format.numberFormat = "yyyy-mm-dd hh:mm";
dataSheet.getRange(`B2:J${usedRows}`).format.numberFormat = "0.0";
dataSheet.getRange(`K2:K${usedRows}`).format.numberFormat = "#,##0";
dataSheet.freezePanes.freezeRows(1);
dataSheet.getRange("A:L").format.autofitColumns();

const metaRows = [
  ["Field", "Value"],
  ["Source URL", "http://www.aws-observation.tmd.go.th/rprt/weatherMinuteData"],
  ["Report page", "http://www.aws-observation.tmd.go.th/rprt/weatherMinute"],
  ["Region", station.regions ?? "1"],
  ["Station ID", station.awsid ?? "37"],
  ["Station name", station.fname ?? "Bangna Agrometeorlogical Station"],
  ["Latitude", n(station.lat)],
  ["Longitude", n(station.lon)],
  ["Altitude", n(station.alt)],
  ["Interval", "1 hour"],
  ["Date", "2026-06-24"],
  ["Time range", "00:00-24:00 UTC"],
  ["Rows returned", rows.length],
  ["Generated from raw JSON", rawJsonPath],
];

metaSheet.getRangeByIndexes(0, 0, metaRows.length, 2).values = metaRows;
metaSheet.getRange(`A1:B${metaRows.length}`).format = {
  font: { name: "Aptos", size: 10 },
};
metaSheet.getRange("A1:B1").format = {
  fill: "#375623",
  font: { bold: true, color: "#FFFFFF", name: "Aptos", size: 10 },
};
metaSheet.getRange(`A1:B${metaRows.length}`).format.borders = {
  preset: "inside",
  style: "thin",
  color: "#E2F0D9",
};
metaSheet.getRange(`A2:A${metaRows.length}`).format = {
  font: { bold: true, name: "Aptos", size: 10 },
};
metaSheet.getRange("A:B").format.autofitColumns();

await fs.mkdir(`${projectRoot}/outputs/tmd_hourly`, { recursive: true });

const inspect = await workbook.inspect({
  kind: "sheet,region",
  sheetId: "Hourly Data",
  range: `A1:K${usedRows}`,
  maxChars: 2500,
});
console.log(inspect.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "formula error scan",
});
console.log(errors.ndjson);

const preview = await workbook.render({
  sheetName: "Hourly Data",
  autoCrop: "all",
  scale: 1,
  format: "png",
});
await fs.writeFile(
  `${projectRoot}/outputs/tmd_hourly/tmd_weatherMinute_station37_20260624_hourly_preview.png`,
  new Uint8Array(await preview.arrayBuffer()),
);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(outputPath);
