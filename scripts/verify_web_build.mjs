import { access, readFile, stat } from "node:fs/promises";

const required = [
  "web/index.html",
  "web/styles.css",
  "web/app.js",
  "web/classifier.js",
  "web/model.json",
  "web/favicon.svg",
];

await Promise.all(required.map((file) => access(file)));

const html = await readFile("web/index.html", "utf8");
const model = JSON.parse(await readFile("web/model.json", "utf8"));
const modelSize = (await stat("web/model.json")).size;

if (!html.includes("FAKE REVIEW DETECTOR") || !html.includes('id="analysis-form"')) {
  throw new Error("The production page is missing its brand or analysis form.");
}

if (
  model.schema !== 1 ||
  model.terms.length !== model.idf.length ||
  model.terms.length !== model.weights.length ||
  model.terms.length < 100_000
) {
  throw new Error("The browser model export is incomplete or malformed.");
}

console.log(
  `Production bundle verified: ${required.length} files, ` +
    `${model.terms.length.toLocaleString()} model terms, ` +
    `${(modelSize / 1_000_000).toFixed(2)} MB model payload.`,
);
