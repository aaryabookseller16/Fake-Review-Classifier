import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  classifyReview,
  extractSurfaceFeatures,
  normalizeText,
  prepareModel,
  tokenize,
} from "../classifier.js";

const payload = JSON.parse(await readFile(new URL("../model.json", import.meta.url), "utf8"));
const model = prepareModel(payload);

test("browser export is complete", () => {
  assert.equal(model.terms.length, 139_295);
  assert.equal(model.idf.length, model.weights.length);
  assert.equal(model.threshold, 0.5);
});

test("tokenization matches the trained vectorizer conventions", () => {
  assert.equal(normalizeText("Très BON"), "tres bon");
  assert.deepEqual(tokenize("I paid $60—for it!"), ["paid", "60", "for", "it"]);
});

test("held-out generated example matches Python inference", () => {
  const result = classifyReview(
    "Works great. Exact replace for $60, with the instructions included. If you have a larger computer, you'll want to get a solid replacement.",
    model,
  );
  assert.equal(result.label, "fake");
  assert.ok(Math.abs(result.score - 1.386952048) < 1e-6);
});

test("held-out genuine example matches Python inference", () => {
  const result = classifyReview(
    "They may not be picture perfect machine-finished but the metal is good they hold up and do their job and were much better than I expected for the amazingly low price.",
    model,
  );
  assert.equal(result.label, "genuine");
  assert.ok(Math.abs(result.score - -0.2925063963) < 1e-6);
});

test("surface features remain safe for empty and noisy inputs", () => {
  assert.equal(extractSurfaceFeatures("").words, 0);
  const features = extractSurfaceFeatures("I LOVE this—amazing! 10/10");
  assert.equal(features.caps, 1);
  assert.equal(features.exclamations, 1);
  assert.equal(features.suspicious, 1);
});

test("unknown vocabulary falls back to the learned bias without crashing", () => {
  const result = classifyReview("zxqv jklm qqqqq", model);
  assert.equal(result.matchedTerms, 0);
  assert.ok(Number.isFinite(result.score));
});
