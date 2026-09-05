import { classifyReview, prepareModel } from "./classifier.js";

const examples = {
  genuine:
    "They may not be picture perfect machine-finished but the metal is good they hold up and do their job and were much better than I expected for the amazingly low price.",
  fake:
    "Works great. Exact replace for $60, with the instructions included. If you have a larger computer, you'll want to get a solid replacement.",
};

const models = [
  { name: "Adaline", detail: "3 surface features", accuracy: 0.5313466057 },
  { name: "Logistic regression", detail: "3 surface features", accuracy: 0.5383949549 },
  { name: "Adaline", detail: "10 surface features", accuracy: 0.7556572276 },
  { name: "Adaline", detail: "TF-IDF · from scratch", accuracy: 0.9488067268, best: true },
  { name: "Logistic regression", detail: "TF-IDF · reference", accuracy: 0.9447261036 },
];

let classifierModel = null;

const navButtons = [...document.querySelectorAll("[data-view]")];
const panels = [...document.querySelectorAll("[data-view-panel]")];
const form = document.querySelector("#analysis-form");
const input = document.querySelector("#review-input");
const analyseButton = document.querySelector("#analyse-button");
const clearButton = document.querySelector("#clear-button");
const count = document.querySelector("#character-count");
const status = document.querySelector("#model-status");
const emptyResult = document.querySelector("#empty-result");
const resultContent = document.querySelector("#result-content");
const resultCard = document.querySelector("#result-card");

function selectView(view, updateHash = true) {
  const target = panels.find((panel) => panel.dataset.viewPanel === view) ?? panels[0];

  panels.forEach((panel) => {
    const active = panel === target;
    panel.hidden = !active;
    panel.classList.toggle("is-active", active);
  });
  navButtons.forEach((button) => {
    const active = button.dataset.view === target.dataset.viewPanel;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-current", active ? "page" : "false");
  });

  if (updateHash) history.replaceState(null, "", `#${target.dataset.viewPanel}`);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

navButtons.forEach((button) => {
  button.addEventListener("click", () => selectView(button.dataset.view));
});

document.querySelectorAll('a[href="#analyse"]').forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    selectView("analyse");
  });
});

function syncInputState() {
  count.textContent = input.value.length.toLocaleString();
  analyseButton.disabled = !classifierModel || input.value.trim().length < 3;
}

input.addEventListener("input", syncInputState);

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = examples[button.dataset.example];
    syncInputState();
    input.focus();
  });
});

clearButton.addEventListener("click", () => {
  input.value = "";
  syncInputState();
  resultContent.hidden = true;
  emptyResult.hidden = false;
  resultCard.classList.remove("is-fake", "is-genuine");
  input.focus();
});

function createSignal(signal) {
  const pill = document.createElement("span");
  pill.className = `signal-pill ${signal.contribution >= 0 ? "is-fake" : "is-genuine"}`;

  const term = document.createElement("span");
  term.textContent = signal.term;

  const direction = document.createElement("i");
  direction.textContent = signal.contribution >= 0 ? "↑ generated" : "↓ genuine";

  pill.append(term, direction);
  return pill;
}

function showResult(result) {
  const isFake = result.label === "fake";
  emptyResult.hidden = true;
  resultContent.hidden = false;
  resultCard.classList.toggle("is-fake", isFake);
  resultCard.classList.toggle("is-genuine", !isFake);

  document.querySelector("#result-badge").textContent = isFake ? "Flag for review" : "Likely authentic";
  document.querySelector("#verdict-label").textContent = isFake ? "Generated" : "Genuine";
  document.querySelector("#verdict-copy").textContent = isFake
    ? "The learned word and phrase patterns place this review above the model’s generated-text threshold. Treat this as a moderation signal, not proof."
    : "The learned word and phrase patterns place this review below the generated-text threshold. No classifier can verify authorship on its own.";
  document.querySelector("#score-value").textContent = result.score.toFixed(3);
  document.querySelector("#matched-count").textContent = `${result.matchedTerms} vocabulary matches`;

  const pinPosition = Math.max(2, Math.min(98, result.score * 100));
  requestAnimationFrame(() => {
    document.querySelector("#score-pin").style.left = `${pinPosition}%`;
  });

  const signalList = document.querySelector("#signal-list");
  signalList.replaceChildren();
  if (result.signals.length) {
    result.signals.forEach((signal) => signalList.append(createSignal(signal)));
  } else {
    const note = document.createElement("span");
    note.className = "signal-pill";
    note.textContent = "No learned terms matched—verdict rests on the model bias";
    signalList.append(note);
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!classifierModel || input.value.trim().length < 3) return;

  analyseButton.querySelector("span:first-child").textContent = "Analysing…";
  analyseButton.disabled = true;

  requestAnimationFrame(() => {
    const result = classifyReview(input.value, classifierModel);
    showResult(result);
    analyseButton.querySelector("span:first-child").textContent = "Analyse again";
    analyseButton.disabled = false;
  });
});

function renderAccuracyChart() {
  const chart = document.querySelector("#accuracy-chart");
  models.forEach((model, index) => {
    const row = document.createElement("div");
    row.className = `bar-row${model.best ? " is-best" : ""}`;

    const label = document.createElement("div");
    label.className = "bar-label";
    const name = document.createElement("strong");
    name.textContent = model.name;
    const detail = document.createElement("span");
    detail.textContent = model.detail;
    label.append(name, detail);

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("span");
    fill.className = "bar-fill";
    fill.style.setProperty("--width", `${((model.accuracy - 0.5) / 0.5) * 100}%`);
    fill.style.setProperty("--delay", `${index * 70}ms`);
    track.append(fill);

    const value = document.createElement("span");
    value.className = "bar-value";
    value.textContent = `${(model.accuracy * 100).toFixed(2)}%`;

    row.append(label, track, value);
    chart.append(row);
  });
}

async function loadModel() {
  try {
    const response = await fetch("/model.json");
    if (!response.ok) throw new Error(`Model request failed (${response.status})`);
    classifierModel = prepareModel(await response.json());
    status.classList.add("is-ready");
    status.lastChild.textContent = " Model ready";
    syncInputState();
  } catch (error) {
    console.error(error);
    status.classList.add("is-error");
    status.lastChild.textContent = " Model unavailable";
    analyseButton.disabled = true;
  }
}

renderAccuracyChart();
selectView(location.hash.slice(1) || "analyse", false);
loadModel();
