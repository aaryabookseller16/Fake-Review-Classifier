const WORD_PATTERN = /[a-z0-9_]{2,}/g;

export function normalizeText(text) {
  return String(text ?? "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();
}

export function tokenize(text) {
  return normalizeText(text).match(WORD_PATTERN) ?? [];
}

export function extractSurfaceFeatures(text) {
  const source = String(text ?? "");
  const words = source.match(/[A-Za-z']+/g) ?? [];
  const lowerWords = words.map((word) => word.toLowerCase());
  const suspicious = ["free", "amazing", "best", "buy now", "limited", "guaranteed"];
  const lowered = source.toLowerCase();
  const firstPerson = new Set(["i", "me", "my", "mine", "we", "us", "our", "ours"]);

  return {
    words: source.trim() ? source.trim().split(/\s+/).length : 0,
    characters: source.length,
    suspicious: suspicious.reduce((total, term) => {
      let count = 0;
      let position = 0;
      while ((position = lowered.indexOf(term, position)) !== -1) {
        count += 1;
        position += term.length;
      }
      return total + count;
    }, 0),
    caps: source.split(/\s+/).filter((word) => word.length > 1 && word === word.toUpperCase() && /[A-Z]/.test(word)).length,
    exclamations: (source.match(/!/g) ?? []).length,
    lexicalDiversity: words.length ? new Set(lowerWords).size / words.length : 0,
    firstPersonRatio: words.length
      ? lowerWords.filter((word) => firstPerson.has(word)).length / words.length
      : 0,
  };
}

export function prepareModel(payload) {
  if (!payload || payload.schema !== 1) {
    throw new Error("Unsupported model export.");
  }
  if (
    payload.terms.length !== payload.idf.length ||
    payload.terms.length !== payload.weights.length
  ) {
    throw new Error("Model arrays do not align.");
  }

  const vocabulary = new Map();
  payload.terms.forEach((term, index) => vocabulary.set(term, index));
  return { ...payload, vocabulary };
}

export function vectorize(text, model) {
  const words = tokenize(text);
  const counts = new Map();

  for (let index = 0; index < words.length; index += 1) {
    const unigram = words[index];
    counts.set(unigram, (counts.get(unigram) ?? 0) + 1);
    if (index + 1 < words.length) {
      const bigram = `${unigram} ${words[index + 1]}`;
      counts.set(bigram, (counts.get(bigram) ?? 0) + 1);
    }
  }

  const values = [];
  let squaredNorm = 0;
  counts.forEach((count, term) => {
    const index = model.vocabulary.get(term);
    if (index === undefined) return;
    const tfidf = (1 + Math.log(count)) * model.idf[index];
    squaredNorm += tfidf * tfidf;
    values.push({ term, index, tfidf });
  });

  const norm = Math.sqrt(squaredNorm) || 1;
  return values.map((entry) => ({ ...entry, value: entry.tfidf / norm }));
}

export function classifyReview(text, model) {
  const vector = vectorize(text, model);
  let score = model.bias;
  const signals = vector.map(({ term, index, value }) => {
    const contribution = value * model.weights[index];
    score += contribution;
    return { term, contribution };
  });

  signals.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

  return {
    label: score >= model.threshold ? "fake" : "genuine",
    score,
    threshold: model.threshold,
    matchedTerms: vector.length,
    signals: signals.slice(0, 6),
    features: extractSurfaceFeatures(text),
  };
}
