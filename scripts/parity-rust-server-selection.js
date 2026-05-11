export function selectParityEntries(entries, { edgeOnly = false, maxTests = Infinity } = {}) {
  const filtered = edgeOnly
    ? entries.filter((entry) =>
        entry.expected_fraud_score === 0.4 || entry.expected_fraud_score === 0.6)
    : entries;

  return filtered.slice(0, maxTests);
}
