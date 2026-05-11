export function compareRustResult(entry, rustResult, { mode = 'oracle', getOracleBucket } = {}) {
  let expectedScore;
  let expectedApproved;
  let oracleBucket = null;

  if (mode === 'expected') {
    expectedScore = entry.expected_fraud_score;
    expectedApproved = entry.expected_approved;
  } else if (mode === 'oracle') {
    if (typeof getOracleBucket !== 'function') {
      throw new Error('getOracleBucket is required in oracle mode');
    }
    oracleBucket = getOracleBucket(entry.request);
    expectedScore = oracleBucket / 5;
    expectedApproved = oracleBucket < 3;
  } else {
    throw new Error(`unknown comparison mode: ${mode}`);
  }

  return {
    ok: rustResult.fraud_score === expectedScore && rustResult.approved === expectedApproved,
    expectedScore,
    expectedApproved,
    oracleBucket,
  };
}
