// const BASE_URL = "http://127.0.0.1:8000";

// export async function analyzeText(text, model = "xgboost") {
//   const predictRes = await fetch(
//     `${BASE_URL}/predict/text?model=${model}`,
//     {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({ text }),
//     }
//   ).then(res => res.json());

//   const explainRes = await fetch(
//     `${BASE_URL}/explain/text?model=${model}`,
//     {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({ text }),
//     }
//   ).then(res => res.json());

//   return {
//   verdict: predictRes.label.toLowerCase(),
//   confidence: predictRes.confidence, // already %
//   explanation: explainRes.important_words.map(w => w.word),
//   scores: predictRes.scores
// };

// }
const BASE_URL = "http://127.0.0.1:8000";

export async function analyzeText(text, model = "bert") {
  try {
    // -------- Predict ----------
    const predictRes = await fetch(
      `${BASE_URL}/predict/text?model=${model}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      }
    );

    if (!predictRes.ok) {
      throw new Error("Prediction API failed");
    }

    const predictData = await predictRes.json();

    // -------- Explain ----------
    const explainRes = await fetch(
      `${BASE_URL}/explain/text?model=${model}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      }
    );

    if (!explainRes.ok) {
      throw new Error("Explain API failed");
    }

    const explainData = await explainRes.json();

    // -------- Unified return ----------
    return {
      verdict: predictData.label.toLowerCase(), // fake / real
      confidence: predictData.confidence,       // %
      explanation: explainData.important_words.map(w => w.word),
      scores: predictData.scores,                // { fake, real }
      model                                   // which model was used
    };

  } catch (error) {
    console.error("API Error:", error);
    throw error;
  }
}

export async function analyzeImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE_URL}/predict/image`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) throw new Error("Image API failed");

  return await res.json();
}


export async function explainImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("http://127.0.0.1:8000/explain/image", {
    method: "POST",
    body: formData
  });

  return await res.json();
}

