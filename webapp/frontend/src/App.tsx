import React, { useState } from "react";

const API_URL = "http://localhost:8000/predict";

const App: React.FC = () => {
  const [prompt, setPrompt] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult("");
    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      const data = await response.json();
      setResult(data.result);
    } catch (err) {
      setResult("Error: " + err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: "2rem auto", fontFamily: "sans-serif" }}>
      <h1>Model Inference Demo</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          rows={4}
          style={{ width: "100%" }}
          placeholder="Enter your prompt here..."
        />
        <button type="submit" disabled={loading} style={{ marginTop: 8 }}>
          {loading ? "Generating..." : "Submit"}
        </button>
      </form>
      {result && (
        <div style={{ marginTop: 16 }}>
          <strong>Result:</strong>
          <div style={{ whiteSpace: "pre-wrap", marginTop: 8 }}>{result}</div>
        </div>
      )}
    </div>
  );
};

export default App;
