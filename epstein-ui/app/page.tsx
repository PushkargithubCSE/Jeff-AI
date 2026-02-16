"use client";

import { useState } from "react";

export default function Home() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState("");

  const askAI = async () => {
    const res = await fetch("http://127.0.0.1:8000/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });

    const data = await res.json();
    setResult(data.answer);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6">
      <h1 className="text-4xl font-bold mb-6">
        Epstein AI Investigator
      </h1>

      <input
        className="w-full max-w-xl p-3 rounded bg-gray-800"
        placeholder="Ask something..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />

      <button
        onClick={askAI}
        className="mt-4 px-6 py-2 bg-blue-600 rounded"
      >
        Ask
      </button>

      {result && (
        <div className="mt-6 max-w-xl bg-gray-900 p-4 rounded">
          {result}
        </div>
      )}
    </div>
  );
}
