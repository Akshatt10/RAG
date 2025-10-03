import React, { useState } from "react";
import { Search, Send, Loader, Code, Settings } from "lucide-react";

export default function QuestionForm({ messages, setMessages }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [settings, setSettings] = useState({
    summarize: true,
    developerMode: true,
    minScore: 0.4,
    topK: 5,
  });
  const [showSettings, setShowSettings] = useState(false);

  const handleAsk = async () => {
    if (!question.trim()) return;

    const userMessage = {
      id: Date.now(),
      type: "user",
      content: question,
      summary: null,
      sources: [],
      confidence: null,
      queryType: "general",
      timestamp: new Date(),
      isError: false,
    };
    setMessages((prev) => [...prev, userMessage]);

    setLoading(true);
    setQuestion("");

    const payload = {
      question: question,
      top_k: settings.topK,
      min_score: settings.minScore,
      summarize: settings.summarize,
      developer_mode: settings.developerMode,
    };

    try {
      const res = await fetch(`http://localhost:8000/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);

      const data = await res.json();

      const summaryPoints = data.summary
        ? data.summary
            .split(/\n|\*|\-+/)
            .filter(Boolean)
            .map((s) => s.trim())
        : ["No summary available."];

      const botMessage = {
        id: Date.now() + 1,
        type: "bot",
        content: data.answer || ["No detailed answer."],
        summary: summaryPoints,
        sources: data.sources || [],
        confidence: data.confidence || null,
        queryType: data.query_type || "general",
        timestamp: new Date(),
        isError: false,
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      const errorMessage = {
        id: Date.now() + 1,
        type: "bot",
        content: [`❌ Error: ${err.message}`],
        isError: true,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mb-6 border border-gray-100">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-indigo-100 rounded-lg">
            <Search className="text-indigo-600" size={24} />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-800">
              Ask Your Knowledge Base
            </h2>
            <p className="text-sm text-gray-500">
              Get answers from your uploaded documents
            </p>
          </div>
        </div>
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="p-2 text-gray-400 hover:text-indigo-600 hover:bg-indigo-50 rounded-lg transition-all duration-200"
        >
          <Settings size={20} />
        </button>
      </div>

      {/* Question Input */}
      <div className="flex gap-3">
        <div className="flex-1 relative">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Ask about your documents, code, or any technical topic..."
            className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 resize-none transition-all duration-200"
            rows="2"
            disabled={loading}
          />
          {settings.developerMode && (
            <div className="absolute right-3 top-3">
              <Code size={20} className="text-indigo-400" />
            </div>
          )}
        </div>
        <button
          onClick={handleAsk}
          disabled={loading || !question.trim()}
          className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl hover:from-indigo-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center gap-2 min-w-[120px] justify-center shadow-lg hover:shadow-xl transform hover:scale-105"
        >
          {loading ? (
            <>
              <Loader className="animate-spin" size={16} />
              <span>Thinking...</span>
            </>
          ) : (
            <>
              <Send size={16} />
              <span>Ask</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}
