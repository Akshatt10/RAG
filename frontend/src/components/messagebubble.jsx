import React from "react";
import { Bot, Code, Lightbulb, Wrench, FileText, TrendingDown, Copy, Check } from "lucide-react";
import { useState } from "react";

export default function MessageBubble({ message }) {
  const isUser = message.type === "user";
  const [copiedIndex, setCopiedIndex] = useState(null);

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.7)
      return "text-emerald-700 bg-emerald-50 border-emerald-200";
    if (confidence >= 0.4)
      return "text-amber-700 bg-amber-50 border-amber-200";
    return "text-rose-700 bg-rose-50 border-rose-200";
  };

  const copyToClipboard = (text, index) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const dedupeSources = (sources) => {
    const seen = new Set();
    return sources.filter((s) => {
      const key = s.name || s.file || JSON.stringify(s);
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  };

  const parseMarkdown = (text) => {
    if (typeof text !== "string") return String(text);

    const elements = [];
    let currentIndex = 0;
    let codeBlockCounter = 0;

    // Regex patterns
    const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
    const inlineCodeRegex = /`([^`]+)`/g;
    const boldRegex = /\*\*(.+?)\*\*/g;
    const numberedListRegex = /^\d+\.\s+(.+)$/gm;

    // Extract code blocks first
    const codeBlocks = [];
    let match;
    while ((match = codeBlockRegex.exec(text)) !== null) {
      codeBlocks.push({
        start: match.index,
        end: match.index + match[0].length,
        language: match[1] || 'text',
        code: match[2].trim()
      });
    }

    // Split text by code blocks
    let segments = [];
    let lastEnd = 0;

    codeBlocks.forEach((block) => {
      if (block.start > lastEnd) {
        segments.push({ type: 'text', content: text.slice(lastEnd, block.start) });
      }
      segments.push({ type: 'code', ...block });
      lastEnd = block.end;
    });

    if (lastEnd < text.length) {
      segments.push({ type: 'text', content: text.slice(lastEnd) });
    }

    if (segments.length === 0) {
      segments.push({ type: 'text', content: text });
    }

    // Process each segment
    return segments.map((segment, segIdx) => {
      if (segment.type === 'code') {
        const blockIndex = `code-${segIdx}`;
        return (
          <div key={segIdx} className="my-4 rounded-lg overflow-hidden border border-gray-200 bg-gray-900">
            <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
              <span className="text-xs font-mono text-gray-300">{segment.language}</span>
              <button
                onClick={() => copyToClipboard(segment.code, blockIndex)}
                className="flex items-center gap-1.5 px-2 py-1 text-xs text-gray-300 hover:text-white hover:bg-gray-700 rounded transition-colors"
              >
                {copiedIndex === blockIndex ? (
                  <>
                    <Check size={14} />
                    Copied!
                  </>
                ) : (
                  <>
                    <Copy size={14} />
                    Copy
                  </>
                )}
              </button>
            </div>
            <pre className="p-4 overflow-x-auto">
              <code className="text-sm text-gray-100 font-mono leading-relaxed">
                {segment.code}
              </code>
            </pre>
          </div>
        );
      }

      // Process text content
      let textContent = segment.content;
      
      // Handle numbered lists
      const listItems = [];
      textContent = textContent.replace(numberedListRegex, (match, content) => {
        listItems.push(content.trim());
        return '\n__LIST_ITEM__\n';
      });

      const parts = textContent.split('\n__LIST_ITEM__\n').filter(p => p.trim());
      
      return (
        <div key={segIdx}>
          {parts.map((part, partIdx) => {
            if (part.trim() === '') return null;

            // Process inline formatting
            const processInline = (str) => {
              const tokens = [];
              let lastIndex = 0;
              const combinedRegex = /(`[^`]+`|\*\*[^*]+\*\*)/g;
              let inlineMatch;

              while ((inlineMatch = combinedRegex.exec(str)) !== null) {
                if (inlineMatch.index > lastIndex) {
                  tokens.push({ type: 'text', content: str.slice(lastIndex, inlineMatch.index) });
                }

                const matched = inlineMatch[0];
                if (matched.startsWith('`') && matched.endsWith('`')) {
                  tokens.push({ type: 'inline-code', content: matched.slice(1, -1) });
                } else if (matched.startsWith('**') && matched.endsWith('**')) {
                  tokens.push({ type: 'bold', content: matched.slice(2, -2) });
                }

                lastIndex = combinedRegex.lastIndex;
              }

              if (lastIndex < str.length) {
                tokens.push({ type: 'text', content: str.slice(lastIndex) });
              }

              return tokens.map((token, idx) => {
                if (token.type === 'inline-code') {
                  return (
                    <code key={idx} className="px-1.5 py-0.5 bg-gray-100 text-gray-800 rounded text-sm font-mono border border-gray-200">
                      {token.content}
                    </code>
                  );
                } else if (token.type === 'bold') {
                  return <strong key={idx} className="font-semibold text-gray-900">{token.content}</strong>;
                }
                return <span key={idx}>{token.content}</span>;
              });
            };

            // Check if this part should be a list item
            if (listItems.length > 0 && partIdx > 0) {
              return (
                <ol key={partIdx} className="list-decimal list-inside space-y-2 my-3 ml-2">
                  {listItems.map((item, idx) => (
                    <li key={idx} className="text-gray-700 leading-relaxed">
                      <span className="ml-2">{processInline(item)}</span>
                    </li>
                  ))}
                </ol>
              );
            }

            return (
              <p key={partIdx} className="mb-3 leading-relaxed text-gray-700">
                {processInline(part)}
              </p>
            );
          })}
        </div>
      );
    });
  };

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-6`}>
      <div
        className={`rounded-2xl shadow-sm max-w-3xl transition-all hover:shadow-md ${
          isUser
            ? "bg-gradient-to-br from-blue-600 to-blue-700 text-white"
            : "bg-white text-gray-800 border border-gray-200"
        }`}
      >
        {/* Main Content */}
        <div className="p-5">
          {!isUser && (
            <div className="flex items-center gap-2 mb-3 pb-3 border-b border-gray-100">
              <Bot size={20} className="text-blue-600" />
              <span className="font-medium text-sm text-gray-600">AI Assistant</span>
            </div>
          )}
          
          <div className={`${isUser ? 'text-white' : 'text-gray-800'} text-[15px]`}>
            {Array.isArray(message.content)
              ? message.content.map((line, idx) => (
                  <div key={idx}>{parseMarkdown(String(line))}</div>
                ))
              : parseMarkdown(String(message.content))}
          </div>
        </div>

        {/* Quick Summary */}
        {message.summary && (
          <div className="px-5 pb-4">
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-4 border border-blue-100">
              <div className="flex items-center gap-2 mb-3">
                <Lightbulb size={18} className="text-blue-600" />
                <h4 className="font-semibold text-gray-900 text-sm">Key Points</h4>
              </div>
              <ul className="space-y-2">
                {Array.isArray(message.summary)
                  ? message.summary
                      .filter(s => String(s).trim() && String(s).trim() !== '•')
                      .map((s, i) => {
                          const text = String(s).replace(/^[•\s]+/, '').trim();
                          if (!text) return null;
                          return (
                            <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                              <span className="text-blue-500 mt-1 flex-shrink-0">•</span>
                              <span className="flex-1">{parseMarkdown(text)}</span>
                            </li>
                          );
                        })
                  : String(message.summary)
                      .split(/\n+/)
                      .filter(s => s.trim() && s.trim() !== '•')
                      .map((s, i) => {
                        const text = s.replace(/^[•\s]+/, '').trim();
                        if (!text) return null;
                        return (
                          <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                            <span className="text-blue-500 mt-1 flex-shrink-0">•</span>
                            <span className="flex-1">{parseMarkdown(text)}</span>
                          </li>
                        );
                      })}
              </ul>
            </div>
          </div>
        )}

        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <div className="px-5 pb-4">
            <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
              <div className="flex items-center gap-2 mb-3">
                <FileText size={16} className="text-gray-600" />
                <h4 className="font-semibold text-gray-900 text-sm">Referenced Sources</h4>
              </div>
              <div className="flex flex-wrap gap-2">
                {dedupeSources(message.sources).map((src, idx) => {
                  const displayName = src.name || src.file || JSON.stringify(src);
                  return (
                    <span
                      key={idx}
                      className="inline-flex items-center gap-1 px-3 py-1.5 bg-white rounded-lg text-xs font-medium text-gray-700 border border-gray-200 hover:border-blue-300 hover:bg-blue-50 transition-colors"
                    >
                      <FileText size={12} className="text-gray-500" />
                      {displayName}
                    </span>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* Footer: Confidence & Timestamp */}
        {(message.confidence !== null && message.confidence !== undefined) || message.timestamp ? (
          <div className="px-5 pb-4 flex items-center justify-between gap-3 text-xs">
            {message.confidence !== null && message.confidence !== undefined && (
              <div
                className={`px-3 py-1.5 rounded-full border font-medium ${getConfidenceColor(
                  message.confidence
                )}`}
              >
                <span className="flex items-center gap-1.5">
                  <TrendingDown size={14} />
                  Confidence: {(message.confidence * 100).toFixed(1)}%
                </span>
              </div>
            )}
            {message.timestamp && (
              <div className="text-gray-400">
                {new Date(message.timestamp).toLocaleTimeString([], { 
                  hour: '2-digit', 
                  minute: '2-digit' 
                })}
              </div>
            )}
          </div>
        ) : null}
      </div>
    </div>
  );
}

// Demo component
function App() {
  const sampleMessage = {
    type: "assistant",
    content: `Based on the context provided, here are the steps to start the E-voting system:

1. **Clone the repository:**
\`\`\`bash
git clone https://github.com/Akshatt10/E-Voting-System.git
\`\`\`

2. **Navigate into the project directory:**
\`\`\`bash
cd E-Voting-System
\`\`\`

3. **Go to the backend directory:**
\`\`\`bash
cd backend
\`\`\`

4. **Run docker compose to build and start the services:**
\`\`\`bash
docker-compose up --build
\`\`\`

5. **Install Dependencies:**
\`\`\`bash
npm install
\`\`\``,
    summary: [
      "Here is a summary of the steps to start the E-voting system:",
      "Clone the repository from GitHub and navigate into the project's backend directory.",
      "Install the necessary dependencies by running `npm install`.",
      "Build and start the application services using the `docker-compose up --build` command."
    ],
    sources: [
      { file: "readme.md" },
      { file: "Total-care-report.pdf" },
      { file: "README-2.md" }
    ],
    confidence: 0.655,
    timestamp: new Date().toISOString()
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-8">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">RAG Response with Markdown</h1>
        <MessageBubble message={sampleMessage} />
        
        <MessageBubble message={{
          type: "user",
          content: "How do I start the E-voting system?",
          timestamp: new Date(Date.now() - 60000).toISOString()
        }} />
      </div>
    </div>
  );
}