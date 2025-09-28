import React from 'react';
import { Bot, User, FileText, Code, Lightbulb, Wrench, MessageCircle } from 'lucide-react';

export default function MessageBubble({ message }) {
  const isUser = message.type === 'user';
  
  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.7) return 'text-green-700 bg-green-100 border-green-200';
    if (confidence >= 0.4) return 'text-yellow-700 bg-yellow-100 border-yellow-200';
    return 'text-red-700 bg-red-100 border-red-200';
  };

  const getQueryTypeIcon = (type) => {
    switch(type) {
      case 'code_help': return <Code size={16} className="text-blue-500" />;
      case 'learning': return <Lightbulb size={16} className="text-yellow-500" />;
      case 'project_guidance': return <Wrench size={16} className="text-green-500" />;
      default: return <Bot size={16} className="text-gray-500" />;
    }
  };

  const formatText = (text) => {
    if (!text) return null;

    // If text is an array, join it into a string with newlines
    const str = Array.isArray(text) ? text.join('\n') : text;

    return str.split('\n').map((line, i) => {
      if (line.startsWith('• ')) {
        return (
          <div key={i} className="ml-4 mb-1">
            {line}
          </div>
        );
      } else if (line.includes('```')) {
        const codeContent = line.replace(/```\w*\n?/g, '');
        return (
          <div key={i} className="bg-gray-900 text-white p-3 rounded-lg my-2 overflow-x-auto font-mono text-sm">
            {codeContent}
          </div>
        );
      } else if (line.trim().startsWith('**') && line.trim().endsWith('**')) {
        const boldText = line.replace(/\*\*(.*?)\*\*/g, '$1');
        return (
          <div key={i} className="font-bold mb-1">
            {boldText}
          </div>
        );
      } else {
        return (
          <div key={i} className={line.trim() === '' ? 'mb-2' : 'mb-1'}>
            {line}
          </div>
        );
      }
    });
  };

  return (
    <div className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start'} mb-6`}>
      {!isUser && (
        <div className="w-10 h-10 rounded-full bg-gradient-to-r from-indigo-600 to-purple-600 flex items-center justify-center text-white flex-shrink-0 shadow-lg">
          <Bot size={18} />
        </div>
      )}
      
      <div className={`max-w-[75%] ${isUser ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white' : 'bg-white border border-gray-200'} rounded-xl p-5 shadow-lg`}>
        {!isUser && (
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              {message.queryType && getQueryTypeIcon(message.queryType)}
              <span className="text-sm font-semibold text-gray-700">RAG Assistant</span>
            </div>
            {message.confidence && (
              <span className={`text-xs px-3 py-1 rounded-full border font-medium ${getConfidenceColor(message.confidence)}`}>
                {Math.round(message.confidence * 100)}% confidence
              </span>
            )}
          </div>
        )}

        {message.summary && (
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 border border-yellow-200 rounded-lg p-4 mb-4">
            <div className="flex items-center gap-2 mb-2">
              <Lightbulb size={16} className="text-yellow-600" />
              <span className="font-semibold text-yellow-800">Quick Summary</span>
            </div>
            <div className="text-yellow-700 text-sm">
              {formatText(message.summary)}
            </div>
          </div>
        )}

        <div className={`${isUser ? 'text-white' : 'text-gray-800'} ${message.isError ? 'text-red-600' : ''} leading-relaxed`}>
          {formatText(message.content)}
        </div>

        {message.sources && message.sources.length > 0 && (
          <div className="mt-4 pt-4 border-t border-gray-200">
            <div className="text-sm font-semibold text-gray-600 mb-3 flex items-center gap-2">
              <FileText size={16} />
              Sources
            </div>
            <div className="space-y-2">
              {message.sources.slice(0, 3).map((source, idx) => (
                <div key={idx} className="flex items-center gap-3 text-sm bg-gray-50 rounded-lg p-3">
                  <span className={`px-2 py-1 rounded-md text-xs font-bold uppercase ${
                    source.type === 'pdf' ? 'bg-red-100 text-red-700' :
                    source.type === 'txt' ? 'bg-green-100 text-green-700' :
                    source.type === 'md' ? 'bg-yellow-100 text-yellow-700' :
                    'bg-blue-100 text-blue-700'
                  }`}>
                    {source.type}
                  </span>
                  <span className="flex-1 text-gray-700 font-medium">{source.file}</span>
                  <span className="text-xs text-gray-500 font-semibold">
                    {Math.round(source.score * 100)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="text-xs text-gray-400 mt-3 flex items-center gap-1">
          <MessageCircle size={12} />
          {message.timestamp.toLocaleTimeString()}
        </div>
      </div>

      {isUser && (
        <div className="w-10 h-10 rounded-full bg-gray-600 flex items-center justify-center text-white flex-shrink-0 shadow-lg">
          <User size={18} />
        </div>
      )}
    </div>
  );
}