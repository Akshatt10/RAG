import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle } from 'lucide-react';
import QuestionForm from './questionform';
import UploadForm from './uploadform';
import MessageBubble from './messagebubble';

export default function RAGInterface() {
  const [messages, setMessages] = useState([
    {
      id: 0,
      type: 'bot',
      content: 'Hello! I\'m your RAG assistant. I can help you find information from your uploaded documents, answer technical questions, and provide code examples. Upload some documents and start asking questions!',
      timestamp: new Date()
    }
  ]);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleUploadSuccess = (data) => {
    const successMessage = {
      id: Date.now(),
      type: 'bot',
      content: `Document "${data.file_name}" has been successfully uploaded and indexed. You can now ask questions about it!`,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, successMessage]);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 p-4">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent mb-3">
            Developer RAG Assistant
          </h1>
          <p className="text-gray-600 text-lg">Your personal knowledge companion for coding and development</p>
        </div>

        {/* Upload Form */}
        <UploadForm onUploadSuccess={handleUploadSuccess} />

        {/* Question Form */}
        <QuestionForm 
          messages={messages}
          setMessages={setMessages}
        />

        {/* Chat Messages */}
        <div className="bg-white rounded-xl shadow-xl border border-gray-100 overflow-hidden">
          <div className="bg-gradient-to-r from-gray-50 to-indigo-50 p-4 border-b border-gray-200">
            <h3 className="font-semibold text-gray-700 flex items-center gap-2">
              <MessageCircle size={20} className="text-indigo-600" />
              Conversation
            </h3>
          </div>
          <div className="h-96 overflow-y-auto p-6 space-y-4">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>
      </div>
    </div>
  );
}