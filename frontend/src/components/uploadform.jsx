import React, { useState, useRef, useEffect } from 'react';
import { Search, Upload, FileText, Loader, Send, Bot, User, CheckCircle, AlertCircle, Code, Lightbulb, Wrench, Settings, MessageCircle } from 'lucide-react';


export default function UploadForm({ onUploadSuccess }) {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const supportedTypes = ['.pdf', '.docx', '.txt', '.md', '.csv', '.json', '.html'];

  const handleUpload = async (selectedFile = null) => {
    const fileToUpload = selectedFile || file;
    if (!fileToUpload) return;

    setUploading(true);
    setStatus("");

    const formData = new FormData();
    formData.append("file", fileToUpload);

    try {
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      
      if (data.status === "success") {
        setStatus(`Successfully uploaded: ${data.file_name}`);
        setFile(null);
        if (fileInputRef.current) fileInputRef.current.value = '';
        if (onUploadSuccess) onUploadSuccess(data);
      } else {
        setStatus(`Upload failed: ${data.message}`);
      }
    } catch (err) {
      setStatus("Upload failed. Please check your connection.");
    } finally {
      setUploading(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      setFile(droppedFile);
      handleUpload(droppedFile);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mb-6 border border-gray-100">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-green-100 rounded-lg">
          <Upload className="text-green-600" size={24} />
        </div>
        <div>
          <h2 className="text-xl font-bold text-gray-800">Upload Document</h2>
          <p className="text-sm text-gray-500">Add documents to your knowledge base</p>
        </div>
      </div>

      <div
        className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${
          dragOver 
            ? 'border-indigo-500 bg-indigo-50 scale-105' 
            : 'border-gray-300 hover:border-indigo-400 hover:bg-gray-50'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="space-y-4">
          <div className="flex justify-center">
            <div className={`p-4 rounded-full transition-colors ${dragOver ? 'bg-indigo-100' : 'bg-gray-100'}`}>
              <FileText size={48} className={dragOver ? 'text-indigo-500' : 'text-gray-400'} />
            </div>
          </div>
          
          <div>
            <p className="text-lg font-medium text-gray-700">
              {dragOver ? 'Drop your file here' : 'Drag & drop your document here'}
            </p>
            <p className="text-sm text-gray-500 mt-1">or click to browse files</p>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            onChange={(e) => setFile(e.target.files[0])}
            className="hidden"
            accept={supportedTypes.join(',')}
          />
          
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-6 py-3 bg-indigo-100 text-indigo-700 rounded-lg hover:bg-indigo-200 transition-all duration-200 font-medium"
            disabled={uploading}
          >
            Choose File
          </button>

          <div className="flex flex-wrap justify-center gap-2 text-xs text-gray-500">
            {supportedTypes.map(type => (
              <span key={type} className="px-2 py-1 bg-gray-100 rounded text-gray-600">
                {type}
              </span>
            ))}
          </div>
        </div>
      </div>

      {file && (
        <div className="bg-gradient-to-r from-gray-50 to-indigo-50 rounded-xl p-4 mb-4 border border-indigo-100">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-indigo-100 rounded-lg">
                <FileText size={20} className="text-indigo-500" />
              </div>
              <div>
                <p className="font-medium text-gray-800">{file.name}</p>
                <p className="text-sm text-gray-500">{formatFileSize(file.size)}</p>
              </div>
            </div>
            <button
              onClick={() => handleUpload()}
              disabled={uploading}
              className="px-4 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg hover:from-indigo-700 hover:to-purple-700 disabled:opacity-50 transition-all duration-200 flex items-center gap-2 shadow-lg hover:shadow-xl transform hover:scale-105"
            >
              {uploading ? (
                <>
                  <Loader className="animate-spin" size={16} />
                  <span>Uploading...</span>
                </>
              ) : (
                <>
                  <Upload size={16} />
                  <span>Upload</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {status && (
        <div className={`p-4 rounded-xl flex items-center gap-3 transition-all duration-300 ${
          status.includes('Successfully') 
            ? 'bg-green-50 text-green-800 border border-green-200' 
            : 'bg-red-50 text-red-800 border border-red-200'
        }`}>
          {status.includes('Successfully') ? (
            <CheckCircle size={20} className="flex-shrink-0" />
          ) : (
            <AlertCircle size={20} className="flex-shrink-0" />
          )}
          <span>{status}</span>
        </div>
      )}
    </div>
  );
}