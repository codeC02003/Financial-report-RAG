import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 600000, // 10 min for large docs + model inference
});

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

export async function askQuestion(question, topK = 5, history = []) {
  const response = await api.post('/ask', {
    question,
    top_k: topK,
    history,
  });
  return response.data;
}

export async function getPageImage(pageNumber) {
  const response = await api.get(`/page/${pageNumber}`);
  return response.data;
}

export async function getDocumentInfo() {
  const response = await api.get('/document');
  return response.data;
}

export async function healthCheck() {
  const response = await api.get('/health');
  return response.data;
}
