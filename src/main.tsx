// main.jsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./index.css"; // Ensure you have this if you want to apply your CSS

const root = document.getElementById("root");
if (!root) throw new Error('Root element with id "root" not found');

ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
