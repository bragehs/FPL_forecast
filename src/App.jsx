import React from "react";
import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";
import './App.css'; // Keep your styles

// Component for the home page
function Home() {
  return (
    <div className="card">
      <h1>Cumstink.com 👋
      </h1>
        <p>nettsiden til verdens farligste mann
       </p>
      <Link to="/new-page">
        <button>Go to New Page</button>
      </Link>
    </div>
  );
}

// Component for the new page with a different header
function NewPage() {
  return (
    <div className="card">
      <h1>Welcome to the New Page 🎉
      </h1>
      <p>This is a completely different page!
      </p>
      <Link to="/">
        <button>Back to Home</button>
      </Link>
    </div>
  );
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/new-page" element={<NewPage />} />
      </Routes>
    </Router>
  );
}

export default App;

