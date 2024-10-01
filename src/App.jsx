// App.jsx
import React, { useEffect, useState } from "react"; // Import React and hooks
import Papa from "papaparse"; // Import PapaParse for parsing CSV
import "./App.css"; // Import CSS styles

function App() {
  // State to hold CSV data
  const [csvData, setCsvData] = useState([]);

  // useEffect to fetch and parse the CSV file when the component mounts
  useEffect(() => {
    const fetchCsvData = async () => {
      try {
        const response = await fetch("/data.csv"); // Fetch the CSV file
        const reader = response.body.getReader(); // Read the response body
        const decoder = new TextDecoder("utf-8"); // Create a decoder
        const result = await reader.read(); // Read the data
        const csvText = decoder.decode(result.value); // Decode to text

        // Parse the CSV text using PapaParse
        Papa.parse(csvText, {
          header: true, // Treat first row as headers
          skipEmptyLines: true, // Skip empty lines
          complete: (result) => {
            setCsvData(result.data); // Save parsed data to state
          },
        });
      } catch (error) {
        console.error("Error fetching the CSV file:", error); // Handle errors
      }
    };

    fetchCsvData(); // Call the fetch function
  }, []); // Empty dependency array to run once on mount

  return (
    <div className="App"> {/* Main container */}
      <h1>CSV Data Display</h1> {/* Title */}
      {csvData.length > 0 ? ( // Check if data is loaded
        <table border="1" cellPadding="5" style={{ borderCollapse: "collapse" }}> {/* Table */}
          <thead>
            <tr>
              {Object.keys(csvData[0]).map((header, index) => (
                <th key={index}>{header}</th> // Render headers
              ))}
            </tr>
          </thead>
          <tbody>
            {csvData.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {Object.values(row).map((value, cellIndex) => (
                  <td key={cellIndex}>{value}</td> // Render row data
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p>Loading data...</p> // Loading message
      )}
    </div>
  );
}

export default App; // Export the App component
