import React, {useEffect, useState} from "react";
import Papa from "papaparse";
import "./App.css";

function App() {
  const [csvData, setCsvData] = useState([]);

  useEffect( () => {
    const fetchCsvData = async () => {
      try {
        //define variables to fetch data from csv and process the data
        const response = await fetch('/archive/2021-2022.csv'); 
        const reader = response.body.getReader(); 
        const decoder = new TextDecoder("utf-8");
        const result = await reader.read();
        const csvText = decoder.decode(result.value);

        // parse csv with papaparse
        Papa.parse(csvText, {
          header: true,
          skipEmtpyLines: true,
          complete: (result) => {
            setCsvData(result.data);
          },
        });
      } catch (error){
        console.error("Error fetching the CSV file:", error);
      }    
    };

    fetchCsvData();
}, []);
}

export default App;