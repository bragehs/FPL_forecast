import React, { useEffect, useState } from 'react';
import axios from 'axios';

const PremierLeagueTable = () => {
  console.log("PremierLeagueTable component is rendering");

  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const headers = [
    'Rk', 'Squad', 'MP', 'W', 'D', 'L','Last 5', 'Pts','GF', 'GA', 'GD', 'xG', 'xGA', 'xGD', 'Pts/MP', 'Top Team Scorer', 'Goalkeeper', 'Attendance', 'Notes'
];

  useEffect(() => {
       
    axios.get('https://efaa-2001-700-200-f120-c86f-570f-fc2a-29c5.ngrok-free.app')
      .then(response => {
        setData(response.data.data);
        setLoading(false);
      })
      .catch(error => {
       console.error('Error fetching data:', error);
       setLoading(false);
    });
  }, []);


  if (loading) {
    return <div>Loading...!!!!</div>;
  }

  return (
    <table>
      <thead>
        <tr>
            {headers.map((key) => (<th key = {key}>{key}</th>))}
        </tr>
        </thead>
      <tbody>
        {data.map((row, index) => (
          <tr key={index}>
            {headers.map((key, i) => (
              <td key={i}>{row[key]}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export default PremierLeagueTable;