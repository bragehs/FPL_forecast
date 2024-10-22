import React, { useEffect, useState } from 'react';
import axios from 'axios';

const PremierLeagueTable = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const headers = [
    'Rk', 'Squad', 'MP', 'W', 'D', 'L','Last 5', 'GF', 'GA', 'GD', 'Pts', 'xG', 'xGA', 'xGD', 'Pts/MP', 'Top Team Scorer', 'Goalkeeper', 'Attendance', 'Notes'
];

  useEffect(() => {
    axios.get('http://127.0.0.1:5000/')
      .then(response => {
        setData(response.data);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error fetching data:', error);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <div>Loading...</div>;
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