import React, { useEffect, useState } from 'react';
import axios from 'axios';

const PremierLeagueTable = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const headers = [
    'Rk', 'Squad', 'MP', 'W', 'D', 'L','Last 5', 'GF', 'GA', 'GD', 'Pts', 'xG', 'xGA', 'xGD', 'Pts/MP', 'Top Team Scorer', 'Goalkeeper', 'Attendance', 'Notes'
];

  useEffect(() => {
    const dummyData = [
      {
        "Rk": 1,
        "Squad": "Manchester City",
        "MP": 38,
        "W": 29,
        "D": 6,
        "L": 3,
        "Last 5": "WWWWW",
        "GF": 99,
        "GA": 28,
        "GD": 71,
        "Pts": 93,
        "xG": 80.5,
        "xGA": 26.3,
        "xGD": 54.2,
        "Pts/MP": 2.45,
        "Top Team Scorer": "Erling Haaland",
        "Goalkeeper": "Ederson",
        "Attendance": "53,000",
        "Notes": "Champions"
      }
    ];
    setData(dummyData);
    setLoading(false);
  }, []);
  //  axios.get('https://45ec-88-92-69-129.ngrok-free.app')
    //  .then(response => {
       // console.log('Data fetched:', response.data);
       // setData(response.data.data);
     //   setLoading(false);
   //   })
    //  .catch(error => {
    //    console.error('Error fetching data:', error);
   //     setLoading(false);
    //  });
 // }, []);


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
        {Array.isArray(data) && data.map((row, index) => (
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