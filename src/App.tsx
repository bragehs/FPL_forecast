import { useState } from "react";
import { PlayerSearch } from "./PlayerSearch";
import { Prediction } from "./Prediction";
import "./App.css";

function App() {
  const [playerId, setPlayerId] = useState<number | string | null>(null);
  const [playerName, setPlayerName] = useState<string | null>(null);
  const setPlayer = (id: number | string, name: string) => {
    setPlayerId(id);
    setPlayerName(name);
  };
  return (
    <div className="outer">
      <h1 style={{ margin: 0 }}>FPL Points Predictor</h1>
      <p className="subhead">
        Points forecast based on past performances. Search for and select a player to see their performance forecast for the next gameweek.
      </p>
      <div className="content-row">
        <div className="left-col">
          <PlayerSearch onPlayerSelect={setPlayer} />
        </div>
        <div className="right-col">
          <Prediction playerId={playerId} playerName={playerName} />
        </div>
      </div>
    </div>
  );
}

export default App;