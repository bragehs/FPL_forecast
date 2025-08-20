import React, { useState } from "react";
import { PlayerSearch } from "./PlayerSearch";
import { Prediction } from "./Prediction";

function App() {
  const [playerId, setPlayerId] = useState<number | string | null>(null);
  const [playerName, setPlayerName] = useState<string | null>(null);
  const setPlayer = (id: number | string, name: string) => {
    setPlayerId(id);
    setPlayerName(name);
  };
  return (
    <div style={outerStyle}>
      <h1 style={{ margin: 0 }}>FPL Points Predictor</h1>
      <p style={subheadStyle}>
        Prediction uses a compact performance scale (Bad → Excellent). It is for comparing players, not exact FPL match points. Higher = stronger expected gameweek.
      </p>
      <div style={contentRow}>
        <div style={leftCol}>
          <PlayerSearch onPlayerSelect={setPlayer} />
        </div>
        <div style={rightCol}>
          <Prediction playerId={playerId} playerName={playerName} />
        </div>
      </div>
    </div>
  );
}

const outerStyle: React.CSSProperties = {
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  padding: 16,
  gap: 12
};

const subheadStyle: React.CSSProperties = {
  margin: 0,
  fontSize: 13,
  lineHeight: 1.4,
  fontWeight: 400,
  color: '#555',
  maxWidth: 640
};

const contentRow: React.CSSProperties = {
  flex: 1,
  display: 'flex',
  alignItems: 'flex-start',
  gap: 24,
  minHeight: 0
};

const leftCol: React.CSSProperties = {
  flex: 1,
  maxWidth: 520
};

const rightCol: React.CSSProperties = {
  width: 340,
  flexShrink: 0
};

export default App;