import React from "react";
import { usePrediction } from "./data/get_prediction";

interface PredictionProps {
  playerId: number | string | null;
  playerName?: string | null;
}

const beautifyMinutes = (minutes: number)  => {
  return minutes > 60 ? "More than 60" : String(Math.round(minutes));
};

// Darker palette; white text on dark backgrounds.
function beautifyPoints(points?: number) {
  if (points == null || isNaN(points)) {
    return {
      category: "unknown" as const,
      label: "No data",
      range: "",
      color: "#eeeeee",
      textColor: "#222"
    };
  }
  const p = Math.round(points * 100) / 100;
  console.log("raw points:", p);
  if (p < 2)
    return { category: "bad" as const, label: "Bad",  color: "#7a1212", textColor: "#ffffff" };
  if (p < 3)
    return { category: "okay" as const, label: "Okay", color: "#fff6c7", textColor: "#5a4d00" };
  if (p < 4)
    return { category: "good" as const, label: "Good", color: "#d6ecff", textColor: "#133a52" };
  return { category: "excellent" as const, label: "Excellent", color: "#0f5f2a", textColor: "#ffffff" };
}

export const Prediction: React.FC<PredictionProps> = ({ playerId, playerName }) => {
  const { prediction, error, isLoading } = usePrediction(playerId);

  if (!playerId) return <div style={baseBoxStyle}>Select a player.</div>;
  if (isLoading) return <div style={baseBoxStyle}>Loading prediction...</div>;
  if (error) return <div style={baseBoxStyle}>Error loading prediction.</div>;
  if (!prediction) return <div style={baseBoxStyle}>No data.</div>;

  const { points, minutes } = prediction as { points?: number; minutes?: number };
  const perf = beautifyPoints(points);
  const minutesDisplay = minutes != null ? beautifyMinutes(minutes) : "—";

  const boxStyle: React.CSSProperties = {
    ...baseBoxStyle,
    background: perf.color,
    color: perf.textColor,
    borderColor: "rgba(0,0,0,0.15)",
    transition: "background .25s,color .25s"
  };

  const metaStyle: React.CSSProperties = {
    fontSize: 12,
    opacity: perf.textColor === "#ffffff" ? 0.85 : 0.7,
    marginBottom: 8
  };

  return (
    <div>
        <div style={boxStyle}>
        <h2 style={{ margin: "0 0 4px", fontSize: 22 }}>
            Prediction {playerName ? `– ${playerName}` : ""}
        </h2>
        <div style={rowStyle}>
            <span style={labelStyle}>Performance:</span>
            <span>{perf.label}</span>
        </div>
        <div style={rowStyle}>
            <span style={labelStyle}>Minutes:</span>
            <span>{minutesDisplay}</span>
        </div>
        </div>
        <footer style={footerStyle}>
        <p>The forecast is on scale (Bad → Okay → Good → Excellent)</p>
        </footer>
    </div>
  );
};

const baseBoxStyle: React.CSSProperties = {
  marginTop: 16,
  padding: "20px 24px",          // increased padding
  border: "1px solid #ddd",
  borderRadius: 10,              // slightly larger radius
  maxWidth: 640,                 // wider
  fontFamily: "system-ui, sans-serif",
  fontSize: 16,                  // larger base font
  lineHeight: 1.45
};

const rowStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  padding: "8px 0"               // more vertical spacing
};

const labelStyle: React.CSSProperties = {
  fontWeight: 600,
  fontSize: 16
};

const footerStyle: React.CSSProperties = {
  fontSize: 13,
  lineHeight: 1.4,
  fontWeight: 400,
  color: '#555',
  maxWidth: 640
};
