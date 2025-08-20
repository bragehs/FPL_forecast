import React from "react";
import { usePrediction } from "./data/get_prediction";

interface PredictionProps {
  playerId: number | string | null;
  playerName?: string | null;
}

const beautifyMinutes = (minutes: number)  => {
    const minutesStr = minutes > 60 ? `60>` : String(Math.round(minutes));
    return minutesStr;
};

// Map raw points (≈0–4) into performance tiers.
function beautifyPoints(points?: number) {
  if (points == null || isNaN(points)) {
    return {
      category: "unknown" as const,
      label: "No data",
      range: "",
      color: "#eeeeee"
    };
  }
  const p = Math.round(points * 10) / 10; // 1 decimal
  if (p < 1)
    return { category: "bad" as const, label: "Bad", range: "0 – <1", color: "#ffe5e5" };
  if (p < 2)
    return { category: "okay" as const, label: "Okay", range: "1 – <2", color: "#fff5cc" };
  if (p < 3)
    return { category: "good" as const, label: "Good", range: "2 – <3", color: "#e5f8e5" };
  return { category: "excellent" as const, label: "Excellent", range: "3+", color: "#d6f0ff" };
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
    transition: "background .25s"
  };

  return (
    <div style={boxStyle}>
      <h2 style={{ margin: "0 0 4px" }}>Prediction {playerName ? `– ${playerName}` : ""}</h2>
      <div style={{ fontSize: 12, opacity: 0.75, marginBottom: 8 }}>
        Raw pts: {points != null ? (Math.round(points * 100) / 100) : "—"}
      </div>
      <div style={rowStyle}>
        <span style={labelStyle}>Performance:</span>
        <span>{perf.label} ({perf.range})</span>
      </div>
      <div style={rowStyle}>
        <span style={labelStyle}>Minutes:</span>
        <span>{minutesDisplay}</span>
      </div>
    </div>
  );
};

const baseBoxStyle: React.CSSProperties = {
  marginTop: 16,
  padding: "12px 16px",
  border: "1px solid #ddd",
  borderRadius: 6,
  maxWidth: 320,
  fontFamily: "system-ui, sans-serif",
  fontSize: 14
};

const rowStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  padding: "4px 0"
};

const labelStyle: React.CSSProperties = {
  fontWeight: 600
};