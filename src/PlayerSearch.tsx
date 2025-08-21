import React, { useState, useMemo, useCallback } from 'react';
import rawPlayerIds from "./data/player_ids.json";

interface PlayerEntry {
  id: number | string;
  name: string;
  [key: string]: any;
}

interface PlayerSearchProps {
  onPlayerSelect: (playerId: PlayerEntry['id'], playerName: PlayerEntry['name']) => void;
  placeholder?: string;
}

export const PlayerSearch: React.FC<PlayerSearchProps> = ({
  onPlayerSelect,
  placeholder = 'Search players...'
}) => {
  const [query, setQuery] = useState('');
  const [focusedIndex, setFocusedIndex] = useState<number>(-1);
  const [open, setOpen] = useState(false);

  // Map object { name: id } -> array [{ name, id }]
  const players: PlayerEntry[] = useMemo(() => {
    if (Array.isArray(rawPlayerIds)) {
      return rawPlayerIds as PlayerEntry[];
    }
    if (rawPlayerIds && typeof rawPlayerIds === 'object') {
      return Object.entries(rawPlayerIds as Record<string, number | string>)
        .map(([name, id]) => ({ name, id }));
    }
    return [];
  }, []);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) {
      const salah = players.find(
        p =>
          String(p.id) === '381' ||
          p.name.toLowerCase() === 'mohamed salah' ||
          p.name.toLowerCase().includes('salah')
      );
      return salah ? [salah] : [];
    }
    return players
      .filter(p =>
        p.name.toLowerCase().includes(q) ||
        String(p.id).toLowerCase().includes(q)
      )
      .slice(0, 100);
  }, [players, query]);

  const handleSelect = useCallback((player: PlayerEntry) => {
    onPlayerSelect(player.id, player.name);
    setFocusedIndex(-1);
    setOpen(false);
  }, [onPlayerSelect]);

  const onKeyDown: React.KeyboardEventHandler<HTMLInputElement> = e => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setOpen(true);
      setFocusedIndex(i => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setFocusedIndex(i => Math.max(i - 1, 0));
    } else if (e.key === 'Enter') {
      if (focusedIndex >= 0 && filtered[focusedIndex]) {
        handleSelect(filtered[focusedIndex]);
      }
    } else if (e.key === 'Escape') {
      setFocusedIndex(-1);
      setOpen(false);
    }
  };

  return (
    <div style={containerStyle}>
      <input
        type="text"
        value={query}
        placeholder={placeholder}
        onChange={e => {
          setQuery(e.target.value);
          setFocusedIndex(-1);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={onKeyDown}
        style={inputStyle}
        aria-autocomplete="list"
        aria-expanded={open && filtered.length > 0}
        aria-activedescendant={focusedIndex >= 0 ? `player-opt-${filtered[focusedIndex].id}` : undefined}
      />
      {open && filtered.length > 0 && (
        <ul style={listStyle} role="listbox">
          {filtered.map((p, idx) => {
            const active = idx === focusedIndex;
            return (
              <li
                id={`player-opt-${p.id}`}
                key={p.id}
                role="option"
                aria-selected={active}
                style={{
                  ...itemStyle,
                  background: active ? '#e6f2ff' : 'white'
                }}
                onMouseEnter={() => setFocusedIndex(idx)}
                onMouseDown={e => {
                  e.preventDefault();
                  handleSelect(p);
                }}
              >
                <span style={{ fontWeight: 500 }}>{p.name}</span>
                <span style={idBadgeStyle}>{p.id}</span>
              </li>
            );
          })}
          {query && filtered.length === 0 && <li style={itemStyle}>No results</li>}
        </ul>
      )}
    </div>
  );
};

const containerStyle: React.CSSProperties = { position: 'relative', maxWidth: 420, fontFamily: 'system-ui, sans-serif' };
const inputStyle: React.CSSProperties = { width: '100%', padding: '8px 10px', fontSize: 14, border: '1px solid #ccc', borderRadius: 4 };
const listStyle: React.CSSProperties = { position: 'absolute', top: '100%', left: 0, right: 0, maxHeight: 300, overflowY: 'auto', margin: 0, padding: 0, listStyle: 'none', border: '1px solid #ccc', borderTop: 'none', background: 'white', zIndex: 10 };
const itemStyle: React.CSSProperties = { display: 'flex', justifyContent: 'space-between', gap: 12, padding: '6px 10px', cursor: 'pointer', fontSize: 14, lineHeight: 1.3 };
const idBadgeStyle: React.CSSProperties = { fontSize: 12, color: '#555', background: '#f2f2f2', padding: '2px 6px', borderRadius: 4 };