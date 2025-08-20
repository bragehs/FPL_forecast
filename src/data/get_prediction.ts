import useSWR from "swr";

const fetcher = (url: string) => fetch(url).then(r => r.json());

export function usePrediction(playerId: string | number | null) {
  const { data, error, isLoading } = useSWR(
    playerId ? `https://bragehs-fpl-forecast-huggingface.hf.space/predict?player_id=${playerId}` : null,
    fetcher
  );
  return { prediction: data, error, isLoading };
}