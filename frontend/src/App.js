import { useState } from "react";
import PredictForm from "./components/PredictForm";
import ResultCard from "./components/ResultCard";

export default function App() {
  const [result, setResult] = useState(null);

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center p-8">
      <h1 className="text-3xl font-bold mb-6 text-blue-600">
        EV Battery SOC Prediction
      </h1>
      <PredictForm setResult={setResult} />
      {result && <ResultCard result={result} />}
      <MetricsCard metrics={metrics} />

    </div>
  );
}
