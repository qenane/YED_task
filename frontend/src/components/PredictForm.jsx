import { useEffect, useState } from "react";
// PredictForm.jsx (öneri)
const API_BASE =
  (typeof process !== 'undefined' && process.env && process.env.REACT_APP_API_BASE) ||
  window.__API_BASE__ ||  // (opsiyonel) runtime inject etmek istersen
  `${window.location.protocol}//${window.location.hostname}:8000`;

// fetch(`${API_BASE}/datasets`) ve fetch(`${API_BASE}/predict`) kullan


export default function PredictForm({ setResult }) {
  const [mode, setMode] = useState("currcap");      // mevcut state’in
const [useLoad, setUseLoad] = useState(false);    // mevcut state’in
const [datasets, setDatasets] = useState([]);
const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/datasets`)
      .then((res) => res.json())
      .then((data) => {
        setDatasets(data.datasets);
        if (data.datasets.length > 0) {
          setDataset(data.datasets[0]);
        }
      });
  }, []);

  useEffect(() => {
  // 1) metrikler
  fetch(`${API}/metrics?mode=${mode}&use_load=${useLoad}`)
    .then(r => r.json())
    .then(setMetrics)
    .catch(() => setMetrics(null));

  // 2) test’te seçilebilecek datasetler (train setler otomatik çıkartılır)
  fetch(`${API}/datasets?scope=test&mode=${mode}&use_load=${useLoad}`)
    .then(r => r.json())
    .then(d => setDatasets(d.datasets || []));
}, [mode, useLoad]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log("Submitting...");
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset,
        mode,
        cycle_id: parseInt(cycleId),
        use_load: useLoad,
      }),
    });
    
    const data = await res.json();
    console.log("📩 Response:", data); // 🔍 backend response’u gör
    setResult(data);
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="bg-white shadow-md rounded-2xl p-6 w-96"
    >
      <label className="block mb-4">
        <span className="text-gray-700 font-semibold">Dataset:</span>
        <select
          value={dataset}
          onChange={(e) => setDataset(e.target.value)}
          className="mt-1 block w-full border-gray-300 rounded-lg p-2"
        >
          {datasets.map((ds) => (
            <option key={ds} value={ds}>
              {ds}
            </option>
          ))}
        </select>
      </label>

      <label className="block mb-4">
        <span className="text-gray-700 font-semibold">Mode:</span>
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value)}
          className="mt-1 block w-full border-gray-300 rounded-lg p-2"
        >
          <option value="currcap">CurrCap (capacity-free)</option>
          <option value="currcap_w_cap">CurrCap (with capacity)</option>
        </select>
      </label>

      <label className="block mb-4">
        <span className="text-gray-700 font-semibold">Use Load Features:</span>
        <input
          type="checkbox"
          checked={useLoad}
          onChange={(e) => setUseLoad(e.target.checked)}
          className="ml-2"
        />
      </label>

      <label className="block mb-6">
        <span className="text-gray-700 font-semibold">Cycle ID:</span>
        <input
          type="number"
          value={cycleId}
          onChange={(e) => setCycleId(e.target.value)}
          className="mt-1 block w-full border-gray-300 rounded-lg p-2"
        />
      </label>

      <button
        type="submit"
        className="w-full bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded-xl"
      >
        Predict SOC
      </button>
    </form>
  );
}
