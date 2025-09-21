// ResultCard.jsx
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Legend,
  Tooltip,
  TimeScale,
} from "chart.js";
import "chartjs-adapter-date-fns";

ChartJS.register(
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Legend,
  Tooltip,
  TimeScale
);

export default function ResultCard({ result }) {
  if (!result || result.error) {
    return (
      <div className="mt-6 bg-white shadow-md rounded-2xl p-6 w-full max-w-4xl">
        <h2 className="text-xl font-bold text-gray-700 mb-2">Result</h2>
        <p className="text-red-500">No prediction available</p>
      </div>
    );
  }

  // SOC Whole
  const labels_whole = result.soc_whole_t_pred?.map((_, i) => i + 1) || [];
  const dataWhole = {
    labels: labels_whole,
    datasets: [
      {
        label: "SOC Whole (Predicted)",
        data: result.soc_whole_t_pred || [],
        borderColor: "rgba(37,99,235,1)",
        pointRadius: 0,
      },
      {
        label: "SOC Whole (True)",
        data: result.soc_whole_t_true || [],
        borderColor: "rgba(239,68,68,1)",
        borderDash: [5, 5],
        pointRadius: 0,
      },
    ],
  };

  // SOC CycleCap
  const labels_cyclecap = result.soc_cyclecap_t_pred?.map((_, i) => i + 1) || [];
  const dataCycleCap = {
    labels: labels_cyclecap,
    datasets: [
      {
        label: "SOC CycleCap (Predicted)",
        data: result.soc_cyclecap_t_pred || [],
        borderColor: "rgba(16,185,129,1)",
        pointRadius: 0,
      },
      {
        label: "SOC CycleCap (True)",
        data: result.soc_cyclecap_t_true || [],
        borderColor: "rgba(245,158,11,1)",
        borderDash: [5, 5],
        pointRadius: 0,
      },
    ],
  };

  // SOC Cycle grafiği (tüm cycle'lar boyunca)
  const labelsCycle = result.history?.map((h) => h.datetime) || [];
  const dataCycle = {
    labels: labelsCycle,
    datasets: [
      {
        label: "SOC Cycle (True)",
        data: result.history?.map((h) => h.soc_cycle_true) || [],
        borderColor: "rgba(16,185,129,1)",
        pointRadius: 3,
      },
      {
        label: "SOC Cycle (Predicted)",
        data: result.history?.map((h) => h.soc_cycle_pred ?? null) || [],
        borderColor: "rgba(37,99,235,1)",
        pointRadius: 5,
        borderDash: [5, 5],
      },
    ],
  };

  // Chart options
  const options = {
    responsive: true,
    plugins: { legend: { display: true } },
    scales: {
      y: { min: 0, max: 1.05, title: { display: true, text: "SOC" } },
      x: { title: { display: true, text: "Timestep" } },
    },
  };

  const optionsCycle = {
    responsive: true,
    plugins: { legend: { display: true } },
    scales: {
      x: {
        type: "time",
        time: { unit: "day" },
        title: { display: true, text: "Datetime" },
      },
      y: { min: 0, max: 1.05, title: { display: true, text: "SOC Cycle" } },
    },
  };

  return (
    <div className="mt-6 bg-white shadow-md rounded-2xl p-6 w-full max-w-4xl">
      <h2 className="text-xl font-bold text-gray-700 mb-2">Result</h2>
      <p className="text-gray-600 mb-4">
        Dataset: <b>{result.dataset}</b> | Mode: <b>{result.mode}</b> | Cycle:{" "}
        <b>{result.cycle_id}</b> | Use Load:{" "}
        <b>{result.use_load ? "Yes" : "No"}</b>
      </p>

      <p className="text-2xl font-bold text-blue-600 mb-6">
        SOC Cycle (last prediction): {(((result?.soc_cycle ?? 0) * 100).toFixed(2))} %
        </p>


      {/* SOC Cycle over dataset */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-2">
          SOC Cycle over Dataset (True vs Predicted)
        </h3>
        <Line data={dataCycle} options={optionsCycle} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 className="text-lg font-semibold text-gray-700 mb-2">
            SOC Whole (Pred vs True)
          </h3>
          <Line data={dataWhole} options={options} />
        </div>

        <div>
          <h3 className="text-lg font-semibold text-gray-700 mb-2">
            SOC CycleCap (Pred vs True)
          </h3>
          <Line data={dataCycleCap} options={options} />
        </div>
      </div>
    </div>
  );
}
