export default function MetricsCard({ metrics }) {
  if (!metrics) return null;
  const m = metrics.metrics || {};
  return (
    <div className="card">
      <h3>Son Model Metrikleri</h3>
      <div>Model: <code>{metrics.model_file}</code></div>
      <div>Eğitim Set(ler)i: { (metrics.train_sets || []).join(", ") || "-" }</div>
      <div>Test Set (eğitimde kullanılmadı): { metrics.test_set || "-" }</div>
      <div style={{marginTop: 8}}>
        { m.rmse_I !== undefined && <div>RMSE (I): {m.rmse_I.toFixed(4)}</div> }
        { m.rmse_cap !== undefined && <div>RMSE (Cap): {m.rmse_cap.toFixed(4)}</div> }
        { m.best_score !== undefined && <div>Best Score: {m.best_score.toFixed(4)}</div> }
      </div>
    </div>
  );
}
