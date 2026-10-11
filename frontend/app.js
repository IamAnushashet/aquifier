/**
 * AI-Powered Aquifer Groundwater Forecasting & Smart Water Management System
 * Frontend Application Controller
 */

// API Base URL
const API_BASE = window.location.origin.includes("5500") || window.location.origin.includes("3000")
  ? "http://127.0.0.1:8000" 
  : "";

// Global App State
const state = {
  currentWellId: "WELL-01",
  currentHorizon: "30d",
  currentModel: "Random Forest",
  wellsList: [],
  activeWellMeta: null,
  map: null,
  markers: {},
  charts: {}
};

// ================= INITIALIZATION =================
document.addEventListener("DOMContentLoaded", async () => {
  initTabs();
  initMap();
  initEventListeners();
  
  // Load initial data
  await loadWellsData();
  await refreshActiveWell();
  await loadResearchData();
});

// ================= TAB NAVIGATION =================
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetId = tab.getAttribute("data-tab");
      
      // Update tab buttons
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      
      // Update tab panes
      document.querySelectorAll(".tab-pane").forEach(pane => {
        pane.classList.remove("active");
      });
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add("active");
      
      // Invalidate map & chart sizes
      if (targetId === "tab-overview" && state.map) {
        setTimeout(() => state.map.invalidateSize(), 150);
      }
      resizeActiveCharts();
    });
  });
}

function resizeActiveCharts() {
  Object.values(state.charts).forEach(c => {
    if (c && typeof c.resize === "function") {
      c.resize();
    }
  });
}

// ================= GIS LEAFLET MAP =================
function initMap() {
  const mapContainer = document.getElementById("gisMap");
  if (!mapContainer) return;

  state.map = L.map("gisMap", {
    zoomControl: true,
    scrollWheelZoom: true
  }).setView([28.58, 77.20], 10);

  // CartoDB Dark Matter tiles for modern dark aesthetic
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; <a href="https://carto.com/">CARTO</a> & OpenStreetMap',
    maxZoom: 18
  }).addTo(state.map);
}

function renderMapMarkers() {
  if (!state.map || !state.wellsList.length) return;

  // Clear existing markers
  Object.values(state.markers).forEach(m => m.remove());
  state.markers = {};

  const bounds = [];

  state.wellsList.forEach(well => {
    const lat = well.latitude;
    const lng = well.longitude;
    bounds.push([lat, lng]);

    // Color based on risk category
    let color = "#10B981";
    if (well.risk_category === "Moderate Risk") color = "#F59E0B";
    if (well.risk_category === "High Risk") color = "#F97316";
    if (well.risk_category === "Critical Risk") color = "#EF4444";

    const customMarker = L.circleMarker([lat, lng], {
      radius: well.well_id === state.currentWellId ? 12 : 9,
      fillColor: color,
      color: "#FFFFFF",
      weight: well.well_id === state.currentWellId ? 3 : 1.5,
      opacity: 1,
      fillOpacity: 0.85
    }).addTo(state.map);

    const popupHtml = `
      <div style="font-family: 'Plus Jakarta Sans', sans-serif; color: #0F172A; min-width: 170px;">
        <strong style="font-size: 13px; color: #0369A1;">${well.well_id}: ${well.name}</strong><br>
        <span style="font-size: 11px; color: #64748B;">Aquifer: ${well.aquifer_type}</span>
        <hr style="margin: 4px 0; border: none; border-top: 1px solid #E2E8F0;">
        <div style="font-size: 11px; margin-bottom: 2px;">Depth: <strong>${well.current_depth_m.toFixed(2)} m</strong></div>
        <div style="font-size: 11px; margin-bottom: 2px;">Risk Index: <strong style="color: ${color};">${well.risk_index} (${well.risk_category})</strong></div>
        <div style="font-size: 11px;">Health Score: <strong>${well.health_score}/100</strong></div>
      </div>
    `;

    customMarker.bindPopup(popupHtml);
    customMarker.on("click", () => {
      document.getElementById("wellSelect").value = well.well_id;
      state.currentWellId = well.well_id;
      refreshActiveWell();
    });

    state.markers[well.well_id] = customMarker;
  });

  if (bounds.length) {
    state.map.fitBounds(bounds, { padding: [40, 40] });
  }
}

// ================= EVENT LISTENERS =================
function initEventListeners() {
  // Well Station Select Change
  const wellSelect = document.getElementById("wellSelect");
  if (wellSelect) {
    wellSelect.addEventListener("change", (e) => {
      state.currentWellId = e.target.value;
      refreshActiveWell();
      if (state.markers[state.currentWellId]) {
        state.markers[state.currentWellId].openPopup();
      }
    });
  }

  // Horizon Buttons
  const horizonButtons = document.querySelectorAll("#horizonButtonGroup button");
  horizonButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      horizonButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.currentHorizon = btn.getAttribute("data-horizon");
      loadForecastData();
    });
  });

  // Model Selector
  const forecastModelSelect = document.getElementById("forecastModelSelect");
  if (forecastModelSelect) {
    forecastModelSelect.addEventListener("change", (e) => {
      state.currentModel = e.target.value;
      loadForecastData();
    });
  }

  // Simulator Sliders Live Text Updates
  const sliderExt = document.getElementById("sliderExtraction");
  const valExt = document.getElementById("valSliderExtraction");
  if (sliderExt && valExt) {
    sliderExt.addEventListener("input", (e) => {
      valExt.textContent = `${e.target.value > 0 ? '+' : ''}${e.target.value}%`;
    });
  }

  const sliderRech = document.getElementById("sliderRecharge");
  const valRech = document.getElementById("valSliderRecharge");
  if (sliderRech && valRech) {
    sliderRech.addEventListener("input", (e) => {
      valRech.textContent = `${e.target.value} m³/day`;
    });
  }

  const sliderRain = document.getElementById("sliderRainfall");
  const valRain = document.getElementById("valSliderRainfall");
  if (sliderRain && valRain) {
    sliderRain.addEventListener("input", (e) => {
      valRain.textContent = `${e.target.value > 0 ? '+' : ''}${e.target.value}%`;
    });
  }

  const sliderHoriz = document.getElementById("sliderHorizon");
  const valHoriz = document.getElementById("valSliderHorizon");
  if (sliderHoriz && valHoriz) {
    sliderHoriz.addEventListener("input", (e) => {
      valHoriz.textContent = `${e.target.value} Days`;
    });
  }

  // Simulator Run Button
  const btnRunSim = document.getElementById("btnRunSimulation");
  if (btnRunSim) {
    btnRunSim.addEventListener("click", () => {
      loadWhatIfSimulation();
    });
  }

  // Recovery Sliders Live Input
  const recRain = document.getElementById("recRainEvent");
  const valRecRain = document.getElementById("valRecRain");
  if (recRain && valRecRain) {
    recRain.addEventListener("input", (e) => {
      valRecRain.textContent = `${e.target.value} mm`;
      loadRecoverySimulation();
    });
  }

  const recCond = document.getElementById("recCond");
  const valRecCond = document.getElementById("valRecCond");
  if (recCond && valRecCond) {
    recCond.addEventListener("input", (e) => {
      valRecCond.textContent = `${e.target.value} m/d`;
      loadRecoverySimulation();
    });
  }

  // XAI Day Offset Selector
  const xaiDayOffset = document.getElementById("xaiDayOffset");
  if (xaiDayOffset) {
    xaiDayOffset.addEventListener("change", () => {
      loadXAILocalExplanation();
    });
  }
}

// ================= DATA LOADING =================
async function loadWellsData() {
  try {
    const res = await fetch(`${API_BASE}/api/wells`);
    const data = await res.json();
    state.wellsList = data.wells || [];
    renderMapMarkers();
  } catch (err) {
    console.error("Failed to fetch wells:", err);
  }
}

async function refreshActiveWell() {
  const currentWell = state.wellsList.find(w => w.well_id === state.currentWellId) || state.wellsList[0];
  if (!currentWell) return;

  state.activeWellMeta = currentWell;

  // 1. Update Header Badge
  const headerRiskBadge = document.getElementById("headerRiskBadge");
  const headerRiskText = document.getElementById("headerRiskText");
  if (headerRiskBadge && headerRiskText) {
    headerRiskBadge.className = `status-pill status-${currentWell.risk_category.toLowerCase().split(' ')[0]}`;
    headerRiskText.textContent = `${currentWell.risk_category} (${currentWell.risk_index})`;
  }

  // 2. Update KPI Ribbon
  document.getElementById("valCurrentDepth").textContent = currentWell.current_depth_m.toFixed(2);
  document.getElementById("subCurrentDepth").textContent = `Safe: <${currentWell.safe_depth_m}m | Critical: >${currentWell.critical_depth_m}m`;
  document.getElementById("valRiskIndex").textContent = currentWell.risk_index;
  document.getElementById("subRiskCategory").textContent = currentWell.risk_category;
  document.getElementById("valHealthScore").textContent = currentWell.health_score;
  document.getElementById("subHealthGrade").textContent = `Grade: ${currentWell.health_grade}`;
  document.getElementById("valExtraction").textContent = currentWell.extraction_rate.toFixed(2);
  document.getElementById("valRainfall").textContent = currentWell.latest_rainfall_mm.toFixed(1);

  // 3. Update Well Meta Panel
  document.getElementById("selectedWellBadge").textContent = currentWell.well_id;
  document.getElementById("metaStationName").textContent = currentWell.name;
  document.getElementById("metaAquiferType").textContent = currentWell.aquifer_type;
  document.getElementById("metaElevation").textContent = `${currentWell.elevation_m} m AMSL`;
  document.getElementById("metaSafeDepth").textContent = `${currentWell.safe_depth_m} m bgl`;
  document.getElementById("metaCritDepth").textContent = `${currentWell.critical_depth_m} m bgl`;
  document.getElementById("metaDescription").textContent = currentWell.description;

  // 4. Fetch Deep Data for this Well
  await Promise.all([
    loadOverviewMiniChart(),
    loadForecastData(),
    loadWhatIfSimulation(),
    loadRiskAndAlerts(),
    loadXAIData(),
    loadAnomaliesData(),
    loadRecoverySimulation(),
    loadClimateSimulation(),
    loadHealthAndRecommendations()
  ]);
}

// ================= OVERVIEW MINI CHART =================
async function loadOverviewMiniChart() {
  try {
    const res = await fetch(`${API_BASE}/api/wells/${state.currentWellId}/history?days=90`);
    const data = await res.json();
    const history = data.history || [];

    const labels = history.map(h => h.date.slice(5)); // MM-DD
    const gwLevels = history.map(h => h.groundwater_level);
    const rainVals = history.map(h => h.rainfall);

    const ctx = document.getElementById("overviewMiniChart");
    if (!ctx) return;

    if (state.charts.overviewMini) state.charts.overviewMini.destroy();

    state.charts.overviewMini = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Water Table (m bgl)",
            data: gwLevels,
            borderColor: "#38BDF8",
            backgroundColor: "rgba(56, 189, 248, 0.1)",
            borderWidth: 2,
            tension: 0.3,
            fill: true,
            pointRadius: 0,
            yAxisID: "y"
          },
          {
            label: "Rainfall (mm)",
            data: rainVals,
            type: "bar",
            backgroundColor: "rgba(16, 185, 129, 0.4)",
            borderWidth: 0,
            yAxisID: "y1"
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { mode: "index", intersect: false }
        },
        scales: {
          x: { display: false },
          y: {
            display: true,
            reverse: true, // Lower is deeper
            grid: { color: "rgba(148, 163, 184, 0.08)" },
            ticks: { color: "#94A3B8", font: { size: 10 } }
          },
          y1: {
            display: false,
            position: "right",
            grid: { display: false }
          }
        }
      }
    });
  } catch (err) {
    console.error("Overview mini chart failed:", err);
  }
}

// ================= FORECASTING STUDIO =================
async function loadForecastData() {
  try {
    const res = await fetch(`${API_BASE}/api/forecast?well_id=${state.currentWellId}&horizon=${state.currentHorizon}&model=${encodeURIComponent(state.currentModel)}`);
    const data = await res.json();

    const obsDates = data.historical_observed.dates || [];
    const obsLevels = data.historical_observed.levels || [];
    const fcDates = data.forecast.forecast_dates || [];
    const fcLevels = data.forecast.forecast_values || [];
    const lower95 = data.forecast.lower_bound_95 || [];
    const upper95 = data.forecast.upper_bound_95 || [];

    // Update stats bar
    document.getElementById("fcStartObserved").textContent = `${data.forecast.start_observed_level.toFixed(2)} m`;
    document.getElementById("fcEndForecast").textContent = `${data.forecast.end_forecasted_level.toFixed(2)} m`;
    const netDelta = data.forecast.net_predicted_change_m;
    document.getElementById("fcNetChange").textContent = `${netDelta > 0 ? '+' : ''}${netDelta.toFixed(2)} m (${netDelta > 0 ? 'Depletion' : 'Replenishment'})`;
    const margin = (data.critical_threshold_m - data.forecast.end_forecasted_level).toFixed(2);
    document.getElementById("fcMargin").textContent = `${margin} m above critical threshold`;

    // Update 30d KPI card if horizon is 30d
    if (state.currentHorizon === "30d") {
      document.getElementById("valForecast30d").textContent = data.forecast.end_forecasted_level.toFixed(2);
      document.getElementById("subForecastDelta").textContent = `Net Change: ${netDelta > 0 ? '+' : ''}${netDelta.toFixed(2)}m (${netDelta > 0 ? 'Depletion' : 'Recovery'})`;
    }

    // Chart combination
    const allLabels = [...obsDates, ...fcDates];
    
    // Fill observed series
    const observedSeries = [...obsLevels, ...Array(fcDates.length).fill(null)];
    
    // Fill forecast series with bridge connection
    const forecastSeries = [...Array(obsDates.length - 1).fill(null), obsLevels[obsLevels.length - 1], ...fcLevels];
    const upperSeries = [...Array(obsDates.length - 1).fill(null), obsLevels[obsLevels.length - 1], ...upper95];
    const lowerSeries = [...Array(obsDates.length - 1).fill(null), obsLevels[obsLevels.length - 1], ...lower95];

    const ctx = document.getElementById("forecastMainChart");
    if (!ctx) return;

    if (state.charts.forecastMain) state.charts.forecastMain.destroy();

    state.charts.forecastMain = new Chart(ctx, {
      type: "line",
      data: {
        labels: allLabels,
        datasets: [
          {
            label: "Historical Observed Telemetry",
            data: observedSeries,
            borderColor: "#38BDF8",
            borderWidth: 2.5,
            pointRadius: 0,
            tension: 0.2
          },
          {
            label: `AI Forecast (${state.currentModel})`,
            data: forecastSeries,
            borderColor: "#10B981",
            borderWidth: 3,
            borderDash: [5, 5],
            pointRadius: 0,
            tension: 0.2
          },
          {
            label: "95% Upper Confidence Bound",
            data: upperSeries,
            borderColor: "transparent",
            backgroundColor: "rgba(16, 185, 129, 0.12)",
            pointRadius: 0,
            fill: "+1" // fill to lowerSeries
          },
          {
            label: "95% Lower Confidence Bound",
            data: lowerSeries,
            borderColor: "transparent",
            pointRadius: 0,
            fill: false
          },
          {
            label: `Critical Threshold (${data.critical_threshold_m}m)`,
            data: Array(allLabels.length).fill(data.critical_threshold_m),
            borderColor: "#EF4444",
            borderWidth: 1.5,
            borderDash: [8, 4],
            pointRadius: 0,
            fill: false
          },
          {
            label: `Safe Threshold (${data.safe_threshold_m}m)`,
            data: Array(allLabels.length).fill(data.safe_threshold_m),
            borderColor: "#34D399",
            borderWidth: 1.5,
            borderDash: [4, 4],
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            position: "top",
            labels: { color: "#94A3B8", font: { size: 11 }, boxWidth: 14 }
          },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${ctx.raw ? Number(ctx.raw).toFixed(2) + ' m' : ''}`
            }
          }
        },
        scales: {
          x: {
            grid: { color: "rgba(148, 163, 184, 0.08)" },
            ticks: {
              color: "#64748B",
              maxTicksLimit: 12,
              font: { size: 10 }
            }
          },
          y: {
            reverse: true, // Deep is down
            grid: { color: "rgba(148, 163, 184, 0.08)" },
            ticks: {
              color: "#94A3B8",
              font: { size: 11 },
              callback: (v) => `${v} m`
            },
            title: {
              display: true,
              text: "Depth to Water Table (m bgl - lower is deeper)",
              color: "#94A3B8",
              font: { size: 11 }
            }
          }
        }
      }
    });
  } catch (err) {
    console.error("Failed to load forecast data:", err);
  }
}

// ================= WHAT-IF SIMULATOR =================
async function loadWhatIfSimulation() {
  const extChange = parseFloat(document.getElementById("sliderExtraction")?.value || -15);
  const rechargeM3 = parseFloat(document.getElementById("sliderRecharge")?.value || 600);
  const rainMod = parseFloat(document.getElementById("sliderRainfall")?.value || 0);
  const horizon = parseInt(document.getElementById("sliderHorizon")?.value || 180);

  try {
    const res = await fetch(`${API_BASE}/api/simulation/what-if`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        well_id: state.currentWellId,
        horizon_days: horizon,
        extraction_change_pct: extChange,
        artificial_recharge_m3_day: rechargeM3,
        rainfall_modifier_pct: rainMod
      })
    });
    const data = await res.json();

    const dates = data.dates || [];
    const scenarios = data.scenarios || {};
    const datasets = [];

    // Build chart datasets for each scenario curve
    Object.entries(scenarios).forEach(([name, sc]) => {
      datasets.push({
        label: name,
        data: sc.values,
        borderColor: sc.color,
        borderWidth: name.includes("Scenario E") ? 3.5 : 2.2,
        borderDash: name.includes("Scenario E") ? [4, 3] : [],
        pointRadius: 0,
        tension: 0.25
      });
    });

    // Reference lines
    datasets.push({
      label: `Critical Threshold (${data.critical_threshold_m}m)`,
      data: Array(dates.length).fill(data.critical_threshold_m),
      borderColor: "#EF4444",
      borderWidth: 1.5,
      borderDash: [6, 4],
      pointRadius: 0,
      fill: false
    });

    const ctx = document.getElementById("whatIfChart");
    if (ctx) {
      if (state.charts.whatIf) state.charts.whatIf.destroy();

      state.charts.whatIf = new Chart(ctx, {
        type: "line",
        data: { labels: dates, datasets: datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: "index", intersect: false },
          plugins: {
            legend: {
              position: "top",
              labels: { color: "#94A3B8", font: { size: 11 }, boxWidth: 14 }
            }
          },
          scales: {
            x: {
              grid: { color: "rgba(148, 163, 184, 0.08)" },
              ticks: { color: "#64748B", maxTicksLimit: 10, font: { size: 10 } }
            },
            y: {
              reverse: true,
              grid: { color: "rgba(148, 163, 184, 0.08)" },
              ticks: { color: "#94A3B8", font: { size: 11 }, callback: (v) => `${v} m` },
              title: { display: true, text: "Simulated Depth to Water Table (m bgl)", color: "#94A3B8" }
            }
          }
        }
      });
    }

    // Populate Comparative Table
    const tableBody = document.getElementById("scenarioTableBody");
    if (tableBody) {
      tableBody.innerHTML = "";
      const baselineEnd = scenarios["Scenario A (Current Baseline)"] ? scenarios["Scenario A (Current Baseline)"].final_level_m : 25.0;

      Object.entries(scenarios).forEach(([name, sc]) => {
        const savedM = (baselineEnd - sc.final_level_m).toFixed(2);
        const status = sc.final_level_m < data.safe_threshold_m ? "Safe" : (sc.final_level_m < data.critical_threshold_m ? "Moderate" : "Critical");
        const statusClass = status === "Safe" ? "status-low" : (status === "Moderate" ? "status-moderate" : "status-critical");

        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong style="color: ${sc.color};">${name}</strong></td>
          <td><small>${sc.description}</small></td>
          <td style="font-family: monospace; font-weight: 600;">${sc.final_level_m.toFixed(2)} m</td>
          <td style="font-family: monospace; color: ${savedM > 0 ? '#34D399' : '#94A3B8'}; font-weight: 600;">
            ${savedM > 0 ? '+' + savedM + ' m preserved' : '0.00 m (Baseline)'}
          </td>
          <td><span class="status-pill ${statusClass}" style="padding: 2px 8px; font-size: 11px;">${status}</span></td>
        `;
        tableBody.appendChild(tr);
      });
    }
  } catch (err) {
    console.error("Failed to run what-if simulation:", err);
  }
}

// ================= RISK & EARLY WARNINGS =================
async function loadRiskAndAlerts() {
  try {
    const res = await fetch(`${API_BASE}/api/risk/${state.currentWellId}`);
    const data = await res.json();
    const risk = data.risk_evaluation || {};
    const alerts = data.early_warnings || [];

    // Update circular gauge
    const score = risk.risk_index || 0;
    document.getElementById("riskGaugeScore").textContent = score.toFixed(1);
    document.getElementById("riskCategoryBadge").textContent = risk.category || "Moderate Risk";
    document.getElementById("riskCategoryBadge").style.color = risk.color_code || "#F59E0B";

    const circle = document.getElementById("gaugeRiskCircle");
    if (circle) {
      circle.setAttribute("stroke-dasharray", `${score}, 100`);
      circle.style.stroke = risk.color_code || "#F59E0B";
    }

    // Risk Factor Bars
    const factorsList = document.getElementById("riskFactorsList");
    if (factorsList && risk.component_scores) {
      factorsList.innerHTML = "";
      const components = [
        { label: "Water Level Threshold Proximity", val: risk.component_scores.level_proximity_score, max: 35 },
        { label: "Rate of Decline Velocity", val: risk.component_scores.decline_velocity_score, max: 25 },
        { label: "Precipitation Deficit Anomaly", val: risk.component_scores.rainfall_deficit_score, max: 20 },
        { label: "Extraction & Temperature Stress", val: risk.component_scores.anthropogenic_pressure_score, max: 20 }
      ];

      components.forEach(comp => {
        const pct = ((comp.val / comp.max) * 100).toFixed(0);
        const item = document.createElement("div");
        item.className = "factor-item";
        item.innerHTML = `
          <div class="factor-header">
            <span style="color: #94A3B8;">${comp.label}</span>
            <span style="font-family: monospace; font-weight: 600;">${comp.val} / ${comp.max} pts</span>
          </div>
          <div class="factor-progress-bg">
            <div class="factor-progress-bar" style="width: ${pct}%;"></div>
          </div>
        `;
        factorsList.appendChild(item);
      });
    }

    // Early-Warning Alerts Feed
    const alertsFeed = document.getElementById("alertsFeed");
    const activeAlertCount = document.getElementById("activeAlertCount");
    if (activeAlertCount) {
      activeAlertCount.textContent = `${alerts.length} Active Alert${alerts.length === 1 ? '' : 's'}`;
    }

    if (alertsFeed) {
      alertsFeed.innerHTML = "";
      if (alerts.length === 0) {
        alertsFeed.innerHTML = `
          <div class="nlp-diagnostic-card">
            <div class="nlp-icon">✅</div>
            <div class="nlp-content">
              <div class="nlp-title">All Clear</div>
              <p class="nlp-text">No imminent threshold breaches or anomalous decline conditions detected for this station.</p>
            </div>
          </div>
        `;
      } else {
        alerts.forEach(alert => {
          const card = document.createElement("div");
          const sevClass = alert.severity === "CRITICAL" ? "alert-critical" : (alert.severity === "HIGH" ? "alert-high" : "alert-warning");
          card.className = `alert-card ${sevClass}`;
          card.innerHTML = `
            <div class="alert-title-row">
              <h4 class="alert-title">${alert.title}</h4>
              <span class="alert-severity-tag status-${sevClass.split('-')[1]}">${alert.severity}</span>
            </div>
            <p class="alert-message">${alert.message}</p>
            <div class="alert-meta-grid">
              <div class="alert-meta-item">
                <span>Predicted Risk Window:</span><br>
                <strong>${alert.projected_window}</strong>
              </div>
              <div class="alert-meta-item">
                <span>Trigger Level:</span><br>
                <strong>${alert.trigger_level_m} m bgl</strong>
              </div>
            </div>
            <div class="alert-action-rec">
              🛡️ Recommendation: ${alert.recommended_action}
            </div>
          `;
          alertsFeed.appendChild(card);
        });
      }
    }
  } catch (err) {
    console.error("Failed to load risk and alerts:", err);
  }
}

// ================= EXPLAINABLE AI (SHAP) =================
async function loadXAIData() {
  await Promise.all([
    loadXAIGlobalImportance(),
    loadXAILocalExplanation()
  ]);
}

async function loadXAIGlobalImportance() {
  try {
    const res = await fetch(`${API_BASE}/api/xai/importance?well_id=${state.currentWellId}`);
    const data = await res.json();
    const topFeats = data.top_features || [];

    const labels = topFeats.map(f => f.feature.replace(/_/g, ' '));
    const values = topFeats.map(f => f.relative_importance_pct);

    const ctx = document.getElementById("xaiGlobalChart");
    if (ctx) {
      if (state.charts.xaiGlobal) state.charts.xaiGlobal.destroy();

      state.charts.xaiGlobal = new Chart(ctx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [{
            label: "Relative Attribution (%)",
            data: values,
            backgroundColor: "rgba(56, 189, 248, 0.75)",
            borderRadius: 4
          }]
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: {
              grid: { color: "rgba(148, 163, 184, 0.08)" },
              ticks: { color: "#94A3B8", callback: (v) => `${v}%` }
            },
            y: {
              grid: { display: false },
              ticks: { color: "#E2E8F0", font: { size: 11 } }
            }
          }
        }
      });
    }

    const quoteEl = document.getElementById("xaiGlobalQuote");
    if (quoteEl && data.interpretation) {
      quoteEl.textContent = data.interpretation;
    }
  } catch (err) {
    console.error("Failed to load XAI global importance:", err);
  }
}

async function loadXAILocalExplanation() {
  const dayOffset = parseInt(document.getElementById("xaiDayOffset")?.value || 15);
  try {
    const res = await fetch(`${API_BASE}/api/xai/explain?well_id=${state.currentWellId}&day_offset=${dayOffset}`);
    const data = await res.json();

    // Natural Language Diagnostic Box
    const diagEl = document.getElementById("xaiLocalDiagnosis");
    if (diagEl) {
      diagEl.textContent = data.diagnostic_summary || "Diagnosis complete.";
    }

    // Waterfall Chart
    const waterfall = data.waterfall_contributions || [];
    const labels = waterfall.map(w => w.feature.replace(/_/g, ' '));
    const values = waterfall.map(w => w.shap_impact_m);
    const bgColors = values.map(v => v > 0 ? "rgba(239, 68, 68, 0.8)" : "rgba(16, 185, 129, 0.8)");

    const ctx = document.getElementById("xaiWaterfallChart");
    if (ctx) {
      if (state.charts.xaiWaterfall) state.charts.xaiWaterfall.destroy();

      state.charts.xaiWaterfall = new Chart(ctx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [{
            label: "SHAP Impact on Water Table (meters)",
            data: values,
            backgroundColor: bgColors,
            borderRadius: 4
          }]
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (c) => `${c.raw > 0 ? '+' : ''}${c.raw} m (${c.raw > 0 ? 'Drawdown Stress' : 'Recharge Benefit'})`
              }
            }
          },
          scales: {
            x: {
              grid: { color: "rgba(148, 163, 184, 0.08)" },
              ticks: { color: "#94A3B8", callback: (v) => `${v}m` }
            },
            y: {
              grid: { display: false },
              ticks: { color: "#E2E8F0", font: { size: 11 } }
            }
          }
        }
      });
    }
  } catch (err) {
    console.error("Failed to load XAI local explanation:", err);
  }
}

// ================= ANOMALY DETECTION =================
async function loadAnomaliesData() {
  try {
    const res = await fetch(`${API_BASE}/api/anomalies/${state.currentWellId}`);
    const data = await res.json();

    document.getElementById("anomTotalCount").textContent = `Total Anomalies: ${data.total_anomalies_detected}`;
    document.getElementById("anomPct").textContent = `Rate: ${data.anomaly_rate_pct}%`;

    const anomPoints = data.anomaly_points || [];

    // History to plot continuous line
    const histRes = await fetch(`${API_BASE}/api/wells/${state.currentWellId}/history?days=730`);
    const histData = await histRes.json();
    const history = histData.history || [];

    const labels = history.map(h => h.date);
    const levels = history.map(h => h.groundwater_level);

    // Map anomaly markers to exact dates
    const anomMap = {};
    anomPoints.forEach(p => { anomMap[p.date] = p; });

    const anomalyScatter = history.map(h => anomMap[h.date] ? h.groundwater_level : null);

    const ctx = document.getElementById("anomalyScatterChart");
    if (ctx) {
      if (state.charts.anomalyScatter) state.charts.anomalyScatter.destroy();

      state.charts.anomalyScatter = new Chart(ctx, {
        type: "line",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Groundwater Level Telemetry",
              data: levels,
              borderColor: "#38BDF8",
              borderWidth: 1.8,
              pointRadius: 0,
              tension: 0.2
            },
            {
              label: "Detected Outlier / Anomaly Event",
              data: anomalyScatter,
              type: "scatter",
              backgroundColor: "#EF4444",
              borderColor: "#FFFFFF",
              borderWidth: 2,
              pointRadius: 6,
              pointHoverRadius: 9
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "top",
              labels: { color: "#94A3B8", boxWidth: 14 }
            },
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  const date = labels[ctx.dataIndex];
                  const anom = anomMap[date];
                  if (anom) {
                    return `⚠️ ${anom.anomaly_type} (${anom.severity}) - Score: ${anom.anomaly_score}`;
                  }
                  return `Level: ${ctx.raw} m`;
                }
              }
            }
          },
          scales: {
            x: {
              grid: { color: "rgba(148, 163, 184, 0.08)" },
              ticks: { color: "#64748B", maxTicksLimit: 12 }
            },
            y: {
              reverse: true,
              grid: { color: "rgba(148, 163, 184, 0.08)" },
              ticks: { color: "#94A3B8", callback: (v) => `${v} m` },
              title: { display: true, text: "Water Table (m bgl)", color: "#94A3B8" }
            }
          }
        }
      });
    }

    // Populate Table
    const tableBody = document.getElementById("anomaliesTableBody");
    if (tableBody) {
      tableBody.innerHTML = "";
      anomPoints.slice(-15).reverse().forEach(anom => {
        const tr = document.createElement("tr");
        const sevClass = anom.severity === "Critical" ? "status-critical" : (anom.severity === "High" ? "status-high" : "status-moderate");
        tr.innerHTML = `
          <td>${anom.date}</td>
          <td style="font-family: monospace; font-weight: 600;">${anom.groundwater_level.toFixed(2)} m</td>
          <td><strong>${anom.anomaly_type}</strong></td>
          <td><span class="status-pill ${sevClass}" style="padding: 2px 8px; font-size: 11px;">${anom.severity}</span></td>
          <td style="font-family: monospace;">${anom.anomaly_score.toFixed(3)}</td>
          <td>${anom.rainfall.toFixed(1)} mm</td>
          <td>${anom.extraction_rate.toFixed(2)}</td>
        `;
        tableBody.appendChild(tr);
      });
    }
  } catch (err) {
    console.error("Failed to load anomalies:", err);
  }
}

// ================= RECOVERY & CLIMATE =================
async function loadRecoverySimulation() {
  const rainEvent = parseFloat(document.getElementById("recRainEvent")?.value || 80);
  const cond = parseFloat(document.getElementById("recCond")?.value || 2.0);

  try {
    const res = await fetch(`${API_BASE}/api/simulation/recovery`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        well_id: state.currentWellId,
        rainfall_event_mm: rainEvent,
        hydraulic_conductivity: cond,
        vadose_zone_thickness_m: 12.0
      })
    });
    const data = await res.json();

    document.getElementById("recLagVal").textContent = `${data.post_rainfall_response_lag_days} Days`;
    document.getElementById("recPeriodVal").textContent = `${data.expected_recovery_period_days} Days`;
    document.getElementById("recRiseVal").textContent = `+${data.total_estimated_recovery_m} m`;
    document.getElementById("recPeakRateVal").textContent = `${data.peak_recovery_rate_m_day} m/d`;
    document.getElementById("recInterpretationText").textContent = data.scientific_interpretation;

    const curve = data.daily_recovery_trajectory || [];
    const labels = Array.from({ length: curve.length }, (_, i) => `Day +${i + 1}`);

    const ctx = document.getElementById("recoveryCurveChart");
    if (ctx) {
      if (state.charts.recoveryCurve) state.charts.recoveryCurve.destroy();

      state.charts.recoveryCurve = new Chart(ctx, {
        type: "line",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Post-Event Water Table Recovery Curve",
              data: curve,
              borderColor: "#10B981",
              backgroundColor: "rgba(16, 185, 129, 0.12)",
              borderWidth: 2.5,
              fill: true,
              pointRadius: 0,
              tension: 0.3
            },
            {
              label: `Safe Target (${data.safe_target_depth_m}m)`,
              data: Array(curve.length).fill(data.safe_target_depth_m),
              borderColor: "#38BDF8",
              borderDash: [6, 4],
              borderWidth: 1.5,
              pointRadius: 0,
              fill: false
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { position: "top", labels: { color: "#94A3B8" } } },
          scales: {
            x: {
              grid: { color: "rgba(148, 163, 184, 0.08)" },
              ticks: { color: "#64748B", maxTicksLimit: 10 }
            },
            y: {
              reverse: true,
              grid: { color: "rgba(148, 163, 184, 0.08)" },
              ticks: { color: "#94A3B8", callback: (v) => `${v} m` }
            }
          }
        }
      });
    }
  } catch (err) {
    console.error("Failed to load recovery simulation:", err);
  }
}

async function loadClimateSimulation() {
  try {
    const res = await fetch(`${API_BASE}/api/simulation/climate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ well_id: state.currentWellId, horizon_days: 365 })
    });
    const data = await res.json();
    const dates = data.dates || [];
    const scenarios = data.scenarios || {};

    const datasets = [];
    Object.entries(scenarios).forEach(([name, sc]) => {
      datasets.push({
        label: name,
        data: sc.values,
        borderColor: sc.color,
        borderWidth: 2.2,
        pointRadius: 0,
        tension: 0.25
      });
    });

    const ctx = document.getElementById("climateChart");
    if (ctx) {
      if (state.charts.climate) state.charts.climate.destroy();

      state.charts.climate = new Chart(ctx, {
        type: "line",
        data: { labels: dates, datasets: datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: "top", labels: { color: "#94A3B8", font: { size: 11 }, boxWidth: 14 } }
          },
          scales: {
            x: {
              grid: { color: "rgba(148, 163, 184, 0.08)" },
              ticks: { color: "#64748B", maxTicksLimit: 12 }
            },
            y: {
              reverse: true,
              grid: { color: "rgba(148, 163, 184, 0.08)" },
              ticks: { color: "#94A3B8", callback: (v) => `${v} m` },
              title: { display: true, text: "Projected Water Table (m bgl)", color: "#94A3B8" }
            }
          }
        }
      });
    }
  } catch (err) {
    console.error("Failed to load climate simulation:", err);
  }
}

// ================= HEALTH & RECOMMENDATIONS =================
async function loadHealthAndRecommendations() {
  await Promise.all([
    loadHealthScore(),
    loadRechargeRecommendations()
  ]);
}

async function loadHealthScore() {
  try {
    const res = await fetch(`${API_BASE}/api/health-score/${state.currentWellId}`);
    const data = await res.json();

    document.getElementById("healthBigScore").textContent = data.score.toFixed(1);
    document.getElementById("healthGradeBadge").textContent = `Grade: ${data.grade}`;
    document.getElementById("healthStatusTitle").textContent = data.classification;
    document.getElementById("healthStatusDesc").textContent = data.summary_description;

    // Radar Chart
    const radarPillars = data.radar_pillars || {};
    const radarLabels = Object.keys(radarPillars);
    const radarVals = Object.values(radarPillars);

    const ctx = document.getElementById("healthRadarChart");
    if (ctx) {
      if (state.charts.healthRadar) state.charts.healthRadar.destroy();

      state.charts.healthRadar = new Chart(ctx, {
        type: "radar",
        data: {
          labels: radarLabels,
          datasets: [{
            label: "Pillar Resilience Score (%)",
            data: radarVals,
            backgroundColor: "rgba(16, 185, 129, 0.25)",
            borderColor: "#10B981",
            borderWidth: 2,
            pointBackgroundColor: "#10B981"
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            r: {
              angleLines: { color: "rgba(148, 163, 184, 0.15)" },
              grid: { color: "rgba(148, 163, 184, 0.15)" },
              pointLabels: { color: "#E2E8F0", font: { size: 10 } },
              ticks: { display: false },
              suggestedMin: 20,
              suggestedMax: 100
            }
          }
        }
      });
    }

    // Factors Table
    const tableBody = document.getElementById("healthFactorsTableBody");
    if (tableBody && data.factors_breakdown) {
      tableBody.innerHTML = "";
      data.factors_breakdown.forEach(f => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${f.pillar}</strong></td>
          <td style="font-family: monospace;">${f.score} / ${f.max_score}</td>
          <td><span class="badge" style="font-size: 11px;">${f.status}</span></td>
          <td style="color: #94A3B8;"><small>${f.details}</small></td>
        `;
        tableBody.appendChild(tr);
      });
    }
  } catch (err) {
    console.error("Failed to load health score:", err);
  }
}

async function loadRechargeRecommendations() {
  try {
    const res = await fetch(`${API_BASE}/api/recommendations/${state.currentWellId}`);
    const data = await res.json();

    document.getElementById("recsExecutiveQuote").textContent = data.executive_recommendation_quote || "";

    const grid = document.getElementById("recsActionsGrid");
    if (grid && data.actions_list) {
      grid.innerHTML = "";
      data.actions_list.forEach(action => {
        const card = document.createElement("div");
        card.className = "action-card";
        card.innerHTML = `
          <div class="action-header-row">
            <h4 class="action-title">${action.action}</h4>
            <span class="action-badge">${action.urgency_badge}</span>
          </div>
          <div style="font-size: 11px; color: #06B6D4; font-weight: 600;">${action.category}</div>
          <p class="action-desc">${action.description}</p>
          <div class="action-impact">⚡ Impact: ${action.expected_impact}</div>
        `;
        grid.appendChild(card);
      });
    }
  } catch (err) {
    console.error("Failed to load recommendations:", err);
  }
}

// ================= RESEARCH BENCHMARK & ABLATION =================
async function loadResearchData() {
  try {
    // 1. Model Benchmark Comparison (Section 17)
    const resComp = await fetch(`${API_BASE}/api/models/comparison?well_id=${state.currentWellId}`);
    const dataComp = await resComp.json();
    const compTable = dataComp.comparison_table || {};

    const tableBody = document.getElementById("researchBenchmarkTableBody");
    if (tableBody) {
      tableBody.innerHTML = "";
      Object.entries(compTable).forEach(([mName, met]) => {
        const isChamp = mName === dataComp.champion_model;
        const tr = document.createElement("tr");
        if (isChamp) tr.style.backgroundColor = "rgba(16, 185, 129, 0.12)";

        let cat = "Baseline Statistical";
        if (mName.includes("Forest") || mName.includes("XGBoost")) cat = "Machine Learning";
        if (mName.includes("LSTM") || mName.includes("GRU")) cat = "Deep Recurrent Network";

        tr.innerHTML = `
          <td><strong>${mName}</strong> ${isChamp ? '🏆 <span class="badge badge-accent">Champion</span>' : ''}</td>
          <td><small>${cat}</small></td>
          <td style="font-family: monospace; font-weight: 600;">${met.MAE.toFixed(4)}</td>
          <td style="font-family: monospace; font-weight: 600;">${met.RMSE.toFixed(4)}</td>
          <td style="font-family: monospace;">${met.MAPE.toFixed(2)}%</td>
          <td style="font-family: monospace; font-weight: 700; color: ${met.R2 > 0.9 ? '#34D399' : '#F59E0B'};">${met.R2.toFixed(4)}</td>
          <td style="font-family: monospace;">${met.Training_Time_s.toFixed(3)}s</td>
          <td><span class="status-pill status-${isChamp ? 'low' : 'moderate'}" style="padding: 2px 8px; font-size: 11px;">${isChamp ? 'Optimal' : 'Evaluated'}</span></td>
        `;
        tableBody.appendChild(tr);
      });
    }

    // Test Predictions Overlay Chart
    const testDates = dataComp.test_dates || [];
    const testActuals = dataComp.test_actuals || [];
    const preds = dataComp.predictions || {};

    const ctx = document.getElementById("modelComparisonTestChart");
    if (ctx) {
      if (state.charts.modelComp) state.charts.modelComp.destroy();

      const testDatasets = [
        {
          label: "Ground Truth Actual Level",
          data: testActuals,
          borderColor: "#FFFFFF",
          borderWidth: 2.5,
          pointRadius: 0
        }
      ];

      const modelColors = {
        "Random Forest": "#10B981",
        "XGBoost": "#06B6D4",
        "GRU": "#8B5CF6",
        "LSTM": "#F59E0B",
        "SARIMA": "#EC4899",
        "ARIMA": "#64748B"
      };

      Object.entries(preds).forEach(([mName, vals]) => {
        testDatasets.push({
          label: `${mName} Prediction`,
          data: vals,
          borderColor: modelColors[mName] || "#94A3B8",
          borderWidth: mName.includes("Random Forest") ? 2.5 : 1.5,
          borderDash: mName.includes("Baseline") ? [4, 4] : [],
          pointRadius: 0
        });
      });

      state.charts.modelComp = new Chart(ctx, {
        type: "line",
        data: { labels: testDates, datasets: testDatasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: "top", labels: { color: "#94A3B8", font: { size: 11 } } },
            title: { display: true, text: "Empirical Test Evaluation (Actuals vs Predictions on Hold-out Horizon)", color: "#E2E8F0" }
          },
          scales: {
            x: {
              grid: { color: "rgba(148, 163, 184, 0.08)" },
              ticks: { color: "#64748B", maxTicksLimit: 12 }
            },
            y: {
              reverse: true,
              grid: { color: "rgba(148, 163, 184, 0.08)" },
              ticks: { color: "#94A3B8", callback: (v) => `${v} m` }
            }
          }
        }
      });
    }

    // 2. Ablation Study (Section 18)
    const resAbl = await fetch(`${API_BASE}/api/models/ablation?well_id=${state.currentWellId}`);
    const dataAbl = await resAbl.json();
    const ablSummary = dataAbl.ablation_summary || [];

    const ablBody = document.getElementById("ablationTableBody");
    if (ablBody) {
      ablBody.innerHTML = "";
      ablSummary.forEach(item => {
        const tr = document.createElement("tr");
        const isFull = item.feature_configuration.includes("Full");
        if (isFull) tr.style.backgroundColor = "rgba(6, 182, 212, 0.12)";

        tr.innerHTML = `
          <td><strong>${item.feature_configuration}</strong></td>
          <td style="font-family: monospace;">${item.features_count}</td>
          <td style="font-family: monospace;">${item.MAE.toFixed(4)}</td>
          <td style="font-family: monospace; font-weight: 600;">${item.RMSE.toFixed(4)}</td>
          <td style="font-family: monospace;">${item.MAPE.toFixed(2)}%</td>
          <td style="font-family: monospace; font-weight: 700; color: ${item.R2 > 0.9 ? '#34D399' : '#F59E0B'};">${item.R2.toFixed(4)}</td>
          <td style="font-family: monospace; color: ${item.rmse_reduction_vs_baseline_pct > 0 ? '#34D399' : '#94A3B8'}; font-weight: 600;">
            ${item.rmse_reduction_vs_baseline_pct > 0 ? '-' + item.rmse_reduction_vs_baseline_pct + '% error' : 'Baseline'}
          </td>
        `;
        ablBody.appendChild(tr);
      });
    }
  } catch (err) {
    console.error("Failed to load research data:", err);
  }
}
