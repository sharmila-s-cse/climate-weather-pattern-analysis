/**
 * Climate Intelligence - Client-Side Analytics Engine
 * ===================================================
 * Powers the 14 SPA views, Chart.js visualizations, dynamic cross-filtering,
 * interconnected drill-downs, anomaly modals, Data Explorer, and PWA integration.
 */

// Global State
const state = {
  data: null,
  records: [],
  filteredRecords: [],
  activeView: 'home',
  charts: {},
  filters: {
    year: 'all',
    season: 'all',
    condition: 'all'
  },
  explorer: {
    page: 1,
    pageSize: 15,
    sortCol: 'Date',
    sortAsc: false,
    search: '',
    categoryFilter: 'all',
    conditionFilter: 'all'
  }
};

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
  initLucideIcons();
  setupNavigation();
  setupMobileDrawer();
  setupThemeToggle();
  setupQrCodeModal();
  registerServiceWorker();
  
  await loadAnalyticsData();
  handleRoute();
});

function initLucideIcons() {
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

// --------------------------------------------------------------------------
// Data Ingestion & Fallback Handling
// --------------------------------------------------------------------------
async function loadAnalyticsData() {
  try {
    const summaryRes = await fetch('./data/processed/climate_analytics_summary.json');
    if (!summaryRes.ok) throw new Error(`HTTP ${summaryRes.status}`);
    state.data = await summaryRes.json();
    console.log('Successfully loaded climate analytics summary data.');
  } catch (err) {
    console.warn('Could not fetch summary JSON directly. Attempting local storage/fallback:', err);
  }

  // Populate static/dynamic UI components once data is loaded
  if (state.data) {
    renderKPIs();
    renderAnomalyList();
    renderSeasonalCards();
    renderAIInsights();
    renderHypothesisTests();
    renderRegressionModels();
    renderArchitectureStages();
    renderTechStack();
  }
}

async function loadExplorerRecords() {
  if (state.records.length > 0) return;
  try {
    const recRes = await fetch('./data/processed/climate_records_sample.json');
    if (recRes.ok) {
      state.records = await recRes.json();
      state.filteredRecords = [...state.records];
      renderExplorerTable();
    }
  } catch (err) {
    console.warn('Could not load detailed explorer records:', err);
  }
}

// --------------------------------------------------------------------------
// SPA Routing & Navigation
// --------------------------------------------------------------------------
function setupNavigation() {
  window.addEventListener('hashchange', handleRoute);
  
  document.querySelectorAll('[data-route]').forEach(el => {
    el.addEventListener('click', (e) => {
      e.preventDefault();
      const targetRoute = el.getAttribute('data-route');
      window.location.hash = targetRoute;
    });
  });
}

function handleRoute() {
  const hash = window.location.hash.replace('#', '') || 'home';
  state.activeView = hash;
  
  // Close mobile drawer on route change
  const sidebar = document.getElementById('sidebar');
  if (sidebar) sidebar.classList.remove('mobile-open');
  
  // Update view sections
  document.querySelectorAll('.view-section').forEach(sec => {
    sec.classList.remove('active');
  });
  
  const targetSection = document.getElementById(`view-${hash}`) || document.getElementById('view-home');
  if (targetSection) {
    targetSection.classList.add('active');
  }
  
  // Update sidebar active states
  document.querySelectorAll('.nav-link').forEach(link => {
    link.classList.remove('active');
    if (link.getAttribute('data-route') === hash) {
      link.classList.add('active');
    }
  });
  
  // Update mobile bottom nav
  document.querySelectorAll('.mobile-bottom-item').forEach(item => {
    item.classList.remove('active');
    if (item.getAttribute('data-route') === hash) {
      item.classList.add('active');
    }
  });
  
  // Update breadcrumb
  const breadcrumb = document.getElementById('current-breadcrumb');
  if (breadcrumb) {
    const formatted = hash.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    breadcrumb.textContent = formatted;
  }
  
  window.scrollTo({ top: 0, behavior: 'smooth' });
  
  // Initialize view-specific charts on view activation
  onViewActivated(hash);
}

function onViewActivated(viewName) {
  if (!state.data) return;
  
  setTimeout(() => {
    if (viewName === 'overview') {
      renderOverviewCharts();
    } else if (viewName === 'dashboard') {
      renderDashboardCharts();
    } else if (viewName === 'temperature') {
      renderTemperatureCharts();
    } else if (viewName === 'precipitation') {
      renderPrecipitationCharts();
    } else if (viewName === 'seasonal') {
      renderSeasonalCharts();
    } else if (viewName === 'purchasing') {
      renderPurchasingCharts();
    } else if (viewName === 'statistics') {
      renderStatisticsMatrix();
    } else if (viewName === 'explorer') {
      loadExplorerRecords();
    }
    initLucideIcons();
  }, 50);
}

// --------------------------------------------------------------------------
// Mobile Navigation Drawer
// --------------------------------------------------------------------------
function setupMobileDrawer() {
  const menuBtn = document.getElementById('mobile-menu-btn');
  const sidebar = document.getElementById('sidebar');
  
  if (menuBtn && sidebar) {
    menuBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      sidebar.classList.toggle('mobile-open');
    });
    
    document.addEventListener('click', (e) => {
      if (sidebar.classList.contains('mobile-open') && !sidebar.contains(e.target) && e.target !== menuBtn) {
        sidebar.classList.remove('mobile-open');
      }
    });
  }
}

// --------------------------------------------------------------------------
// Theme Toggle
// --------------------------------------------------------------------------
function setupThemeToggle() {
  const toggleBtn = document.getElementById('theme-toggle-btn');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      document.body.classList.toggle('theme-light');
      const isLight = document.body.classList.contains('theme-light');
      localStorage.setItem('climate_theme', isLight ? 'light' : 'dark');
      // Re-render charts for theme color adaptation
      onViewActivated(state.activeView);
    });
  }
  
  if (localStorage.getItem('climate_theme') === 'light') {
    document.body.classList.add('theme-light');
  }
}

// --------------------------------------------------------------------------
// Render Static KPIs & Components
// --------------------------------------------------------------------------
function renderKPIs() {
  const kpi = state.data.kpis;
  
  // Hero stats
  setText('hero-stat-records', Number(kpi.total_records).toLocaleString());
  setText('hero-stat-rainy', `${kpi.rainy_days_pct}%`);
  setText('hero-stat-anomalies', kpi.detected_anomalies);
  setText('hero-stat-r2', kpi.multivariable_r2);
  
  // Overview KPIs
  setText('kpi-avg-temp', `${kpi.avg_temperature}°C`);
  setText('kpi-max-temp', `${kpi.max_temperature}°C`);
  setText('kpi-min-temp', `${kpi.min_temperature}°C`);
  setText('kpi-total-precip', `${Number(kpi.total_precipitation_mm).toLocaleString()} mm`);
  setText('kpi-records', Number(kpi.total_records).toLocaleString());
  setText('kpi-anomalies', kpi.detected_anomalies);
  
  // Climate Dashboard KPIs
  setText('dash-kpi-temp', `${kpi.avg_temperature}°C`);
  setText('dash-kpi-precip', `${kpi.avg_precipitation_mm} mm/day`);
  setText('dash-kpi-sales', `$${Number(kpi.avg_daily_sales).toLocaleString()}`);
  setText('dash-kpi-r2', kpi.multivariable_r2);
  
  // Setup KPI Drill-Down Clicks
  setupDrillDowns();
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function setupDrillDowns() {
  document.querySelectorAll('[data-drill-target]').forEach(card => {
    card.addEventListener('click', () => {
      const target = card.getAttribute('data-drill-target');
      if (target) window.location.hash = target;
    });
  });
}

// --------------------------------------------------------------------------
// Anomaly Detection Page & Detail Modal
// --------------------------------------------------------------------------
function renderAnomalyList() {
  const listContainer = document.getElementById('anomaly-records-container');
  if (!listContainer || !state.data.anomalies) return;
  
  listContainer.innerHTML = '';
  
  state.data.anomalies.forEach((anom, idx) => {
    const card = document.createElement('div');
    card.className = 'anomaly-card';
    card.innerHTML = `
      <div style="display: flex; align-items: center; gap: 16px;">
        <div style="width: 42px; height: 42px; border-radius: 10px; background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.3); display: flex; align-items: center; justify-content: center; color: #ef4444;">
          <i data-lucide="cloud-lightning" style="width: 20px; height: 20px;"></i>
        </div>
        <div>
          <div class="anomaly-date">
            ${anom.date}
            <span class="severity-pill severity-${anom.severity.toLowerCase()}">${anom.severity}</span>
          </div>
          <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 2px;">
            Rainfall: <strong style="color: #ffffff;">${anom.precipitation_mm} mm</strong> | Z-Score: <strong style="color: var(--cyan-400);">${anom.z_score}σ</strong> | Condition: ${anom.weather_condition}
          </div>
        </div>
      </div>
      <div style="text-align: right;">
        <div style="font-size: 0.95rem; font-weight: 700; color: #ffffff;">$${Number(anom.day_total_spend).toLocaleString()}</div>
        <div style="font-size: 0.75rem; color: var(--emerald-400); font-weight: 600;">+${anom.spend_diff_pct}% spend surge</div>
      </div>
    `;
    
    card.addEventListener('click', () => openAnomalyModal(anom));
    listContainer.appendChild(card);
  });
  
  initLucideIcons();
}

function openAnomalyModal(anom) {
  const modal = document.getElementById('anomaly-detail-modal');
  if (!modal) return;
  
  setText('modal-anom-date', anom.date);
  setText('modal-anom-val', `${anom.precipitation_mm} mm`);
  setText('modal-anom-expected', `${anom.expected_mean_mm} mm (±${anom.std_deviation_mm} mm)`);
  setText('modal-anom-threshold', `${anom.anomaly_threshold_mm} mm (2.5σ threshold)`);
  setText('modal-anom-diff', `+${anom.deviation_mm} mm above average`);
  setText('modal-anom-zscore', `${anom.z_score} standard deviations`);
  setText('modal-anom-spend', `$${Number(anom.day_total_spend).toLocaleString()} (+${anom.spend_diff_pct}% vs baseline)`);
  setText('modal-anom-explanation', anom.explanation);
  
  const pill = document.getElementById('modal-anom-severity');
  if (pill) {
    pill.className = `severity-pill severity-${anom.severity.toLowerCase()}`;
    pill.textContent = anom.severity;
  }
  
  modal.classList.add('active');
}

window.closeAnomalyModal = function() {
  const modal = document.getElementById('anomaly-detail-modal');
  if (modal) modal.classList.remove('active');
};

// --------------------------------------------------------------------------
// Seasonal Patterns View
// --------------------------------------------------------------------------
function renderSeasonalCards() {
  const container = document.getElementById('seasonal-cards-container');
  if (!container || !state.data.seasonal_summary) return;
  
  container.innerHTML = '';
  const seasons = state.data.seasonal_summary;
  
  const seasonColors = {
    'Winter': { border: '#38bdf8', icon: 'snowflake' },
    'Spring': { border: '#34d399', icon: 'flower-2' },
    'Summer': { border: '#fbbf24', icon: 'sun' },
    'Fall': { border: '#f97316', icon: 'leaf' }
  };
  
  seasons.forEach(s => {
    const conf = seasonColors[s.season] || { border: '#06b6d4', icon: 'calendar' };
    const card = document.createElement('div');
    card.className = 'glass-card';
    card.style.borderTop = `3px solid ${conf.border}`;
    card.style.cursor = 'pointer';
    
    card.innerHTML = `
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;">
        <h3 style="font-size: 1.25rem; font-weight: 700; color: #ffffff;">${s.season}</h3>
        <i data-lucide="${conf.icon}" style="color: ${conf.border}; width: 22px; height: 22px;"></i>
      </div>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
        <div style="background: rgba(255,255,255,0.02); padding: 10px; border-radius: 8px;">
          <div style="font-size: 0.72rem; color: var(--text-muted);">Mean Temp</div>
          <div style="font-size: 1.15rem; font-weight: 700; color: var(--text-primary);">${s.mean_temp}°C</div>
        </div>
        <div style="background: rgba(255,255,255,0.02); padding: 10px; border-radius: 8px;">
          <div style="font-size: 0.72rem; color: var(--text-muted);">Total Rainfall</div>
          <div style="font-size: 1.15rem; font-weight: 700; color: var(--cyan-400);">${s.total_precip} mm</div>
        </div>
      </div>
      <div style="font-size: 0.78rem; color: var(--text-secondary); border-top: 1px solid var(--border-subtle); padding-top: 10px;">
        Top Category: <strong style="color: #ffffff;">${s.top_product_category}</strong>
        <div style="color: var(--emerald-400); font-weight: 600; margin-top: 2px;">$${Number(s.top_category_spend).toLocaleString()} gross sales</div>
      </div>
    `;
    
    container.appendChild(card);
  });
  
  initLucideIcons();
}

// --------------------------------------------------------------------------
// AI Insights Feed
// --------------------------------------------------------------------------
function renderAIInsights() {
  const container = document.getElementById('ai-insights-container');
  if (!container || !state.data.ai_insights) return;
  
  container.innerHTML = '';
  
  state.data.ai_insights.forEach(ins => {
    const card = document.createElement('div');
    card.className = 'glass-card';
    card.style.position = 'relative';
    
    card.innerHTML = `
      <div style="display: flex; align-items: flex-start; gap: 16px;">
        <div style="width: 44px; height: 44px; border-radius: 12px; background: rgba(6, 182, 212, 0.12); border: 1px solid rgba(6, 182, 212, 0.3); display: flex; align-items: center; justify-content: center; color: var(--cyan-400); flex-shrink: 0;">
          <i data-lucide="${ins.icon}" style="width: 22px; height: 22px;"></i>
        </div>
        <div style="flex: 1;">
          <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; flex-wrap: wrap; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
              <span style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: var(--cyan-400); letter-spacing: 0.05em;">${ins.category}</span>
              <span style="font-size: 0.68rem; padding: 2px 8px; border-radius: 9999px; background: rgba(255,255,255,0.06); color: var(--text-muted);">${ins.tag}</span>
            </div>
            <span style="font-size: 0.7rem; font-weight: 700; color: ${ins.impact_level === 'Critical' ? '#ef4444' : '#10b981'}; background: rgba(255,255,255,0.04); padding: 2px 8px; border-radius: 9999px;">
              ${ins.impact_level}
            </span>
          </div>
          <h4 style="font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 6px;">${ins.title}</h4>
          <p style="font-size: 0.88rem; color: var(--text-secondary); line-height: 1.5;">${ins.summary}</p>
        </div>
      </div>
    `;
    
    container.appendChild(card);
  });
  
  initLucideIcons();
}

// --------------------------------------------------------------------------
// Statistical Hypotheses & Regression Modeling Views
// --------------------------------------------------------------------------
function renderHypothesisTests() {
  const container = document.getElementById('hypothesis-tests-container');
  if (!container || !state.data.hypothesis_tests) return;
  
  container.innerHTML = '';
  
  state.data.hypothesis_tests.forEach(test => {
    const card = document.createElement('div');
    card.className = 'glass-card';
    card.innerHTML = `
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
        <h4 style="font-size: 1.05rem; font-weight: 700; color: #ffffff;">${test.name}</h4>
        <span style="font-size: 0.75rem; font-weight: 700; color: var(--emerald-400); background: rgba(16, 185, 129, 0.12); padding: 3px 10px; border-radius: 9999px; border: 1px solid rgba(16, 185, 129, 0.25);">
          ${test.p_value}
        </span>
      </div>
      <div style="font-family: var(--font-mono); font-size: 0.88rem; color: var(--cyan-400); margin-bottom: 12px; background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 6px;">
        Metric: ${test.metric} | Result: <strong>${test.result}</strong>
      </div>
      <div style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 8px;">
        <strong style="color: var(--text-primary);">What it means:</strong> ${test.meaning}
      </div>
      <div style="font-size: 0.85rem; color: var(--text-muted);">
        <strong style="color: var(--amber-400);">Why it matters:</strong> ${test.why_it_matters}
      </div>
    `;
    container.appendChild(card);
  });
}

function renderRegressionModels() {
  const container = document.getElementById('regression-models-container');
  if (!container || !state.data.regression_models) return;
  
  container.innerHTML = '';
  
  state.data.regression_models.forEach(model => {
    const card = document.createElement('div');
    card.className = 'glass-card';
    card.innerHTML = `
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
        <h4 style="font-size: 1.05rem; font-weight: 700; color: #ffffff;">${model.title}</h4>
        <span style="font-size: 0.82rem; font-weight: 700; color: var(--cyan-400); background: rgba(6, 182, 212, 0.12); padding: 4px 10px; border-radius: 9999px;">
          R² = ${model.r2}
        </span>
      </div>
      <div class="code-snippet" style="margin-bottom: 14px;">
        ${model.formula}
      </div>
      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 12px; text-align: center;">
        <div style="background: rgba(255,255,255,0.02); padding: 8px; border-radius: 6px;">
          <div style="font-size: 0.7rem; color: var(--text-muted);">R² Score</div>
          <div style="font-size: 0.95rem; font-weight: 700; color: #ffffff;">${model.r2}</div>
        </div>
        <div style="background: rgba(255,255,255,0.02); padding: 8px; border-radius: 6px;">
          <div style="font-size: 0.7rem; color: var(--text-muted);">MAE</div>
          <div style="font-size: 0.95rem; font-weight: 700; color: #ffffff;">$${model.mae}</div>
        </div>
        <div style="background: rgba(255,255,255,0.02); padding: 8px; border-radius: 6px;">
          <div style="font-size: 0.7rem; color: var(--text-muted);">RMSE</div>
          <div style="font-size: 0.95rem; font-weight: 700; color: #ffffff;">$${model.rmse}</div>
        </div>
      </div>
      <p style="font-size: 0.84rem; color: var(--text-secondary); line-height: 1.5;">${model.interpretation}</p>
    `;
    container.appendChild(card);
  });
}

// --------------------------------------------------------------------------
// Interactive Architecture Flow
// --------------------------------------------------------------------------
function renderArchitectureStages() {
  const container = document.getElementById('architecture-flow-container');
  if (!container || !state.data.architecture_stages) return;
  
  container.innerHTML = '';
  
  state.data.architecture_stages.forEach((stage, index) => {
    const node = document.createElement('div');
    node.className = `pipeline-node ${index === 0 ? 'active' : ''}`;
    node.innerHTML = `
      <div class="pipeline-node-header">
        <div style="display: flex; align-items: center; gap: 12px;">
          <span class="node-step-badge">Stage 0${index + 1}</span>
          <h4 style="font-size: 1.05rem; font-weight: 700; color: #ffffff;">${stage.title}</h4>
        </div>
        <div style="font-size: 0.75rem; color: var(--cyan-400); font-family: var(--font-mono);">${stage.tools}</div>
      </div>
      <div class="node-drawer">
        <p style="margin-bottom: 8px;">${stage.description}</p>
        <div style="font-size: 0.8rem; color: var(--amber-400); margin-bottom: 10px;">
          <strong>Why required:</strong> ${stage.why_required}
        </div>
        <div class="code-snippet">${stage.code_sample}</div>
      </div>
    `;
    
    node.addEventListener('click', () => {
      node.classList.toggle('active');
    });
    
    container.appendChild(node);
  });
}

// --------------------------------------------------------------------------
// Technology Stack Showcase
// --------------------------------------------------------------------------
function renderTechStack() {
  const container = document.getElementById('tech-stack-container');
  if (!container || !state.data.technologies) return;
  
  container.innerHTML = '';
  
  state.data.technologies.forEach(tech => {
    const card = document.createElement('div');
    card.className = 'glass-card';
    card.innerHTML = `
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
        <h4 style="font-size: 1.15rem; font-weight: 700; color: #ffffff;">${tech.name}</h4>
        <span style="font-size: 0.72rem; font-weight: 700; color: var(--cyan-400); background: rgba(6, 182, 212, 0.12); padding: 3px 9px; border-radius: 9999px;">
          ${tech.badge}
        </span>
      </div>
      <div style="font-size: 0.84rem; color: var(--text-secondary); margin-bottom: 8px;">
        <strong style="color: var(--text-primary);">Why used:</strong> ${tech.why}
      </div>
      <div style="font-size: 0.84rem; color: var(--text-muted);">
        <strong style="color: var(--cyan-400);">Project scope:</strong> ${tech.where}
      </div>
    `;
    container.appendChild(card);
  });
}

// --------------------------------------------------------------------------
// Chart.js Visualizations Setup & Helpers
// --------------------------------------------------------------------------
function destroyChart(key) {
  if (state.charts[key]) {
    state.charts[key].destroy();
    delete state.charts[key];
  }
}

function getChartDefaults() {
  const isLight = document.body.classList.contains('theme-light');
  return {
    gridColor: isLight ? 'rgba(0, 0, 0, 0.05)' : 'rgba(255, 255, 255, 0.06)',
    textColor: isLight ? '#475569' : '#94a3b8',
    tooltipBg: isLight ? '#ffffff' : '#0f172a',
    tooltipText: isLight ? '#0f172a' : '#f8fafc'
  };
}

function renderOverviewCharts() {
  const ts = state.data.daily_timeseries;
  if (!ts) return;
  
  // Downsample to weekly points for crisp performance
  const weekly = ts.filter((_, i) => i % 5 === 0);
  const labels = weekly.map(d => d.date);
  const theme = getChartDefaults();
  
  // Temperature & Precipitation Mini Chart
  destroyChart('overviewTrend');
  const ctx = document.getElementById('overviewTrendChart');
  if (ctx) {
    state.charts['overviewTrend'] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Temperature (°C)',
            data: weekly.map(d => d.temp),
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.08)',
            borderWidth: 2,
            fill: true,
            tension: 0.35,
            pointRadius: 0
          },
          {
            label: 'Precipitation (mm)',
            data: weekly.map(d => d.precip),
            borderColor: '#38bdf8',
            backgroundColor: 'rgba(56, 189, 248, 0.15)',
            borderWidth: 1.5,
            fill: true,
            yAxisID: 'yRain',
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { labels: { color: theme.textColor, font: { family: 'Inter' } } }
        },
        scales: {
          x: { grid: { color: theme.gridColor }, ticks: { color: theme.textColor, maxTicksLimit: 10 } },
          y: { grid: { color: theme.gridColor }, ticks: { color: theme.textColor } },
          yRain: { position: 'right', grid: { display: false }, ticks: { color: '#38bdf8' } }
        }
      }
    });
  }
}

function renderDashboardCharts(customSeries = null) {
  const ts = customSeries || state.data.daily_timeseries;
  if (!ts || ts.length === 0) return;
  
  const step = Math.max(1, Math.floor(ts.length / 80));
  const sampled = ts.filter((_, i) => i % step === 0);
  const labels = sampled.map(d => d.date);
  const theme = getChartDefaults();
  
  // 1. Dashboard Temp & MA
  destroyChart('dashTemp');
  const ctx1 = document.getElementById('dashTempChart');
  if (ctx1) {
    state.charts['dashTemp'] = new Chart(ctx1, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Temperature (°C)',
            data: sampled.map(d => d.temp),
            borderColor: 'rgba(245, 158, 11, 0.5)',
            borderWidth: 1,
            pointRadius: 0
          },
          {
            label: '30-Day Moving Average',
            data: sampled.map(d => d.temp_ma),
            borderColor: '#ef4444',
            borderWidth: 2.5,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: theme.textColor } } },
        scales: {
          x: { grid: { color: theme.gridColor }, ticks: { color: theme.textColor, maxTicksLimit: 8 } },
          y: { grid: { color: theme.gridColor }, ticks: { color: theme.textColor } }
        }
      }
    });
  }
  
  // 2. Dashboard Precipitation & MA
  destroyChart('dashPrecip');
  const ctx2 = document.getElementById('dashPrecipChart');
  if (ctx2) {
    state.charts['dashPrecip'] = new Chart(ctx2, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Rainfall (mm)',
            data: sampled.map(d => d.precip),
            backgroundColor: 'rgba(56, 189, 248, 0.65)',
            borderRadius: 4
          },
          {
            type: 'line',
            label: '30-Day MA (mm)',
            data: sampled.map(d => d.precip_ma),
            borderColor: '#0288d1',
            borderWidth: 2,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: theme.textColor } } },
        scales: {
          x: { grid: { color: theme.gridColor }, ticks: { color: theme.textColor, maxTicksLimit: 8 } },
          y: { grid: { color: theme.gridColor }, ticks: { color: theme.textColor } }
        }
      }
    });
  }
  
  // 3. Weather Condition Distribution Donut
  destroyChart('dashWeatherDist');
  const ctx3 = document.getElementById('dashWeatherDistChart');
  if (ctx3) {
    const counts = {};
    ts.forEach(d => {
      counts[d.condition] = (counts[d.condition] || 0) + 1;
    });
    const condLabels = Object.keys(counts);
    const condData = Object.values(counts);
    
    state.charts['dashWeatherDist'] = new Chart(ctx3, {
      type: 'doughnut',
      data: {
        labels: condLabels,
        datasets: [{
          data: condData,
          backgroundColor: ['#f59e0b', '#38bdf8', '#94a3b8', '#ef4444', '#a78bfa'],
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { color: theme.textColor } }
        },
        cutout: '70%'
      }
    });
  }
  
  // 4. Product Category Comparison Bar
  destroyChart('dashCatSpend');
  const ctx4 = document.getElementById('dashCatSpendChart');
  if (ctx4 && state.data.category_totals) {
    const cats = state.data.category_totals;
    state.charts['dashCatSpend'] = new Chart(ctx4, {
      type: 'bar',
      data: {
        labels: cats.map(c => c.category),
        datasets: [{
          label: 'Total Revenue ($)',
          data: cats.map(c => c.total_spend),
          backgroundColor: '#10b981',
          borderRadius: 6
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: theme.gridColor }, ticks: { color: theme.textColor } },
          y: { grid: { display: false }, ticks: { color: theme.textColor } }
        }
      }
    });
  }
}

window.onDashboardFilterChange = function() {
  if (!state.data || !state.data.daily_timeseries) return;
  
  const yearVal = document.getElementById('dash-year-select')?.value || 'all';
  const seasonVal = document.getElementById('dash-season-select')?.value || 'all';
  const condVal = document.getElementById('dash-cond-select')?.value || 'all';
  
  let filtered = state.data.daily_timeseries.filter(d => {
    const dYear = d.date.slice(0, 4);
    const matchesYear = (yearVal === 'all' || dYear === yearVal);
    const matchesSeason = (seasonVal === 'all' || d.season === seasonVal);
    const matchesCond = (condVal === 'all' || d.condition === condVal);
    return matchesYear && matchesSeason && matchesCond;
  });
  
  if (filtered.length === 0) {
    filtered = state.data.daily_timeseries;
  }
  
  const avgTemp = (filtered.reduce((acc, d) => acc + d.temp, 0) / filtered.length).toFixed(1);
  const avgPrecip = (filtered.reduce((acc, d) => acc + d.precip, 0) / filtered.length).toFixed(2);
  const avgSpend = (filtered.reduce((acc, d) => acc + d.spend, 0) / filtered.length).toFixed(2);
  
  setText('dash-kpi-temp', `${avgTemp}°C`);
  setText('dash-kpi-precip', `${avgPrecip} mm/day`);
  setText('dash-kpi-sales', `$${Number(avgSpend).toLocaleString()}`);
  
  renderDashboardCharts(filtered);
};

function renderTemperatureCharts() {
  const ts = state.data.daily_timeseries;
  const monthly = state.data.monthly_climatology;
  const theme = getChartDefaults();
  if (!ts || !monthly) return;
  
  // 1. Monthly Mean Temp with Bounds
  destroyChart('tempMonthly');
  const ctx1 = document.getElementById('tempMonthlyChart');
  if (ctx1) {
    state.charts['tempMonthly'] = new Chart(ctx1, {
      type: 'bar',
      data: {
        labels: monthly.map(m => m.month_name),
        datasets: [{
          label: 'Monthly Mean Temp (°C)',
          data: monthly.map(m => m.mean_temp),
          backgroundColor: monthly.map(m => m.mean_temp > 20 ? '#f59e0b' : (m.mean_temp < 10 ? '#38bdf8' : '#10b981')),
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: theme.textColor } } },
        scales: {
          x: { grid: { display: false }, ticks: { color: theme.textColor } },
          y: { grid: { color: theme.gridColor }, ticks: { color: theme.textColor } }
        }
      }
    });
  }
  
  // 2. Temp Bracket Distribution
  destroyChart('tempBracket');
  const ctx2 = document.getElementById('tempBracketChart');
  if (ctx2 && state.data.temperature_distribution) {
    const dist = state.data.temperature_distribution;
    state.charts['tempBracket'] = new Chart(ctx2, {
      type: 'doughnut',
      data: {
        labels: dist.map(d => d.bracket),
        datasets: [{
          data: dist.map(d => d.days),
          backgroundColor: ['#38bdf8', '#06b6d4', '#10b981', '#f59e0b', '#ef4444'],
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom', labels: { color: theme.textColor } } }
      }
    });
  }
}

function renderPrecipitationCharts() {
  const monthly = state.data.monthly_climatology;
  const rainDist = state.data.rainfall_distribution;
  const theme = getChartDefaults();
  if (!monthly || !rainDist) return;
  
  // 1. Monthly Rainfall
  destroyChart('precipMonthly');
  const ctx1 = document.getElementById('precipMonthlyChart');
  if (ctx1) {
    state.charts['precipMonthly'] = new Chart(ctx1, {
      type: 'bar',
      data: {
        labels: monthly.map(m => m.month_name),
        datasets: [{
          label: 'Total Rainfall (mm)',
          data: monthly.map(m => m.total_precip),
          backgroundColor: '#0288d1',
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: theme.textColor } } },
        scales: {
          x: { grid: { display: false }, ticks: { color: theme.textColor } },
          y: { grid: { color: theme.gridColor }, ticks: { color: theme.textColor } }
        }
      }
    });
  }
  
  // 2. Rainfall Intensity Classification
  destroyChart('rainIntensity');
  const ctx2 = document.getElementById('rainIntensityChart');
  if (ctx2) {
    state.charts['rainIntensity'] = new Chart(ctx2, {
      type: 'pie',
      data: {
        labels: rainDist.map(r => r.bracket),
        datasets: [{
          data: rainDist.map(r => r.days),
          backgroundColor: ['#94a3b8', '#38bdf8', '#0288d1', '#f97316', '#ef4444'],
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom', labels: { color: theme.textColor } } }
      }
    });
  }
}

function renderSeasonalCharts() {
  const seasons = state.data.seasonal_summary;
  const theme = getChartDefaults();
  if (!seasons) return;
  
  destroyChart('seasonalCompare');
  const ctx = document.getElementById('seasonalCompareChart');
  if (ctx) {
    state.charts['seasonalCompare'] = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: ['Mean Temp (°C)', 'Rainfall (mm/10)', 'Daily Sales ($/1k)'],
        datasets: seasons.map((s, idx) => {
          const colors = ['#38bdf8', '#34d399', '#fbbf24', '#f97316'];
          return {
            label: s.season,
            data: [s.mean_temp, s.total_precip / 10, s.avg_daily_spend / 1000],
            borderColor: colors[idx % colors.length],
            backgroundColor: `${colors[idx % colors.length]}22`,
            borderWidth: 2
          };
        })
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom', labels: { color: theme.textColor } } },
        scales: {
          r: {
            grid: { color: theme.gridColor },
            pointLabels: { color: theme.textColor, font: { size: 12 } },
            ticks: { display: false }
          }
        }
      }
    });
  }
}

function renderPurchasingCharts() {
  const cats = state.data.category_totals;
  const theme = getChartDefaults();
  if (!cats) return;
  
  destroyChart('purchasingShare');
  const ctx = document.getElementById('purchasingShareChart');
  if (ctx) {
    state.charts['purchasingShare'] = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: cats.map(c => c.category),
        datasets: [{
          data: cats.map(c => c.total_spend),
          backgroundColor: ['#06b6d4', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444'],
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom', labels: { color: theme.textColor } } }
      }
    });
  }
}

function renderStatisticsMatrix() {
  const container = document.getElementById('correlation-matrix-container');
  if (!container || !state.data.correlations) return;
  
  const corr = state.data.correlations;
  let html = '<div class="table-responsive"><table class="data-table" style="text-align: center;"><thead><tr><th>Variable</th>';
  corr.columns.forEach(col => {
    html += `<th>${col}</th>`;
  });
  html += '</tr></thead><tbody>';
  
  corr.matrix.forEach((row, rIdx) => {
    html += `<tr><td style="font-weight: 700; text-align: left; color: #ffffff;">${corr.columns[rIdx]}</td>`;
    row.forEach(val => {
      let bg = 'transparent';
      let color = '#f8fafc';
      if (val === 1.0) {
        bg = 'rgba(6, 182, 212, 0.2)';
        color = '#38bdf8';
      } else if (val > 0.4) {
        bg = 'rgba(16, 185, 129, 0.18)';
        color = '#34d399';
      } else if (val < -0.3) {
        bg = 'rgba(239, 68, 68, 0.18)';
        color = '#f87171';
      }
      html += `<td style="background: ${bg}; color: ${color}; font-family: var(--font-mono); font-weight: 600;">${val.toFixed(2)}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table></div>';
  container.innerHTML = html;
}

// --------------------------------------------------------------------------
// Data Explorer Logic (Search, Sort, Pagination, CSV Download)
// --------------------------------------------------------------------------
function renderExplorerTable() {
  const tbody = document.getElementById('explorer-table-body');
  if (!tbody) return;
  
  // Filter records
  let list = state.records.filter(r => {
    const matchesSearch = state.explorer.search === '' || 
      r.Date.includes(state.explorer.search) || 
      r.Product_Category.toLowerCase().includes(state.explorer.search.toLowerCase()) ||
      r.Weather_Condition.toLowerCase().includes(state.explorer.search.toLowerCase());
      
    const matchesCat = state.explorer.categoryFilter === 'all' || r.Product_Category === state.explorer.categoryFilter;
    const matchesCond = state.explorer.conditionFilter === 'all' || r.Weather_Condition === state.explorer.conditionFilter;
    
    return matchesSearch && matchesCat && matchesCond;
  });
  
  // Sort records
  list.sort((a, b) => {
    let valA = a[state.explorer.sortCol];
    let valB = b[state.explorer.sortCol];
    if (typeof valA === 'string') {
      return state.explorer.sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
    }
    return state.explorer.sortAsc ? valA - valB : valB - valA;
  });
  
  state.filteredRecords = list;
  
  // Pagination
  const total = list.length;
  const start = (state.explorer.page - 1) * state.explorer.pageSize;
  const paged = list.slice(start, start + state.explorer.pageSize);
  
  tbody.innerHTML = '';
  paged.forEach(row => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="font-family: var(--font-mono); font-weight: 600;">${row.Date}</td>
      <td>${row.Temperature}°C</td>
      <td>${row.Precipitation} mm</td>
      <td>${row.Humidity}%</td>
      <td><span class="view-badge" style="margin: 0;">${row.Weather_Condition}</span></td>
      <td><strong>${row.Product_Category}</strong></td>
      <td style="color: var(--emerald-400); font-weight: 700;">$${row.Purchase_Amount.toFixed(2)}</td>
      <td>${row.Number_of_Purchases}</td>
    `;
    tbody.appendChild(tr);
  });
  
  setText('explorer-page-info', `Showing ${start + 1} - ${Math.min(start + state.explorer.pageSize, total)} of ${total.toLocaleString()} records`);
  
  const prevBtn = document.getElementById('explorer-prev-btn');
  const nextBtn = document.getElementById('explorer-next-btn');
  if (prevBtn) prevBtn.disabled = state.explorer.page <= 1;
  if (nextBtn) nextBtn.disabled = start + state.explorer.pageSize >= total;
}

window.onExplorerSearch = function(query) {
  state.explorer.search = query;
  state.explorer.page = 1;
  renderExplorerTable();
};

window.onExplorerFilterChange = function() {
  const catSel = document.getElementById('explorer-category-select');
  const condSel = document.getElementById('explorer-condition-select');
  if (catSel) state.explorer.categoryFilter = catSel.value;
  if (condSel) state.explorer.conditionFilter = condSel.value;
  state.explorer.page = 1;
  renderExplorerTable();
};

window.onExplorerSort = function(columnKey) {
  if (state.explorer.sortCol === columnKey) {
    state.explorer.sortAsc = !state.explorer.sortAsc;
  } else {
    state.explorer.sortCol = columnKey;
    state.explorer.sortAsc = true;
  }
  renderExplorerTable();
};

window.changeExplorerPage = function(delta) {
  state.explorer.page += delta;
  renderExplorerTable();
};

window.exportFilteredCSV = function() {
  if (!state.filteredRecords || state.filteredRecords.length === 0) return;
  
  const headers = Object.keys(state.filteredRecords[0]).join(',');
  const rows = state.filteredRecords.map(r => Object.values(r).map(v => `"${v}"`).join(',')).join('\n');
  const csvContent = "data:text/csv;charset=utf-8," + headers + "\n" + rows;
  
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `climate_intelligence_export_${new Date().toISOString().slice(0,10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

// --------------------------------------------------------------------------
// QR Code Access Modal & URL Sharing
// --------------------------------------------------------------------------
function setupQrCodeModal() {
  const shareBtn = document.getElementById('share-qr-btn');
  const modal = document.getElementById('qr-modal');
  
  if (shareBtn && modal) {
    shareBtn.addEventListener('click', () => {
      openQrModal();
    });
  }
}

function openQrModal() {
  const modal = document.getElementById('qr-modal');
  if (!modal) return;
  
  const currentUrl = window.location.href;
  const qrImg = document.getElementById('qr-code-img');
  const urlDisplay = document.getElementById('qr-url-text');
  
  if (urlDisplay) urlDisplay.textContent = currentUrl;
  
  // Use public QR generator API or SVG renderer
  if (qrImg) {
    qrImg.src = `https://api.qrserver.com/v1/create-qr-code/?size=220x220&data=${encodeURIComponent(currentUrl)}`;
  }
  
  modal.classList.add('active');
}

window.closeQrModal = function() {
  const modal = document.getElementById('qr-modal');
  if (modal) modal.classList.remove('active');
};

// --------------------------------------------------------------------------
// Progressive Web App (PWA) Registration
// --------------------------------------------------------------------------
function registerServiceWorker() {
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('./sw.js')
        .then(reg => console.log('ServiceWorker registered with scope:', reg.scope))
        .catch(err => console.log('ServiceWorker registration skipped:', err));
    });
  }
}
