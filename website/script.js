'use strict';
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const $ = (s, c = document) => c.querySelector(s);
const $$ = (s, c = document) => [...c.querySelectorAll(s)];

/* Mobile menu */
function initMenu() {
  const btn = $('.burger'), links = $('#links');
  btn.addEventListener('click', () => btn.setAttribute('aria-expanded', links.classList.toggle('open')));
  $$('a', links).forEach(a => a.addEventListener('click', () => { links.classList.remove('open'); btn.setAttribute('aria-expanded', 'false'); }));
}

/* Scroll reveal + animated counters */
function initReveal() {
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    e.target.classList.add('in');
    $$('[data-count]', e.target).forEach(countUp);
    io.unobserve(e.target);
  }), { threshold: .15 });
  $$('.reveal').forEach(el => io.observe(el));
}
function countUp(el) {
  const end = +el.dataset.count;
  if (reduce) { el.textContent = end.toLocaleString(); return; }
  const t0 = performance.now(), dur = 1400;
  (function tick(t) {
    const p = Math.min((t - t0) / dur, 1);
    el.textContent = Math.round(end * (1 - Math.pow(1 - p, 3))).toLocaleString();
    if (p < 1) requestAnimationFrame(tick);
  })(t0);
}

/* Active nav highlighting */
function initSpy() {
  const map = new Map($$('.links a[href^="#"]').map(a => [a.getAttribute('href').slice(1), a]));
  const io = new IntersectionObserver(es => es.forEach(e => {
    const a = map.get(e.target.id);
    if (a && e.isIntersecting) { map.forEach(x => x.classList.remove('active')); a.classList.add('active'); }
  }), { rootMargin: '-45% 0px -50% 0px' });
  map.forEach((_, id) => { const s = document.getElementById(id); if (s) io.observe(s); });
}

/* Charts (Chart.js, responsive by default) */
function initCharts() {
  const BLUE = '#4A63B0', TEAL = '#5BA59C', INK = '#1F2633', GRID = '#E8EAEE';
  Chart.defaults.font.family = 'Inter, sans-serif';
  Chart.defaults.color = '#687386';
  if (reduce) Chart.defaults.animation = false;
  const base = { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } };
  const pct = { callbacks: { label: c => `${c.dataset.label}: ${c.parsed.y}%` } };
  const yAxis = { beginAtZero: true, grid: { color: GRID }, ticks: { callback: v => v + '%' } };
  const bar = (id, labels, sets, extra = {}) => new Chart($(id), {
    type: 'bar',
    data: { labels, datasets: sets.map(([label, data, color]) => ({ label, data, backgroundColor: color, borderRadius: 6 })) },
    options: { ...base, ...extra, plugins: { ...base.plugins, tooltip: pct }, scales: { y: yAxis, x: { grid: { display: false } } } }
  });

  // Hero mini chart (simulated conversion rates)
  new Chart($('#heroChart'), {
    type: 'bar',
    data: { labels: ['Control', 'Treatment'], datasets: [{ data: [6.72, 8.93], backgroundColor: [BLUE, TEAL], borderRadius: 6 }] },
    options: { responsive: true, plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => c.parsed.y + '%' } } }, scales: { y: { beginAtZero: true, grid: { color: GRID }, ticks: { callback: v => v + '%' } }, x: { grid: { display: false } } } }
  });

  // Cluster distribution (observed)
  new Chart($('#clusterChart'), {
    type: 'doughnut',
    data: { labels: ['Casual', 'Regular', 'Dedicated'], datasets: [{ data: [6578, 3945, 1870], backgroundColor: [BLUE, TEAL, INK], borderWidth: 2, borderColor: '#fff' }] },
    options: { ...base, cutout: '62%', plugins: { ...base.plugins, tooltip: { callbacks: { label: c => `${c.label}: ${c.parsed.toLocaleString()} players` } } } }
  });

  // Overall A/B comparison (simulated)
  bar('#abChart', ['CTR', 'Conversion Rate'], [['Control A', [11.96, 6.72], BLUE], ['Treatment B', [16.19, 8.93], TEAL]]);
  // Segment-level (simulated)
  bar('#segChart', ['Casual', 'Regular', 'Dedicated'], [['Control', [5.01, 8.04, 10.03], BLUE], ['Treatment', [7.00, 9.10, 15.22], TEAL]]);
}

/* Chart lightbox */
function initLightbox() {
  const lb = $('#lb');
  $$('.sfig').forEach(f => f.addEventListener('click', () => { const i = $('img', f); $('img', lb).src = f.dataset.full; $('img', lb).alt = i.alt; lb.showModal(); }));
  $('button', lb).addEventListener('click', () => lb.close());
  lb.addEventListener('click', e => { if (e.target === lb) lb.close(); });
}

document.addEventListener('DOMContentLoaded', () => {
  [initMenu, initReveal, initSpy, initLightbox].forEach(fn => { try { fn(); } catch (e) { console.error(fn.name, e); } });
  const run = () => { try { initCharts(); } catch (e) { console.error('charts', e); } };
  if (window.Chart) run(); else window.addEventListener('load', () => window.Chart ? run() : console.error('Chart.js failed to load: check assets/chart.umd.js'));
});
