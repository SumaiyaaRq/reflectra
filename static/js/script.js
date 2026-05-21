/* ── Reflectra script.js ── */

// ── Mobile menu ──
function toggleMobileMenu() {
    const navbar = document.getElementById('navbar');
    const toggle = document.querySelector('.mobile-menu-toggle');
    if (!navbar) return;
    navbar.classList.toggle('active');
    if (toggle) toggle.classList.toggle('active');
}

document.addEventListener('click', function (e) {
    const navbar = document.getElementById('navbar');
    const toggle = document.querySelector('.mobile-menu-toggle');
    if (!navbar || !toggle) return;
    if (!navbar.contains(e.target) && !toggle.contains(e.target)) {
        navbar.classList.remove('active');
        toggle.classList.remove('active');
    }
});

// ── Dropdown menus ──
function toggleDropdown(e) {
    e.preventDefault();
    e.stopPropagation();
    const dropdown = e.currentTarget.closest('.nav-dropdown');
    const allDropdowns = document.querySelectorAll('.nav-dropdown');
    allDropdowns.forEach(d => { if (d !== dropdown) d.classList.remove('open'); });
    dropdown.classList.toggle('open');
}

document.addEventListener('click', function (e) {
    if (!e.target.closest('.nav-dropdown')) {
        document.querySelectorAll('.nav-dropdown').forEach(d => d.classList.remove('open'));
    }
});

// ── Search overlay ──
function toggleSearch() {
    const overlay = document.getElementById('search-overlay');
    if (!overlay) return;
    const isHidden = overlay.style.display === 'none' || overlay.style.display === '';
    overlay.style.display = isHidden ? 'block' : 'none';
    if (isHidden) {
        const input = document.getElementById('global-search');
        if (input) setTimeout(() => input.focus(), 100);
    }
}

function performSearch(query) {
    const resultsDiv = document.getElementById('search-results');
    if (!resultsDiv) return;
    if (!query.trim()) { resultsDiv.innerHTML = ''; return; }
    resultsDiv.innerHTML = `<div style="padding:1rem;color:#666;font-size:0.9rem;">
        🔍 Searching for "<strong>${query}</strong>" in your entries…
        <a href="/past_entries?q=${encodeURIComponent(query)}" style="color:var(--primary-color);margin-left:0.5rem;">View all results →</a>
    </div>`;
}

// ── Theme toggle ───
let currentTheme = localStorage.getItem('reflectra-theme') || 'light';
applyTheme(currentTheme);

function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    applyTheme(currentTheme);
    localStorage.setItem('reflectra-theme', currentTheme);
}

function applyTheme(theme) {
    const icon = document.getElementById('theme-icon');
    if (theme === 'dark') {
        document.body.style.background = 'linear-gradient(135deg,#1a1a2e,#16213e,#0f3460)';
        document.body.style.color = '#e0e0e0';
        document.body.style.backgroundSize = 'unset';
        document.body.style.animation = 'none';
        if (icon) icon.textContent = '☀️';
    } else {
        document.body.style.background = '';
        document.body.style.color = '';
        document.body.style.backgroundSize = '';
        document.body.style.animation = '';
        if (icon) icon.textContent = '🌙';
    }
}

// ── Flash messages ──
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.flash').forEach(flash => {
        setTimeout(() => dismissFlash(flash), 5000);
        flash.addEventListener('click', () => dismissFlash(flash));
    });
});

function dismissFlash(el) {
    el.style.opacity = '0';
    el.style.transform = 'translateX(380px)';
    el.style.transition = 'all 0.4s ease';
    setTimeout(() => el.remove(), 400);
}

// ── Smooth scroll ──
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        const target = document.querySelector(this.getAttribute('href'));
        if (target) { e.preventDefault(); target.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
    });
});

// ── Intersection observer fade-ins ──
const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.animation = 'fadeIn 0.5s ease both';
            observer.unobserve(entry.target);
        }
    });
}, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

document.querySelectorAll('.card, .glass-card, .feature-card, .entry-card').forEach(el => observer.observe(el));

// ── Button ripple ───
document.querySelectorAll('.btn').forEach(btn => {
    btn.addEventListener('click', function (e) {
        const ripple = document.createElement('span');
        const rect = this.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        ripple.style.cssText = `
            position:absolute;width:${size}px;height:${size}px;
            left:${e.clientX-rect.left-size/2}px;top:${e.clientY-rect.top-size/2}px;
            background:rgba(255,255,255,0.45);border-radius:50%;
            transform:scale(0);animation:ripple 0.55s ease-out;pointer-events:none;`;
        this.style.position = 'relative';
        this.style.overflow = 'hidden';
        this.appendChild(ripple);
        setTimeout(() => ripple.remove(), 600);
    });
});

const rippleCss = document.createElement('style');
rippleCss.textContent = '@keyframes ripple { to { transform:scale(4); opacity:0; } }';
document.head.appendChild(rippleCss);

// ── Stat counter animation ──
window.addEventListener('load', () => {
    document.querySelectorAll('.stat-value').forEach(stat => {
        const val = parseInt(stat.textContent);
        if (isNaN(val) || val <= 0 || val > 9999) return;
        let cur = 0;
        stat.textContent = '0';
        const inc = Math.max(1, Math.ceil(val / 35));
        const t = setInterval(() => {
            cur = Math.min(cur + inc, val);
            stat.textContent = cur;
            if (cur >= val) clearInterval(t);
        }, 28);
    });
});

// ── Keyboard shortcuts ───
document.addEventListener('keydown', e => {
    // Ctrl/Cmd+S → submit journal
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        const btn = document.getElementById('submit-btn');
        if (btn) btn.click();
    }
    // Escape → close overlays
    if (e.key === 'Escape') {
        document.querySelectorAll('.nav-dropdown').forEach(d => d.classList.remove('open'));
        const so = document.getElementById('search-overlay');
        if (so) so.style.display = 'none';
    }
});

// ── Global toast helper ───
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `flash ${type}`;
    toast.textContent = message;
    toast.style.cssText = 'position:fixed;top:70px;right:20px;z-index:9999;cursor:pointer;';
    document.body.appendChild(toast);
    toast.addEventListener('click', () => dismissFlash(toast));
    setTimeout(() => dismissFlash(toast), 4000);
}

window.Reflectra = { showToast };
console.log('✨ Reflectra loaded');