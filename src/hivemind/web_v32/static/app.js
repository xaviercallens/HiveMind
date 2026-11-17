// HiveMind web portal v3.3. No inline code; every value from the server is written with textContent, never as HTML.
"use strict";
(function () {
  const $ = (id) => document.getElementById(id);
  const show = (el, on) => { if (el) el.hidden = !on; };
  const TOKEN_RE = /token=([A-Za-z0-9_-]{20,128})/;
  let strings = {};
  let lang = document.documentElement.dataset.lang === "fr" ? "fr" : "en";
  let caps = {};

  // ---- i18n -----------------------------------------------------------------------------------------------------------
  const t = (key, fallback) => (strings[lang] && strings[lang][key]) || (strings.en && strings.en[key]) || fallback || key;

  function applyLanguage() {
    document.documentElement.lang = lang;
    document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n, el.textContent); });
  }

  async function loadStrings() {
    try { const r = await fetch("/v32/static/i18n.json", { credentials: "same-origin" }); strings = await r.json(); } catch (e) { strings = {}; }
  }

  function setLanguage(next) {
    lang = next;
    document.cookie = "hm_lang=" + lang + "; Path=/; Max-Age=31536000; SameSite=Lax; Secure";
    applyLanguage();
    loadFacts();
    refresh();
  }

  // ---- HTTP -----------------------------------------------------------------------------------------------------------
  function csrf() {
    const m = document.cookie.match(/(?:^|; )hm_csrf=([^;]+)/);
    return m ? decodeURIComponent(m[1]) : "";
  }

  async function api(path, body) {
    const opts = { method: body === undefined ? "GET" : "POST", credentials: "same-origin", headers: { Accept: "application/json" } };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.headers["X-CSRF-Token"] = csrf();
      opts.body = JSON.stringify(body);
    }
    let res, data = {};
    try { res = await fetch(path, opts); data = await res.json(); } catch (e) { return { ok: false, status: 0, data: { code: "network" } }; }
    return { ok: res.ok, status: res.status, data: data || {} };
  }

  const errorText = (data, fallbackKey) =>
    data && data.code === "network" ? t("msg_network") : (data && data.code ? t("err_" + data.code, data.error) : t(fallbackKey));

  function say(text, kind) {
    const m = $("msg");
    m.textContent = text;
    m.className = "msg " + (kind || "");
    show(m, !!text);
    if (text) m.focus();
  }

  function formData(form) {
    const out = {};
    new FormData(form).forEach((v, k) => { out[k] = String(v); });
    return out;
  }

  // ---- facts ----------------------------------------------------------------------------------------------------------
  function link(href, key) {
    const a = document.createElement("a");
    a.className = "src"; a.href = href; a.rel = "noopener"; a.textContent = " " + t(key);
    return a;
  }

  async function loadFacts() {
    const r = await api("/v32/api/facts");
    const list = $("facts-list");
    list.textContent = "";
    if (!r.ok) { const li = document.createElement("li"); li.className = "muted"; li.textContent = t("facts_down"); list.appendChild(li); return; }
    for (const f of r.data.facts || []) {
      const li = document.createElement("li");
      const left = document.createElement("span");
      const note = f.note ? t("fact_" + f.key + "_note", f.note) : "";
      left.textContent = t("fact_" + f.key, f.label) + (note ? " (" + note + ")" : "");
      const right = document.createElement("span");
      const v = document.createElement("span");
      v.className = "value";
      v.textContent = f.available ? Number(f.value).toLocaleString(lang === "fr" ? "fr-FR" : "en-US") : t("unavailable");
      right.append(v, link(f.source, "source"));
      li.append(left, right);
      list.appendChild(li);
    }
    const st = $("statements-list");
    st.textContent = "";
    for (const s of r.data.statements || []) {
      const li = document.createElement("li");
      li.textContent = t("stmt_" + s.key, s.text);
      li.appendChild(link(s.source, "source"));
      st.appendChild(li);
    }
  }

  // ---- account --------------------------------------------------------------------------------------------------------
  const tabs = () => Array.from(document.querySelectorAll(".tabs [role=tab]"));

  function selectTab(name, focus) {
    tabs().forEach((b) => {
      const on = b.dataset.tab === name;
      b.setAttribute("aria-selected", String(on));
      b.tabIndex = on ? 0 : -1;
      if (on && focus) b.focus();
    });
    document.querySelectorAll("[data-panel]").forEach((p) => show(p, p.dataset.panel === name));
    show($("forgot-unavailable"), name === "forgot" && !caps.password_reset_email);
  }

  function onTabKey(ev) {
    const list = tabs();
    const i = list.indexOf(ev.target);
    let next = -1;
    if (ev.key === "ArrowRight") next = (i + 1) % list.length;
    else if (ev.key === "ArrowLeft") next = (i - 1 + list.length) % list.length;
    else if (ev.key === "Home") next = 0;
    else if (ev.key === "End") next = list.length - 1;
    if (next >= 0) { ev.preventDefault(); selectTab(list[next].dataset.tab, true); }
  }

  function showKeyOnce(key) {
    if (!key) return;
    $("key-once").textContent = key;
    show($("keybox"), true);
  }

  async function loadAccount() {
    const r = await api("/v32/api/account");
    if (!r.ok) { show($("balance-line"), false); show($("usage-table"), false); return; }
    const bal = r.data.balance;
    show($("balance-line"), bal !== null && bal !== undefined);
    if (bal !== null && bal !== undefined) {
      $("balance").textContent = Number(bal).toLocaleString(lang === "fr" ? "fr-FR" : "en-US", { maximumFractionDigits: 2 });
      $("account-source").href = r.data.source;
    }
    const rows = $("usage-rows");
    rows.textContent = "";
    const usage = r.data.usage || [];
    show($("usage-table"), usage.length > 0);
    show($("usage-none"), usage.length === 0);
    for (const u of usage) {
      const tr = document.createElement("tr");
      const when = u.utc ? new Date(u.utc * 1000).toLocaleString(lang === "fr" ? "fr-FR" : "en-US") : "";
      const cells = [when, u.model, u.tokens, u.charge !== null && u.charge !== undefined ? Number(u.charge).toFixed(3) : "", t(u.as === "producer" ? "as_producer" : "as_consumer")];
      cells.forEach((c, j) => { const td = document.createElement("td"); td.textContent = c === undefined || c === null ? "" : String(c); if (j === 2 || j === 3) td.className = "num"; tr.appendChild(td); });
      rows.appendChild(tr);
    }
  }

  async function refresh() {
    const s = await api("/v32/api/session");
    const signedIn = s.ok && s.data.authenticated;
    const resetToken = (location.hash.match(TOKEN_RE) || [])[1];
    show($("unavailable"), !caps.sessions);
    show($("signed-out"), caps.sessions && !signedIn && !resetToken);
    show($("form-reset"), caps.sessions && !!resetToken);
    show($("signed-in"), signedIn);
    show($("sso"), !!caps.google);
    show($("link-google"), signedIn && !!caps.google);
    const nav = $("nav-account");
    nav.dataset.i18n = signedIn ? "nav_account" : "nav_signin";
    nav.textContent = t(nav.dataset.i18n);
    if (signedIn) {
      $("who").textContent = s.data.handle || "";
      $("role").textContent = s.data.role ? "(" + s.data.role + ")" : "";
      loadAccount();
    }
  }

  function bind(id, handler) {
    const f = $(id);
    if (f) f.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const b = f.querySelector("button[type=submit]");
      if (b) b.disabled = true;
      try { await handler(formData(f), f); } finally { if (b) b.disabled = false; }
    });
  }

  async function init() {
    await loadStrings();
    applyLanguage();
    const c = await api("/v32/api/capabilities");
    caps = c.ok ? c.data : {};
    tabs().forEach((b) => { b.addEventListener("click", () => selectTab(b.dataset.tab, false)); b.addEventListener("keydown", onTabKey); });
    $("lang-toggle").addEventListener("click", () => setLanguage(lang === "fr" ? "en" : "fr"));
    if (!caps.password_reset_email) { const b = $("form-forgot").querySelector("button"); if (b) b.disabled = true; }

    bind("form-login", async (d) => {
      const r = await api("/v32/api/login", d);
      if (r.ok) { say(t("msg_signed_in"), "ok"); await refresh(); } else say(errorText(r.data, "err_bad_login"), "err");
    });
    bind("form-register", async (d) => {
      const r = await api("/v32/api/register", d);
      if (r.ok) { say(t("msg_account_ready"), "ok"); await refresh(); showKeyOnce(r.data.api_key_once); } else say(errorText(r.data, "err_register_refused"), "err");
    });
    bind("form-forgot", async (d) => {
      const r = await api("/v32/api/password-reset/request", d);
      say(r.ok ? t("msg_reset_sent") : errorText(r.data, "err_reset_unavailable"), r.ok ? "ok" : "err");
    });
    bind("form-reset", async (d) => {
      const token = (location.hash.match(TOKEN_RE) || [])[1] || "";
      const r = await api("/v32/api/password-reset/confirm", { token: token, new_password: d.new_password });
      if (r.ok) { history.replaceState(null, "", location.pathname + "#account"); say(t("msg_reset_done"), "ok"); await refresh(); selectTab("login", false); }
      else say(errorText(r.data, "err_reset_invalid"), "err");
    });
    bind("form-change", async (d, f) => {
      const r = await api("/v32/api/change-password", d);
      if (r.ok) { f.reset(); showKeyOnce(r.data.api_key_once); say(t("msg_password_changed"), "ok"); } else say(errorText(r.data, "err_weak_password"), "err");
    });
    $("rotate").addEventListener("click", async () => {
      const r = await api("/v32/api/rotate-key", {});
      if (r.ok) { showKeyOnce(r.data.api_key_once); say(t("msg_key_rotated"), "ok"); } else say(errorText(r.data, "err_session_expired"), "err");
    });
    $("logout").addEventListener("click", async () => { await api("/v32/api/logout", {}); show($("keybox"), false); say(t("msg_signed_out"), "ok"); await refresh(); });
    $("copy-key").addEventListener("click", async () => {
      try { await navigator.clipboard.writeText($("key-once").textContent); say(t("msg_copied"), "ok"); } catch (e) { say(t("msg_copy_failed"), "err"); }
    });

    const h = location.hash;
    if (h.indexOf("signin-error") >= 0) say(t("msg_google_failed"), "err");
    if (h === "#linked") say(t("msg_google_linked"), "ok");
    await refresh();
    selectTab("login", false);
    loadFacts();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
