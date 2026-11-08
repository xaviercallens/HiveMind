// HiveMind web portal v3.2. No inline code; every value from the server is written with textContent, never as HTML.
"use strict";
(function () {
  const $ = (id) => document.getElementById(id);
  const show = (el, on) => { if (el) el.hidden = !on; };

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
    try { res = await fetch(path, opts); data = await res.json(); } catch (e) { return { ok: false, status: 0, data: { error: "Network error. Try again." } }; }
    return { ok: res.ok, status: res.status, data: data || {} };
  }

  function say(text, kind) {
    const m = $("msg");
    m.textContent = text;
    m.className = "msg " + (kind || "");
    show(m, !!text);
  }

  function formData(form) {
    const out = {};
    new FormData(form).forEach((v, k) => { out[k] = String(v); });
    return out;
  }

  // ---- facts -------------------------------------------------------------------------------------------------------
  async function loadFacts() {
    const r = await api("/v32/api/facts");
    const list = $("facts-list");
    list.textContent = "";
    if (!r.ok) { const li = document.createElement("li"); li.className = "muted"; li.textContent = "Facts are unavailable right now."; list.appendChild(li); return; }
    for (const f of r.data.facts || []) {
      const li = document.createElement("li");
      const left = document.createElement("span");
      left.textContent = f.label + (f.note ? " (" + f.note + ")" : "");
      const right = document.createElement("span");
      const v = document.createElement("span");
      v.className = "value";
      v.textContent = f.available ? Number(f.value).toLocaleString("en-US") : "unavailable";
      const src = document.createElement("a");
      src.className = "src";
      src.href = f.source;
      src.rel = "noopener";
      src.textContent = " source";
      right.append(v, src);
      li.append(left, right);
      list.appendChild(li);
    }
    const st = $("statements-list");
    st.textContent = "";
    for (const s of r.data.statements || []) {
      const li = document.createElement("li");
      li.textContent = s.text + " ";
      const a = document.createElement("a");
      a.className = "src"; a.href = s.source; a.rel = "noopener"; a.textContent = "source";
      li.appendChild(a);
      st.appendChild(li);
    }
  }

  // ---- account -----------------------------------------------------------------------------------------------------
  let caps = {};

  function selectTab(name) {
    document.querySelectorAll(".tabs [role=tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
    document.querySelectorAll("[data-panel]").forEach((p) => show(p, p.dataset.panel === name));
    show($("forgot-unavailable"), name === "forgot" && !caps.password_reset_email);
  }

  function showKeyOnce(key) {
    if (!key) return;
    $("key-once").textContent = key;
    show($("keybox"), true);
  }

  async function refresh() {
    const s = await api("/v32/api/session");
    const signedIn = s.ok && s.data.authenticated;
    const resetToken = (location.hash.match(/token=([A-Za-z0-9_-]{20,128})/) || [])[1];
    show($("unavailable"), !caps.sessions);
    show($("signed-out"), caps.sessions && !signedIn && !resetToken);
    show($("form-reset"), caps.sessions && !!resetToken);
    show($("signed-in"), signedIn);
    show($("sso"), !!caps.google);
    show($("link-google"), signedIn && !!caps.google);
    $("nav-account").textContent = signedIn ? "Account" : "Sign in";
    if (signedIn) { $("who").textContent = s.data.handle || "your account"; $("role").textContent = s.data.role ? "(" + s.data.role + ")" : ""; }
  }

  function bind(id, handler) {
    const f = $(id);
    if (f) f.addEventListener("submit", async (ev) => { ev.preventDefault(); const b = f.querySelector("button[type=submit]"); if (b) b.disabled = true; try { await handler(formData(f), f); } finally { if (b) b.disabled = false; } });
  }

  async function init() {
    const c = await api("/v32/api/capabilities");
    caps = c.ok ? c.data : {};
    document.querySelectorAll(".tabs [role=tab]").forEach((b) => b.addEventListener("click", () => selectTab(b.dataset.tab)));
    if (!caps.password_reset_email) { const b = $("form-forgot").querySelector("button"); if (b) b.disabled = true; }

    bind("form-login", async (d) => {
      const r = await api("/v32/api/login", d);
      if (r.ok) { say("Signed in.", "ok"); await refresh(); } else say(r.data.error || "Sign-in failed.", "err");
    });
    bind("form-register", async (d) => {
      const r = await api("/v32/api/register", d);
      if (r.ok) { say("Account ready.", "ok"); await refresh(); showKeyOnce(r.data.api_key_once); } else say(r.data.error || "Registration failed.", "err");
    });
    bind("form-forgot", async (d) => {
      const r = await api("/v32/api/password-reset/request", d);
      say(r.data.message || r.data.error || "Request sent.", r.ok ? "ok" : "err");
    });
    bind("form-reset", async (d) => {
      const token = (location.hash.match(/token=([A-Za-z0-9_-]{20,128})/) || [])[1] || "";
      const r = await api("/v32/api/password-reset/confirm", { token: token, new_password: d.new_password });
      if (r.ok) { history.replaceState(null, "", location.pathname + "#account"); say(r.data.message, "ok"); await refresh(); selectTab("login"); }
      else say(r.data.error || "The reset failed.", "err");
    });
    bind("form-change", async (d, f) => {
      const r = await api("/v32/api/change-password", d);
      if (r.ok) { f.reset(); say("Password changed.", "ok"); } else say(r.data.error || "The change failed.", "err");
    });
    $("rotate").addEventListener("click", async () => {
      const r = await api("/v32/api/rotate-key", {});
      if (r.ok) { showKeyOnce(r.data.api_key_once); say("New API key issued; the previous one no longer works.", "ok"); } else say(r.data.error || "Failed.", "err");
    });
    $("logout").addEventListener("click", async () => { await api("/v32/api/logout", {}); show($("keybox"), false); say("Signed out.", "ok"); await refresh(); });
    $("copy-key").addEventListener("click", async () => { try { await navigator.clipboard.writeText($("key-once").textContent); say("Copied.", "ok"); } catch (e) { say("Select the key and copy it.", "err"); } });

    const h = location.hash;
    if (h.indexOf("signin-error") >= 0) say("Google sign-in did not complete. Try again.", "err");
    if (h === "#linked") say("Your Google account is linked.", "ok");
    await refresh();
    selectTab("login");
    loadFacts();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
