    /* ── Interfaz y estado ──────────────────────────────────── */
    function notify(message, type = 'success', duration = 4200) {
      const region = document.getElementById('toast-region');
      if (!region) return;
      const toast = document.createElement('div');
      toast.className = `toast ${type}`;
      toast.textContent = message;
      region.appendChild(toast);
      window.setTimeout(() => {
        toast.classList.add('leaving');
        window.setTimeout(() => toast.remove(), 220);
      }, duration);
    }

    function setButtonBusy(button, busy, busyLabel = 'Procesando…') {
      if (!button) return;
      if (busy) {
        button.dataset.idleLabel = button.textContent;
        button.disabled = true;
        button.setAttribute('aria-busy', 'true');
        button.textContent = `⏳ ${busyLabel}`;
      } else {
        button.disabled = false;
        button.removeAttribute('aria-busy');
        if (button.dataset.idleLabel) button.textContent = button.dataset.idleLabel;
      }
    }

    const FIELD_RULES = {
      fs: { min: 5, max: 500, required: true, label: 'Fs' },
      vas: { min: 0.1, max: 2000, required: true, label: 'Vas' },
      qts: { min: 0.05, max: 2, required: true, label: 'Qts' },
      qes: { min: 0.05, max: 5, label: 'Qes' },
      qms: { min: 0.1, max: 100, label: 'Qms' },
      xmax: { min: 0.01, max: 100, label: 'Xmax' },
      sd: { min: 1, max: 5000, label: 'Sd' },
      spl: { min: 50, max: 130, label: 'SPL' },
      mms: { min: 0.01, max: 1000, label: 'Mms' },
      bl: { min: 0.01, max: 50, label: 'Bl' },
      re: { min: 0.1, max: 100, label: 'Re' },
      le: { min: 0.01, max: 10, label: 'Le' },
      portDiam: { min: 0.5, max: 50, label: 'Diámetro del puerto' },
      numPorts: { min: 1, max: 8, label: 'Número de puertos' },
      slotW: { min: 0.5, max: 200, label: 'Ancho del slot' },
      slotH: { min: 0.5, max: 200, label: 'Alto del slot' },
    };

    function clearFieldErrors() {
      document.querySelectorAll('.field-error').forEach(error => error.remove());
      document.querySelectorAll('[aria-invalid="true"]').forEach(field => {
        field.removeAttribute('aria-invalid');
        field.removeAttribute('aria-describedby');
      });
    }

    function showFieldError(id, message) {
      const field = document.getElementById(id);
      if (!field || field.getAttribute('aria-invalid') === 'true') return;
      const error = document.createElement('span');
      error.className = 'field-error';
      error.id = `${id}-error`;
      error.textContent = message;
      field.setAttribute('aria-invalid', 'true');
      field.setAttribute('aria-describedby', error.id);
      field.closest('.form-field')?.appendChild(error);
    }

    function validateCalculatorForm() {
      clearFieldErrors();
      let valid = true;
      const boxType = document.getElementById('boxType').value;
      const portType = document.getElementById('portType').value;
      const inactive = new Set(
        boxType === 'closed' ? ['portDiam', 'numPorts', 'slotW', 'slotH']
          : portType === 'circular' ? ['slotW', 'slotH'] : ['portDiam']
      );

      Object.entries(FIELD_RULES).forEach(([id, rule]) => {
        if (inactive.has(id)) return;
        const field = document.getElementById(id);
        const raw = field?.value.trim();
        if (!raw) {
          if (rule.required) {
            showFieldError(id, `${rule.label} es obligatorio.`);
            valid = false;
          }
          return;
        }
        const value = Number(raw);
        if (!Number.isFinite(value) || value < rule.min || value > rule.max) {
          showFieldError(id, `${rule.label} debe estar entre ${rule.min} y ${rule.max}.`);
          valid = false;
        }
      });

      const qts = Number(document.getElementById('qts').value);
      for (const id of ['qes', 'qms']) {
        const field = document.getElementById(id);
        if (field.value && Number(field.value) <= qts) {
          showFieldError(id, `${id.toUpperCase()} debe ser mayor que Qts.`);
          valid = false;
        }
      }
      if (boxType === 'closed') {
        const qtc = Number(document.getElementById('qtcTarget').value);
        if (qtc <= qts) {
          showFieldError('qtcTarget', 'Qtc objetivo debe ser mayor que Qts.');
          valid = false;
        }
      }

      if (!valid) {
        document.querySelector('[aria-invalid="true"]')?.focus();
        notify('Corrige los campos señalados antes de calcular.', 'warning');
      }
      return valid;
    }

    let calcResults = null;
    let currentView = 'calc';
    let currentEnc = 'sec-intro';

    /* ── Navegación principal ───────────────────────────────── */
    function setView(v) {
      currentView = v;
      document.querySelectorAll('.view').forEach(el => el.classList.remove('active'));
      document.getElementById('view-' + v).classList.add('active');
      document.querySelectorAll('.tb-nav button').forEach(b => b.classList.remove('active'));
      document.getElementById('tb-' + v).classList.add('active');

      const sidebar = document.getElementById('enc-sidebar');
      sidebar.classList.toggle('hidden', v !== 'enc');
      document.getElementById('enc-menu-toggle').hidden = v !== 'enc';
      if (v !== 'enc') closeEncMenu(false);

      if (v === 'db') renderDB();
    }

    function openEncMenu() {
      const sidebar = document.getElementById('enc-sidebar');
      const toggle = document.getElementById('enc-menu-toggle');
      sidebar.classList.add('mobile-open');
      document.getElementById('enc-menu-overlay').hidden = false;
      toggle.setAttribute('aria-expanded', 'true');
      document.body.classList.add('enc-menu-open');
      sidebar.querySelector('.enc-menu-close')?.focus();
    }

    function closeEncMenu(restoreFocus = true) {
      const sidebar = document.getElementById('enc-sidebar');
      const toggle = document.getElementById('enc-menu-toggle');
      sidebar.classList.remove('mobile-open');
      document.getElementById('enc-menu-overlay').hidden = true;
      toggle.setAttribute('aria-expanded', 'false');
      document.body.classList.remove('enc-menu-open');
      if (restoreFocus && !toggle.hidden) toggle.focus();
    }

    /* ── Enciclopedia ───────────────────────────────────────── */
    function showEnc(id) {
      document.querySelectorAll('#view-enc .enc-article').forEach(a => a.hidden = true);
      document.getElementById(id).hidden = false;
      document.querySelectorAll('#enc-sidebar .nav-item').forEach(b => b.classList.remove('active'));
      document.querySelector(`[data-enc="${id}"]`).classList.add('active');
      document.getElementById('content-area').scrollTop = 0;
      currentEnc = id;
      if (window.matchMedia('(max-width: 700px)').matches) closeEncMenu(false);
    }

    /* ── Bridge: enciclopedia → calculadora ─────────────────── */
    function goToCalc(boxType, contextMsg) {
      setView('calc');
      document.getElementById('boxType').value = boxType;
      toggleBoxOpts();
      const banner = document.getElementById('context-banner');
      const msg = document.getElementById('context-msg');
      const sectionNames = {
        'sec-cerradas': 'Cajas Cerradas', 'sec-reflex': 'Bass-Reflex',
        'sec-ts': 'Parámetros T/S', 'sec-puertos': 'Diseño del Puerto',
        'sec-materiales': 'Materiales y Proporciones',
      };
      msg.textContent = `Vinculado desde: Enciclopedia → ${sectionNames[currentEnc] || 'Enciclopedia'}`;
      banner.classList.add('show');
      document.getElementById('content-area').scrollTop = 0;
    }

    function dismissContext() {
      document.getElementById('context-banner').classList.remove('show');
    }

    /* ── Tooltip de concepto ────────────────────────────────── */
    const TIPS = {
      Fs: { title: 'Fs — Resonancia Libre', body: 'Frecuencia donde el cono resuena naturalmente. La caja reflex la eleva a Fc. Por debajo de Fs la respuesta cae abruptamente.' },
      Vas: { title: 'Vas — Volumen Equivalente', body: 'Elasticidad de la suspensión expresada en litros de aire. No es el volumen de la caja. Vas alto = suspensión blanda = caja grande.' },
      Qts: { title: 'Qts — Factor de Calidad', body: 'Amortiguamiento en resonancia. Qts < 0.4 → Bass-Reflex. Qts > 0.4 → Sellada. Qts = Qes·Qms/(Qes+Qms).' },
      Xmax: { title: 'Xmax — Excursión Máxima', body: 'Desplazamiento lineal máximo (mm) por dirección. Con Sd determina Vd = Sd×Xmax y el SPL máximo alcanzable en graves.' },
      Sd: { title: 'Sd — Área Efectiva', body: 'Superficie radiante del cono (cm²). Con Xmax: Vd = Sd × Xmax. Mayor Sd = más aire movido a igual excursión.' },
      TipoCaja: { title: 'Tipo de recinto', body: 'Reflex: mayor extensión y eficiencia, ideal Qts<0.4. Sellada: bajos más precisos y controlados, ideal Qts 0.4-0.7.' },
      Alineacion: { title: 'Alineación Thiele-Small', body: 'QB3: equilibrio general. SBB4: extensión máxima, más volumen. B4: Butterworth clásico, más plano.' },
      FactorK: { title: 'Corrección de extremo k', body: 'Compensación por la masa de aire extra en las bocas del tubo. 0.732 es el valor estándar para tubo con un extremo abierto y otro flangado.' },
      Qtc: { title: 'Qtc objetivo (Caja Sellada)', body: '0.577 (Bessel): transitorios perfectos pero caja muy grande. 0.707 (Butterworth): plano óptimo. 1.0 (Chebyshev): pico +1.2 dB, caja pequeña.' },
      Mms: { title: 'Mms — Masa Móvil', body: 'Masa total del conjunto móvil (cono + bobina + suspensión) en gramos. Dato crítico para precisión. Si se omite, el simulador lo estima desde Vas/Sd con errores que pueden superar el 100%.' },
      Bl: { title: 'Bl — Factor de Fuerza', body: 'Producto del campo magnético por la longitud del hilo en el entrehierro (T·m). Es el engranaje eléctrico→mecánico: F = Bl × I. Bl alto → Qes bajo → motor potente.' },
    };

    function openTip(key) {
      const tip = TIPS[key];
      if (!tip) return;
      const el = document.getElementById('enc-tooltip');
      el.innerHTML = `<strong>${tip.title}</strong>${tip.body}
    <br><a href="#" data-tip-section="${key}" class="tip-section-link">Ver en la Enciclopedia →</a>`;
      el.classList.add('show');
      setTimeout(() => el.classList.remove('show'), 5000);
    }

    document.addEventListener('click', e => {
      if (!e.target.classList.contains('tip')) {
        document.getElementById('enc-tooltip').classList.remove('show');
      }
    });

    function goToEncSection(key) {
      const map = {
        Fs: 'sec-ts', Vas: 'sec-ts', Qts: 'sec-ts', Xmax: 'sec-ts', Sd: 'sec-ts',
        Mms: 'sec-ts', Bl: 'sec-modelo',
        TipoCaja: 'sec-reflex', Alineacion: 'sec-simulador',
        FactorK: 'sec-puertos', Qtc: 'sec-cerradas'
      };
      const sec = map[key] || 'sec-ts';
      setView('enc');
      showEnc(sec);
    }

    /* ── Búsqueda de altavoces ──────────────────────────────── */
    function filterDB(q) {
      const dd = document.getElementById('spk-dropdown');
      if (!q || q.length < 2) { dd.classList.remove('open'); return; }
      const hits = DB.filter(s =>
        (s.brand + ' ' + s.model).toLowerCase().includes(q.toLowerCase())
      );
      if (!hits.length) { dd.classList.remove('open'); return; }
      dd.innerHTML = hits.map((s, i) => `
    <div class="dd-item" data-speaker-index="${DB.indexOf(s)}">
      <div><span class="dd-brand">${s.brand}</span><br><span class="dd-model">${s.model}</span></div>
      <div class="speaker-search-meta">${s.inches}" · ${s.align}</div>
    </div>`).join('');
      dd.classList.add('open');
    }

    function loadSpeaker(idx) {
      const s = DB[idx];
      document.getElementById('fs').value = s.fs;
      document.getElementById('vas').value = s.vas;
      document.getElementById('qts').value = s.qts;
      document.getElementById('qes').value = s.qes || '';
      document.getElementById('qms').value = s.qms || '';
      document.getElementById('xmax').value = s.xmax || '';
      document.getElementById('sd').value = s.sd || '';
      document.getElementById('spl').value = s.spl || '';
      document.getElementById('inches').value = s.inches;
      document.getElementById('alignment').value = s.align;
      document.getElementById('spk-search').value = s.brand + ' ' + s.model;
      document.getElementById('spk-dropdown').classList.remove('open');

      // Nuevos campos
      const mmsEl = document.getElementById('mms');
      const blEl = document.getElementById('bl');
      const reEl = document.getElementById('re');
      const leEl = document.getElementById('le');
      if (mmsEl) mmsEl.value = s.mms || '';
      if (blEl) blEl.value = s.bl || '';
      if (reEl) reEl.value = s.re || '';
      if (leEl) leEl.value = s.le || '';
    }

    document.addEventListener('click', e => {
      if (!e.target.closest('.search-wrapper'))
        document.getElementById('spk-dropdown').classList.remove('open');
    });

    /* ── Toggles de UI ──────────────────────────────────────── */
    function toggleBoxOpts() {
      const isReflex = document.getElementById('boxType').value === 'reflex';
      document.getElementById('reflex-opts').hidden = !isReflex;
      document.getElementById('closed-opts').hidden = isReflex;
    }
    function togglePortOpts() {
      const isCirc = document.getElementById('portType').value === 'circular';
      document.getElementById('port-circular').hidden = !isCirc;
      document.getElementById('port-slot').hidden = isCirc;
    }

    function switchTab(evt, id) {
      ['tab-diag', 'tab-port', 'tab-dim', 'tab-chart', 'tab-cuts', 'tab-compare'].forEach(t => {
        const el = document.getElementById(t);
        if (el) el.hidden = true;
      });
      document.querySelectorAll('.calc-tab').forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
        b.tabIndex = -1;
      });
      document.getElementById(id).hidden = false;
      evt.currentTarget.classList.add('active');
      evt.currentTarget.setAttribute('aria-selected', 'true');
      evt.currentTarget.tabIndex = 0;
      if (id === 'tab-chart' && calcResults) drawChart(calcResults);
    }

    function n(v, d = 1) { return (v !== null && v !== undefined && !isNaN(v)) ? v.toFixed(d) : '—'; }

    /* ── CÁLCULO PRINCIPAL ──────────────────────────────────── */
    async function calculate() {
      if (!validateCalculatorForm()) return;
      const fs = parseFloat(document.getElementById('fs').value);
      const vas = parseFloat(document.getElementById('vas').value);
      const qts = parseFloat(document.getElementById('qts').value);
      const calculateButton = document.getElementById('btn-calculate');
      setButtonBusy(calculateButton, true, 'Calculando…');

      const qes = parseFloat(document.getElementById('qes').value) || null;
      const qms = parseFloat(document.getElementById('qms').value) || null;
      const xmax = parseFloat(document.getElementById('xmax').value) || null;
      const sd = parseFloat(document.getElementById('sd').value) || null;
      const inches = parseFloat(document.getElementById('inches').value) || 10;
      const T = parseFloat(document.getElementById('material').value) / 10;
      const boxType = document.getElementById('boxType').value;

      // Parámetros físicos del fabricante
      const mms = parseFloat(document.getElementById('mms')?.value) || null;
      const bl = parseFloat(document.getElementById('bl')?.value) || null;
      const re = parseFloat(document.getElementById('re')?.value) || null;
      const le = parseFloat(document.getElementById('le')?.value) || null;

      let r = { fs, vas, qts, qms, T, boxType, mms, bl, re, le };

      if (boxType === 'reflex') {
        const alignment = document.getElementById('alignment').value;
        const portType = document.getElementById('portType').value;
        const portDiam = parseFloat(document.getElementById('portDiam').value) || 7;
        const slotW = parseFloat(document.getElementById('slotW').value) || 10;
        const slotH = parseFloat(document.getElementById('slotH').value) || 5;
        const k = parseFloat(document.getElementById('kFactor').value);
        const N = parseInt(document.getElementById('numPorts').value) || 1;

        let target;
        try {
          const response = await fetch(`${API_BASE}/api/alignments`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fs, vas, qts }),
          });
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const data = await response.json();
          target = data.alignments?.[alignment];
          if (!target) throw new Error(`Alineamiento ${alignment} no disponible`);
        } catch (error) {
          notify(`No se pudo calcular el alineamiento: ${error.message}`, 'error');
          setButtonBusy(calculateButton, false);
          return;
        }
        const Vb = target.vb;
        const Fb = target.fb;
        const F3 = target.f3;

        let Sp, d_eq;
        if (portType === 'circular') { Sp = Math.PI * (portDiam / 2) ** 2; d_eq = portDiam; }
        else { Sp = slotW * slotH; d_eq = 2 * Math.sqrt(Sp / Math.PI); }
        const SpTotal = N * Sp;
        const L = (29974.86 * N * Sp) / (Fb ** 2 * Vb) - k * d_eq;
        const portVel = (sd && xmax) ? (Fb * 2 * Math.PI * (xmax / 1000 * 0.4) * sd) / SpTotal : null;
        const Vport = N * Sp * Math.max(L, 1) / 1000;
        const Vdriver = 0.0035 * inches ** 2.8;
        const Vb_bruto = Vb + Vport + Vdriver + 0.05 * Vb;
        const Vd = (sd && xmax) ? sd * (xmax / 10) : null;
        const SPLmax = Vd ? 112.2 + 20 * Math.log10((Vd / 1e6) * Fb ** 2) : null;
        const EBP = qes ? fs / qes : null;

        Object.assign(r, { alignment, Vb, Fb, F3, Sp, SpTotal, d_eq, L, portVel, Vb_bruto, Vd, SPLmax, EBP, N, portType, portDiam, slotW, slotH, Vdriver, qes, xmax, sd, inches });
      } else {
        const qtcTarget = parseFloat(document.getElementById('qtcTarget').value);
        let target;
        try {
          const response = await fetch(`${API_BASE}/api/alignments`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fs, vas, qts, qtc_target: qtcTarget }),
          });
          const data = await response.json();
          if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
          target = data.closed;
        } catch (error) {
          notify(`No se pudo calcular la caja sellada: ${error.message}`, 'error');
          setButtonBusy(calculateButton, false);
          return;
        }
        const Vb = target.vb;
        const Qtc_real = target.qtc;
        const F3 = target.f3;
        const Vdriver = 0.0035 * inches ** 2.8;
        const Vb_bruto = Vb + Vdriver + 0.05 * Vb;
        const Vd = (sd && xmax) ? sd * (xmax / 10) : null;
        Object.assign(r, { qtcTarget, Qtc_real, Vb, F3, Vb_bruto, Vdriver, Vd, xmax, sd, inches });
      }

      calcResults = r;

      // Mostrar resultados
      document.getElementById('empty-state').hidden = true;
      document.getElementById('results-content').hidden = false;

      // Hero cards
      document.getElementById('r-vb').innerHTML = `${n(r.Vb)} <span class="hc-unit">L</span>`;
      document.getElementById('r-vb-sub').textContent = `Bruto para cortes: ${n(r.Vb_bruto)} L`;
      if (r.boxType === 'reflex') {
        document.getElementById('r-fb-label').textContent = 'Sintonía Fb';
        document.getElementById('r-fb').innerHTML = `${n(r.Fb)} <span class="hc-unit">Hz</span>`;
        document.getElementById('r-fb-sub').textContent = `Alineación ${r.alignment}`;
        document.getElementById('r-f3-sub').textContent = `f3/Fs = ${n(r.F3 / r.fs, 3)}`;
      } else {
        document.getElementById('r-fb-label').textContent = 'Qtc real';
        document.getElementById('r-fb').innerHTML = `${n(r.Qtc_real, 3)}`;
        document.getElementById('r-fb-sub').textContent = `Objetivo: ${r.qtcTarget}`;
        document.getElementById('r-f3-sub').textContent = 'Derivación Butterworth exacta';
      }
      document.getElementById('r-f3').innerHTML = `${n(r.F3)} <span class="hc-unit">Hz</span>`;

      renderDiag(r);
      renderPort(r);
      renderDim(r);
      renderCuts(r);

      // Activar primer tab visible
      document.querySelectorAll('.calc-tab').forEach(b => b.classList.remove('active'));
      document.querySelector('.calc-tab').classList.add('active');
      ['tab-diag', 'tab-port', 'tab-dim', 'tab-chart', 'tab-cuts', 'tab-compare'].forEach(t => {
        const el = document.getElementById(t);
        if (el) el.hidden = true;
      });
      document.getElementById('tab-diag').hidden = false;
      setButtonBusy(calculateButton, false);
      projectDirty = projectBaseline
        ? JSON.stringify(collectProjectForm()) !== projectBaseline
        : true;
      refreshProjectEditState();
      notify('Diseño calculado correctamente.', 'success');
    }

    /* ── Renderizado ────────────────────────────────────────── */
    function renderDiag(r) {
      const items = [];
      if (r.EBP) {
        const c = r.EBP > 100 ? 'g' : r.EBP > 50 ? 'y' : 'r';
        const m = r.EBP > 100 ? 'Ideal Bass-Reflex' : r.EBP > 50 ? 'Ambos tipos sirven' : 'Preferir Sellada';
        items.push({ label: 'EBP (Fs/Qes)', val: `${n(r.EBP, 0)} — ${m}`, c });
      }
      if (r.portVel != null) {
        const c = r.portVel < 12 ? 'g' : r.portVel < 17 ? 'y' : 'r';
        const m = r.portVel < 12 ? 'Sin turbulencia' : r.portVel < 17 ? 'Límite aceptable' : '⚠ Turbulencia!';
        items.push({ label: 'Velocidad puerto', val: `${n(r.portVel)} m/s — ${m}`, c });
      }
      if (r.SPLmax) items.push({ label: 'SPL máximo (Keele 1975)', val: `${n(r.SPLmax)} dB @ 1m / 1W`, c: 'g' });
      if (r.Vd) items.push({ label: 'Vd = Sd × Xmax', val: `${n(r.Vd, 0)} cm³ — ${r.Vd > 100 ? 'Subwoofer serio' : 'Woofer medio'}`, c: r.Vd > 100 ? 'g' : 'y' });
      if (r.Qtc_real) {
        const c = r.Qtc_real < 0.8 ? 'g' : r.Qtc_real < 1 ? 'y' : 'r';
        items.push({ label: 'Qtc real', val: `${n(r.Qtc_real, 3)} — ${r.Qtc_real < 0.8 ? 'Respuesta plana' : 'Pico en Fc'}`, c });
      }
      const qok = r.qts >= 0.2 && r.qts <= 0.5;
      items.push({ label: 'Rango Qts (tablas Thiele)', val: `${r.qts} — ${qok ? 'Válido' : 'Fuera de tablas'}`, c: qok ? 'g' : 'y' });

      document.getElementById('diag-grid').innerHTML = items.map(d => `
    <div class="diag-item">
      <div class="diag-dot ${d.c}"></div>
      <div>
        <div class="diag-label">${d.label}</div>
        <div class="diag-value">${d.val}</div>
      </div>
    </div>`).join('');
    }

    function renderPort(r) {
      if (r.boxType !== 'reflex') { document.getElementById('tab-port').hidden = true; return; }
      const Fpipe = r.L ? 34400 / (2 * r.L) : null;
      const portDesc = r.portType === 'circular'
        ? `<div class="data-row"><span class="dr-label">Diámetro del tubo</span><span class="dr-val">${r.portDiam} cm</span></div>`
        : `<div class="data-row"><span class="dr-label">Slot ${r.slotW}×${r.slotH} cm</span><span class="dr-val">Área ${n(r.Sp)} cm²</span></div>`;

      document.getElementById('port-body').innerHTML = `
    <div class="data-row"><span class="dr-label">Longitud del tubo / slot</span><span class="dr-val">${n(r.L)} cm &nbsp;(${n(r.L * 10, 0)} mm)</span></div>
    <div class="data-row"><span class="dr-label">Área por puerto (Sp)</span><span class="dr-val">${n(r.Sp)} cm²</span></div>
    <div class="data-row"><span class="dr-label">Área total (${r.N} puerto/s)</span><span class="dr-val">${n(r.SpTotal)} cm²</span></div>
    <div class="data-row"><span class="dr-label">Diámetro equivalente</span><span class="dr-val">${n(r.d_eq)} cm</span></div>
    ${portDesc}
    ${Fpipe ? `<div class="data-row"><span class="dr-label">Resonancia tubo (Pipe)</span><span class="dr-val pipe-warning">${n(Fpipe)} Hz — rellenar 1/3 con espuma</span></div>` : ''}
    <div class="data-row"><span class="dr-label">Filtro subsónico recomendado</span><span class="dr-val">${n(r.Fb * 0.7)} Hz (0.7×Fb)</span></div>
  `;
    }

    function renderDim(r) {
      const T = r.T, Vb = r.Vb_bruto;
      const Di = Math.cbrt(Vb * 1000 / (1.59 * 1.26));
      const Wi = 1.26 * Di, Hi = 1.59 * Di;
      const De = Di + 2 * T, We = Wi + 2 * T, He = Hi + 2 * T;
      const Fbsc = 115 / (We / 100);
      document.getElementById('dim-body').innerHTML = `
    <div class="data-row"><span class="dr-label">Interior H × W × D</span><span class="dr-val">${n(Hi)} × ${n(Wi)} × ${n(Di)} cm</span></div>
    <div class="data-row"><span class="dr-label">Exterior H × W × D</span><span class="dr-val">${n(He)} × ${n(We)} × ${n(De)} cm</span></div>
    <div class="data-row"><span class="dr-label">Grosor de pared (T)</span><span class="dr-val">${r.T * 10} mm</span></div>
    <div class="data-row"><span class="dr-label">Baffle Step (F_bsc)</span><span class="dr-val">${n(Fbsc)} Hz — compensar con shelving</span></div>
    <div class="data-row"><span class="dr-label">Proporción H:W:D (áurea)</span><span class="dr-val">1.59 : 1.26 : 1.00</span></div>
  `;
    }

    function renderCuts(r) {
      const T = r.T, Vb = r.Vb_bruto;
      const Di = Math.cbrt(Vb * 1000 / (1.59 * 1.26));
      const Wi = 1.26 * Di, Hi = 1.59 * Di;
      const De = Di + 2 * T, We = Wi + 2 * T, He = Hi + 2 * T;

      const pcs = [
        { name: 'Frontal', cant: 1, w: n(We), h: n(He), note: 'Orificio del altavoz (ver ∅ nominal)' },
        { name: 'Trasera', cant: 1, w: n(We), h: n(He), note: 'Terminal de bornes' },
        { name: 'Tapa (superior)', cant: 1, w: n(We - 2 * T), h: n(De), note: 'Encaja entre frontal y trasera' },
        { name: 'Base (inferior)', cant: 1, w: n(We - 2 * T), h: n(De), note: 'Encaja entre frontal y trasera' },
        { name: 'Laterales', cant: 2, w: n(He), h: n(De), note: 'Piezas idénticas ×2' },
      ];

      document.getElementById('cuts-table').innerHTML = `
    <thead><tr><th>Pieza</th><th>Cant.</th><th>Ancho (cm)</th><th>Alto (cm)</th><th>Notas</th></tr></thead>
    <tbody>${pcs.map(p => `<tr>
      <td class="piece">${p.name}</td><td>${p.cant}</td>
      <td class="dims">${p.w}</td><td class="dims">${p.h}</td>
      <td class="note">${p.note}</td>
    </tr>`).join('')}</tbody>`;

      const area = (2 * (We * He) + 2 * ((We - 2 * T) * De) + 2 * (He * De)) / 10000;
      document.getElementById('cuts-summary').innerHTML = `
    <div class="sum-item"><div class="sv">${n(Vb)} L</div><div class="sl">Vb bruto</div></div>
    <div class="sum-item"><div class="sv">${n(area * 1.15, 3)} m²</div><div class="sl">Tablero (+15% merma)</div></div>
    <div class="sum-item"><div class="sv">${n(We)}×${n(He)}×${n(De)}</div><div class="sl">Exterior (cm)</div></div>`;
    }

    /* ═══════════════════════════════════════════════════════════
       INTEGRACIÓN SCIPY — llamada al backend FastAPI
       ════════════════════════════════════════════════════════ */
    // Frontend y API se sirven desde el mismo origen en local y en Vercel.
    const API_BASE = '';
    let sciPyData = null;                       // caché del último resultado de la API

    /* Construye el payload a partir del formulario actual */
    function buildDriverPayload() {
      const r = calcResults;
      if (!r) return null;
      const driver = {
        fs: r.fs,
        vas: r.vas,
        qts: r.qts,
        qes: r.qes || undefined,
        qms: r.qms || undefined,
        xmax: r.xmax || undefined,
        sd: r.sd || undefined,
        spl: parseFloat(document.getElementById('spl').value) || 86,
        re: r.re || undefined,
        inches: r.inches,
        model_name: document.getElementById('spk-search').value || 'Altavoz',
        box_type: r.boxType,
        alignment: r.alignment || undefined,
        qtc_target: r.qtcTarget || undefined,
        material_mm: r.T * 10,
        port_type: r.portType || undefined,
        port_diam_cm: r.portDiam || undefined,
        slot_w_cm: r.slotW || undefined,
        slot_h_cm: r.slotH || undefined,
        num_ports: r.N || undefined,
        k_factor: parseFloat(document.getElementById('kFactor').value) || 0.732,
      };

      // Solo enviar si el usuario los proporcionó
      if (r.mms) driver.mms = r.mms;
      if (r.bl) driver.bl = r.bl;
      if (r.le) driver.le = r.le;

      return {
        driver,
        freq_min: 15,
        freq_max: 800,
        freq_points: 500,
        eg_volts: 2.83,
      };
    }

    /* Llama a POST /api/simulate y actualiza la gráfica */
    async function runScipy() {
      const payload = buildDriverPayload();
      if (!payload) {
        notify('Calcula primero antes de ejecutar la simulación científica.', 'warning');
        return;
      }

      const errEl = document.getElementById('chart-api-error');
      const loadEl = document.getElementById('chart-loading');
      const canvasEl = document.getElementById('freqChart');
      const badge = document.getElementById('chart-mode-badge');
      const btnScipy = document.getElementById('btn-scipy');

      errEl.hidden = true;
      loadEl.hidden = false;
      canvasEl.hidden = true;
      btnScipy.disabled = true;
      btnScipy.textContent = '⏳ Calculando…';

      try {
        const res = await fetch(`${API_BASE}/api/simulate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || `HTTP ${res.status}`);
        }

        sciPyData = await res.json();

        // Actualizar métricas hero con valores scipy (más precisos)
        const m = sciPyData.metrics;
        document.getElementById('r-f3').innerHTML = `${m.f3} <span class="hc-unit">Hz</span>`;
        document.getElementById('r-f3-sub').textContent = `scipy · F6=${m.f6}Hz`;
        if (m.fb) {
          document.getElementById('r-fb').innerHTML = `${m.fb} <span class="hc-unit">Hz</span>`;
        }

        // Actualizar badge
        badge.textContent = '⚗️ scipy / Small 1973';
        badge.classList.add('simulation-badge-success');

        // Mostrar avisos si los hay
        if (sciPyData.warnings?.length) {
          errEl.innerHTML = '⚠️ ' + sciPyData.warnings.join('<br>⚠️ ');
          errEl.hidden = false;
          errEl.className = 'u-inline-09 simulation-message-warning';
        }

        loadEl.hidden = true;
        canvasEl.hidden = false;
        drawChartFromData(sciPyData);
        renderExcursionChart(sciPyData);
        notify('Simulación científica completada.', 'success');

      } catch (err) {
        loadEl.hidden = true;
        canvasEl.hidden = false;
        // Backend no disponible — mostrar aviso suave, no error rojo
        const isConnErr = err.message === 'Failed to fetch' || err.message.includes('NetworkError');
        if (isConnErr) {
          errEl.innerHTML = `⚗️ Backend scipy no conectado — mostrando gráfica JS aproximada.
        <a href="https://github.com/cbenaventte/speakerlab_pro" target="_blank" rel="noopener" class="simulation-help-link">
        Ver instrucciones de arranque →</a>`;
          errEl.className = 'u-inline-09 simulation-message-info';
        } else {
          errEl.innerHTML = `❌ Error scipy: <strong>${err.message}</strong>`;
          notify(`Error de simulación: ${err.message}`, 'error');
          errEl.className = 'u-inline-09 simulation-message-error';
        }
        errEl.hidden = false;
        drawChart(calcResults);   // fallback a la gráfica JS
      } finally {
        btnScipy.disabled = false;
        btnScipy.textContent = '⚗️ Simular con scipy';
      }
    }

    /* ── Llama a POST /api/compare y genera la gráfica comparativa ── */
    async function runCompare() {
      const payload = buildDriverPayload();
      if (!payload) {
        notify('Calcula primero para poder comparar alineamientos.', 'warning');
        return;
      }
      const btn = document.getElementById('btn-compare');
      const errEl = document.getElementById('compare-error');
      btn.textContent = '⏳ Analizando...';
      btn.disabled = true;
      errEl.hidden = true;

      try {
        const res = await fetch(`${API_BASE}/api/compare`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload.driver)
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();

        const tbody = document.getElementById('compare-tbody');
        let html = '';
        const colors = { 'QB3': '#0984e3', 'SBB4': '#00b894', 'B4': '#e17055', 'Closed': '#d63031' };

        for (const align of ["QB3", "SBB4", "B4", "Closed"]) {
          const info = data.curves[align];
          if (!info || info.error) continue;
          html += `<tr>
        <td class="compare-align compare-align-${align.toLowerCase()}">${align} ${align === 'Closed' ? '(Sellada)' : ''}</td>
        <td class="dims">${n(info.vb)} L</td>
        <td class="dims">${align === 'Closed' ? 'Qtc ' + n(info.qtc, 3) : n(info.fb) + ' Hz'}</td>
        <td class="dims">${n(info.f3)} Hz</td>
      </tr>`;
        }
        tbody.innerHTML = html;

        const canvas = document.getElementById('compareChart');
        const ctx = canvas.getContext('2d');
        canvas.width = canvas.offsetWidth * 2 || 1200; canvas.height = 640; ctx.scale(2, 2);
        const W = canvas.width / 2, H = canvas.height / 2;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#fcfbf8'; ctx.fillRect(0, 0, W, H);

        const PAD = { l: 40, r: 15, t: 15, b: 25 };
        const fw = W - PAD.l - PAD.r, fh = H - PAD.t - PAD.b;
        const logMin = Math.log10(15), logMax = Math.log10(600);
        const xPos = f => PAD.l + (Math.log10(Math.max(f, 15)) - logMin) / (logMax - logMin) * fw;
        const ref = 0; // Relative layout for comparison
        const yPos = db => PAD.t + fh * 0.12 - (db - 3) * (fh / 36);

        [20, 30, 50, 100, 200, 500].forEach(f => {
          let x = xPos(f);
          if (x < PAD.l || x > PAD.l + fw) return;
          ctx.strokeStyle = 'rgba(0,0,0,0.06)'; ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + fh); ctx.stroke();
          ctx.fillStyle = '#5a6a72'; ctx.font = '10px Kalam,cursive'; ctx.textAlign = 'center'; ctx.fillText(f + 'Hz', x, H - 5);
        });

        [-18, -15, -12, -9, -6, -3, 0, 3].forEach(db => {
          let y = yPos(db);
          ctx.strokeStyle = 'rgba(0,0,0,0.1)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l + fw, y); ctx.stroke();
          ctx.textAlign = 'right'; ctx.fillStyle = '#5a6a72'; ctx.fillText(db + 'dB', PAD.l - 4, y + 4);
        });

        // Draw 0dB reference thicker
        ctx.strokeStyle = 'rgba(0,0,0,0.2)'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(PAD.l, yPos(0)); ctx.lineTo(PAD.l + fw, yPos(0)); ctx.stroke();

        // Dibujar línea de referencia -3 dB
        const y3db = yPos(-3);
        ctx.strokeStyle = 'rgba(200,60,60,0.6)';
        ctx.lineWidth = 1;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(PAD.l, y3db);
        ctx.lineTo(PAD.l + fw, y3db);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(200,60,60,0.8)';
        ctx.font = '9px Kalam,cursive';
        ctx.textAlign = 'left';
        ctx.fillText('−3 dB', PAD.l + 3, y3db - 3);

        for (const align of ["Closed", "B4", "SBB4", "QB3"]) {
          const info = data.curves[align];
          if (!info || info.error) continue;
          ctx.strokeStyle = colors[align];
          ctx.lineWidth = 2.2;
          ctx.beginPath();
          info.spl.forEach((db, i) => {
            // SPL ya viene normalizado a 0 dB desde el backend — no tocar
            const x = xPos(data.freqs[i]);
            const y = yPos(db);
            if (x >= PAD.l && x <= PAD.l + fw) {
              if (i === 0) ctx.moveTo(x, Math.min(Math.max(y, PAD.t), PAD.t + fh));
              else ctx.lineTo(x, Math.min(Math.max(y, PAD.t), PAD.t + fh));
            }
          });
          ctx.stroke();
        }

        document.getElementById('compare-results').hidden = false;
        notify('Comparación de alineamientos completada.', 'success');
      } catch (e) {
        errEl.textContent = '❌ Error de API: ' + e.message;
        errEl.hidden = false;
        notify(`No se pudo completar la comparación: ${e.message}`, 'error');
      } finally {
        btn.textContent = 'Analizar las 4 alineaciones (Scipy)';
        btn.disabled = false;
      }
    }

    /* ── Función de renderizado del canvas — acepta datos scipy O datos JS ── */
    function _drawCanvasCore(canvas, freqs, splArr, markers, sensRef) {
      const ctx = canvas.getContext('2d');
      canvas.width = canvas.offsetWidth * 2 || 1200;
      canvas.height = 480;
      ctx.scale(2, 2);
      const W = canvas.width / 2, H = canvas.height / 2;
      ctx.clearRect(0, 0, W, H);

      // Fondo papel cuadriculado
      ctx.fillStyle = '#fcfbf8';
      ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = 'rgba(0,0,0,0.055)';
      ctx.lineWidth = 0.5;
      for (let y = 0; y < H; y += 22) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
      for (let x = 0; x < W; x += 22) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }

      const PAD = { l: 54, r: 22, t: 24, b: 38 };
      const fw = W - PAD.l - PAD.r, fh = H - PAD.t - PAD.b;
      const logMin = Math.log10(15), logMax = Math.log10(900);
      const xPos = f => PAD.l + (Math.log10(Math.max(f, 15)) - logMin) / (logMax - logMin) * fw;
      const ref = sensRef || 0;
      const dbRange = 30;
      const yPos = db => PAD.t + fh / 2 - (db - ref) * (fh / dbRange);

      // Grid frecuencias
      const gridFreqs = [20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 700];
      ctx.font = '10px Kalam,cursive'; ctx.fillStyle = '#5a6a72'; ctx.textAlign = 'center';
      gridFreqs.forEach(f => {
        const x = xPos(f);
        if (x < PAD.l || x > PAD.l + fw) return;
        ctx.strokeStyle = 'rgba(0,0,0,0.1)'; ctx.lineWidth = 0.7;
        ctx.beginPath(); ctx.setLineDash([3, 3]); ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + fh); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillText(f + 'Hz', x, PAD.t + fh + 22);
      });

      // Grid dB
      const dbLines = ref
        ? [ref - 12, ref - 9, ref - 6, ref - 3, ref, ref + 3]
        : [-12, -9, -6, -3, 0, 3];
      dbLines.forEach(db => {
        const y = yPos(db);
        ctx.strokeStyle = 'rgba(0,0,0,0.1)'; ctx.lineWidth = 0.7;
        ctx.beginPath(); ctx.setLineDash([3, 3]); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l + fw, y); ctx.stroke();
        ctx.setLineDash([]);
        ctx.textAlign = 'right'; ctx.fillStyle = '#5a6a72';
        ctx.fillText((ref ? (db - ref) : db) + 'dB', PAD.l - 5, y + 4);
      });

      // Curva SPL — tinta azul blueprint
      const pts = freqs.map((f, i) => ({ x: xPos(f), y: yPos(splArr[i]) }))
        .filter(p => p.x >= PAD.l && p.x <= PAD.l + fw);

      // Relleno
      ctx.beginPath();
      pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
      ctx.lineTo(pts[pts.length - 1].x, PAD.t + fh);
      ctx.lineTo(pts[0].x, PAD.t + fh);
      ctx.closePath();
      ctx.fillStyle = 'rgba(9,132,227,0.08)'; ctx.fill();

      ctx.beginPath(); ctx.strokeStyle = '#0984e3'; ctx.lineWidth = 2.2; ctx.setLineDash([]);
      pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
      ctx.stroke();

      // Línea de referencia −3dB
      const y3 = yPos(ref - 3);
      ctx.strokeStyle = 'rgba(0,0,0,0.2)'; ctx.lineWidth = 0.8; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(PAD.l, y3); ctx.lineTo(PAD.l + fw, y3); ctx.stroke();
      ctx.setLineDash([]);

      // Marcadores verticales (F3, Fb, etc.)
      markers.forEach(({ f, label, color }) => {
        if (!f || f < 15 || f > 900) return;
        const x = xPos(f);
        ctx.strokeStyle = color; ctx.lineWidth = 1.3; ctx.setLineDash([5, 4]);
        ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + fh); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = color; ctx.font = 'bold 10px Kalam,cursive'; ctx.textAlign = 'left';
        ctx.fillText(label, x + 4, PAD.t + 20);
      });

      // Labels de ejes
      ctx.fillStyle = '#5a6a72'; ctx.font = '11px Kalam,cursive'; ctx.textAlign = 'center';
      ctx.fillText('Frecuencia (Hz)', PAD.l + fw / 2, H - 4);
      ctx.save(); ctx.translate(14, PAD.t + fh / 2); ctx.rotate(-Math.PI / 2);
      ctx.fillText('Nivel (dB)', 0, 0); ctx.restore();
    }

    /* Gráfica con datos del backend scipy */
    function drawChartFromData(data) {
      const canvas = document.getElementById('freqChart');
      const m = data.metrics;
      const markers = [
        { f: m.f3, label: `F3=${m.f3}Hz`, color: '#d63031' },
        { f: m.f6, label: `F6=${m.f6}Hz`, color: 'rgba(214,48,49,0.5)' },
        { f: m.fb, label: `Fb=${m.fb}Hz`, color: '#0984e3' },
      ].filter(mk => mk.f);
      _drawCanvasCore(canvas, data.freqs, data.spl, markers, m.sens_band);
      document.getElementById('extra-charts').hidden = false;
    }

    /* Gráfica de excursión + velocidad de puerto */
    function renderExcursionChart(data) {
      const canvas = document.getElementById('excChart');
      canvas.width = canvas.offsetWidth * 2 || 1200;
      canvas.height = 320;
      const ctx = canvas.getContext('2d');
      ctx.scale(2, 2);
      const W = canvas.width / 2, H = canvas.height / 2;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#fcfbf8'; ctx.fillRect(0, 0, W, H);

      const PAD = { l: 54, r: 22, t: 16, b: 34 };
      const fw = W - PAD.l - PAD.r, fh = H - PAD.t - PAD.b;
      const logMin = Math.log10(15), logMax = Math.log10(900);
      const xPos = f => PAD.l + (Math.log10(Math.max(f, 15)) - logMin) / (logMax - logMin) * fw;

      // Excursión (eje izquierdo)
      const maxExc = Math.max(...data.excursion) * 1.3 || 20;
      const yExc = v => PAD.t + fh - (v / maxExc) * fh;

      ctx.fillStyle = 'rgba(9,132,227,0.07)';
      ctx.beginPath();
      data.freqs.forEach((f, i) => {
        const x = xPos(f), y = yExc(data.excursion[i]);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.lineTo(xPos(data.freqs[data.freqs.length - 1]), PAD.t + fh);
      ctx.lineTo(xPos(data.freqs[0]), PAD.t + fh);
      ctx.closePath(); ctx.fill();

      ctx.beginPath(); ctx.strokeStyle = '#0984e3'; ctx.lineWidth = 1.8; ctx.setLineDash([]);
      data.freqs.forEach((f, i) => {
        const x = xPos(f), y = yExc(data.excursion[i]);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();

      // Línea Xmax
      const xmaxMm = calcResults?.xmax || null;
      if (xmaxMm) {
        const yx = yExc(xmaxMm);
        ctx.strokeStyle = '#e5a700'; ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]);
        ctx.beginPath(); ctx.moveTo(PAD.l, yx); ctx.lineTo(PAD.l + fw, yx); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#7a5500'; ctx.font = 'bold 9px Kalam,cursive'; ctx.textAlign = 'left';
        ctx.fillText(`Xmax=${xmaxMm}mm`, PAD.l + 4, yx - 3);
      }

      // Velocidad de puerto (eje derecho, naranja)
      if (data.port_vel) {
        const maxVel = Math.max(Math.max(...data.port_vel) * 1.3, 20);
        const yVel = v => PAD.t + fh - (v / maxVel) * fh;

        ctx.beginPath(); ctx.strokeStyle = 'rgba(225,112,85,0.7)'; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);
        data.freqs.forEach((f, i) => {
          const x = xPos(f), y = yVel(data.port_vel[i]);
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke(); ctx.setLineDash([]);

        // Límite 17 m/s
        const y17 = yVel(17);
        ctx.strokeStyle = 'rgba(214,48,49,0.5)'; ctx.lineWidth = 0.8; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(PAD.l, y17); ctx.lineTo(PAD.l + fw, y17); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#d63031'; ctx.font = '9px Kalam,cursive'; ctx.textAlign = 'right';
        ctx.fillText('17m/s turbulencia', PAD.l + fw - 4, y17 - 3);
      }

      // Leyenda
      ctx.font = '10px Kalam,cursive';
      ctx.fillStyle = '#0984e3'; ctx.textAlign = 'left'; ctx.fillText('── Excursión (mm)', PAD.l + 8, PAD.t + 14);
      if (data.port_vel) {
        ctx.fillStyle = '#e17055'; ctx.fillText('- - Vel. puerto (m/s)', PAD.l + 8, PAD.t + 28);
      }

      // Eje X
      [20, 30, 50, 100, 200, 500].forEach(f => {
        const x = xPos(f);
        ctx.strokeStyle = 'rgba(0,0,0,0.1)'; ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + fh); ctx.stroke();
        ctx.fillStyle = '#5a6a72'; ctx.font = '9px Kalam,cursive'; ctx.textAlign = 'center';
        ctx.fillText(f + 'Hz', x, PAD.t + fh + 20);
      });
    }

    /* Gráfica de fallback con aproximación JS (sin scipy) */
    function drawChart(r) {
      if (!r) return;
      const canvas = document.getElementById('freqChart');

      // Construir arrays desde la aproximación JS
      const freqArr = [], splArr = [];
      for (let fi = 15; fi <= 850; fi *= 1.015) {
        let db;
        if (r.boxType === 'reflex') {
          db = -10 * Math.log10(1 + (r.F3 / fi) ** 8) * 0.4;
          if (fi < r.Fb * 0.5) db -= 15 * (r.Fb * 0.5 / fi - 1);
        } else {
          const Qtc = r.Qtc_real || 0.707, ratio = fi / r.F3;
          const denom = (1 - ratio ** 2) ** 2 + ratio ** 2 / Qtc ** 2;
          db = denom > 0 ? 10 * Math.log10(ratio ** 4 / denom) : -30;
          db = Math.max(db, -28);
        }
        freqArr.push(fi); splArr.push(db);
      }

      const markers = [
        { f: r.F3, label: `F3=${n(r.F3, 0)}Hz`, color: '#d63031' },
        { f: r.Fb, label: `Fb=${n(r.Fb, 0)}Hz`, color: '#0984e3' },
      ].filter(m => m.f);

      _drawCanvasCore(canvas, freqArr, splArr, markers, 0);
    }

    /* ── Base de Datos view ─────────────────────────────────── */
    function renderDB() {
      const tbody = document.getElementById('db-tbody');
      tbody.innerHTML = DB.map((s, i) => `<tr>
    <td class="brand">${s.brand}</td>
    <td>${s.model}</td>
    <td class="db-size-cell">${s.inches}"</td>
    <td class="num">${s.fs}</td>
    <td class="num">${s.vas}</td>
    <td class="num">${s.qts}</td>
    <td class="num">${s.qes || '—'}</td>
    <td class="num">${s.xmax || '—'}</td>
    <td class="num">${s.sd || '—'}</td>
    <td class="num">${s.spl || '—'}</td>
    <td class="num">${s.mms || '—'}</td>
    <td class="num">${s.bl || '—'}</td>
    <td class="num">${s.re || '—'}</td>
    <td><span class="align-badge ${s.align}">${s.align}</span></td>
    <td><button class="db-use-btn" data-use-speaker="${i}">Usar →</button></td>
  </tr>`).join('');
    }

    function useFromDB(idx) {
      loadSpeaker(idx);
      setView('calc');
      const s = DB[idx];
      const banner = document.getElementById('context-banner');
      document.getElementById('context-msg').textContent = `Cargado desde la Base de Datos: ${s.brand} ${s.model}`;
      banner.classList.add('show');
    }

    /* ── Simulador enciclopedia ─────────────────────────────── */
    const alignments = {
      "1": { path: "M 50 190 C 130 190, 140 38, 220 75 L 550 75", title: "Vb Pequeño (Chebyshev — Sub-amortiguado)", desc: "Produce un pico audible antes del decaimiento. Sonido 'boomy' a una sola nota. Respuesta transitoria pobre." },
      "2": { path: "M 50 190 C 118 190, 140 75, 220 75 L 550 75", title: "Vb Óptimo (Butterworth — Máximamente Plano)", desc: "Curva plana ideal. Mejor compromiso extensión/transitorios. Caída brusca a 24 dB/oct. La referencia de diseño." },
      "3": { path: "M 50 190 C 70 190, 155 98, 220 75 L 550 75", title: "Vb Grande (Bessel — Sobre-amortiguado)", desc: "Caída suave que empieza antes. Pierde impacto en el rango medio-grave pero ofrece los mejores transitorios posibles." },
    };

    document.addEventListener('DOMContentLoaded', () => {
      const slider = document.getElementById('align-slider');
      if (slider) slider.addEventListener('input', function () {
        const a = alignments[this.value];
        document.getElementById('curve-path').setAttribute('d', a.path);
        document.getElementById('align-label').textContent = a.title;
        document.getElementById('align-desc').innerHTML = `<strong>${a.title}:</strong><br>${a.desc}`;
      });
      const desc = document.getElementById('align-desc');
      if (desc) desc.innerHTML = `<strong>${alignments["2"].title}:</strong><br>${alignments["2"].desc}`;
    });

    /* ── Tabs enciclopedia ──────────────────────────────────── */
    function openTab(evt, tabId, groupClass) {
      document.querySelectorAll('.tab-content.' + groupClass).forEach(t => t.classList.remove('active'));
      evt.currentTarget.parentElement.querySelectorAll('.tab-btn').forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
        b.tabIndex = -1;
      });
      document.getElementById(tabId).classList.add('active');
      evt.currentTarget.classList.add('active');
      evt.currentTarget.setAttribute('aria-selected', 'true');
      evt.currentTarget.tabIndex = 0;
    }

    /* ── Descarga libre del PDF ──────────────────────────────── */
    async function downloadPDF() {
      if (!calcResults) {
        notify('Primero calcula la caja de tu altavoz.', 'warning');
        return;
      }
      const payload = buildDriverPayload();
      if (!payload) return;
      const buttons = [...document.querySelectorAll('[data-pdf-download]')];
      buttons.forEach(button => setButtonBusy(button, true, 'Generando PDF…'));
      try {
        const response = await fetch(`${API_BASE}/api/pdf`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ driver: payload.driver }),
        });
        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || `HTTP ${response.status}`);
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        const model = (document.getElementById('spk-search').value || 'altavoz').replace(/\s+/g, '_');
        link.download = `speakerlab_${model}.pdf`;
        link.click();
        URL.revokeObjectURL(url);
        notify('PDF generado. La descarga comenzará automáticamente.', 'success');
      } catch (error) {
        notify(`No se pudo generar el PDF: ${error.message}`, 'error', 6000);
      } finally {
        buttons.forEach(button => setButtonBusy(button, false));
      }
    }

    /* ── Proyectos locales ──────────────────────────────────── */
    const PROJECTS_KEY = 'speakerlab.projects.v1';
    const PROJECT_FIELD_IDS = [
      'spk-search', 'fs', 'vas', 'qts', 'qes', 'qms', 'xmax', 'sd', 'spl',
      'inches', 'mms', 'bl', 're', 'le', 'boxType', 'material', 'alignment',
      'portType', 'portDiam', 'numPorts', 'slotW', 'slotH', 'kFactor', 'qtcTarget',
    ];
    let activeProjectId = null;
    let projectBaseline = null;
    let projectDirty = false;

    function readLocalProjects() {
      try {
        const projects = JSON.parse(localStorage.getItem(PROJECTS_KEY));
        return Array.isArray(projects) ? projects : [];
      } catch (_) {
        return [];
      }
    }

    function writeLocalProjects(projects) {
      try {
        localStorage.setItem(PROJECTS_KEY, JSON.stringify(projects));
        return true;
      } catch (_) {
        notify('No hay espacio disponible para guardar más proyectos.', 'error');
        return false;
      }
    }

    function collectProjectForm() {
      return Object.fromEntries(PROJECT_FIELD_IDS.map(id => [id, document.getElementById(id)?.value ?? '']));
    }

    function currentProjectSummary() {
      return {
        boxType: calcResults?.boxType ?? document.getElementById('boxType').value,
        vb: calcResults?.Vb ?? null,
        fb: calcResults?.Fb ?? null,
        f3: calcResults?.F3 ?? null,
      };
    }

    function refreshProjectEditState() {
      const project = readLocalProjects().find(item => item.id === activeProjectId);
      const updateButton = document.getElementById('btn-update-project');
      const status = document.getElementById('project-edit-status');
      updateButton.disabled = !project;
      status.textContent = project
        ? `Seleccionado: ${project.name}${projectDirty ? ' · cambios sin guardar' : ''}`
        : 'Ningún proyecto seleccionado.';
    }

    function projectId() {
      return window.crypto?.randomUUID?.() || `project-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    let projectsModalTrigger = null;

    function openProjectsModal() {
      projectsModalTrigger = document.activeElement;
      renderProjectsList();
      refreshProjectEditState();
      document.getElementById('projects-modal').hidden = false;
      document.body.classList.add('modal-open');
      window.setTimeout(() => document.getElementById('project-name').focus(), 0);
    }

    function closeProjectsModal() {
      document.getElementById('projects-modal').hidden = true;
      document.body.classList.remove('modal-open');
      projectsModalTrigger?.focus();
      projectsModalTrigger = null;
    }

    function saveLocalProject() {
      const nameField = document.getElementById('project-name');
      const name = nameField.value.trim();
      if (!calcResults) {
        notify('Calcula un diseño antes de guardarlo.', 'warning');
        closeProjectsModal();
        return;
      }
      if (!name) {
        notify('Escribe un nombre para el proyecto.', 'warning');
        nameField.focus();
        return;
      }
      const now = new Date().toISOString();
      const project = {
        version: 1,
        id: projectId(),
        name: name.slice(0, 80),
        createdAt: now,
        updatedAt: now,
        form: collectProjectForm(),
        summary: currentProjectSummary(),
      };
      const projects = readLocalProjects();
      projects.unshift(project);
      if (!writeLocalProjects(projects)) return;
      activeProjectId = project.id;
      projectBaseline = JSON.stringify(project.form);
      projectDirty = false;
      renderProjectsList();
      refreshProjectEditState();
      notify(`Proyecto “${project.name}” guardado localmente.`, 'success');
    }

    function updateLocalProject() {
      const projects = readLocalProjects();
      const project = projects.find(item => item.id === activeProjectId);
      if (!project) return notify('Selecciona primero un proyecto guardado.', 'warning');
      const name = document.getElementById('project-name').value.trim();
      if (!name) return notify('Escribe un nombre para el proyecto.', 'warning');
      project.name = name.slice(0, 80);
      project.updatedAt = new Date().toISOString();
      project.form = collectProjectForm();
      project.summary = currentProjectSummary();
      if (!writeLocalProjects(projects)) return;
      projectBaseline = JSON.stringify(project.form);
      projectDirty = false;
      renderProjectsList();
      refreshProjectEditState();
      notify(`Proyecto “${project.name}” actualizado.`, 'success');
    }

    async function loadLocalProject(id) {
      const project = readLocalProjects().find(item => item.id === id);
      if (!project) return notify('El proyecto ya no existe.', 'error');
      if (projectDirty && activeProjectId !== id && !window.confirm('Hay cambios sin guardar. ¿Quieres cargar otro proyecto?')) return;
      Object.entries(project.form || {}).forEach(([fieldId, value]) => {
        const field = document.getElementById(fieldId);
        if (field) field.value = value;
      });
      toggleBoxOpts();
      togglePortOpts();
      activeProjectId = project.id;
      projectBaseline = JSON.stringify(collectProjectForm());
      projectDirty = false;
      document.getElementById('project-name').value = project.name;
      setView('calc');
      closeProjectsModal();
      await calculate();
      notify(`Proyecto “${project.name}” recuperado.`, 'success');
    }

    function deleteLocalProject(id) {
      const projects = readLocalProjects();
      const project = projects.find(item => item.id === id);
      if (!project) return notify('El proyecto ya no existe.', 'error');
      if (!window.confirm(`¿Eliminar definitivamente “${project.name}” de este dispositivo?`)) return;
      const remaining = projects.filter(item => item.id !== id);
      if (!writeLocalProjects(remaining)) return;
      if (activeProjectId === id) {
        activeProjectId = null;
        projectBaseline = null;
        projectDirty = false;
        document.getElementById('project-name').value = '';
      }
      renderProjectsList();
      refreshProjectEditState();
      notify(`Proyecto “${project.name}” eliminado.`, 'success');
    }

    function duplicateLocalProject(id) {
      const projects = readLocalProjects();
      const source = projects.find(item => item.id === id);
      if (!source) return notify('El proyecto ya no existe.', 'error');
      const now = new Date().toISOString();
      const copy = {
        ...source,
        id: projectId(),
        name: `${source.name} — copia`.slice(0, 80),
        createdAt: now,
        updatedAt: now,
        form: { ...source.form },
        summary: { ...source.summary },
      };
      projects.unshift(copy);
      if (writeLocalProjects(projects)) {
        renderProjectsList();
        notify(`Proyecto duplicado como “${copy.name}”.`, 'success');
      }
    }

    function downloadJson(data, filename) {
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      link.click();
      URL.revokeObjectURL(url);
    }

    function safeProjectFilename(name) {
      return name.normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[^a-zA-Z0-9_-]+/g, '_').slice(0, 60) || 'proyecto';
    }

    function exportLocalProject(id) {
      const project = readLocalProjects().find(item => item.id === id);
      if (!project) return notify('El proyecto ya no existe.', 'error');
      downloadJson(project, `speakerlab_${safeProjectFilename(project.name)}.json`);
      notify('Proyecto exportado como JSON.', 'success');
    }

    function exportAllProjects() {
      const projects = readLocalProjects();
      if (!projects.length) return notify('No hay proyectos para exportar.', 'warning');
      downloadJson({ version: 1, exportedAt: new Date().toISOString(), projects }, 'speakerlab_proyectos.json');
      notify('Todos los proyectos fueron exportados.', 'success');
    }

    function normalizeImportedProject(raw) {
      if (!raw || raw.version !== 1 || typeof raw.name !== 'string' || !raw.name.trim() || typeof raw.form !== 'object') {
        throw new Error('El archivo contiene un proyecto incompatible');
      }
      const form = {};
      PROJECT_FIELD_IDS.forEach(id => {
        const value = raw.form[id];
        form[id] = value === undefined || value === null ? '' : String(value).slice(0, 200);
      });
      if (!form.fs || !form.vas || !form.qts) throw new Error(`El proyecto “${raw.name}” no contiene Fs, Vas y Qts`);
      const now = new Date().toISOString();
      const summary = raw.summary && typeof raw.summary === 'object' ? raw.summary : {};
      return {
        version: 1,
        id: projectId(),
        name: raw.name.trim().slice(0, 80),
        createdAt: typeof raw.createdAt === 'string' ? raw.createdAt : now,
        updatedAt: now,
        form,
        summary: {
          boxType: summary.boxType === 'closed' ? 'closed' : 'reflex',
          vb: Number.isFinite(Number(summary.vb)) ? Number(summary.vb) : null,
          fb: Number.isFinite(Number(summary.fb)) ? Number(summary.fb) : null,
          f3: Number.isFinite(Number(summary.f3)) ? Number(summary.f3) : null,
        },
      };
    }

    async function importProjectsFile(file) {
      if (!file) return;
      if (file.size > 2 * 1024 * 1024) return notify('El archivo JSON supera el máximo de 2 MB.', 'error');
      try {
        const parsed = JSON.parse(await file.text());
        const rawProjects = Array.isArray(parsed?.projects) ? parsed.projects : [parsed];
        if (!rawProjects.length || rawProjects.length > 200) throw new Error('Cantidad de proyectos inválida');
        const imported = rawProjects.map(normalizeImportedProject);
        const projects = [...imported, ...readLocalProjects()];
        if (!writeLocalProjects(projects)) return;
        renderProjectsList();
        notify(`${imported.length} ${imported.length === 1 ? 'proyecto importado' : 'proyectos importados'}.`, 'success');
      } catch (error) {
        notify(`No se pudo importar el archivo: ${error.message}`, 'error', 6000);
      } finally {
        document.getElementById('projects-import').value = '';
      }
    }

    function projectAction(label, handler, variant = '') {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `btn-project-action ${variant}`.trim();
      button.textContent = label;
      button.addEventListener('click', handler);
      return button;
    }

    function renderProjectsList() {
      const projects = readLocalProjects();
      const list = document.getElementById('projects-list');
      const empty = document.getElementById('projects-empty');
      document.getElementById('projects-count').textContent = `${projects.length} ${projects.length === 1 ? 'proyecto' : 'proyectos'}`;
      empty.hidden = projects.length > 0;
      list.replaceChildren();
      projects.forEach(project => {
        const item = document.createElement('article');
        item.className = `project-item${project.id === activeProjectId ? ' selected' : ''}`;
        const meta = document.createElement('div');
        meta.className = 'project-meta';
        const title = document.createElement('strong');
        title.textContent = project.name;
        const details = document.createElement('small');
        const summary = project.summary || {};
        const updated = project.updatedAt ? new Date(project.updatedAt).toLocaleString() : 'fecha desconocida';
        details.textContent = `${summary.boxType === 'closed' ? 'Sellada' : 'Bass-reflex'} · Vb ${n(summary.vb)} L · F3 ${n(summary.f3)} Hz · ${updated}`;
        meta.append(title, details);
        const actions = document.createElement('div');
        actions.className = 'project-actions';
        actions.append(
          projectAction('Cargar', () => loadLocalProject(project.id)),
          projectAction('Duplicar', () => duplicateLocalProject(project.id)),
          projectAction('JSON', () => exportLocalProject(project.id)),
          projectAction('Eliminar', () => deleteLocalProject(project.id), 'danger'),
        );
        item.append(meta, actions);
        list.appendChild(item);
      });
    }

    function trapFocus(container, event) {
      if (event.key !== 'Tab') return;
      const focusable = [...container.querySelectorAll(
        'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
      )].filter(element => !element.hidden && element.getClientRects().length);
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }

    function initKeyboardAccessibility() {
      document.querySelectorAll('.bridge-cta, .ts-card, .tip').forEach(element => {
        element.tabIndex = 0;
        element.setAttribute('role', 'button');
        if (element.classList.contains('ts-card')) {
          element.setAttribute('aria-expanded', element.classList.contains('open') ? 'true' : 'false');
          element.addEventListener('click', () => {
            element.classList.toggle('open');
            element.setAttribute('aria-expanded', element.classList.contains('open') ? 'true' : 'false');
          });
        }
        element.addEventListener('keydown', event => {
          if (event.key !== 'Enter' && event.key !== ' ') return;
          event.preventDefault();
          element.click();
        });
      });

      const calcPanels = ['tab-diag', 'tab-port', 'tab-dim', 'tab-chart', 'tab-cuts', 'tab-compare'];
      document.querySelectorAll('.calc-tab').forEach((tab, index) => {
        const panel = document.getElementById(calcPanels[index]);
        const tabId = `calc-tab-control-${index + 1}`;
        tab.id = tabId;
        tab.setAttribute('role', 'tab');
        tab.setAttribute('aria-controls', panel.id);
        tab.setAttribute('aria-selected', index === 0 ? 'true' : 'false');
        tab.tabIndex = index === 0 ? 0 : -1;
        panel.setAttribute('role', 'tabpanel');
        panel.setAttribute('aria-labelledby', tabId);
      });

      const advancedTabs = [...document.querySelectorAll('#sec-avanzadas .tab-btn')];
      const advancedPanels = ['tab-bp', 'tab-tl'];
      advancedTabs.forEach((tab, index) => {
        const panel = document.getElementById(advancedPanels[index]);
        const tabId = `advanced-tab-control-${index + 1}`;
        tab.id = tabId;
        tab.setAttribute('role', 'tab');
        tab.setAttribute('aria-controls', panel.id);
        tab.setAttribute('aria-selected', index === 0 ? 'true' : 'false');
        tab.tabIndex = index === 0 ? 0 : -1;
        panel.setAttribute('role', 'tabpanel');
        panel.setAttribute('aria-labelledby', tabId);
      });
      advancedTabs[0]?.parentElement.setAttribute('role', 'tablist');

      document.querySelectorAll('[role="tablist"]').forEach(tablist => {
        tablist.addEventListener('keydown', event => {
          if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
          const tabs = [...tablist.querySelectorAll('[role="tab"]')];
          const current = tabs.indexOf(document.activeElement);
          if (current < 0) return;
          event.preventDefault();
          let next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1
            : (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
          tabs[next].focus();
          tabs[next].click();
        });
      });
    }

    function initDeclarativeEvents() {
      const actions = {
        calculate,
        closeEncMenu,
        closeProjectsModal,
        dismissContext,
        downloadPDF,
        exportAllProjects,
        openEncMenu,
        openProjectsImport: () => document.getElementById('projects-import').click(),
        openProjectsModal,
        runCompare,
        runScipy,
        saveLocalProject,
        showBackendInstructions,
        updateLocalProject,
      };

      document.addEventListener('click', event => {
        const target = event.target.closest(
          '[data-action], [data-view], [data-enc], [data-tip], [data-calc-box], [data-result-tab], [data-content-tab], [data-speaker-index], [data-use-speaker], [data-tip-section]'
        );
        if (!target) return;
        if (target.dataset.action) actions[target.dataset.action]?.();
        else if (target.dataset.view) setView(target.dataset.view);
        else if (target.dataset.enc) showEnc(target.dataset.enc);
        else if (target.dataset.tip) openTip(target.dataset.tip);
        else if (target.dataset.calcBox) goToCalc(target.dataset.calcBox);
        else if (target.dataset.resultTab) switchTab({ currentTarget: target }, target.dataset.resultTab);
        else if (target.dataset.contentTab) openTab({ currentTarget: target }, target.dataset.contentTab, target.dataset.tabGroup);
        else if (target.dataset.speakerIndex) loadSpeaker(Number(target.dataset.speakerIndex));
        else if (target.dataset.useSpeaker) useFromDB(Number(target.dataset.useSpeaker));
        else if (target.dataset.tipSection) {
          event.preventDefault();
          goToEncSection(target.dataset.tipSection);
        }
      });

      document.getElementById('spk-search').addEventListener('input', event => filterDB(event.target.value));
      document.getElementById('boxType').addEventListener('change', toggleBoxOpts);
      document.getElementById('portType').addEventListener('change', togglePortOpts);
    }

    /* ── Init ───────────────────────────────────────────────── */
    document.addEventListener('DOMContentLoaded', () => {
      initDeclarativeEvents();
      initKeyboardAccessibility();
      Object.entries(FIELD_RULES).forEach(([id, rule]) => {
        const field = document.getElementById(id);
        if (!field) return;
        field.min = rule.min;
        field.max = rule.max;
        if (rule.required) field.required = true;
      });
      document.querySelectorAll('.form-field input, .form-field select').forEach(field => {
        field.addEventListener('input', () => {
          const error = document.getElementById(`${field.id}-error`);
          if (error) error.remove();
          field.removeAttribute('aria-invalid');
          field.removeAttribute('aria-describedby');
          if (PROJECT_FIELD_IDS.includes(field.id)) {
            projectDirty = projectBaseline
              ? JSON.stringify(collectProjectForm()) !== projectBaseline
              : true;
            refreshProjectEditState();
          }
        });
      });
      toggleBoxOpts();
      togglePortOpts();
      _probeBackend();
      loadDB();
      const projectsModal = document.getElementById('projects-modal');
      projectsModal.addEventListener('click', event => {
        if (event.target === projectsModal) closeProjectsModal();
      });
      document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && !projectsModal.hidden) closeProjectsModal();
        if (event.key === 'Escape' && document.getElementById('enc-sidebar').classList.contains('mobile-open')) {
          closeEncMenu();
        }
        if (!projectsModal.hidden) trapFocus(projectsModal, event);
        const encSidebar = document.getElementById('enc-sidebar');
        if (encSidebar.classList.contains('mobile-open')) trapFocus(encSidebar, event);
      });
      document.getElementById('project-name').addEventListener('keydown', event => {
        if (event.key === 'Enter') {
          activeProjectId ? updateLocalProject() : saveLocalProject();
        }
      });
      document.getElementById('project-name').addEventListener('input', () => {
        if (activeProjectId) {
          projectDirty = true;
          refreshProjectEditState();
        }
      });
      document.getElementById('projects-import').addEventListener('change', event => {
        importProjectsFile(event.target.files?.[0]);
      });
      window.addEventListener('beforeunload', event => {
        if (!projectDirty) return;
        event.preventDefault();
        event.returnValue = '';
      });
    });

    /* Sondea el backend al cargar — muestra/oculta el botón scipy según disponibilidad */
    async function _probeBackend() {
      const btn = document.getElementById('btn-scipy');
      const badge = document.getElementById('chart-mode-badge');
      try {
        const r = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(2000) });
        if (r.ok) {
          // Backend disponible — botón scipy visible y activo
          if (btn) {
            btn.hidden = false;
            btn.classList.remove('backend-unavailable');
            btn.dataset.action = 'runScipy';
            btn.title = 'Backend scipy disponible';
          }
          if (badge) { badge.title = 'Conectado al backend'; }
        } else {
          _hideScipy();
        }
      } catch (_) {
        _hideScipy();
      }
    }

    function _hideScipy() {
      const btn = document.getElementById('btn-scipy');
      if (btn) {
        btn.textContent = '⚗️ scipy (sin backend)';
        btn.title = 'Arranca uvicorn api.index:app --port 8000 para activar la simulación precisa';
        btn.classList.add('backend-unavailable');
        btn.dataset.action = 'showBackendInstructions';
      }
    }

    function showBackendInstructions() {
      const message = document.getElementById('chart-api-error');
      message.innerHTML = `⚗️ El backend Python no está corriendo.<br>
        <small>Lanza: <code>uvicorn api.index:app --reload --port 8000</code></small>`;
      message.className = 'u-inline-09 simulation-message-info';
      message.hidden = false;
    }
  
