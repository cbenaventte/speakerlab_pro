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
        button.dataset.idleHtml = button.innerHTML;
        button.disabled = true;
        button.setAttribute('aria-busy', 'true');
        button.textContent = `⏳ ${busyLabel}`;
      } else {
        button.disabled = false;
        button.removeAttribute('aria-busy');
        if (button.dataset.idleHtml) button.innerHTML = button.dataset.idleHtml;
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
      simulationVoltage: { min: 0.1, max: 200, required: true, labelKey: 'simulation_voltage' },
      qb: { min: 3, max: 30, required: true, labelKey: 'loss_factor_qb' },
      portDiam: { min: 0.5, max: 50, labelKey: 'port_diameter' },
      numPorts: { min: 1, max: 8, labelKey: 'port_count' },
      slotW: { min: 0.5, max: 200, labelKey: 'slot_width' },
      slotH: { min: 0.5, max: 200, labelKey: 'slot_height' },
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
        const fieldLabel = rule.labelKey ? t(rule.labelKey) : rule.label;
        const raw = field?.value.trim();
        if (!raw) {
          if (rule.required) {
            showFieldError(id, t('required', { field: fieldLabel }));
            valid = false;
          }
          return;
        }
        const value = Number(raw);
        if (!Number.isFinite(value) || value < rule.min || value > rule.max) {
          showFieldError(id, t('range', { field: fieldLabel, min: rule.min, max: rule.max }));
          valid = false;
        }
      });

      const qts = Number(document.getElementById('qts').value);
      for (const id of ['qes', 'qms']) {
        const field = document.getElementById(id);
        if (field.value && Number(field.value) <= qts) {
          showFieldError(id, t('greater_qts', { field: id.toUpperCase() }));
          valid = false;
        }
      }
      if (boxType === 'closed') {
        const qtc = Number(document.getElementById('qtcTarget').value);
        if (qtc <= qts) {
          showFieldError('qtcTarget', t('target_qtc_greater'));
          valid = false;
        }
      }

      if (!valid) {
        document.querySelector('[aria-invalid="true"]')?.focus();
        notify(t('fix_fields'), 'warning');
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
        'sec-cerradas': t('section_closed'), 'sec-reflex': t('section_reflex'),
        'sec-ts': t('section_ts'), 'sec-puertos': t('section_port'),
        'sec-materiales': t('section_materials'),
      };
      msg.textContent = t('linked_from', { section: sectionNames[currentEnc] || t('nav_encyclopedia') });
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

    const TIPS_EN = {
      Fs: { title: 'Fs — Free-Air Resonance', body: 'The frequency at which the cone naturally resonates. The enclosure raises it to Fc. Response falls rapidly below Fs.' },
      Vas: { title: 'Vas — Equivalent Volume', body: 'Suspension compliance expressed as an equivalent volume of air. It is not the enclosure volume. High Vas means a compliant suspension and a larger enclosure.' },
      Qts: { title: 'Qts — Total Quality Factor', body: 'Damping at resonance. Qts < 0.4 generally favors bass reflex; Qts > 0.4 generally favors sealed. Qts = Qes·Qms/(Qes+Qms).' },
      Xmax: { title: 'Xmax — Maximum Excursion', body: 'Maximum linear travel in one direction. Together with Sd, it determines Vd = Sd×Xmax and maximum low-frequency SPL.' },
      Sd: { title: 'Sd — Effective Piston Area', body: 'Effective radiating cone area in cm². Together with Xmax: Vd = Sd × Xmax. Greater Sd moves more air at the same excursion.' },
      TipoCaja: { title: 'Enclosure Type', body: 'Reflex offers greater extension and efficiency and generally suits Qts < 0.4. Sealed offers more accurate, controlled bass and generally suits Qts 0.4–0.7.' },
      Alineacion: { title: 'Thiele-Small Alignment', body: 'QB3 balances the main tradeoffs. SBB4 maximizes extension with more volume. B4 is the classic maximally flat Butterworth alignment.' },
      FactorK: { title: 'End Correction k', body: 'Compensates for the additional air mass at the port openings. 0.732 is the standard value for one open and one flanged end.' },
      Qtc: { title: 'Target Qtc (Sealed)', body: '0.577 Bessel favors transient response but needs a large enclosure. 0.707 Butterworth is maximally flat. 1.0 Chebyshev adds a +1.2 dB peak in a smaller enclosure.' },
      Mms: { title: 'Mms — Moving Mass', body: 'Total moving mass in grams. It is critical for accuracy. Estimating it from Vas/Sd can introduce errors above 100%.' },
      Bl: { title: 'Bl — Force Factor', body: 'Magnetic flux density times wire length in the gap (T·m). It couples the electrical and mechanical domains: F = Bl × I. High Bl generally means a powerful motor and low Qes.' },
    };

    function openTip(key) {
      const tip = (getLanguage() === 'en' ? TIPS_EN : TIPS)[key];
      if (!tip) return;
      const el = document.getElementById('enc-tooltip');
      el.innerHTML = `<strong>${tip.title}</strong>${tip.body}
    <br><a href="#" data-tip-section="${key}" class="tip-section-link">${t('view_encyclopedia')}</a>`;
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
      <div><span class="dd-brand">${escapeHtml(s.brand)}</span><br><span class="dd-model">${escapeHtml(s.model)}</span></div>
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
      scheduleCalculatorDraft();
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
      const requestedTab = evt.currentTarget;
      if (requestedTab.hidden || requestedTab.disabled) return;
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
      if (id === 'tab-chart' && calcResults) {
        if (sciPyData) {
          drawChartFromData(sciPyData);
          renderExcursionChart(sciPyData);
        } else {
          drawChart(calcResults);
        }
      }
    }

    function n(v, d = 1) {
      return (v !== null && v !== undefined && !isNaN(v))
        ? formatNumber(v, { minimumFractionDigits: d, maximumFractionDigits: d })
        : '—';
    }

    function renderCalculationResults(r) {
      if (!r) return;
      document.getElementById('r-vb').innerHTML = `${n(r.Vb)} <span class="hc-unit">L</span>`;
      document.getElementById('r-vb-sub').textContent = t('gross_for_cuts', { value: n(r.Vb_bruto) });
      const fbLabel = document.getElementById('r-fb-label');
      if (r.boxType === 'reflex') {
        fbLabel.dataset.i18n = 'tuning_frequency';
        fbLabel.textContent = t('tuning_frequency');
        document.getElementById('r-fb').innerHTML = `${n(r.Fb)} <span class="hc-unit">Hz</span>`;
        document.getElementById('r-fb-sub').textContent = t('alignment_value', { value: r.alignment });
        document.getElementById('r-f3-sub').textContent = `f3/Fs = ${n(r.F3 / r.fs, 3)}`;
      } else {
        fbLabel.dataset.i18n = 'actual_qtc';
        fbLabel.textContent = t('actual_qtc');
        document.getElementById('r-fb').textContent = n(r.Qtc_real, 3);
        document.getElementById('r-fb-sub').textContent = t('target_value', { value: r.qtcTarget });
        document.getElementById('r-f3-sub').textContent = t('butterworth_derivation');
      }
      document.getElementById('r-f3').innerHTML = `${n(r.F3)} <span class="hc-unit">Hz</span>`;
      renderDiag(r);
      renderPort(r);
      renderDim(r);
      renderCuts(r);
    }

    /* ── CÁLCULO PRINCIPAL ──────────────────────────────────── */
    async function calculate() {
      if (!validateCalculatorForm()) return;
      const fs = parseFloat(document.getElementById('fs').value);
      const vas = parseFloat(document.getElementById('vas').value);
      const qts = parseFloat(document.getElementById('qts').value);
      const calculateButton = document.getElementById('btn-calculate');
      setButtonBusy(calculateButton, true, t('calculating'));

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
      const simulationVoltage = parseFloat(document.getElementById('simulationVoltage').value);

      let r = { fs, vas, qts, qms, T, boxType, mms, bl, re, le, simulationVoltage };

      if (boxType === 'reflex') {
        const alignment = document.getElementById('alignment').value;
        const portType = document.getElementById('portType').value;
        const portDiam = parseFloat(document.getElementById('portDiam').value) || 7;
        const slotW = parseFloat(document.getElementById('slotW').value) || 10;
        const slotH = parseFloat(document.getElementById('slotH').value) || 5;
        const k = parseFloat(document.getElementById('kFactor').value);
        const N = parseInt(document.getElementById('numPorts').value) || 1;
        const qb = parseFloat(document.getElementById('qb').value);

        let target;
        try {
          const response = await fetch(`${API_BASE}/api/alignments`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fs, vas, qts, qb }),
          });
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const data = await response.json();
          if (data.reflex_supported === false) {
            throw new Error(t('reflex_qts_unsupported', { value: qts }));
          }
          target = data.alignments?.[alignment];
          if (!target) throw new Error(t('unavailable_alignment', { alignment }));
        } catch (error) {
          notify(t('alignment_error', { error: error.message }), 'error');
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
        // La velocidad depende de la tensión y del modelo completo. Se obtiene
        // exclusivamente desde /api/simulate para no mezclar dos estimaciones.
        const portVel = null;
        const Vport = N * Sp * Math.max(L, 1) / 1000;
        const Vdriver = 0.0035 * inches ** 2.8;
        const Vb_bruto = Vb + Vport + Vdriver + 0.05 * Vb;
        const Vd = (sd && xmax) ? sd * (xmax / 10) : null;
        const SPLmax = Vd ? 112.2 + 20 * Math.log10((Vd / 1e6) * Fb ** 2) : null;
        const EBP = qes ? fs / qes : null;

        Object.assign(r, { alignment, Vb, Fb, F3, Sp, SpTotal, d_eq, L, portVel, Vb_bruto, Vd, SPLmax, EBP, N, qb, portType, portDiam, slotW, slotH, Vdriver, qes, xmax, sd, inches });
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
          notify(t('closed_error', { error: error.message }), 'error');
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

      renderCalculationResults(r);

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
      notify(t('design_success'), 'success');
      sciPyData = null;
      await runScipy();
    }

    /* ── Renderizado ────────────────────────────────────────── */
    function renderDiag(r) {
      const items = [];
      if (r.EBP) {
        const c = r.EBP > 100 ? 'g' : r.EBP > 50 ? 'y' : 'r';
        const m = r.EBP > 100 ? t('ebp_ideal_reflex') : r.EBP > 50 ? t('ebp_both') : t('ebp_closed');
        items.push({ label: 'EBP (Fs/Qes)', val: `${n(r.EBP, 0)} — ${m}`, c });
      }
      if (r.portVel != null) {
        const c = r.portVel < 12 ? 'g' : r.portVel < 17 ? 'y' : 'r';
        const m = r.portVel < 12 ? t('no_turbulence') : r.portVel < 17 ? t('acceptable_limit') : t('turbulence');
        items.push({ label: t('port_velocity'), val: `${n(r.portVel)} m/s — ${m}`, c });
      }
      if (r.boxType === 'reflex' && r.portVel == null) {
        items.push({ label: t('port_velocity'), val: t('port_velocity_pending'), c: 'y' });
      }
      if (r.simulationPower != null) {
        items.push({
          label: t('simulation_drive'),
          val: `${n(r.simulationVoltage, 2)} V RMS ≈ ${n(r.simulationPower, 2)} W (${t('power_over_re')})`,
          c: 'g',
        });
      }
      if (r.SPLmax) items.push({ label: t('max_spl'), val: `${n(r.SPLmax)} dB @ 1m / 1W`, c: 'g' });
      if (r.Vd) items.push({ label: 'Vd = Sd × Xmax', val: `${n(r.Vd, 0)} cm³ — ${r.Vd > 100 ? t('serious_subwoofer') : t('medium_woofer')}`, c: r.Vd > 100 ? 'g' : 'y' });
      if (r.Qtc_real) {
        const c = r.Qtc_real < 0.8 ? 'g' : r.Qtc_real < 1 ? 'y' : 'r';
        items.push({ label: t('actual_qtc'), val: `${n(r.Qtc_real, 3)} — ${r.Qtc_real < 0.8 ? t('flat_response') : t('fc_peak')}`, c });
      }
      const qok = r.qts >= 0.2 && r.qts <= 0.5;
      items.push({ label: t('qts_range'), val: `${r.qts} — ${qok ? t('valid') : t('outside_tables')}`, c: qok ? 'g' : 'y' });

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
      const portTab = document.querySelector('[data-result-tab="tab-port"]');
      const portPanel = document.getElementById('tab-port');
      const portBody = document.getElementById('port-body');
      const isReflex = r.boxType === 'reflex';

      portTab.hidden = !isReflex;
      portTab.disabled = !isReflex;
      portTab.setAttribute('aria-hidden', String(!isReflex));

      if (!isReflex) {
        portPanel.hidden = true;
        portBody.replaceChildren();
        return;
      }

      const portFeasible = r.L >= 1;
      const Fpipe = portFeasible ? 34400 / (2 * r.L) : null;
      const portDesc = r.portType === 'circular'
        ? `<div class="data-row"><span class="dr-label">${t('tube_diameter')}</span><span class="dr-val">${r.portDiam} cm</span></div>`
        : `<div class="data-row"><span class="dr-label">Slot ${r.slotW}×${r.slotH} cm</span><span class="dr-val">${t('area', { value: n(r.Sp) })}</span></div>`;

      portBody.innerHTML = `
    <div class="data-row"><span class="dr-label">${t('tube_length')}</span><span class="dr-val">${n(r.L)} cm &nbsp;(${n(r.L * 10, 0)} mm)</span></div>
    ${!portFeasible ? `<div class="data-row"><span class="dr-label">⚠️ ${t('port_not_feasible')}</span><span class="dr-val pipe-warning">${t('port_change_geometry')}</span></div>` : ''}
    <div class="data-row"><span class="dr-label">${t('area_per_port')}</span><span class="dr-val">${n(r.Sp)} cm²</span></div>
    <div class="data-row"><span class="dr-label">${t('total_area', { count: r.N })}</span><span class="dr-val">${n(r.SpTotal)} cm²</span></div>
    <div class="data-row"><span class="dr-label">${t('equivalent_diameter')}</span><span class="dr-val">${n(r.d_eq)} cm</span></div>
    ${portDesc}
    ${Fpipe ? `<div class="data-row"><span class="dr-label">${t('pipe_resonance')}</span><span class="dr-val pipe-warning">${t('pipe_fill', { value: n(Fpipe) })}</span></div>` : ''}
    <div class="data-row"><span class="dr-label">${t('subsonic_filter')}</span><span class="dr-val">${n(r.Fb * 0.7)} Hz (0.7×Fb)</span></div>
  `;
    }

    function renderDim(r) {
      const T = r.T, Vb = r.Vb_bruto;
      const Di = Math.cbrt(Vb * 1000 / (1.59 * 1.26));
      const Wi = 1.26 * Di, Hi = 1.59 * Di;
      const De = Di + 2 * T, We = Wi + 2 * T, He = Hi + 2 * T;
      const Fbsc = 115 / (We / 100);
      document.getElementById('dim-body').innerHTML = `
    <div class="data-row"><span class="dr-label">${t('interior_dimensions')}</span><span class="dr-val">${n(Hi)} × ${n(Wi)} × ${n(Di)} cm</span></div>
    <div class="data-row"><span class="dr-label">${t('exterior_dimensions')}</span><span class="dr-val">${n(He)} × ${n(We)} × ${n(De)} cm</span></div>
    <div class="data-row"><span class="dr-label">${t('wall_thickness')}</span><span class="dr-val">${r.T * 10} mm</span></div>
    <div class="data-row"><span class="dr-label">Baffle Step (F_bsc)</span><span class="dr-val">${t('baffle_compensation', { value: n(Fbsc) })}</span></div>
    <div class="data-row"><span class="dr-label">${t('golden_ratio')}</span><span class="dr-val">1.59 : 1.26 : 1.00</span></div>
  `;
    }

    function renderCuts(r) {
      const T = r.T, Vb = r.Vb_bruto;
      const Di = Math.cbrt(Vb * 1000 / (1.59 * 1.26));
      const Wi = 1.26 * Di, Hi = 1.59 * Di;
      const De = Di + 2 * T, We = Wi + 2 * T, He = Hi + 2 * T;

      const pcs = [
        { name: t('piece_front'), cant: 1, w: n(We), h: n(He), note: t('note_speaker_hole') },
        { name: t('piece_rear'), cant: 1, w: n(We), h: n(He), note: t('note_terminal') },
        { name: t('piece_top'), cant: 1, w: n(We - 2 * T), h: n(De), note: t('note_between_front_rear') },
        { name: t('piece_base'), cant: 1, w: n(We - 2 * T), h: n(De), note: t('note_between_front_rear') },
        { name: t('piece_sides'), cant: 2, w: n(He), h: n(De), note: t('note_identical') },
      ];

      document.getElementById('cuts-table').innerHTML = `
    <thead><tr><th>${t('cut_piece')}</th><th>${t('quantity_short')}</th><th>${t('width_cm')}</th><th>${t('height_cm')}</th><th>${t('notes')}</th></tr></thead>
    <tbody>${pcs.map(p => `<tr>
      <td class="piece">${p.name}</td><td>${p.cant}</td>
      <td class="dims">${p.w}</td><td class="dims">${p.h}</td>
      <td class="note">${p.note}</td>
    </tr>`).join('')}</tbody>`;

      const area = (2 * (We * He) + 2 * ((We - 2 * T) * De) + 2 * (He * De)) / 10000;
      document.getElementById('cuts-summary').innerHTML = `
    <div class="sum-item"><div class="sv">${n(Vb)} L</div><div class="sl">${t('gross_vb')}</div></div>
    <div class="sum-item"><div class="sv">${n(area * 1.15, 3)} m²</div><div class="sl">${t('board_waste')}</div></div>
    <div class="sum-item"><div class="sv">${n(We)}×${n(He)}×${n(De)}</div><div class="sl">${t('exterior_cm')}</div></div>`;
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
        model_name: document.getElementById('spk-search').value || t('unnamed_driver'),
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
        qb: r.qb || undefined,
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
        eg_volts: r.simulationVoltage,
        language: getLanguage(),
      };
    }

    /* Llama a POST /api/simulate y actualiza la gráfica */
    async function runScipy() {
      const payload = buildDriverPayload();
      if (!payload) {
        notify(t('calculate_before_simulation'), 'warning');
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
      btnScipy.textContent = `⏳ ${t('calculating')}`;

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
        calcResults.simulationPower = m.input_power_w;
        calcResults.simulationVoltage = m.simulation_voltage;
        if (m.max_port_velocity != null) {
          calcResults.portVel = m.max_port_velocity;
        }
        renderDiag(calcResults);
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
        notify(t('simulation_success'), 'success');

      } catch (err) {
        loadEl.hidden = true;
        canvasEl.hidden = false;
        // Backend no disponible — mostrar aviso suave, no error rojo
        const isConnErr = err.message === 'Failed to fetch' || err.message.includes('NetworkError');
        if (isConnErr) {
          errEl.innerHTML = `⚗️ ${t('scipy_disconnected')}
        <a href="https://github.com/cbenaventte/speakerlab_pro" target="_blank" rel="noopener" class="simulation-help-link">
        ${t('startup_instructions')}</a>`;
          errEl.className = 'u-inline-09 simulation-message-info';
        } else {
          errEl.innerHTML = `❌ <strong>${t('scipy_error', { error: err.message })}</strong>`;
          notify(t('simulation_error', { error: err.message }), 'error');
          errEl.className = 'u-inline-09 simulation-message-error';
        }
        errEl.hidden = false;
        drawChart(calcResults);   // fallback a la gráfica JS
      } finally {
        btnScipy.disabled = false;
        btnScipy.innerHTML = `⚗️ <span data-i18n="simulate_scipy">${t('simulate_scipy')}</span>`;
      }
    }

    /* ── Llama a POST /api/compare y genera la gráfica comparativa ── */
    async function runCompare() {
      const payload = buildDriverPayload();
      if (!payload) {
        notify(t('calculate_before_compare'), 'warning');
        return;
      }
      const btn = document.getElementById('btn-compare');
      const errEl = document.getElementById('compare-error');
      btn.textContent = `⏳ ${t('analyze')}`;
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
        <td class="compare-align compare-align-${align.toLowerCase()}">${align} ${align === 'Closed' ? `(${t('sealed')})` : ''}</td>
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
        notify(t('comparison_success'), 'success');
      } catch (e) {
        errEl.textContent = `❌ ${t('api_error', { error: e.message })}`;
        errEl.hidden = false;
        notify(t('comparison_error', { error: e.message }), 'error');
      } finally {
        btn.innerHTML = `<span data-i18n="compare_four">${t('compare_four')}</span>`;
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
      ctx.strokeStyle = 'rgba(198,40,40,0.55)'; ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(PAD.l, y3); ctx.lineTo(PAD.l + fw, y3); ctx.stroke();
      ctx.setLineDash([]);

      // Marcadores verticales (F3, Fb, etc.)
      let previousMarkerX = -Infinity;
      let labelRow = 0;
      [...markers].sort((a, b) => a.f - b.f).forEach(({ f, label, color, kind }) => {
        if (!f || f < 15 || f > 900) return;
        const x = xPos(f);
        labelRow = x - previousMarkerX < 85 ? (labelRow + 1) % 3 : 0;
        previousMarkerX = x;
        const dash = kind === 'f3' ? [] : kind === 'fb' ? [7, 5] : [3, 4];
        ctx.strokeStyle = color;
        ctx.lineWidth = kind === 'f3' ? 2.2 : 1.6;
        ctx.setLineDash(dash);
        ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + fh); ctx.stroke();
        ctx.setLineDash([]);
        if (kind === 'f3') {
          ctx.beginPath(); ctx.arc(x, y3, 4.5, 0, Math.PI * 2);
          ctx.fillStyle = color; ctx.fill();
          ctx.strokeStyle = '#fffdf8'; ctx.lineWidth = 1.5; ctx.stroke();
        }
        const labelY = PAD.t + 17 + labelRow * 17;
        ctx.font = 'bold 10px Kalam,cursive';
        const labelWidth = ctx.measureText(label).width + 8;
        const labelX = Math.min(x + 4, PAD.l + fw - labelWidth);
        ctx.fillStyle = 'rgba(255,253,248,0.9)';
        ctx.fillRect(labelX, labelY - 12, labelWidth, 15);
        ctx.fillStyle = color; ctx.textAlign = 'left';
        ctx.fillText(label, labelX + 4, labelY);
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
        { f: m.f3, label: `F3=${m.f3}Hz`, color: '#c62828', kind: 'f3' },
        { f: m.f6, label: `F6=${m.f6}Hz`, color: '#8d6e63', kind: 'f6' },
        { f: m.fb, label: `Fb=${m.fb}Hz`, color: '#075fa8', kind: 'fb' },
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

      const PAD = { l: 54, r: 48, t: 16, b: 34 };
      const fw = W - PAD.l - PAD.r, fh = H - PAD.t - PAD.b;
      const logMin = Math.log10(15), logMax = Math.log10(900);
      const xPos = f => PAD.l + (Math.log10(Math.max(f, 15)) - logMin) / (logMax - logMin) * fw;

      // Excursión (eje izquierdo)
      const maxExc = Math.max(...data.excursion) * 1.3 || 20;
      const yExc = v => PAD.t + fh - (v / maxExc) * fh;

      [0, 0.5, 1].forEach(ratio => {
        const value = maxExc * ratio, y = yExc(value);
        ctx.fillStyle = '#0984e3'; ctx.font = '9px Kalam,cursive'; ctx.textAlign = 'right';
        ctx.fillText(n(value, 1), PAD.l - 5, y + 3);
      });

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

        [0, 0.5, 1].forEach(ratio => {
          const value = maxVel * ratio, y = yVel(value);
          ctx.fillStyle = '#e17055'; ctx.font = '9px Kalam,cursive'; ctx.textAlign = 'left';
          ctx.fillText(n(value, 1), PAD.l + fw + 5, y + 3);
        });

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
        ctx.fillText(t('port_turbulence_limit'), PAD.l + fw - 4, y17 - 3);
      }

      // Leyenda
      ctx.font = '10px Kalam,cursive';
      ctx.fillStyle = '#0984e3'; ctx.textAlign = 'left'; ctx.fillText(`── ${t('axis_excursion')}`, PAD.l + 8, PAD.t + 14);
      if (data.port_vel) {
        ctx.fillStyle = '#e17055'; ctx.fillText(`- - ${t('axis_port_velocity')}`, PAD.l + 8, PAD.t + 28);
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
      const badge = document.getElementById('chart-mode-badge');
      badge.textContent = t('chart_js_approx');
      badge.classList.remove('simulation-badge-success');

      // Construir arrays desde la aproximación JS
      const freqArr = [], splArr = [];
      for (let fi = 15; fi <= 850; fi *= 1.015) {
        const exponent = r.boxType === 'reflex' ? 8 : 4;
        const db = -10 * Math.log10(1 + (r.F3 / fi) ** exponent);
        freqArr.push(fi); splArr.push(db);
      }

      const markers = [
        { f: r.F3, label: `F3=${n(r.F3, 0)}Hz`, color: '#c62828', kind: 'f3' },
        { f: r.Fb, label: `Fb=${n(r.Fb, 0)}Hz`, color: '#075fa8', kind: 'fb' },
      ].filter(m => m.f);

      _drawCanvasCore(canvas, freqArr, splArr, markers, 0);
    }

    /* ── Base de Datos view ─────────────────────────────────── */
    const CUSTOM_SPEAKER_FIELDS = ['inches', 'fs', 'vas', 'qts', 'qes', 'qms', 'xmax', 'sd', 'spl', 'mms', 'bl', 're', 'le'];
    let speakerModalTrigger = null;

    function openSpeakerModal(id = '') {
      speakerModalTrigger = document.activeElement;
      const custom = id ? readCustomSpeakers().find(speaker => speaker.id === id) : null;
      document.getElementById('custom-speaker-id').value = custom?.id || '';
      document.getElementById('custom-brand').value = custom?.brand || '';
      document.getElementById('custom-model').value = custom?.model || '';
      CUSTOM_SPEAKER_FIELDS.forEach(field => {
        const calculator = document.getElementById(field);
        document.getElementById(`custom-${field}`).value = custom?.[field] ?? calculator?.value ?? '';
      });
      document.getElementById('speaker-modal-title').textContent = custom ? t('custom_speaker_edit') : t('custom_speaker_add');
      document.getElementById('speaker-modal').hidden = false;
      document.body.classList.add('modal-open');
      document.getElementById('custom-brand').focus();
    }

    function closeSpeakerModal() {
      document.getElementById('speaker-modal').hidden = true;
      document.body.classList.remove('modal-open');
      speakerModalTrigger?.focus();
      speakerModalTrigger = null;
    }

    function collectCustomSpeaker() {
      const brand = document.getElementById('custom-brand').value.trim();
      const model = document.getElementById('custom-model').value.trim();
      if (!brand || !model) throw new Error(t('brand_model_required'));
      const speaker = { brand: brand.slice(0, 60), model: model.slice(0, 80) };
      CUSTOM_SPEAKER_FIELDS.forEach(field => {
        const input = document.getElementById(`custom-${field}`);
        if (!input.value && !input.required) return speaker[field] = null;
        if (!input.checkValidity()) throw new Error(t('review_field', { field: input.closest('label').firstChild.textContent.trim() }));
        speaker[field] = Number(input.value);
      });
      return speaker;
    }

    function saveCustomSpeaker() {
      try {
        const speaker = collectCustomSpeaker();
        const idField = document.getElementById('custom-speaker-id');
        const speakers = readCustomSpeakers();
        const existing = speakers.findIndex(item => item.id === idField.value);
        const record = {
          ...speaker,
          id: existing >= 0 ? speakers[existing].id : (crypto.randomUUID?.() || `speaker-${Date.now()}`),
          updatedAt: new Date().toISOString(),
        };
        if (existing >= 0) speakers[existing] = record;
        else speakers.push(record);
        writeCustomSpeakers(speakers);
        closeSpeakerModal();
        notify(existing >= 0 ? t('speaker_updated') : t('speaker_saved'), 'success');
      } catch (error) {
        notify(error.message, 'warning');
      }
    }

    function deleteCustomSpeaker(id) {
      const speakers = readCustomSpeakers();
      const speaker = speakers.find(item => item.id === id);
      if (!speaker) return;
      writeCustomSpeakers(speakers.filter(item => item.id !== id));
      notify(t('speaker_deleted', { speaker: `${speaker.brand} ${speaker.model}` }), 'success');
    }

    function exportCustomSpeakers() {
      const speakers = readCustomSpeakers();
      if (!speakers.length) return notify(t('no_custom_speakers'), 'warning');
      downloadJson({ version: 1, exportedAt: new Date().toISOString(), speakers }, 'speakerlab_altavoces.json');
    }

    async function importCustomSpeakers(file) {
      if (!file) return;
      try {
        if (file.size > 2 * 1024 * 1024) throw new Error(t('file_too_large'));
        const parsed = JSON.parse(await file.text());
        if (parsed?.version !== 1 || !Array.isArray(parsed.speakers) || parsed.speakers.length > 500) {
          throw new Error(t('incompatible_speakers'));
        }
        const imported = parsed.speakers.map(raw => {
          const brand = String(raw.brand || '').trim().slice(0, 60);
          const model = String(raw.model || '').trim().slice(0, 80);
          const fs = Number(raw.fs), vas = Number(raw.vas), qts = Number(raw.qts), sd = Number(raw.sd), inches = Number(raw.inches);
          if (!brand || !model || fs < 5 || fs > 500 || vas < 0.1 || vas > 2000 || qts < 0.05 || qts > 2 || sd < 1 || sd > 5000 || inches < 1 || inches > 30) {
            throw new Error(t('invalid_record', { brand: brand || t('no_brand'), model: model || t('no_model') }));
          }
          const record = { ...raw, brand, model, fs, vas, qts, sd, inches };
          CUSTOM_SPEAKER_FIELDS.forEach(field => {
            record[field] = record[field] === null || record[field] === '' ? null : Number(record[field]);
          });
          record.id = crypto.randomUUID?.() || `speaker-${Date.now()}-${Math.random()}`;
          record.updatedAt = new Date().toISOString();
          return record;
        });
        writeCustomSpeakers([...imported, ...readCustomSpeakers()]);
        notify(t('speakers_imported', { count: imported.length }), 'success');
      } catch (error) {
        notify(t('import_error_short', { error: error.message }), 'error', 6000);
      } finally {
        document.getElementById('speakers-import').value = '';
      }
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>'"]/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[char]);
    }

    function renderDB() {
      const tbody = document.getElementById('db-tbody');
      const customCount = DB.filter(speaker => speaker.custom).length;
      document.getElementById('db-summary').textContent = t('database_summary', { verified: BUILTIN_DB.length, custom: customCount });
      tbody.innerHTML = DB.map((s, i) => `<tr>
    <td class="brand">${escapeHtml(s.brand)}${s.custom ? `<span class="db-source">${t('local')}</span>` : ''}</td>
    <td>${escapeHtml(s.model)}</td>
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
    <td><div class="db-row-actions"><button class="db-use-btn" data-use-speaker="${i}">${t('use')}</button>${s.custom ? `<button class="btn-project-action" data-edit-speaker="${s.id}">${t('edit')}</button><button class="btn-project-action danger" data-delete-speaker="${s.id}">${t('delete')}</button>` : ''}</div></td>
  </tr>`).join('');
    }

    function useFromDB(idx) {
      loadSpeaker(idx);
      setView('calc');
      const s = DB[idx];
      const banner = document.getElementById('context-banner');
      document.getElementById('context-msg').textContent = t('loaded_from_database', { speaker: `${s.brand} ${s.model}` });
      banner.classList.add('show');
    }

    /* ── Simulador enciclopedia ─────────────────────────────── */
    const alignments = {
      "1": { path: "M 50 190 C 130 190, 140 38, 220 75 L 550 75", titleKey: 'align_small_title', descKey: 'align_small_desc' },
      "2": { path: "M 50 190 C 118 190, 140 75, 220 75 L 550 75", titleKey: 'align_optimal_title', descKey: 'align_optimal_desc' },
      "3": { path: "M 50 190 C 70 190, 155 98, 220 75 L 550 75", titleKey: 'align_large_title', descKey: 'align_large_desc' },
    };

    document.addEventListener('DOMContentLoaded', () => {
      const slider = document.getElementById('align-slider');
      if (slider) slider.addEventListener('input', function () {
        const a = alignments[this.value];
        document.getElementById('curve-path').setAttribute('d', a.path);
        document.getElementById('align-label').textContent = t(a.titleKey);
        document.getElementById('align-desc').innerHTML = `<strong>${t(a.titleKey)}:</strong><br>${t(a.descKey)}`;
      });
      const desc = document.getElementById('align-desc');
      if (desc) desc.innerHTML = `<strong>${t(alignments["2"].titleKey)}:</strong><br>${t(alignments["2"].descKey)}`;
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
        notify(t('pdf_calculate_first'), 'warning');
        return;
      }
      const payload = buildDriverPayload();
      if (!payload) return;
      const buttons = [...document.querySelectorAll('[data-pdf-download]')];
      buttons.forEach(button => setButtonBusy(button, true, t('pdf_generating')));
      try {
        const response = await fetch(`${API_BASE}/api/pdf`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            driver: payload.driver,
            language: getLanguage(),
            eg_volts: payload.eg_volts,
          }),
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
        notify(t('pdf_success'), 'success');
      } catch (error) {
        notify(t('pdf_error', { error: error.message }), 'error', 6000);
      } finally {
        buttons.forEach(button => setButtonBusy(button, false));
      }
    }

    /* ── Proyectos locales ──────────────────────────────────── */
    const PROJECTS_KEY = 'speakerlab.projects.v1';
    const CALCULATOR_DRAFT_KEY = 'speakerlab.calculator-draft.v1';
    const PROJECT_FIELD_IDS = [
      'spk-search', 'fs', 'vas', 'qts', 'qes', 'qms', 'xmax', 'sd', 'spl',
      'inches', 'mms', 'bl', 're', 'le', 'simulationVoltage', 'boxType', 'material', 'alignment',
      'portType', 'portDiam', 'numPorts', 'slotW', 'slotH', 'kFactor', 'qb', 'qtcTarget',
    ];
    let activeProjectId = null;
    let projectBaseline = null;
    let projectDirty = false;
    let draftSaveTimer = null;
    let draftWasRestored = false;

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
        notify(t('no_storage_space'), 'error');
        return false;
      }
    }

    function collectProjectForm() {
      return Object.fromEntries(PROJECT_FIELD_IDS.map(id => [id, document.getElementById(id)?.value ?? '']));
    }

    function updateDraftStatus(updatedAt, restored = false) {
      const status = document.getElementById('draft-status');
      const date = new Date(updatedAt);
      draftWasRestored = restored;
      document.getElementById('draft-status-text').textContent = `${t(restored ? 'draft_restored' : 'draft_saved')} · ${date.toLocaleString(getLanguage() === 'en' ? 'en-US' : 'es-CL')}`;
      status.hidden = false;
    }

    function saveCalculatorDraft() {
      const form = collectProjectForm();
      if (!form.fs && !form.vas && !form.qts && !form['spk-search']) {
        localStorage.removeItem(CALCULATOR_DRAFT_KEY);
        document.getElementById('draft-status').hidden = true;
        return;
      }
      const updatedAt = new Date().toISOString();
      try {
        localStorage.setItem(CALCULATOR_DRAFT_KEY, JSON.stringify({ version: 1, updatedAt, form }));
        updateDraftStatus(updatedAt);
      } catch (_) {
        notify(t('draft_save_error'), 'warning');
      }
    }

    function scheduleCalculatorDraft() {
      window.clearTimeout(draftSaveTimer);
      draftSaveTimer = window.setTimeout(saveCalculatorDraft, 600);
    }

    function restoreCalculatorDraft() {
      try {
        const draft = JSON.parse(localStorage.getItem(CALCULATOR_DRAFT_KEY));
        if (draft?.version !== 1 || !draft.form || typeof draft.updatedAt !== 'string') return false;
        PROJECT_FIELD_IDS.forEach(id => {
          const field = document.getElementById(id);
          if (field && Object.hasOwn(draft.form, id)) field.value = String(draft.form[id] ?? '').slice(0, 200);
        });
        updateDraftStatus(draft.updatedAt, true);
        return true;
      } catch (_) {
        localStorage.removeItem(CALCULATOR_DRAFT_KEY);
        return false;
      }
    }

    function clearCalculatorDraft() {
      localStorage.removeItem(CALCULATOR_DRAFT_KEY);
      document.getElementById('draft-status').hidden = true;
      notify(t('draft_deleted'), 'success');
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
        ? `${t('selected_project', { name: project.name })}${projectDirty ? ` · ${t('unsaved_changes')}` : ''}`
        : t('no_project_selected');
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
      document.getElementById('project-name').focus();
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
        notify(t('calculate_before_save'), 'warning');
        closeProjectsModal();
        return;
      }
      if (!name) {
        notify(t('enter_project_name'), 'warning');
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
      saveCalculatorDraft();
      renderProjectsList();
      refreshProjectEditState();
      notify(t('project_saved', { name: project.name }), 'success');
    }

    function updateLocalProject() {
      const projects = readLocalProjects();
      const project = projects.find(item => item.id === activeProjectId);
      if (!project) return notify(t('select_saved_project'), 'warning');
      const name = document.getElementById('project-name').value.trim();
      if (!name) return notify(t('enter_project_name'), 'warning');
      project.name = name.slice(0, 80);
      project.updatedAt = new Date().toISOString();
      project.form = collectProjectForm();
      project.summary = currentProjectSummary();
      if (!writeLocalProjects(projects)) return;
      projectBaseline = JSON.stringify(project.form);
      projectDirty = false;
      renderProjectsList();
      refreshProjectEditState();
      notify(t('project_updated', { name: project.name }), 'success');
    }

    async function loadLocalProject(id) {
      const project = readLocalProjects().find(item => item.id === id);
      if (!project) return notify(t('project_missing'), 'error');
      if (projectDirty && activeProjectId !== id && !window.confirm(t('confirm_load_changes'))) return;
      if (!Object.hasOwn(project.form || {}, 'simulationVoltage')) {
        document.getElementById('simulationVoltage').value = '2.83';
      }
      Object.entries(project.form || {}).forEach(([fieldId, value]) => {
        const field = document.getElementById(fieldId);
        if (field) field.value = value;
      });
      toggleBoxOpts();
      togglePortOpts();
      activeProjectId = project.id;
      projectBaseline = JSON.stringify(collectProjectForm());
      projectDirty = false;
      saveCalculatorDraft();
      document.getElementById('project-name').value = project.name;
      setView('calc');
      closeProjectsModal();
      await calculate();
      notify(t('project_restored', { name: project.name }), 'success');
    }

    function deleteLocalProject(id) {
      const projects = readLocalProjects();
      const project = projects.find(item => item.id === id);
      if (!project) return notify(t('project_missing'), 'error');
      if (!window.confirm(t('confirm_delete_project', { name: project.name }))) return;
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
      notify(t('project_deleted', { name: project.name }), 'success');
    }

    function duplicateLocalProject(id) {
      const projects = readLocalProjects();
      const source = projects.find(item => item.id === id);
      if (!source) return notify(t('project_missing'), 'error');
      const now = new Date().toISOString();
      const copy = {
        ...source,
        id: projectId(),
        name: `${source.name} — ${t('copy_suffix')}`.slice(0, 80),
        createdAt: now,
        updatedAt: now,
        form: { ...source.form },
        summary: { ...source.summary },
      };
      projects.unshift(copy);
      if (writeLocalProjects(projects)) {
        renderProjectsList();
        notify(t('project_duplicated', { name: copy.name }), 'success');
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
      if (!project) return notify(t('project_missing'), 'error');
      downloadJson(project, `speakerlab_${safeProjectFilename(project.name)}.json`);
      notify(t('project_exported'), 'success');
    }

    function exportAllProjects() {
      const projects = readLocalProjects();
      if (!projects.length) return notify(t('no_projects_export'), 'warning');
      downloadJson({ version: 1, exportedAt: new Date().toISOString(), projects }, 'speakerlab_proyectos.json');
      notify(t('all_projects_exported'), 'success');
    }

    function normalizeImportedProject(raw) {
      if (!raw || raw.version !== 1 || typeof raw.name !== 'string' || !raw.name.trim() || typeof raw.form !== 'object') {
        throw new Error(t('incompatible_project'));
      }
      const form = {};
      PROJECT_FIELD_IDS.forEach(id => {
        const value = raw.form[id];
        form[id] = value === undefined || value === null
          ? (id === 'simulationVoltage' ? '2.83' : '')
          : String(value).slice(0, 200);
      });
      if (!form.fs || !form.vas || !form.qts) throw new Error(t('project_missing_params', { name: raw.name }));
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
      if (file.size > 2 * 1024 * 1024) return notify(t('projects_file_too_large'), 'error');
      try {
        const parsed = JSON.parse(await file.text());
        const rawProjects = Array.isArray(parsed?.projects) ? parsed.projects : [parsed];
        if (!rawProjects.length || rawProjects.length > 200) throw new Error(t('invalid_project_count'));
        const imported = rawProjects.map(normalizeImportedProject);
        const projects = [...imported, ...readLocalProjects()];
        if (!writeLocalProjects(projects)) return;
        renderProjectsList();
        notify(t(imported.length === 1 ? 'project_imported_one' : 'project_imported_many', { count: imported.length }), 'success');
      } catch (error) {
        notify(t('import_file_error', { error: error.message }), 'error', 6000);
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
      document.getElementById('projects-count').textContent = t(projects.length === 1 ? 'projects_count_one' : 'projects_count_many', { count: projects.length });
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
        const updated = project.updatedAt ? new Date(project.updatedAt).toLocaleString(getLanguage() === 'en' ? 'en-US' : 'es-CL') : t('unknown_date');
        details.textContent = `${summary.boxType === 'closed' ? t('sealed') : 'Bass-reflex'} · Vb ${n(summary.vb)} L · F3 ${n(summary.f3)} Hz · ${updated}`;
        meta.append(title, details);
        const actions = document.createElement('div');
        actions.className = 'project-actions';
        actions.append(
          projectAction(t('load'), () => loadLocalProject(project.id)),
          projectAction(t('duplicate'), () => duplicateLocalProject(project.id)),
          projectAction('JSON', () => exportLocalProject(project.id)),
          projectAction(t('delete'), () => deleteLocalProject(project.id), 'danger'),
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
        clearCalculatorDraft,
        closeEncMenu,
        closeProjectsModal,
        dismissContext,
        downloadPDF,
        exportAllProjects,
        openEncMenu,
        openProjectsImport: () => document.getElementById('projects-import').click(),
        openProjectsModal,
        openSpeakerModal,
        openSpeakersImport: () => document.getElementById('speakers-import').click(),
        runCompare,
        runScipy,
        saveLocalProject,
        saveCustomSpeaker,
        showBackendInstructions,
        closeSpeakerModal,
        exportCustomSpeakers,
        updateLocalProject,
      };

      document.addEventListener('click', event => {
        const target = event.target.closest(
          '[data-action], [data-view], [data-enc], [data-tip], [data-calc-box], [data-result-tab], [data-content-tab], [data-speaker-index], [data-use-speaker], [data-tip-section], [data-edit-speaker], [data-delete-speaker]'
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
        else if (target.dataset.editSpeaker) openSpeakerModal(target.dataset.editSpeaker);
        else if (target.dataset.deleteSpeaker) deleteCustomSpeaker(target.dataset.deleteSpeaker);
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
            scheduleCalculatorDraft();
          }
        });
      });
      document.getElementById('spk-search').addEventListener('input', scheduleCalculatorDraft);
      restoreCalculatorDraft();
      toggleBoxOpts();
      togglePortOpts();
      _probeBackend();
      loadDB();
      const projectsModal = document.getElementById('projects-modal');
      const speakerModal = document.getElementById('speaker-modal');
      projectsModal.addEventListener('click', event => {
        if (event.target === projectsModal) closeProjectsModal();
      });
      speakerModal.addEventListener('click', event => {
        if (event.target === speakerModal) closeSpeakerModal();
      });
      document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && !projectsModal.hidden) closeProjectsModal();
        if (event.key === 'Escape' && !speakerModal.hidden) closeSpeakerModal();
        if (event.key === 'Escape' && document.getElementById('enc-sidebar').classList.contains('mobile-open')) {
          closeEncMenu();
        }
        if (!projectsModal.hidden) trapFocus(projectsModal, event);
        if (!speakerModal.hidden) trapFocus(speakerModal, event);
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
      document.getElementById('speakers-import').addEventListener('change', event => {
        importCustomSpeakers(event.target.files?.[0]);
      });
      document.addEventListener('speakerlab:languagechange', () => {
        clearFieldErrors();
        if (calcResults) renderCalculationResults(calcResults);
        if (sciPyData && !document.getElementById('freqChart').hidden) {
          drawChartFromData(sciPyData);
          renderExcursionChart(sciPyData);
        }
        renderDB();
        renderProjectsList();
        refreshProjectEditState();
        let draft = null;
        try { draft = JSON.parse(localStorage.getItem(CALCULATOR_DRAFT_KEY) || 'null'); } catch (_) { /* ignore invalid draft */ }
        if (draft?.updatedAt && !document.getElementById('draft-status').hidden) {
          updateDraftStatus(draft.updatedAt, draftWasRestored);
        }
        const speakerModal = document.getElementById('speaker-modal');
        if (!speakerModal.hidden) {
          document.getElementById('speaker-modal-title').textContent = document.getElementById('custom-speaker-id').value
            ? t('custom_speaker_edit') : t('custom_speaker_add');
        }
        const alignment = alignments[document.getElementById('align-slider')?.value || '2'];
        if (alignment) {
          document.getElementById('align-label').textContent = t(alignment.titleKey);
          document.getElementById('align-desc').innerHTML = `<strong>${t(alignment.titleKey)}:</strong><br>${t(alignment.descKey)}`;
        }
        document.getElementById('enc-tooltip').classList.remove('show');
      });
      window.addEventListener('beforeunload', event => {
        window.clearTimeout(draftSaveTimer);
        saveCalculatorDraft();
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
            btn.title = t('backend_available');
          }
          if (badge) { badge.title = t('backend_connected'); }
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
        btn.textContent = `⚗️ ${t('scipy_no_backend')}`;
        btn.title = t('scipy_start_title');
        btn.classList.add('backend-unavailable');
        btn.dataset.action = 'showBackendInstructions';
      }
    }

    function showBackendInstructions() {
      const message = document.getElementById('chart-api-error');
      message.innerHTML = `⚗️ ${t('backend_not_running')}<br>
        <small>${t('launch_command')} <code>uvicorn api.index:app --reload --port 8000</code></small>`;
      message.className = 'u-inline-09 simulation-message-info';
      message.hidden = false;
    }
  
