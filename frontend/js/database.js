    let DB = [];

    async function loadDB() {
      try {
        const res = await fetch(`${API_BASE}/api/speakers`);
        if (res.ok) {
          const rawDB = await res.json();
          DB = rawDB.map(s => {
            const ebp = s.qes ? (s.fs / s.qes) : (s.fs / s.qts);
            let align = "QB3";
            if (ebp < 50) align = "Closed";
            else {
              if (s.qts < 0.30) align = "QB3";
              else if (s.qts < 0.40) align = "B4";
              else align = "SBB4";
            }
            return {
              ...s,
              brand: s.manufacturer,
              model: s.model_name,
              mms: s.mms || null,
              bl: s.bl || null,
              re: s.re || null,
              le: s.le || null,
              inches: s.sd < 150 ? 6 : s.sd < 400 ? 8 : s.sd < 600 ? 10 : s.sd < 800 ? 12 : 15,
              align: align
            };
          });
          if (typeof renderDB === 'function' && currentView === 'db') renderDB();
        }
      } catch (e) { console.error('Error cargando BD:', e); }
    }
