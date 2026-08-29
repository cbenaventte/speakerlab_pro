    const CUSTOM_SPEAKERS_KEY = 'speakerlab.custom-speakers.v1';
    let DB = [];
    let BUILTIN_DB = [];

    function speakerAlignment(speaker) {
      const ebp = speaker.qes ? speaker.fs / speaker.qes : speaker.fs / speaker.qts;
      if (ebp < 50) return 'Closed';
      if (speaker.qts < 0.30) return 'QB3';
      if (speaker.qts < 0.40) return 'B4';
      return 'SBB4';
    }

    function normalizeSpeaker(speaker, custom = false) {
      const sd = Number(speaker.sd) || null;
      return {
        ...speaker,
        brand: speaker.brand || speaker.manufacturer,
        model: speaker.model || speaker.model_name,
        fs: Number(speaker.fs), vas: Number(speaker.vas), qts: Number(speaker.qts),
        qes: Number(speaker.qes) || null, qms: Number(speaker.qms) || null,
        xmax: Number(speaker.xmax) || null, sd, spl: Number(speaker.spl) || null,
        mms: Number(speaker.mms) || null, bl: Number(speaker.bl) || null,
        re: Number(speaker.re) || null, le: Number(speaker.le) || null,
        inches: Number(speaker.inches) || (sd < 150 ? 6 : sd < 400 ? 8 : sd < 600 ? 10 : sd < 800 ? 12 : 15),
        align: custom ? speakerAlignment(speaker) : (speaker.align || speakerAlignment(speaker)),
        custom,
      };
    }

    function readCustomSpeakers() {
      try {
        const speakers = JSON.parse(localStorage.getItem(CUSTOM_SPEAKERS_KEY) || '[]');
        return Array.isArray(speakers) ? speakers : [];
      } catch (_) {
        return [];
      }
    }

    function writeCustomSpeakers(speakers) {
      localStorage.setItem(CUSTOM_SPEAKERS_KEY, JSON.stringify(speakers));
      refreshSpeakerDB();
    }

    function refreshSpeakerDB() {
      DB = [...BUILTIN_DB, ...readCustomSpeakers().map(speaker => normalizeSpeaker(speaker, true))];
      if (typeof renderDB === 'function' && currentView === 'db') renderDB();
    }

    async function loadDB() {
      try {
        const res = await fetch(`${API_BASE}/api/speakers`);
        if (res.ok) {
          const rawDB = await res.json();
          BUILTIN_DB = rawDB.map(speaker => normalizeSpeaker(speaker));
        }
      } catch (e) {
        console.error('Error cargando BD:', e);
      } finally {
        refreshSpeakerDB();
      }
    }
