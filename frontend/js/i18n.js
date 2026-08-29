(function () {
  const STORAGE_KEY = 'speakerlab-language';
  const DEFAULT_LANGUAGE = 'es';
  const SUPPORTED_LANGUAGES = ['es', 'en'];

  function storedLanguage() {
    try {
      const value = localStorage.getItem(STORAGE_KEY);
      return SUPPORTED_LANGUAGES.includes(value) ? value : null;
    } catch (_) {
      return null;
    }
  }

  function browserLanguage() {
    const language = String(navigator.language || '').toLowerCase().split('-')[0];
    return SUPPORTED_LANGUAGES.includes(language) ? language : DEFAULT_LANGUAGE;
  }

  let currentLanguage = storedLanguage() || browserLanguage();

  function t(key, replacements = {}) {
    const locales = window.SPEAKERLAB_LOCALES || {};
    const dictionary = locales[currentLanguage] || locales[DEFAULT_LANGUAGE] || {};
    const fallback = locales[DEFAULT_LANGUAGE] || {};
    const template = dictionary[key] ?? fallback[key] ?? key;
    return Object.entries(replacements).reduce(
      (value, [name, replacement]) => value.replaceAll(`{${name}}`, String(replacement)),
      template,
    );
  }

  function translateDocument() {
    document.documentElement.lang = currentLanguage;
    document.title = t('meta_title');
    document.querySelector('meta[name="description"]')?.setAttribute('content', t('meta_description'));

    document.querySelectorAll('[data-i18n]').forEach(element => {
      element.textContent = t(element.dataset.i18n);
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(element => {
      element.setAttribute('placeholder', t(element.dataset.i18nPlaceholder));
    });
    document.querySelectorAll('[data-i18n-title]').forEach(element => {
      element.setAttribute('title', t(element.dataset.i18nTitle));
    });
    document.querySelectorAll('[data-i18n-aria-label]').forEach(element => {
      element.setAttribute('aria-label', t(element.dataset.i18nAriaLabel));
    });
    document.querySelectorAll('[data-language]').forEach(button => {
      const active = button.dataset.language === currentLanguage;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
  }

  function setLanguage(language, options = {}) {
    const nextLanguage = SUPPORTED_LANGUAGES.includes(language) ? language : DEFAULT_LANGUAGE;
    const changed = nextLanguage !== currentLanguage;
    currentLanguage = nextLanguage;
    if (options.persist !== false) {
      try { localStorage.setItem(STORAGE_KEY, currentLanguage); } catch (_) { /* storage is optional */ }
    }
    translateDocument();
    if (changed) {
      document.dispatchEvent(new CustomEvent('speakerlab:languagechange', {
        detail: { language: currentLanguage },
      }));
    }
  }

  window.t = t;
  window.setLanguage = setLanguage;
  window.getLanguage = () => currentLanguage;
  window.formatNumber = (value, options = {}) => new Intl.NumberFormat(
    currentLanguage === 'en' ? 'en-US' : 'es-CL',
    options,
  ).format(value);

  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-language]').forEach(button => {
      button.addEventListener('click', () => setLanguage(button.dataset.language));
    });
    setLanguage(currentLanguage, { persist: false });
  });
}());
