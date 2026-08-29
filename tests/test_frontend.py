import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend" / "index.html").read_text()
JS = (ROOT / "frontend" / "js" / "app.js").read_text()
I18N = (ROOT / "frontend" / "js" / "i18n.js").read_text()
LOCALE_ES = (ROOT / "frontend" / "js" / "locales" / "es.js").read_text()
LOCALE_EN = (ROOT / "frontend" / "js" / "locales" / "en.js").read_text()
ENCYCLOPEDIA_LOCALE = (ROOT / "frontend" / "js" / "locales" / "encyclopedia.js").read_text()
CSS = (ROOT / "frontend" / "css" / "app.css").read_text()
PDF_GENERATOR = (ROOT / "api" / "pdf_generator.py").read_text()


class FrontendExperienceTests(unittest.TestCase):
    def test_assets_and_notification_region_are_present(self):
        self.assertIn('href="/css/app.css"', HTML)
        self.assertIn('src="/js/app.js"', HTML)
        self.assertIn('id="toast-region"', HTML)
        self.assertIn('aria-live="polite"', HTML)
        self.assertIn('/assets/brand/speakerlab-pro-logo.svg', HTML)
        self.assertIn('/assets/brand/speakerlab-pro-mark.svg', HTML)
        self.assertIn('/assets/brand/favicon.svg', HTML)
        self.assertIn('<h1 data-i18n="calculator_title">Calculadora de Cajas Acústicas</h1>', HTML)

    def test_bilingual_interface_foundations_are_present(self):
        self.assertIn('src="/js/locales/es.js"', HTML)
        self.assertIn('src="/js/locales/en.js"', HTML)
        self.assertIn('src="/js/i18n.js"', HTML)
        self.assertIn('data-language="es"', HTML)
        self.assertIn('data-language="en"', HTML)
        self.assertIn("speakerlab-language", I18N)
        self.assertIn("navigator.language", I18N)
        self.assertIn("document.documentElement.lang", I18N)
        self.assertIn("Intl.NumberFormat", I18N)
        self.assertIn("nav_calculator", LOCALE_ES)
        self.assertIn("nav_calculator", LOCALE_EN)
        self.assertIn("function renderCalculationResults", JS)

    def test_second_bilingual_stage_covers_local_tools_and_pdf(self):
        for key in (
            "database_title", "local_projects", "speaker_add", "project_saved",
            "simulation_success", "cut_piece", "pdf_success",
        ):
            self.assertIn(key, LOCALE_ES)
            self.assertIn(key, LOCALE_EN)
        self.assertIn("language: getLanguage()", JS)
        self.assertIn("body: JSON.stringify({ driver: payload.driver, language: getLanguage() })", JS)
        self.assertIn('"language": d.get("language", "es")', PDF_GENERATOR)
        self.assertIn("PDF_EN", PDF_GENERATOR)
        self.assertIn('VALID_BOX_TYPES = {"reflex", "closed"}', PDF_GENERATOR)

    def test_encyclopedia_has_a_dedicated_technical_translation_layer(self):
        self.assertIn('src="/js/locales/encyclopedia.js"', HTML)
        self.assertIn("window.translateEncyclopedia", ENCYCLOPEDIA_LOCALE)
        self.assertIn("acoustic compliance", ENCYCLOPEDIA_LOCALE)
        self.assertIn("Port air velocity", ENCYCLOPEDIA_LOCALE)
        self.assertIn("Transmission Line", ENCYCLOPEDIA_LOCALE)
        for chapter in range(1, 13):
            self.assertIn(f"enc_chapter_{chapter}", LOCALE_ES)
            self.assertIn(f"enc_chapter_{chapter}", LOCALE_EN)

    def test_calculator_controls_have_associated_labels(self):
        self.assertIn('for="spk-search"', HTML)
        for field_id in ("fs", "vas", "qts", "qes", "qms", "xmax", "sd", "boxType", "material"):
            self.assertIn(f'for="{field_id}"', HTML)

    def test_input_parameter_descriptions_are_localized(self):
        for key in (
            "example_fs", "example_vas", "hint_fs", "hint_vas",
            "diameter_inches_short", "hint_mms", "hint_bl", "hint_re", "hint_le",
        ):
            self.assertIn(f'data-i18n="{key}"', HTML) if not key.startswith("example_") else self.assertIn(f'data-i18n-placeholder="{key}"', HTML)
            self.assertIn(key, LOCALE_ES)
            self.assertIn(key, LOCALE_EN)

    def test_blocking_browser_alerts_are_not_used(self):
        self.assertNotIn("alert(", JS)
        self.assertIn("function notify(", JS)

    def test_long_running_actions_have_busy_controls(self):
        self.assertIn('id="btn-calculate"', HTML)
        self.assertGreaterEqual(HTML.count("data-pdf-download"), 2)
        self.assertIn("setButtonBusy(calculateButton, true", JS)
        self.assertIn("setButtonBusy(button, true, t('pdf_generating'))", JS)

    def test_inline_field_validation_is_accessible(self):
        self.assertIn("function validateCalculatorForm()", JS)
        self.assertIn("aria-invalid", JS)
        self.assertIn("aria-describedby", JS)
        self.assertIn("field-error", JS)

    def test_local_project_manager_supports_required_actions(self):
        self.assertIn('id="projects-modal"', HTML)
        self.assertIn("speakerlab.projects.v1", JS)
        for function in (
            "saveLocalProject",
            "loadLocalProject",
            "duplicateLocalProject",
            "exportLocalProject",
            "exportAllProjects",
            "updateLocalProject",
            "deleteLocalProject",
            "importProjectsFile",
        ):
            self.assertIn(f"function {function}", JS)
        self.assertNotIn("/api/auth", JS)

    def test_keyboard_accessibility_foundations_are_present(self):
        self.assertIn('class="skip-link"', HTML)
        self.assertIn('role="tablist"', HTML)
        self.assertIn("function trapFocus(", JS)
        self.assertIn("function initKeyboardAccessibility(", JS)
        self.assertIn(":focus-visible", CSS)

    def test_event_handlers_are_not_embedded_in_html(self):
        for attribute in ("onclick=", "oninput=", "onchange="):
            self.assertNotIn(attribute, HTML)
        self.assertIn("function initDeclarativeEvents()", JS)

    def test_presentation_is_not_embedded_in_html_or_javascript(self):
        self.assertNotIn("style=", HTML)
        self.assertNotIn(".style.", JS)
        self.assertIn('href="/css/utilities.css"', HTML)

    def test_custom_speaker_database_supports_local_management(self):
        self.assertIn('id="speaker-modal"', HTML)
        for function in (
            "saveCustomSpeaker", "deleteCustomSpeaker", "exportCustomSpeakers",
            "importCustomSpeakers", "openSpeakerModal",
        ):
            self.assertIn(f"function {function}", JS)

    def test_calculator_draft_supports_autosave_and_recovery(self):
        self.assertIn('id="draft-status"', HTML)
        self.assertIn("speakerlab.calculator-draft.v1", JS)
        for function in ("saveCalculatorDraft", "restoreCalculatorDraft", "clearCalculatorDraft"):
            self.assertIn(f"function {function}", JS)


if __name__ == "__main__":
    unittest.main()
