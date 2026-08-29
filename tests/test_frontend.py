import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend" / "index.html").read_text()
JS = (ROOT / "frontend" / "js" / "app.js").read_text()
CSS = (ROOT / "frontend" / "css" / "app.css").read_text()


class FrontendExperienceTests(unittest.TestCase):
    def test_assets_and_notification_region_are_present(self):
        self.assertIn('href="/css/app.css"', HTML)
        self.assertIn('src="/js/app.js"', HTML)
        self.assertIn('id="toast-region"', HTML)
        self.assertIn('aria-live="polite"', HTML)
        self.assertIn('/assets/brand/speakerlab-pro-logo.svg', HTML)
        self.assertIn('/assets/brand/speakerlab-pro-mark.svg', HTML)
        self.assertIn('/assets/brand/favicon.svg', HTML)
        self.assertIn('<h1>Calculadora de Cajas Acústicas</h1>', HTML)

    def test_calculator_controls_have_associated_labels(self):
        self.assertIn('for="spk-search"', HTML)
        for field_id in ("fs", "vas", "qts", "qes", "qms", "xmax", "sd", "boxType", "material"):
            self.assertIn(f'for="{field_id}"', HTML)

    def test_blocking_browser_alerts_are_not_used(self):
        self.assertNotIn("alert(", JS)
        self.assertIn("function notify(", JS)

    def test_long_running_actions_have_busy_controls(self):
        self.assertIn('id="btn-calculate"', HTML)
        self.assertGreaterEqual(HTML.count("data-pdf-download"), 2)
        self.assertIn("setButtonBusy(calculateButton, true", JS)
        self.assertIn("setButtonBusy(button, true, 'Generando PDF…')", JS)

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
