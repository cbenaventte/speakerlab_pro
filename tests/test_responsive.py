import os
import threading
import unittest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
MOBILE_VIEWPORTS = (
    {"width": 320, "height": 568},
    {"width": 375, "height": 667},
    {"width": 430, "height": 932},
    {"width": 768, "height": 1024},
)


class QuietHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND), **kwargs)

    def log_message(self, _format, *args):
        pass

    def end_headers(self):
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob:; "
            "connect-src 'self'; script-src 'self'; object-src 'none'",
        )
        super().end_headers()


class ResponsiveBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

        cls.playwright = sync_playwright().start()
        launch_options = {"headless": True}
        chrome_path = os.environ.get("PLAYWRIGHT_CHROME_PATH")
        if chrome_path:
            launch_options["executable_path"] = chrome_path
        cls.browser = cls.playwright.chromium.launch(**launch_options)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(timeout=2)

    def open_page(self, viewport, **context_options):
        context = self.browser.new_context(viewport=viewport, locale="es-CL", **context_options)
        self.addCleanup(context.close)
        page = context.new_page()
        page.goto(self.base_url, wait_until="domcontentloaded")
        return page

    def test_language_detection_switch_and_persistence(self):
        context = self.browser.new_context(viewport={"width": 1440, "height": 900}, locale="en-US")
        self.addCleanup(context.close)
        page = context.new_page()
        page.goto(self.base_url, wait_until="domcontentloaded")

        self.assertEqual(page.locator("html").get_attribute("lang"), "en")
        self.assertEqual(page.locator('[data-i18n="nav_calculator"]').inner_text(), "Calculator")
        self.assertEqual(page.locator("h1").inner_text(), "Speaker Enclosure Calculator")
        self.assertEqual(page.locator('[data-language="en"]').get_attribute("aria-pressed"), "true")
        self.assertEqual(page.locator("#fs").get_attribute("placeholder"), "e.g. 28")
        self.assertEqual(page.locator("#vas").get_attribute("placeholder"), "e.g. 75")
        self.assertEqual(page.locator("#fs + .unit-hint").inner_text(), "Hz — free-air resonance")
        self.assertEqual(page.locator("#vas + .unit-hint").inner_text(), "Liters — equivalent volume")
        self.assertEqual(page.locator('label[for="inches"]').inner_text(), "Diameter (in)")
        self.assertEqual(page.locator("#mms + .unit-hint").inner_text(), "g — moving mass")
        self.assertEqual(page.locator("#bl + .unit-hint").inner_text(), "T·m — force factor")
        self.assertEqual(page.locator("#re + .unit-hint").inner_text(), "Ω — DC resistance")
        self.assertEqual(page.locator("#le + .unit-hint").inner_text(), "mH — voice-coil inductance")
        page.locator('[data-tip="Fs"]').click()
        self.assertIn("Free-Air Resonance", page.locator("#enc-tooltip").inner_text())

        page.locator("#tb-db").click()
        self.assertEqual(page.locator("#view-db h2").inner_text(), "Speaker Driver Database")
        self.assertEqual(page.get_by_text("Add driver", exact=True).count(), 1)
        page.locator('[data-action="openProjectsModal"]').click()
        self.assertEqual(page.locator("#projects-title").inner_text(), "My Local Projects")
        page.locator('[data-action="closeProjectsModal"]').click()

        page.locator('[data-language="es"]').click()
        self.assertEqual(page.locator("html").get_attribute("lang"), "es")
        self.assertEqual(page.locator("h1").inner_text(), "Calculadora de Cajas Acústicas")
        page.reload(wait_until="domcontentloaded")
        self.assertEqual(page.locator("html").get_attribute("lang"), "es")
        self.assertEqual(page.evaluate("localStorage.getItem('speakerlab-language')"), "es")

    def test_all_encyclopedia_chapters_switch_to_technical_english(self):
        page = self.open_page({"width": 1440, "height": 900})
        page.locator('[data-language="en"]').click()
        page.locator("#tb-enc").click()
        expected_headings = (
            "Anatomy of a Dynamic Driver", "Electro-Mechanical-Acoustic Model",
            "Thiele-Small Parameters", "Sealed Enclosure Design",
            "Bass Reflex (Vented) Enclosures", "Advanced Enclosure Designs",
            "Edge Diffraction and Baffle Step", "Materials and Proportions",
            "Internal Bracing and Damping", "Bass Reflex Port Mathematics",
            "Classic Alignments", "Measurement and Testing",
        )
        for chapter, heading in enumerate(expected_headings, start=1):
            page.locator(f"#enc-sidebar .nav-item:nth-of-type({chapter})").click()
            article = page.locator("#view-enc .enc-article:visible")
            self.assertIn(heading, article.locator("h2").inner_text())

        page.locator('[data-language="es"]').click()
        self.assertIn("Medición y Pruebas", page.locator("#sec-pruebas h2").inner_text())

    def assert_no_document_overflow(self, page, viewport_width):
        dimensions = page.evaluate(
            """() => ({
                body: document.body.scrollWidth,
                document: document.documentElement.scrollWidth
            })"""
        )
        self.assertLessEqual(dimensions["body"], viewport_width + 1)
        self.assertLessEqual(dimensions["document"], viewport_width + 1)

    def test_mobile_header_uses_two_rows_without_overflow(self):
        for viewport in MOBILE_VIEWPORTS[:3]:
            with self.subTest(viewport=viewport):
                page = self.open_page(viewport)
                logo = page.locator(".tb-logo").bounding_box()
                self.assertGreater(page.locator(".tb-brand img").evaluate("img => img.naturalWidth"), 0)
                self.assertTrue(page.locator(".tb-brand img").evaluate("img => img.currentSrc.endsWith('/assets/brand/speakerlab-pro-mark.svg')"))
                nav = page.locator(".tb-nav").bounding_box()
                self.assertIsNotNone(logo)
                self.assertIsNotNone(nav)
                self.assertGreaterEqual(nav["y"], logo["y"] + logo["height"] - 1)
                self.assertEqual(page.locator(".tb-action-label").first.evaluate("el => getComputedStyle(el).display"), "none")
                self.assert_no_document_overflow(page, viewport["width"])

    def test_small_mobile_controls_and_grids_are_touch_friendly(self):
        page = self.open_page({"width": 430, "height": 932})
        controls = page.locator("#view-calc .form-field input, #view-calc .form-field select, #spk-search, #btn-calculate")
        sizes = controls.evaluate_all("elements => elements.filter(el => el.getClientRects().length).map(el => ({id: el.id, height: el.getBoundingClientRect().height}))")
        self.assertTrue(sizes)
        undersized = [control for control in sizes if control["height"] < 44]
        self.assertEqual(undersized, [])

        page.evaluate("document.getElementById('results-content').hidden = false")
        cards = page.locator(".hero-cards .hero-card")
        first = cards.nth(0).bounding_box()
        second = cards.nth(1).bounding_box()
        self.assertGreaterEqual(second["y"], first["y"] + first["height"] - 1)

        page.locator("#tb-db").click()
        page.get_by_text("Añadir altavoz", exact=True).click()
        columns = page.locator(".speaker-form-grid").evaluate("el => getComputedStyle(el).gridTemplateColumns.split(' ').length")
        self.assertEqual(columns, 1)

    def test_calculator_fields_expose_accessible_names(self):
        page = self.open_page({"width": 1440, "height": 900})
        unlabeled = page.locator("#view-calc input:not([type='hidden']), #view-calc select").evaluate_all(
            "elements => elements.filter(el => !el.labels || el.labels.length === 0).map(el => el.id)"
        )
        self.assertEqual(unlabeled, [])

    def test_mobile_encyclopedia_index_opens_and_closes(self):
        for viewport in MOBILE_VIEWPORTS[:3]:
            with self.subTest(viewport=viewport):
                page = self.open_page(viewport)
                page.locator("#tb-enc").click()
                toggle = page.locator("#enc-menu-toggle")
                self.assertTrue(toggle.is_visible())
                self.assertEqual(toggle.get_attribute("aria-expanded"), "false")

                toggle.click()
                page.locator("#enc-sidebar").wait_for(state="visible")
                self.assertTrue(page.locator("#enc-menu-overlay").is_visible())
                self.assertEqual(toggle.get_attribute("aria-expanded"), "true")

                page.locator("#enc-sidebar .nav-item").nth(1).click()
                page.locator("#enc-sidebar").wait_for(state="hidden")
                self.assertEqual(toggle.get_attribute("aria-expanded"), "false")
                self.assert_no_document_overflow(page, viewport["width"])

    def test_projects_dialog_fits_mobile_viewports(self):
        for viewport in MOBILE_VIEWPORTS:
            with self.subTest(viewport=viewport):
                page = self.open_page(viewport)
                page.locator(".btn-projects").click()
                dialog = page.locator("#projects-modal .projects-dialog").bounding_box()
                self.assertIsNotNone(dialog)
                self.assertLessEqual(dialog["width"], viewport["width"])
                self.assertLessEqual(dialog["height"], viewport["height"])
                page.locator("#projects-modal .projects-close").click()

    def test_desktop_keeps_permanent_encyclopedia_sidebar(self):
        page = self.open_page({"width": 1440, "height": 900})
        page.locator("#tb-enc").click()
        self.assertTrue(page.locator("#enc-sidebar").is_visible())
        self.assertFalse(page.locator("#enc-menu-toggle").is_visible())
        sidebar = page.locator("#enc-sidebar").bounding_box()
        self.assertGreaterEqual(sidebar["width"], 260)
        self.assert_no_document_overflow(page, 1440)

    def test_keyboard_navigation_and_modal_focus(self):
        page = self.open_page({"width": 1440, "height": 900})

        page.keyboard.press("Tab")
        self.assertEqual(page.locator(":focus").get_attribute("class"), "skip-link")

        page.locator("#tb-enc").click()
        page.evaluate("showEnc('sec-ts')")
        card = page.locator(".ts-card").first
        card.focus()
        page.keyboard.press("Enter")
        self.assertEqual(card.get_attribute("aria-expanded"), "true")

        page.evaluate("showEnc('sec-avanzadas')")
        first_tab = page.locator("#sec-avanzadas .tab-btn").first
        first_tab.focus()
        page.keyboard.press("ArrowRight")
        second_tab = page.locator("#sec-avanzadas .tab-btn").nth(1)
        self.assertTrue(second_tab.evaluate("el => el === document.activeElement"))
        self.assertEqual(second_tab.get_attribute("aria-selected"), "true")

        trigger = page.locator(".btn-projects")
        trigger.focus()
        trigger.click()
        self.assertTrue(page.locator("#project-name").evaluate("el => el === document.activeElement"))
        page.locator("#projects-modal .projects-close").click()
        self.assertTrue(trigger.evaluate("el => el === document.activeElement"))

    def test_custom_speaker_can_be_created_edited_and_deleted_locally(self):
        page = self.open_page({"width": 1440, "height": 900})
        page.locator("#tb-db").click()
        page.get_by_text("Añadir altavoz", exact=True).click()
        values = {
            "custom-brand": "Prueba Local", "custom-model": "Woofer 10",
            "custom-inches": "10", "custom-fs": "28", "custom-vas": "65",
            "custom-qts": "0.36", "custom-sd": "345",
        }
        for field, value in values.items():
            page.locator(f"#{field}").fill(value)
        page.get_by_text("Guardar altavoz", exact=True).click()
        self.assertTrue(page.get_by_text("Prueba Local", exact=False).is_visible())
        self.assertEqual(page.evaluate("JSON.parse(localStorage.getItem('speakerlab.custom-speakers.v1')).length"), 1)

        page.get_by_text("Editar", exact=True).click()
        page.locator("#custom-model").fill("Woofer 10 MkII")
        page.get_by_text("Guardar altavoz", exact=True).click()
        self.assertTrue(page.get_by_text("Woofer 10 MkII", exact=True).is_visible())

        page.get_by_text("Eliminar", exact=True).click()
        self.assertEqual(page.evaluate("JSON.parse(localStorage.getItem('speakerlab.custom-speakers.v1')).length"), 0)
        self.assertFalse(page.get_by_text("Woofer 10 MkII", exact=True).is_visible())

    def test_calculator_draft_is_restored_after_reload(self):
        page = self.open_page({"width": 1440, "height": 900})
        page.locator("#fs").fill("31.5")
        page.locator("#vas").fill("72")
        page.locator("#qts").fill("0.39")
        page.wait_for_timeout(750)
        self.assertEqual(
            page.evaluate("JSON.parse(localStorage.getItem('speakerlab.calculator-draft.v1')).form.fs"),
            "31.5",
        )

        page.reload(wait_until="domcontentloaded")
        self.assertEqual(page.locator("#fs").input_value(), "31.5")
        self.assertEqual(page.locator("#vas").input_value(), "72")
        self.assertTrue(page.locator("#draft-status").is_visible())
        self.assertIn("Borrador recuperado", page.locator("#draft-status-text").inner_text())

        page.get_by_text("Eliminar borrador", exact=True).click()
        self.assertIsNone(page.evaluate("localStorage.getItem('speakerlab.calculator-draft.v1')"))
        self.assertEqual(page.locator("#fs").input_value(), "31.5")


if __name__ == "__main__":
    unittest.main()
