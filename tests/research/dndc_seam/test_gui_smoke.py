"""Playwright GUI smoke for DNDCForms — assumes `npm run dev` is running."""
import os
import pytest

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    pytest.skip("playwright not installed; pip install playwright",
                allow_module_level=True)


URL = os.environ.get("HYDRUS_DEV_URL", "http://localhost:1420")


def test_dndc_forms_renders_all_11_sections():
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            page = b.new_page()
            page.goto(f"{URL}/#/research/dndc")    # exact route depends on App.vue routing
            page.wait_for_load_state("networkidle")
            summaries = page.locator("summary").all_inner_texts()
            b.close()
            # Expect 11 collapsible sections
            assert len(summaries) >= 11, f"got {len(summaries)} sections: {summaries}"
            for keyword in ("Atmospheric", "ET partition", "Root growth", "Feddes",
                            "Fertilizer", "Irrigation", "N transformation",
                            "Plant N", "State exchange", "Soil temperature", "Residue"):
                assert any(keyword.lower() in s.lower() for s in summaries), \
                    f"missing section keyword: {keyword!r}"
    except Exception as e:
        if "ERR_CONNECTION_REFUSED" in str(e) or "Connection refused" in str(e):
            pytest.skip(f"dev server not running at {URL}; start with: cd desktop && npm run dev")
        raise
