import unittest
from types import SimpleNamespace

from app.utils import validate_uploaded_image


class AppUtilsTests(unittest.TestCase):
    def test_validate_uploaded_image_accepts_supported_png(self) -> None:
        upload = SimpleNamespace(type="image/png", size=1024)
        is_valid, error = validate_uploaded_image(upload)
        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_validate_uploaded_image_rejects_unsupported_type(self) -> None:
        upload = SimpleNamespace(type="application/pdf", size=1024)
        is_valid, error = validate_uploaded_image(upload)
        self.assertFalse(is_valid)
        self.assertIn("Supported", error)

    def test_validate_uploaded_image_rejects_large_files(self) -> None:
        upload = SimpleNamespace(type="image/jpeg", size=20 * 1024 * 1024)
        is_valid, error = validate_uploaded_image(upload)
        self.assertFalse(is_valid)
        self.assertIn("larger", error)


if __name__ == "__main__":
    unittest.main()
