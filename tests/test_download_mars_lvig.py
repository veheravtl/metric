"""Тесты возобновляемой загрузки больших MARS-LVIG ROS bag."""

from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
from types import ModuleType
from urllib.request import Request

import pytest


def _load_download_module() -> ModuleType:
    """Загружает исполняемый script как модуль без превращения scripts в пакет."""

    script_path = Path(__file__).parents[1] / "scripts/download_mars_lvig.py"
    spec = importlib.util.spec_from_file_location("download_mars_lvig", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DOWNLOAD_MODULE = _load_download_module()
download_with_resume = DOWNLOAD_MODULE.download_with_resume
parse_content_range = DOWNLOAD_MODULE.parse_content_range


class _FakeHttpResponse(io.BytesIO):
    """Минимальный HTTP-ответ для проверки Range без внешней сети."""

    def __init__(
        self,
        payload: bytes,
        *,
        status: int,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(payload)
        self.status = status
        self.headers = headers or {}


def test_parse_content_range_accepts_exact_server_range() -> None:
    """Парсер должен читать реальные числовые границы, а не буквальную строку."""

    assert parse_content_range(
        "bytes 4-9/10",
        expected_start=4,
        expected_total=10,
    ) == (4, 9, 10)


def test_download_resumes_partial_file_without_duplicating_prefix(
    tmp_path: Path,
) -> None:
    """HTTP 206 должен дописать хвост и сохранить проверяемую контрольную сумму."""

    destination = tmp_path / "flight.bag"
    partial_path = tmp_path / "flight.bag.part"
    partial_path.write_bytes(b"abcd")
    captured_request: Request | None = None

    def open_request(request: Request, *, timeout: int) -> _FakeHttpResponse:
        nonlocal captured_request
        captured_request = request
        assert timeout > 0
        return _FakeHttpResponse(
            b"efghij",
            status=206,
            headers={"Content-Range": "bytes 4-9/10"},
        )

    status, sha256 = download_with_resume(
        url="https://example.invalid/flight.bag",
        destination=destination,
        expected_size_bytes=10,
        expected_sha256=hashlib.sha256(b"abcdefghij").hexdigest(),
        open_request=open_request,
    )

    assert "загружен" in status
    assert sha256 == hashlib.sha256(b"abcdefghij").hexdigest()
    assert destination.read_bytes() == b"abcdefghij"
    assert not partial_path.exists()
    assert captured_request is not None
    assert captured_request.get_header("Range") == "bytes=4-"


def test_download_keeps_partial_if_server_ignores_range(tmp_path: Path) -> None:
    """Ответ 200 на продолжение нельзя добавлять: он повторил бы начало файла."""

    destination = tmp_path / "flight.bag"
    partial_path = tmp_path / "flight.bag.part"
    partial_path.write_bytes(b"abcd")

    def open_request(request: Request, *, timeout: int) -> _FakeHttpResponse:
        return _FakeHttpResponse(b"abcdefghij", status=200)

    with pytest.raises(RuntimeError, match="HTTP 206"):
        download_with_resume(
            url="https://example.invalid/flight.bag",
            destination=destination,
            expected_size_bytes=10,
            expected_sha256=None,
            open_request=open_request,
        )

    assert partial_path.read_bytes() == b"abcd"
    assert not destination.exists()


def test_complete_partial_with_wrong_hash_is_not_published(tmp_path: Path) -> None:
    """Один правильный размер недостаточен: повреждённый bag остаётся .part."""

    destination = tmp_path / "flight.bag"
    partial_path = tmp_path / "flight.bag.part"
    partial_path.write_bytes(b"ten-bytes!")

    with pytest.raises(RuntimeError, match="SHA-256"):
        download_with_resume(
            url="https://example.invalid/flight.bag",
            destination=destination,
            expected_size_bytes=10,
            expected_sha256="0" * 64,
        )

    assert partial_path.exists()
    assert not destination.exists()

def test_download_rejects_html_before_creating_partial_file(tmp_path: Path) -> None:
    """Страница квоты Google Drive не должна становиться ложным bag-файлом."""

    destination = tmp_path / "flight.bag"

    def open_request(request: Request, *, timeout: int) -> _FakeHttpResponse:
        return _FakeHttpResponse(
            b"<!DOCTYPE html><title>Quota exceeded</title>",
            status=200,
            headers={
                "Content-Type": "text/html; charset=utf-8",
                "Content-Length": "44",
            },
        )

    with pytest.raises(RuntimeError, match="HTML"):
        download_with_resume(
            url="https://example.invalid/flight.bag",
            destination=destination,
            expected_size_bytes=10,
            expected_sha256=None,
            open_request=open_request,
        )

    assert not destination.exists()
    assert not (tmp_path / "flight.bag.part").exists()
