import json

import pytest

from cfbsim.client import BudgetExceededError, CFBDClient, CFBDError, MissingKeyError


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.payload = payload if payload is not None else [{"id": 1}]
        self.text = json.dumps(self.payload)

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response=None):
        self.response = response or FakeResponse()
        self.calls = []

    def get(self, url, params, headers, timeout):
        self.calls.append((url, params, headers))
        return self.response


def make_client(tmp_path, session=None, **kwargs):
    kwargs.setdefault("api_key", "secret-key")
    return CFBDClient(cache_dir=tmp_path, session=session or FakeSession(), month=lambda: "2026-11", **kwargs)


def test_second_request_comes_from_cache_and_costs_nothing(tmp_path):
    session = FakeSession()
    client = make_client(tmp_path, session)
    assert client.get("games", year=2024) == client.get("games", year=2024)
    assert len(session.calls) == 1
    assert client.calls_this_month() == 1


def test_refresh_bypasses_cache(tmp_path):
    session = FakeSession()
    client = make_client(tmp_path, session)
    client.get("games", year=2024)
    client.get("games", year=2024, refresh=True)
    assert len(session.calls) == 2


def test_cached_data_is_readable_without_a_key(tmp_path):
    make_client(tmp_path).get("games", year=2024)
    assert make_client(tmp_path, api_key="").get("games", year=2024) == [{"id": 1}]


def test_missing_key_raises_instead_of_returning_empty_data(tmp_path):
    session = FakeSession()
    with pytest.raises(MissingKeyError):
        make_client(tmp_path, session, api_key="").get("games", year=2024)
    assert session.calls == []


def test_budget_blocks_the_request_before_it_is_sent(tmp_path):
    session = FakeSession()
    client = make_client(tmp_path, session, monthly_budget=1)
    client.get("games", year=2024)
    with pytest.raises(BudgetExceededError):
        client.get("games", year=2023)
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    "status, error", [(401, MissingKeyError), (429, BudgetExceededError), (500, CFBDError)]
)
def test_failed_responses_raise_and_are_not_cached(tmp_path, status, error):
    client = make_client(tmp_path, FakeSession(FakeResponse(status, {"message": "no"})))
    with pytest.raises(error):
        client.get("games", year=2024)
    assert not client.is_cached("games", year=2024)


def test_key_is_sent_as_header_and_never_written_to_disk(tmp_path):
    session = FakeSession()
    make_client(tmp_path, session).get("games", year=2024)
    assert session.calls[0][2]["Authorization"] == "Bearer secret-key"
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert "secret-key" not in path.read_text() and "secret-key" not in str(path)
