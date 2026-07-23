import scripts.backfill_fundamentals as bf


# --- cik_for_ticker ---


def test_cik_for_ticker_zero_pads_to_10_digits():
    tickers = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    assert bf.cik_for_ticker(tickers, "AAPL") == "0000320193"


def test_cik_for_ticker_is_case_insensitive():
    tickers = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    assert bf.cik_for_ticker(tickers, "aapl") == "0000320193"


def test_cik_for_ticker_unknown_is_none():
    assert bf.cik_for_ticker({"0": {"cik_str": 1, "ticker": "AAPL"}}, "NOPE") is None


# --- extract_observations: point-in-time, stamped at the FILING date ---


def _facts(concept, rows, taxonomy="us-gaap", unit="USD"):
    return {"facts": {taxonomy: {concept: {"units": {unit: rows}}}}}


def test_extract_observations_stamps_release_date_at_the_filing_not_the_period_end():
    facts = _facts("NetIncomeLoss", [
        {"end": "2023-09-30", "val": 96995000000, "filed": "2023-11-03", "form": "10-K"},
    ])
    obs = bf.extract_observations(facts, ["NetIncomeLoss"])
    assert obs == [
        {
            "concept": "NetIncomeLoss",
            "unit": "USD",
            "refPeriod": "2023-09-30",
            "releaseDate": "2023-11-03",
            "value": 96995000000,
            "form": "10-K",
        }
    ]


def test_extract_observations_dedupes_exact_repeats():
    row = {"end": "2023-09-30", "val": 1, "filed": "2023-11-03", "form": "10-K"}
    obs = bf.extract_observations(_facts("Revenues", [row, dict(row)]), ["Revenues"])
    assert len(obs) == 1


def test_extract_observations_keeps_restatements_as_distinct_dated_rows():
    facts = _facts("Revenues", [
        {"end": "2023-09-30", "val": 100, "filed": "2023-11-03", "form": "10-K"},
        {"end": "2023-09-30", "val": 105, "filed": "2024-02-01", "form": "10-K/A"},
    ])
    obs = bf.extract_observations(facts, ["Revenues"])
    assert len(obs) == 2
    assert {o["value"] for o in obs} == {100, 105}


def test_extract_observations_skips_missing_concept():
    assert bf.extract_observations({"facts": {"us-gaap": {}}}, ["NetIncomeLoss"]) == []


def test_extract_observations_skips_rows_without_end_or_filed():
    facts = _facts("Revenues", [{"val": 1, "form": "10-K"}, {"end": "2023-09-30", "val": 2, "form": "10-K"}])
    assert bf.extract_observations(facts, ["Revenues"]) == []


def test_extract_observations_handles_malformed_facts():
    assert bf.extract_observations({}, ["Revenues"]) == []
    assert bf.extract_observations(None, ["Revenues"]) == []


def test_extract_observations_distinguishes_quarterly_from_cumulative_by_duration():
    # A flow concept returns co-terminating durations (a 3-month quarter AND the 12-month year both ending
    # 2020-12-31, filed the same day). They must NOT collapse into indistinguishable rows — each carries its
    # period start + duration so a consumer can pick one consistent bucket.
    facts = _facts("NetIncomeLoss", [
        {"start": "2020-10-01", "end": "2020-12-31", "val": 28800, "filed": "2021-01-28", "form": "10-K"},
        {"start": "2020-01-01", "end": "2020-12-31", "val": 57400, "filed": "2021-01-28", "form": "10-K"},
    ])
    obs = bf.extract_observations(facts, ["NetIncomeLoss"])
    assert len(obs) == 2
    by_duration = {o["durationDays"]: o for o in obs}
    assert set(by_duration) == {91, 365}
    assert by_duration[91]["value"] == 28800
    assert by_duration[91]["periodStart"] == "2020-10-01"
    assert by_duration[365]["value"] == 57400


def test_extract_observations_same_day_correction_keeps_latest_accession():
    # Two filings for the SAME period+duration on the SAME day (a same-day amendment) collapse to the later
    # accession — deterministic + revision-aware, never input-order-dependent.
    facts = _facts("Revenues", [
        {"start": "2023-07-01", "end": "2023-09-30", "val": 100, "filed": "2023-11-03", "form": "10-Q", "accn": "0000-23-000001"},
        {"start": "2023-07-01", "end": "2023-09-30", "val": 105, "filed": "2023-11-03", "form": "10-Q/A", "accn": "0000-23-000009"},
    ])
    obs = bf.extract_observations(facts, ["Revenues"])
    assert len(obs) == 1
    assert obs[0]["value"] == 105
