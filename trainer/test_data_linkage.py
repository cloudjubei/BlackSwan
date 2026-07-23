from trainer import data_catalog, data_linkage


def test_every_edge_points_at_a_catalogued_proxy_series():
    for edge in data_linkage.edges():
        assert data_catalog.instrument(edge["proxy"]) is not None, edge["proxy"]
        assert edge["proxySource"] is not None
        assert edge["proxyBarCloseTz"]


def test_every_edge_asset_is_catalogued_in_its_class():
    for edge in data_linkage.edges():
        assert data_catalog.instrument(edge["asset"], edge["assetClass"]) is not None, edge["asset"]


def test_edges_carry_the_required_fields():
    for edge in data_linkage.edges():
        for key in ("asset", "assetClass", "proxy", "edgeType", "rationale"):
            assert edge[key], (edge, key)


def test_edges_for_filters_by_asset():
    jpm = data_linkage.edges_for("JPM")
    assert jpm and all(e["asset"] == "JPM" for e in jpm)
    assert jpm[0]["proxy"] == "T10Y2Y"


def test_edges_for_scopes_by_class():
    # JPM is a stock, not crypto — a class-scoped query for the wrong class finds nothing.
    assert data_linkage.edges_for("JPM", "crypto") == []
    assert data_linkage.edges_for("JPM", "stocks")


def test_linkage_wraps_the_edges():
    assert data_linkage.linkage()["edges"] == data_linkage.edges()


def test_gold_real_rate_edge_is_present():
    gold = data_linkage.edges_for("GOLD")
    assert any(e["proxy"] == "DFII10" and e["edgeType"] == "real-rate" for e in gold)


def test_fx_terms_of_trade_edges_present():
    aud = data_linkage.edges_for("AUDUSD")
    cad = data_linkage.edges_for("USDCAD")
    assert any(e["proxy"] == "COPPER" for e in aud)
    assert any(e["proxy"] == "WTI" for e in cad)


def test_semiconductor_and_lithium_etf_proxy_edges_present():
    assert any(e["proxy"] == "SOXX" for e in data_linkage.edges_for("NVDA"))
    assert any(e["proxy"] == "SOXX" for e in data_linkage.edges_for("AVGO"))
    assert any(e["proxy"] == "SMH" for e in data_linkage.edges_for("AAPL", "stocks"))
    assert any(e["proxy"] == "LIT" and e["edgeType"] == "input-cost" for e in data_linkage.edges_for("TSLA"))


def test_etf_assets_drive_their_own_edges():
    # The ETFs are catalogued assets, so edges can also point FROM them.
    assert any(e["proxy"] == "GOLD" for e in data_linkage.edges_for("GDX", "etfs"))
    assert any(e["proxy"] == "WTI" for e in data_linkage.edges_for("XLE", "etfs"))
    assert any(e["proxy"] == "T10Y2Y" for e in data_linkage.edges_for("XLF", "etfs"))
