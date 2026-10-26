"""Offline behavior checks for CLI dataset ingestion and campaign receipts."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from unittest.mock import Mock

import huggingface_hub
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hivemind import config
from hivemind.cli import demo_hf_domain_swarm as demo
from hivemind.cli import multi_moe_hf_campaign as campaign
from hivemind.data.hf_real_trainer import CURATED_SEASON3_DOMAIN_CORPORA


@pytest.fixture(autouse=True)
def offline_downloads(monkeypatch):
    """Every network boundary must be explicitly supplied by the test."""
    hub = Mock(side_effect=OSError("offline fixture: no hub download"))
    http = Mock(side_effect=urllib.error.URLError("offline fixture: no HTTP"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub)
    monkeypatch.setattr(urllib.request, "urlopen", http)
    return hub, http


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    """Run cache setup itself, restoring every environment variable it modifies."""
    monkeypatch.setattr(config, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(config, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(demo, "CACHE_ROOT", tmp_path / "cache")
    if importlib.util.find_spec("dotenv") is not None:
        import dotenv

        monkeypatch.setattr(dotenv, "load_dotenv", Mock(return_value=False))
    for key in (
        "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME",
        "UV_CACHE_DIR", "TMP", "TEMP", "CUBLAS_WORKSPACE_CONFIG",
        "HF_HUB_DISABLE_SYMLINKS_WARNING",
    ):
        # Setting even absent keys records their removal for fixture teardown.
        monkeypatch.setenv(key, os.environ.get(key, ""))
    monkeypatch.setenv("HF_TOKEN", "test-offline-token")
    return tmp_path / "cache" / "swarm_store" / "hf_subsets"


def _parquet(tmp_path, rows):
    path = tmp_path / "observations.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _kepler(**changes):
    return {
        "period_days": 12.3456, "planet_radius_rearth": 1.6,
        "transit_depth_ppm": 123.45, "model_snr": 9.85,
        "insolation_earth": 1.234, "stellar_teff_k": 5772.0,
        "stellar_logg": 4.44, "stellar_radius_rsun": 1.01,
        "magnitude_kep": 12.25, "target": " confirmed ", **changes,
    }


def test_kepler_parquet_classifies_boundaries_skips_nulls_and_limits(tmp_path, offline_downloads):
    radii = [1.6, 1.61, 2.5, 2.51, 6.0, 6.01, 22.0]
    rows = [_kepler(period_days=None), _kepler(planet_radius_rearth=None), _kepler(stellar_teff_k=None)]
    rows += [_kepler(planet_radius_rearth=r) for r in radii]
    rows += [_kepler(planet_radius_rearth=1.0, target=" false positive ")]
    rows += [_kepler(target=None, transit_depth_ppm=None, model_snr=None,
                     insolation_earth=None, stellar_logg=None, stellar_radius_rsun=None,
                     magnitude_kep=None), _kepler(planet_radius_rearth=99.0)]
    offline_downloads[0].side_effect = None
    offline_downloads[0].return_value = str(_parquet(tmp_path, rows))

    result = demo._format_astrophysics_hf_dataset_subset("MZRK36/EXOPLANETS-UNIFIED-DATASET", 9, "fixture-token")

    expected = ["Rocky Terrestrial / Earth-Size World", "Super-Earth Exoplanet",
                "Super-Earth Exoplanet", "Sub-Neptune / Mini-Neptune",
                "Sub-Neptune / Mini-Neptune", "Jovian Gas Giant / Hot Jupiter",
                "Eclipsing Binary / False Positive Transit", "Eclipsing Binary / False Positive Transit",
                "Rocky Terrestrial / Earth-Size World"]
    assert len(result) == 9
    assert [text.split("Planetary Class: ")[1] for text in result] == expected
    assert "Orbital Period: 12.346 days | Planet Radius: 1.60 R_earth" in result[0]
    assert "Transit Depth: 123.5 ppm (SNR=9.8) | Insolation: 1.23 S_earth" in result[0]
    assert "Disposition: CONFIRMED" in result[0]
    assert "Transit Depth: N/A (SNR=N/A) | Insolation: N/A" in result[-1]
    assert "logg=4.40, R*=1.00 R_sun, KepMag=13.50" in result[-1]
    assert "Disposition: CANDIDATE" in result[-1]
    assert demo.describe_data_provenance(result)["curated_fallback_rows"] == 0
    offline_downloads[0].assert_called_once_with(
        repo_id="mzrk36/exoplanets-unified-dataset", filename="data/train-00000-of-00001.parquet",
        repo_type="dataset", token="fixture-token",
    )
    offline_downloads[1].assert_not_called()


def _nasa(**changes):
    return {
        "pl_name": "Fixture b", "hostname": "Fixture star", "discoverymethod": "Transit",
        "disc_facility": "Fixture telescope", "pl_orbper": 50.0, "pl_rade": 1.0,
        "pl_bmasse": 2.0, "pl_eqt": 280.0, "pl_orbsmax": 1.0,
        "st_teff": 5800.0, "sy_dist": 11.25, **changes,
    }


def test_nasa_parquet_taxonomy_thermal_regimes_and_default_values(tmp_path, offline_downloads):
    rows = [_nasa(pl_orbper=None, pl_rade=None, pl_bmasse=None)]
    rows += [_nasa(pl_rade=radius) for radius in (1.49, 1.5, 2.5, 6.0)]
    rows += [
        _nasa(pl_eqt=500.0, pl_orbsmax=None, pl_orbper=10.0),
        _nasa(pl_eqt=100.0, pl_orbsmax=3.0, pl_orbper=900.0),
        _nasa(pl_eqt=500.0, pl_orbsmax=0.65),
        _nasa(pl_name=None, hostname=None, discoverymethod=None, disc_facility=None,
              pl_orbper=None, pl_rade=None, pl_bmasse=1.0, pl_eqt=None,
              pl_orbsmax=None, st_teff=None, sy_dist=None),
        _nasa(pl_rade=None, pl_bmasse=None, pl_eqt=None, pl_orbper=10.0, pl_orbsmax=None),
        _nasa(pl_name="must not appear"),
    ]
    offline_downloads[0].side_effect = None
    offline_downloads[0].return_value = str(_parquet(tmp_path, rows))

    result = demo._format_astrophysics_hf_dataset_subset("julienSimon/nasa-exoplanets", 9)

    assert len(result) == 9
    assert [r.split("Physical Class: ")[1].split(" | ")[0] for r in result[:4]] == [
        "Rocky Terrestrial Planet", "Super-Earth Exoplanet", "Sub-Neptune / Ice Giant", "Jovian Gas Giant",
    ]
    assert "Thermal Regime: Hot Inner Irradiated Orbit" in result[4]
    assert "Thermal Regime: Cold Outer Giant Orbit" in result[5]
    assert "Thermal Regime: Temperate Habitable Zone Orbit" in result[6]
    assert "EXO-8 around HostStar" in result[7]
    assert "Discovery: Transit (Space Observatory) | Period: 365.000 d | Radius: est." in result[7]
    assert "Mass: 1.00 M_earth | Teq: 280 K | Host Star Teff: 5700 K, Distance: 100.0 pc" in result[7]
    assert "Mass: est. | Teq: 950 K" in result[8]
    assert "Physical Class: Super-Earth Exoplanet" in result[8]
    assert demo.describe_data_provenance(result)["curated_fallback_rows"] == 0


@pytest.mark.parametrize(
    "teff,logg,feh,spectral,luminosity,population",
    [
        (7500.0, 4.1, -0.5, "A-Type White Star", "Class V (Main-Sequence Dwarf)", "Thin-Disk Population I"),
        (6000.0, 3.5, -1.2, "F-Type Yellow-White Star", "Class IV (Subgiant Turnoff)", "Thick-Disk Intermediate Population"),
        (5200.0, 3.49, -1.21, "G-Type Solar-Analog Star", "Class III (Red Giant Branch)", "Galactic Halo Metal-Poor Population II"),
        (3700.0, 4.5, 0.1, "K-Type Orange Star", "Class V (Main-Sequence Dwarf)", "Thin-Disk Population I"),
        (3699.0, 3.0, -2.0, "M-Type Red Dwarf/Giant Star", "Class III (Red Giant Branch)", "Galactic Halo Metal-Poor Population II"),
    ],
)
def test_sdss_real_parquet_classification(tmp_path, offline_downloads, teff, logg, feh, spectral, luminosity, population):
    row = {"spec_id": 42, "plate": 7, "fiber": 3, "ra": 12.3456, "dec": -4.5678,
           "snr": 17.5, "rv_adop": -12.3, "catalog_teff": teff, "catalog_feh": feh, "catalog_logg": logg}
    offline_downloads[0].side_effect = None
    offline_downloads[0].return_value = str(_parquet(tmp_path, [row, row]))

    result = demo._format_astrophysics_hf_dataset_subset("fixture/sdss_dr12_stars", 1)

    assert len(result) == 1
    assert "SpecID 42, Plate 7-3" in result[0]
    assert "Coordinates: RA=12.346, Dec=-4.568" in result[0]
    assert "Radial Velocity: -12.3 km/s (SNR=17.5)" in result[0]
    assert result[0].endswith(f"Spectral Type: {spectral} | Luminosity Class: {luminosity} | Galactic Population: {population}")


def test_sdss_missing_catalog_fields_use_documented_defaults(tmp_path, offline_downloads):
    row = dict.fromkeys(("spec_id", "plate", "fiber", "ra", "dec", "snr", "rv_adop",
                         "catalog_teff", "catalog_feh", "catalog_logg"))
    offline_downloads[0].side_effect = None
    offline_downloads[0].return_value = str(_parquet(tmp_path, [row]))
    result = demo._format_astrophysics_hf_dataset_subset("fixture/stars_regression", 1)
    assert "SpecID 0, Plate 1000-1" in result[0]
    assert "RA=0.000, Dec=+0.000 | Teff: 5800.0 K" in result[0]
    assert "log(g): 4.20 dex | Metallicity [Fe/H]: -0.20 dex" in result[0]
    assert "Radial Velocity: +0.0 km/s (SNR=30.0)" in result[0]


@pytest.mark.parametrize("dataset,key", [
    ("fixture/kepler", "Kepler_Exoplanet_Transit_Classifier"),
    ("julienSimon/nasa-exoplanets", "NASA_Exoplanet_Taxonomy_Classifier"),
    ("fixture/sdss_dr12_stars", "SDSS_Stellar_MK_Spectral_Classifier"),
    ("fixture/atmospheric_spectroscopy", "JWST_Atmosphere_Biosignature_Classifier"),
])
def test_download_failures_repeat_curated_rows_with_honest_provenance(offline_downloads, dataset, key):
    bank = CURATED_SEASON3_DOMAIN_CORPORA[key]
    count = len(bank) + 2
    result = demo._format_astrophysics_hf_dataset_subset(dataset, count)
    assert result == [bank[i % len(bank)] for i in range(count)]
    assert demo.describe_data_provenance(iter(result)) == {
        "rows": count, "distinct_rows": len(set(bank)), "curated_fallback_rows": count,
        "source_status": "all_rows_are_curated_fallback",
    }
    assert offline_downloads[0].call_count == (2 if "spectroscopy" in dataset else 1)


def _csv(tmp_path, rows):
    path = tmp_path / "spectra.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("habitability,temp,classification", [
    (0.70, 230, "Class-I High-Habitability Temperate World"),
    (0.90, 330, "Class-I High-Habitability Temperate World"),
    (0.70, 331, "Class-II Marginal Sub-Temperate Atmosphere"),
    (0.40, 285, "Class-II Marginal Sub-Temperate Atmosphere"),
    (0.39, 285, "Class-III Sterile Non-Habitable Atmosphere"),
])
def test_jwst_csv_habitability_thresholds(tmp_path, offline_downloads, habitability, temp, classification):
    row = {"exoplanet_id": "Fixture-b", "host_star_type": "K-Type", "distance_ly": 12.25,
           "transit_depth_ppm": 432.0, "h2o_abundance_log": -3.1, "ch4_abundance_log": -4.2,
           "co2_abundance_log": -5.3, "cloud_deck_pressure_bar": 0.15,
           "equilibrium_temp_k": temp, "habitability_score": habitability}
    offline_downloads[0].side_effect = None
    offline_downloads[0].return_value = str(_csv(tmp_path, [row, row]))
    result = demo._format_astrophysics_hf_dataset_subset("fixture/atmospheric_spectroscopy", 1, "fixture-token")
    assert len(result) == 1
    assert "Fixture-b around K-Type Star, 12.2 ly" in result[0]
    assert "Transit Depth: 432 ppm | log(H2O): -3.10 | log(CH4): -4.20 | log(CO2): -5.30" in result[0]
    assert f"Habitability Class: {classification} | Biosignature Status:" in result[0]
    assert demo.describe_data_provenance(result)["curated_fallback_rows"] == 0
    offline_downloads[0].assert_called_once_with(
        repo_id="Shoriful025/exoplanet_atmospheric_spectroscopy_results", filename="main.csv",
        repo_type="dataset", token="fixture-token",
    )


def test_jwst_partial_csv_then_qa_then_curated_rows(tmp_path, offline_downloads):
    csv_path = _csv(tmp_path, [{"exoplanet_id": "Fixture-b"}])
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(json.dumps({"qa_pairs": [
        {"question": " ", "answer": "skip"}, {"question": "skip", "answer": " "},
        {"question": "  " + "Q" * 210, "answer": "  " + "A" * 250},
    ]}), encoding="utf-8")
    offline_downloads[0].side_effect = [str(csv_path), str(qa_path)]
    result = demo._format_astrophysics_hf_dataset_subset("fixture/atmospheric_spectroscopy", 3)
    assert len(result) == 3
    assert "40.0 ly" in result[0]
    assert "Transit Depth: 1200 ppm | log(H2O): -4.50" in result[0]
    assert "Cloud Deck: 0.20 bar | Teq: 285 K | Habitability Index: 0.50" in result[0]
    assert result[1] == ("### Astrophysics & Planetary Science Query:\n" + "Q" * 200
                         + "\n### Verified Astrophysical Classification & Solution:\n" + "A" * 240)
    assert result[2] == CURATED_SEASON3_DOMAIN_CORPORA["JWST_Atmosphere_Biosignature_Classifier"][2]
    assert demo.describe_data_provenance(result)["source_status"] == "mixed_fetched_and_curated_fallback"
    assert demo.describe_data_provenance(result)["curated_fallback_rows"] == 1


def test_jwst_bad_csv_recovers_through_qa_and_stops_at_limit(tmp_path, offline_downloads):
    csv_path = _csv(tmp_path, [{"distance_ly": "not-a-number"}])
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(json.dumps({"qa_pairs": [
        {"question": " Why stars? ", "answer": " Nuclear fusion. "},
        {"question": "must not appear", "answer": "extra"},
    ]}), encoding="utf-8")
    offline_downloads[0].side_effect = [str(csv_path), str(qa_path)]
    result = demo._format_astrophysics_hf_dataset_subset("LazySloth/AstrophysicsQuestionAndAnswer", 1)
    assert result == ["### Astrophysics & Planetary Science Query:\nWhy stars?\n"
                      "### Verified Astrophysical Classification & Solution:\nNuclear fusion."]
    assert demo.describe_data_provenance(result)["curated_fallback_rows"] == 0


def test_unknown_astrophysics_dataset_returns_no_rows_without_download(offline_downloads):
    assert demo._format_astrophysics_hf_dataset_subset("fixture/unrelated", 2) == []
    offline_downloads[0].assert_not_called()


def test_astro_fetch_caches_formatted_data_and_reuses_without_download(tmp_path, isolated_cache, offline_downloads):
    offline_downloads[0].side_effect = None
    offline_downloads[0].return_value = str(_parquet(tmp_path, [_kepler()]))
    dataset = "mzrk36/exoplanets-unified-dataset"
    first = demo.fetch_hf_domain_subset(dataset, 2)
    assert len(first) == 2
    assert "Disposition: CONFIRMED" in first[0]
    assert demo.describe_data_provenance(first)["curated_fallback_rows"] == 1
    cache_file = isolated_cache / "mzrk36_exoplanets_unified_dataset_2.json"
    assert json.loads(cache_file.read_text(encoding="utf-8")) == first
    offline_downloads[0].reset_mock(side_effect=True)
    offline_downloads[0].side_effect = AssertionError("cached rows should avoid hub access")
    assert demo.fetch_hf_domain_subset(dataset, 2) == first
    offline_downloads[0].assert_not_called()
    offline_downloads[1].assert_not_called()


@pytest.mark.parametrize("cached", ["invalid JSON", "{}", '["too few"]'])
def test_invalid_subset_cache_recovers_via_rest(isolated_cache, offline_downloads, cached):
    isolated_cache.mkdir(parents=True)
    (isolated_cache / "fixture_questions_2.json").write_text(cached, encoding="utf-8")
    response = io.BytesIO(json.dumps({"rows": [
        {"row": {"instruction": " Explain stars ", "output": " Fusion "}},
        {"row": {"question": " Explain planets ", "long_answer": " Accretion "}},
        {"row": {"question": "skip without answer"}},
    ]}).encode())
    offline_downloads[1].side_effect = None
    offline_downloads[1].return_value = response
    result = demo.fetch_hf_domain_subset("fixture/questions", 2)
    assert result == ["### Instruction:\nExplain stars\n### Response:\nFusion",
                      "### Instruction:\nExplain planets\n### Response:\nAccretion"]
    assert response.closed
    request = offline_downloads[1].call_args.args[0]
    assert request.get_header("Authorization") == "Bearer test-offline-token"
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query) == {
        "dataset": ["fixture/questions"], "config": ["default"], "split": ["train"],
        "offset": ["0"], "length": ["2"],
    }
    assert offline_downloads[1].call_args.kwargs == {"timeout": 15}


def test_truncated_code_json_preserves_complete_rows_and_uses_rest(tmp_path, isolated_cache, offline_downloads):
    path = tmp_path / "partial.json"
    path.write_text('[{"instruction":" First task ","input":" Context ","output":" First answer "}, {"instruction":', encoding="utf-8")
    offline_downloads[0].side_effect = None
    offline_downloads[0].return_value = str(path)
    response = io.BytesIO(json.dumps({"rows": [{"row": {"question": "Second task", "answer": "Second answer"}}]}).encode())
    offline_downloads[1].side_effect = None
    offline_downloads[1].return_value = response
    result = demo.fetch_hf_domain_subset("fixture/questions", 2)
    assert result == ["### Instruction:\nFirst task\n### Input:\nContext\n### Response:\nFirst answer",
                      "### Instruction:\nSecond task\n### Response:\nSecond answer"]
    assert response.closed
    assert demo.describe_data_provenance(result)["curated_fallback_rows"] == 0


@pytest.mark.parametrize("body", [b"not JSON", b"\xff", b'{"rows":[]}', b'{"rows":[{"row":{}}]}'])
def test_bad_rest_response_closes_and_returns_labelled_fallback(isolated_cache, offline_downloads, body):
    response = io.BytesIO(body)
    offline_downloads[1].side_effect = None
    offline_downloads[1].return_value = response
    result = demo.fetch_hf_domain_subset("fixture/questions", 3)
    assert result == [demo._GENERIC_FALLBACK_SAMPLE] * 3
    assert response.closed
    assert demo.describe_data_provenance(result)["source_status"] == "all_rows_are_curated_fallback"


def test_campaign_live_rows_encode_request_filter_and_truncate(offline_downloads):
    response = io.BytesIO(json.dumps({"rows": [
        {"row": {"number": 42, "short": "tiny", "text": "  " + "é" * 400 + "  ", "other": "ignored" * 10}},
        {"row": {"empty": None, "body": "  " + "b" * 32 + "  "}},
        {"row": {"body": "excluded" * 10}},
    ]}).encode("utf-8"))
    offline_downloads[1].side_effect = None
    offline_downloads[1].return_value = response
    result = campaign._try_fetch_live_hf_rows("fixture/é space&name", max_rows=2, timeout_sec=0.125)
    assert result == ["é" * 384, "b" * 32]
    assert response.closed
    request = offline_downloads[1].call_args.args[0]
    assert request.get_header("User-agent") == "HiveMind-Community-Campaign/1.8"
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query) == {
        "dataset": ["fixture/é space&name"], "config": ["default"], "split": ["train"],
    }
    assert offline_downloads[1].call_args.kwargs == {"timeout": 0.125}


@pytest.mark.parametrize("body", [b"not JSON", b"\xff", b"{}", b'{"rows":null}', b'{"rows":[null]}',
                                      b'{"rows":[{"row":null},{"row":{"text":"short"}}]}'])
def test_campaign_live_bad_or_empty_responses_close_and_return_no_rows(offline_downloads, body):
    response = io.BytesIO(body)
    offline_downloads[1].side_effect = None
    offline_downloads[1].return_value = response
    assert campaign._try_fetch_live_hf_rows("fixture/questions") == []
    assert response.closed


def test_campaign_live_network_failure_returns_no_rows(offline_downloads):
    assert campaign._try_fetch_live_hf_rows("fixture/questions") == []
    offline_downloads[1].assert_called_once()


@pytest.mark.parametrize("rows", [[], [{"row": {"text": "Synthetic scientific observation for offline ingestion."}}]])
def test_campaign_ingestion_error_closes_http_and_cleans_temporary_cas(tmp_path, monkeypatch, offline_downloads, rows):
    scratch = tmp_path / "temporary-cas"
    scratch.mkdir()
    monkeypatch.setattr(campaign.tempfile, "tempdir", str(scratch))
    response = io.BytesIO(json.dumps({"rows": rows}).encode("utf-8"))
    active_directories = []

    def serve(request, timeout):
        active_directories.extend(scratch.iterdir())
        assert len(active_directories) == 1
        assert (active_directories[0] / "cas").is_dir()
        return response

    offline_downloads[1].side_effect = serve
    output = tmp_path / "receipt.json"
    # This creates the real tiny CPU reference model, then rejects tokenization
    # before the first optimization step. Preserve the caller's RNG state.
    with campaign.torch.random.fork_rng(devices=[]):
        with pytest.raises(ValueError, match="sequence_length must be between 2 and 128"):
            campaign.run_multi_moe_hf_campaign(
                preferred_device="cpu", steps_per_domain=2, domain_limit=2,
                live_hf_stream=True, sequence_length=1, output_path=output,
            )
    offline_downloads[1].assert_called_once()
    assert response.closed
    assert active_directories
    assert all(not path.exists() for path in active_directories)
    assert list(scratch.iterdir()) == []
    assert not output.exists()


def test_campaign_fixture_digest_is_for_raw_bytes_and_validates_splits(tmp_path):
    domains = {f"domain-{i}": {"train": [{"id": "train", "text": f"Training text {i}"}],
                              "holdout": [{"id": "holdout", "text": f"Holdout text {i}"}]} for i in range(4)}
    raw = json.dumps({"domains": domains}, indent=2).encode("utf-8")
    path = tmp_path / "fixture.json"
    path.write_bytes(raw)
    payload, digest = campaign.load_polyglot_8domain_fixture(path)
    assert payload == {"domains": domains}
    assert digest == hashlib.sha256(raw).hexdigest()
    domains["domain-0"]["holdout"][0]["text"] = "  Training   text 0  "
    path.write_text(json.dumps({"domains": domains}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate document/ID"):
        campaign.load_polyglot_8domain_fixture(path)


@pytest.mark.parametrize("domains", [None, {}, {"one": {}, "two": {}, "three": {}}])
def test_campaign_fixture_rejects_insufficient_domains(tmp_path, domains):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps({"domains": domains}), encoding="utf-8")
    with pytest.raises(ValueError, match="at least 4 domains"):
        campaign.load_polyglot_8domain_fixture(path)


@pytest.mark.parametrize("kwargs,message", [
    ({"contributor_handle": ""}, "contributor_handle must be non-empty"),
    ({"contributor_handle": " \t "}, "contributor_handle must be non-empty"),
    ({"steps_per_domain": 1}, "steps_per_domain must be >= 2"),
    ({"domain_limit": 1}, "domain_limit must be between 2 and 8"),
    ({"domain_limit": 9}, "domain_limit must be between 2 and 8"),
])
def test_campaign_invalid_options_rejected_before_any_training(tmp_path, kwargs, message):
    output = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match=message):
        campaign.run_multi_moe_hf_campaign(output_path=output, **kwargs)
    assert not output.exists()


def test_campaign_cli_help_describes_modes_without_running(capsys):
    with pytest.raises(SystemExit) as exc:
        campaign.main(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    for text in ("--contributor-handle", "--device", "--tier", "--steps-per-domain", "--domain-limit",
                 "--live-hf-stream", "--output", "quick (4 steps/domain)", "deep (24 steps/domain)"):
        assert text in output


@pytest.mark.parametrize("argv", [["--device", "tpu"], ["--tier", "invalid"],
                                  ["--steps-per-domain", "two"], ["--unknown-option"]])
def test_campaign_cli_invalid_arguments_report_usage(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        campaign.main(argv)
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "usage:" in error
    assert "error:" in error


@pytest.mark.parametrize("tier", ["quick", "standard", "deep"])
def test_campaign_cli_passes_options_into_real_validation(tmp_path, capsys, tier):
    path = tmp_path / "nested" / "receipt.json"
    with pytest.raises(ValueError, match="steps_per_domain must be >= 2"):
        campaign.main(["--contributor-handle", "offline-fixture", "--device", "cpu", "--tier", tier,
                       "--steps-per-domain", "1", "--domain-limit", "2", "--live-hf-stream", "--output", str(path)])
    assert not path.exists()
    assert capsys.readouterr().out == ""


def _receipt():
    return {
        "contributor_handle": "offline-fixture", "campaign_sha256": "a" * 64,
        "stage1_hardware_and_corpus": {"device_name": "Fixture CPU", "backend": "cpu",
            "peak_allocated_vram_mib": 0.0, "domains_evaluated": 4, "total_tokens_trained": 128,
            "steps_per_domain": 2},
        "stage2_and_3_multi_domain_lora_and_ef21_cas": {
            "mean_holdout_nll_reduction_pct": 12.5, "mean_train_nll_reduction_pct": 20.0,
            "domains": [{"super_cluster_name": "Fixture cluster", "domain_name": "Fixture domain",
                "hf_dataset_id": "fixture/questions", "holdout_nll_before_nats": 3.123456,
                "holdout_nll_after_nats": 2.123456, "holdout_perplexity_before": 22.123,
                "holdout_perplexity_after": 8.456, "holdout_nll_reduction_pct": 32.016,
                "peer_transfer_nll_drift": 0.0000012}]},
        "stage4_byzantine_ties_merge_vs_naive": {"naive_poisoned_holdout_nll_nats": 4.0,
            "naive_poisoned_perplexity": 54.6, "ties_filtered_holdout_nll_nats": 2.0,
            "ties_filtered_perplexity": 7.39},
        "stage5_128_expert_moe_and_zk_opoi_v4": {"gating_Compute_reduction_factor": 8.0,
            "speculative_token_tree_prefetch_calls": 3, "zk_opoi_v4_batch_passed": True},
        "symbolic_compute_reciprocity_points": {"symbolic_points_preview_pts": 5},
    }


@pytest.mark.parametrize("new_key", [False, True])
def test_campaign_markdown_formats_both_receipt_schemas_without_mutation(new_key):
    report = _receipt()
    if new_key:
        report["stage2_and_3_multi_moe_lora"] = report.pop("stage2_and_3_multi_domain_lora_and_ef21_cas")
    before = copy.deepcopy(report)
    output = campaign.format_campaign_markdown_summary(report)
    assert "`@offline-fixture`" in output
    assert "`Fixture CPU` (`cpu`)" in output
    assert "`128` UTF-8 tokens (`2` steps/domain)" in output
    assert "`3.1235 -> 2.1235`" in output
    assert "`22.12 -> 8.46`" in output
    assert "**`-32.02%`** | `0.000001`" in output
    assert "`fixture/questions`" in output
    assert "a" * 64 in output
    assert "HiveMind Reflexion Points (HMT)" in output
    assert report == before


def test_evidence_labels_preserve_explicit_provenance_and_zero_verification_factor():
    provenance = demo.describe_data_provenance(iter(["fetched fixture row", demo._GENERIC_FALLBACK_SAMPLE]))
    exp = {"data_provenance": provenance, "inference_settlement": {"rtt_speedup": 99, "fee_paid_hive": 0.25},
           "pre_lora_perplexity": 12.0, "post_lora_perplexity": 8.0}
    report = {"experiments": [exp, {"inference_settlement": None}],
              "regional_swarm_speculative_generations": {"zero": {"latency_reduction_factor": 0.0}, "empty": {}}}
    assert demo.label_astrophysics_report_evidence(report) is report
    assert exp["data_provenance"] is provenance
    assert provenance["source_status"] == "mixed_fetched_and_curated_fallback"
    assert exp["inference_settlement"]["rtt_speedup"] is None
    assert exp["inference_settlement"]["fee_paid_hive"] == 0.25
    assert (exp["pre_lora_perplexity"], exp["post_lora_perplexity"]) == (12.0, 8.0)
    assert report["regional_swarm_speculative_generations"]["zero"]["tokens_per_verification_pass"] == 0.0
    assert "tokens_per_verification_pass" not in report["regional_swarm_speculative_generations"]["empty"]
    assert report["evidence_scope"]["network_timing_measured"] is False
    assert demo.label_astrophysics_report_evidence(copy.deepcopy(report)) == report
