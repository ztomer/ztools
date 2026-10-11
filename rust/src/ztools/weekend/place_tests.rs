use super::*;

fn write_conf(body: &str) -> (tempfile::TempDir, Vec<String>) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("weekend.toml");
    std::fs::write(&path, body).unwrap();
    let paths = vec![path.display().to_string()];
    (dir, paths)
}

/// The forecast is asked for the configured place and zone -- not a copy of
/// them in code. A different place in the config is a different URL.
#[test]
fn the_forecast_url_is_built_from_the_configured_place() {
    let (_dir, paths) = write_conf(
        "[location]\ncity = \"Elsewhere\"\nlatitude = 45.5\nlongitude = -73.6\ntimezone = \"America/Montreal\"\n",
    );
    let place = load_place(&paths).unwrap();
    assert_eq!(
        open_meteo_url(
            "https://api.open-meteo.com/",
            &place,
            "2026-08-07",
            "2026-08-09"
        ),
        concat!(
            "https://api.open-meteo.com/v1/forecast",
            "?latitude=45.5&longitude=-73.6",
            "&daily=temperature_2m_max,precipitation_sum",
            "&timezone=America/Montreal",
            "&start_date=2026-08-07&end_date=2026-08-09",
        )
    );
}

/// A place the config does not complete is a stated failure naming the key.
#[test]
fn a_location_without_coordinates_is_named() {
    let (_dir, paths) = write_conf("[location]\ncity = \"X\"\ntimezone = \"America/Toronto\"\n");
    let why = load_place(&paths).unwrap_err();
    assert!(why.contains("`latitude`"), "{why}");
    assert!(load_place(&[]).unwrap_err().contains("no weekend config"));
}

/// The shipped config is complete: the planner's own forecast has a place.
#[test]
fn the_shipped_config_names_a_place() {
    let conf = concat!(env!("CARGO_MANIFEST_DIR"), "/../conf/weekend.toml").to_string();
    let place = load_place(&[conf]).unwrap();
    assert_eq!(place.timezone, "America/Toronto");
}

/// The ORIGIN substitutes; the path and the window do not. A mirror or a
/// self-hosted instance lands on the same contract, and a trailing slash an
/// operator typed must not become `//v1/forecast` -- a 404, which reads as a
/// dead forecast. The expectation is a LITERAL: one assembled from the same
/// value the function trims agrees with it whether or not the trim happens
/// (calibrated -- removing `trim_end_matches` left that version green).
#[test]
fn the_configured_origin_replaces_the_host_and_survives_a_trailing_slash() {
    let place = Place {
        latitude: 43.8361,
        longitude: -79.4982,
        timezone: "America/Toronto".to_string(),
    };
    let expected = concat!(
        "http://127.0.0.1:9/v1/forecast",
        "?latitude=43.8361&longitude=-79.4982",
        "&daily=temperature_2m_max,precipitation_sum",
        "&timezone=America/Toronto",
        "&start_date=2026-08-07&end_date=2026-08-09",
    );
    for endpoint in ["http://127.0.0.1:9", "http://127.0.0.1:9/"] {
        assert_eq!(
            open_meteo_url(endpoint, &place, "2026-08-07", "2026-08-09"),
            expected,
            "an endpoint of {endpoint:?} must not change the path or the window"
        );
    }
}
