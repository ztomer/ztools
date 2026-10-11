//! Where the plan is for, as the forecast needs it.
//!
//! `conf/weekend.toml [location]` names the operator's coordinates and time
//! zone. The forecast URL used to hard-code its own copy -- a different
//! longitude (-79.4982 against -79.5083) and zone (`America/New_York` against
//! `America/Toronto`) -- so the config's values were read by nothing and a
//! move would have kept the old forecast. The config is now the only source.

use std::fmt::Write as _;

/// The coordinates and zone the forecast is asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct Place {
    pub latitude: f64,
    pub longitude: f64,
    pub timezone: String,
}

/// The `[location]` place from the first weekend config that has one.
///
/// # Errors
///
/// No config with a `[location]` table, or one missing `latitude`,
/// `longitude` or `timezone` -- each named, because a forecast for the wrong
/// place reads exactly like one for the right place.
pub fn load_place(paths: &[String]) -> Result<Place, String> {
    for raw in paths {
        let path = crate::manifest::expand_tilde(raw);
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(val) = toml::from_str::<toml::Value>(&content) else {
            continue;
        };
        let Some(location) = val.get("location") else {
            continue;
        };
        let missing = |key: &str| {
            format!(
                "{} has a [location] table without `{key}`; the forecast needs it",
                path.display()
            )
        };
        let coordinate = |key: &str| {
            location
                .get(key)
                .and_then(|v| {
                    v.as_float().or_else(|| {
                        v.as_integer()
                            .and_then(|i| i32::try_from(i).ok())
                            .map(f64::from)
                    })
                })
                .ok_or_else(|| missing(key))
        };
        return Ok(Place {
            latitude: coordinate("latitude")?,
            longitude: coordinate("longitude")?,
            timezone: location
                .get("timezone")
                .and_then(toml::Value::as_str)
                .ok_or_else(|| missing("timezone"))?
                .to_string(),
        });
    }
    Err(format!(
        "no weekend config with a [location] table among {paths:?}; no forecast without a place"
    ))
}

/// The Open-Meteo forecast URL for `place` over the window, against `endpoint`.
///
/// `endpoint` is an ORIGIN -- scheme, host, and any path prefix -- exactly as
/// [`crate::config::ZtoolsConfig::weather_url`] holds it, so a mirror, a proxy
/// or a self-hosted instance substitutes the host and still gets Open-Meteo's
/// `/v1/forecast` contract. The trailing slash is TRIMMED rather than
/// concatenated: an operator who writes `https://host/` meant one slash, and
/// `//v1/forecast` is a 404 that reads as a dead forecast.
#[must_use]
pub fn open_meteo_url(
    endpoint: &str,
    place: &Place,
    friday_date: &str,
    sunday_date: &str,
) -> String {
    let mut url = format!("{}/v1/forecast", endpoint.trim_end_matches('/'));
    let _ = write!(
        url,
        "?latitude={}&longitude={}&daily=temperature_2m_max,precipitation_sum\
         &timezone={}&start_date={friday_date}&end_date={sunday_date}",
        place.latitude, place.longitude, place.timezone
    );
    url
}

#[cfg(test)]
#[path = "place_tests.rs"]
mod tests;
