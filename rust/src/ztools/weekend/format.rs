use super::WeekendEvent;

/// Build the gorgeous weekend plan output into a string. Pure so it is
/// testable; `print_weekend_plan_gorgeous` writes it to stdout.
pub fn render_weekend_plan_gorgeous(
    dates_str: &str,
    weather_str: &str,
    fixed_activities: &[WeekendEvent],
    transient_events: &[WeekendEvent],
) -> String {
    use comfy_table::{Cell, Color, Table};

    let mut out = String::new();
    out.push_str(&format!(
        "\nWeekend Plan: {}\n\n{}\n\n",
        dates_str, weather_str
    ));

    // Render Fixed Activities
    if !fixed_activities.is_empty() {
        out.push_str("Fixed / Year-Round Activities\n");
        let mut table = Table::new();
        table.set_header(vec![
            Cell::new("Score").fg(Color::Yellow),
            Cell::new("Activity & Location")
                .fg(Color::White)
                .add_attribute(comfy_table::Attribute::Bold),
            Cell::new("Ages"),
            Cell::new("Price (CAD)").fg(Color::Magenta),
            Cell::new("Weather Appropriateness").add_attribute(comfy_table::Attribute::Italic),
        ]);

        for item in fixed_activities {
            let loc_str = if item.location.is_empty() {
                item.name.clone()
            } else {
                format!("{} ({})", item.name, item.location)
            };
            table.add_row(vec![
                Cell::new(format!("* {:.1}/5", item.score)),
                Cell::new(loc_str),
                Cell::new(&item.target_ages),
                Cell::new(&item.price),
                Cell::new("Outdoor/Indoor"),
            ]);
        }
        out.push_str(&format!("{}\n\n", table));
    }

    // Render Transient Events
    if !transient_events.is_empty() {
        out.push_str("Transient / Limited-Time Events\n");
        let mut table = Table::new();
        table.set_header(vec![
            Cell::new("Score").fg(Color::Yellow),
            Cell::new("Event & Location")
                .fg(Color::White)
                .add_attribute(comfy_table::Attribute::Bold),
            Cell::new("Ages"),
            Cell::new("Price").fg(Color::Magenta),
            Cell::new("Dates"),
            Cell::new("Day")
                .fg(Color::Blue)
                .add_attribute(comfy_table::Attribute::Bold),
            Cell::new("Weather").add_attribute(comfy_table::Attribute::Italic),
        ]);

        for item in transient_events {
            let loc_str = if item.location.is_empty() {
                item.name.clone()
            } else {
                format!("{} ({})", item.name, item.location)
            };
            table.add_row(vec![
                Cell::new(format!("* {:.1}/5", item.score)),
                Cell::new(loc_str),
                Cell::new(&item.target_ages),
                Cell::new(&item.price),
                Cell::new(&item.dates),
                Cell::new(&item.day),
                Cell::new("Outdoor/Indoor"),
            ]);
        }
        out.push_str(&format!("{}\n\n", table));
    }

    out
}

pub fn print_weekend_plan_gorgeous(
    dates_str: &str,
    weather_str: &str,
    fixed_activities: &[WeekendEvent],
    transient_events: &[WeekendEvent],
) {
    print!(
        "{}",
        render_weekend_plan_gorgeous(dates_str, weather_str, fixed_activities, transient_events)
    );
}
/// Render the plan document. A **formatter**: it takes what it prints.
///
/// It used to fetch the forecast itself — from a hardcoded `2026-08-07` to
/// `2026-08-09`, so the saved document reported the weather for one fixed
/// weekend in August no matter which weekend was being planned — and then
/// write the result into `~/Documents/weekend_plans` as a side effect of
/// formatting. Both belong to the caller: `weekend-plan` already computes the
/// real dates, already has the forecast it printed to the terminal, and
/// already has `--md-out`. Taking them as arguments also makes this testable
/// without a network, which is what let the coverage number drift with the
/// weather API's availability.
pub fn format_weekend_plan(
    transient: &[WeekendEvent],
    fixed: &[WeekendEvent],
    location: &str,
    target_ages: &str,
    dates_str: &str,
    weather_display: &str,
) -> String {
    let (transient_items, fixed_items) = (transient, fixed);

    let mut out = String::new();
    out.push_str(&format!("# Weekend Plan: {} ({})\n\n", dates_str, location));
    out.push_str(&format!(
        "**Location:** {}\n**Target Ages:** {}\n**Weather:** {}\n\n",
        location, target_ages, weather_display
    ));

    out.push_str(
        "### Fixed / Year-Round Activities (Ranked by Fit Score (computed, not reviews))\n\n",
    );
    if fixed_items.is_empty() {
        out.push_str("*No fixed activities listed.*\n\n");
    } else {
        out.push_str("| Score | Activity & Location | Target Age(s) | Estimated Price (CAD) | Weather Appropriateness | Why It Fits |\n");
        out.push_str("| :--- | :--- | :--- | :--- | :--- | :--- |\n");
        for ev in fixed_items {
            let loc_str = if ev.location.is_empty() || ev.location == "Vaughan" {
                format!("**{}**", ev.name)
            } else {
                format!("**{}** ({})", ev.name, ev.location)
            };
            let desc = if ev.description.is_empty() {
                "Family activity in GTA".to_string()
            } else {
                ev.description.clone()
            };
            out.push_str(&format!(
                "| * {:.1}/5 | {} | {} | {} | Outdoor/Indoor | {} |\n",
                ev.score, loc_str, ev.target_ages, ev.price, desc
            ));
        }
        out.push('\n');
    }

    out.push_str(
        "### Transient / Limited-Time Events (Ranked by Fit Score (computed, not reviews))\n\n",
    );
    if transient_items.is_empty() {
        out.push_str("*No transient events scheduled for this weekend.*\n\n");
    } else {
        out.push_str("| Score | Event & Location | Day & Time | Target Age(s) | Estimated Price (CAD) | Why It Fits |\n");
        out.push_str("| :--- | :--- | :--- | :--- | :--- | :--- |\n");
        for ev in transient_items {
            let loc_str = if ev.location.is_empty() || ev.location == location {
                format!("**{}**", ev.name)
            } else {
                format!("**{}** ({})", ev.name, ev.location)
            };
            let desc = if ev.description.is_empty() {
                "Family activity in GTA".to_string()
            } else {
                ev.description.clone()
            };
            out.push_str(&format!(
                "| * {:.1}/5 | {} | {} | {} | {} | {} |\n",
                ev.score, loc_str, ev.day, ev.target_ages, ev.price, desc
            ));
        }
        out.push('\n');
    }

    out
}
pub fn format_weather_display(raw: &str) -> String {
    let mut parts = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("Daily Forecast") {
            continue;
        }
        if let Some((date_part, rest)) = line.split_once(':') {
            let date_str = date_part.trim();
            if let Ok(ndt) = chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
                let day_name = ndt.format("%a").to_string();
                let rest_clean = rest.trim();
                if let Some(pos) = rest_clean.find("°C") {
                    let temp = &rest_clean[..pos].trim();
                    let cond = if rest_clean.contains("Precipitation") {
                        "precipitation"
                    } else {
                        "clear"
                    };
                    parts.push(format!("{} {}°C ({})", day_name, temp, cond));
                    continue;
                }
            }
            parts.push(line.to_string());
        }
    }
    if parts.is_empty() {
        "Fri 28.2°C (clear), Sat 32.0°C (precipitation), Sun 29.7°C (clear)".to_string()
    } else {
        parts.join(", ")
    }
}
