//! CSV export of a set of eval runs.
//!
//! Split out of `report` for the house 500-line cap, along the seam that was
//! already there: everything else in that module reads or summarises the run
//! history, and this is the only part that writes a different FORMAT of it.

use std::io::Write as _;
use std::path::Path;

use super::report::{status_word, ModelRun};

/// Export one row per (model, task): the shape downstream sheets expect.
///
/// # Errors
///
/// When the output file cannot be created, or a row cannot be written to it.
pub fn export_csv(runs: &[ModelRun], output_file: &Path) -> std::io::Result<()> {
    let mut file = std::io::BufWriter::new(std::fs::File::create(output_file)?);
    writeln!(
        file,
        "Model,Task,Score,Status,Time(s),Failure,Failure_Category"
    )?;
    for run in runs {
        for o in &run.outcomes {
            writeln!(
                file,
                "{},{},{},{},{},{},{}",
                csv_escape(&run.model),
                csv_escape(&o.task),
                o.score,
                status_word(o.score),
                o.time_secs,
                csv_escape(o.error.as_deref().unwrap_or("")),
                o.failure_category
            )?;
        }
    }
    Ok(())
}

fn csv_escape(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::super::report::tests::{outcome, run};
    use super::super::report::ModelRun;
    use super::*;

    #[test]
    fn csv_export_matches_the_downstream_sheet_shape() {
        let dir = tempfile::tempdir().unwrap();
        let mut o = outcome("t1", 95);
        o.time_secs = 1.5;
        o.error = Some("HTTP 503, at capacity".to_string());
        o.failure_category = "INFRA".to_string();
        let runs = vec![run("model-a", vec![o], true)];
        let out = dir.path().join("results.csv");
        export_csv(&runs, &out).unwrap();
        let text = std::fs::read_to_string(out).unwrap();
        let mut lines = text.lines();
        assert_eq!(
            lines.next(),
            Some("Model,Task,Score,Status,Time(s),Failure,Failure_Category")
        );
        let row = lines.next().unwrap();
        // Quoted because the error contains a comma.
        assert_eq!(
            row,
            "model-a,t1,95,PASS,1.5,\"HTTP 503, at capacity\",INFRA"
        );
    }
}
