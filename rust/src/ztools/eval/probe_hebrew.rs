//! The Hebrew roster tasks (LM.PROBES, design section 4).
//!
//! `hebrew_summarize`, `hebrew_mixed` and `hebrew_json` measure what the house does with Hebrew
//! text -- attributed bullets from a Hebrew timeline, picking the Hebrew items out of a
//! mixed one, and field extraction from Hebrew event text -- by the facts planted in
//! the prompt ([`super::validators::recall`]).
//!
//! The timelines are synthetic. The design built them from the accepted summaries, but
//! this repo is public and OW.95 keeps the real timeline on the Mac, so the fixtures
//! here are written for the probe: invented handles, plausible news, planted names and
//! numbers. A real-data fixture, if ever wanted, belongs in the untracked user overlay.

use super::graded::Graded;
use super::task_loader::{Check, EvalTask};

fn strs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_string()).collect()
}

fn graded(name: &str, prompt: &str, check: Graded) -> EvalTask {
    EvalTask::new(name, prompt, vec![Check::Graded(check)])
}

const SUMMARIZE_PROMPT: &str = "Summarize the Hebrew timeline below as bullet points, one bullet per \
tweet, written in Hebrew. End every bullet with its attribution in exactly this form: \
(@handle | HH:MM). Keep every name and number exactly.\n\
\n\
@noa_cohen 08:12 ראש עיריית רמת השרון הודיע: 14 קווי אוטובוס יבוטלו מיום ראשון בגלל עבודות הרכבת הקלה.\n\
@itay_b 08:40 בנק ישראל הותיר את הריבית על 4.5% - ההחלטה השלישית ברציפות.\n\
@yael_g 09:05 מכבי חיפה ניצחה 3-1 את הפועל באר שבע באצטדיון סמי עופר, שחר פרץ כבש צמד.\n\
@dr_amir 09:31 משרד הבריאות: 212 מקרי חצבת נרשמו החודש, רובם בירושלים ובבני ברק.\n\
@tamar_news 10:02 נחתה בנתב\"ג טיסה ראשונה של חברת סקיילינק מליסבון עם 186 נוסעים.\n\
@ron_k 10:17 תושבי קריית שמונה: הרעש מהכביש החדש הגיע ל-78 דציבל, העירייה הבטיחה חומה אקוסטית.\n\
@orna_v 10:45 מחיר הדלק יעלה ב-17 אגורות בחצות, וליטר בנזין 95 יעלה ל-7.42 שקל.\n\
@benny_r 11:03 הפועל תל אביב השלימה את החתמת מאור בוזגלו מבית\"ר, העסקה בסך 1,250,000 דולר.\n\
@sigal_t 11:20 הדיון בחוק הגיוס נדחה ל-9 בנובמבר, כך הודיעה יו\"ר הוועדה מירב כהן.\n\
@uri_ph 11:48 רעידת אדמה בעוצמה 3.6 הורגשה ליד טבריה בשעה 04:12, ללא נפגעים.\n\
@hila_y 12:15 אוניברסיטת בן גוריון קיבלה תרומה של 25 מיליון שקל מהקרן על שם משפחת ליפשיץ.\n\
@avi_dev 12:40 4,800 שחיינים ישתתפו בגביע העולם בשחייה בבודפשט, ישראל שלחה אליו 11 נציגים.";

const SUMMARIZE_ITEMS: &[&str] = &[
    "רמת השרון",
    "14",
    "4.5%",
    "3-1",
    "שחר פרץ",
    "סמי עופר",
    "212",
    "חצבת",
    "סקיילינק",
    "186",
    "קריית שמונה",
    "78",
    "17 אגורות",
    "7.42",
    "בוזגלו",
    "1,250,000",
    "9 בנובמבר",
    "מירב כהן",
    "3.6",
    "טבריה",
    "25 מיליון",
    "ליפשיץ",
    "4,800",
    "בודפשט",
];
const SUMMARIZE_HANDLES: &[&str] = &[
    "noa_cohen",
    "itay_b",
    "yael_g",
    "dr_amir",
    "tamar_news",
    "ron_k",
    "orna_v",
    "benny_r",
    "sigal_t",
    "uri_ph",
    "hila_y",
    "avi_dev",
];

const MIXED_PROMPT: &str = "Below is a mixed Hebrew and English timeline. Summarize ONLY the \
Hebrew-language tweets, as bullets, in Hebrew, each with its (@handle | HH:MM). Ignore every \
English tweet entirely.\n\
\n\
@dana_m 07:55 עיריית תל אביב אישרה 9 נתיבי אופניים חדשים ברחוב אבן גבירול.\n\
@harold_w 08:03 The Hartley Bridge reopens on Friday after 11 months of repairs.\n\
@guy_s 08:20 חברת החשמל: 6,400 בתים נותרו ללא חשמל אחרי הסערה בגליל.\n\
@priya_n 08:47 Quarterly results: Vantex Corp reported revenue of 8.2 billion dollars.\n\
@liat_o 09:10 הטמפרטורה בערד הגיעה ל-41 מעלות, שיא של חמש שנים.\n\
@miles_t 09:36 Councillor Pruitt confirmed the 700-seat Aldgate theatre will open in May.\n\
@eyal_r 10:01 מנכ\"ל תנובה, אמנון שחר, הודיע על פתיחת מפעל חדש בקריית גת.\n\
@susan_l 10:24 The Marlow ferry carried 5,300 passengers on its first weekend.\n\
@tomer_h 10:50 פסטיבל הסרטים בירושלים יציג 63 סרטים מ-28 מדינות.\n\
@kate_b 11:15 Rainfall of 120 mm was recorded in Brisbane overnight, officials said.";

const MIXED_ITEMS: &[&str] = &[
    "9",
    "אבן גבירול",
    "6,400",
    "גליל",
    "41",
    "ערד",
    "אמנון שחר",
    "קריית גת",
    "63",
    "28",
];
const MIXED_HANDLES: &[&str] = &["dana_m", "guy_s", "liat_o", "eyal_r", "tomer_h"];
const MIXED_FORBIDDEN: &[&str] = &[
    "Hartley", "Vantex", "Pruitt", "Aldgate", "Marlow", "Brisbane", "8.2", "5,300",
];

const JSON_PROMPT: &str = "Extract each event announced below into JSON. Output ONLY a JSON \
object {\"events\": [...]} whose items have the keys title, date (YYYY-MM-DD), venue, price_nis \
(a number), organizer and duration_min (a number). Copy names exactly as written.\n\
\n\
1. הצגת הבכורה \"הגן של סבתא רוזה\" תתקיים ב-14 במרץ 2027 בתיאטרון הקאמרי בתל אביב. \
מחיר כרטיס: 120 ש\"ח. ההפקה של אסנת ברק, ההצגה נמשכת 95 דקות.\n\
2. הקונצרט \"לילות ים תיכוניים\" יתקיים ב-2 באפריל 2027 בהיכל התרבות בחיפה. \
מחיר כרטיס: 85 ש\"ח. מפיק האירוע: יובל שמעוני, משך הקונצרט 140 דקות.";

fn json_expected() -> Vec<(String, String)> {
    [
        ("title", "הגן של סבתא רוזה"),
        ("date", "2027-03-14"),
        ("venue", "הקאמרי"),
        ("price_nis", "120"),
        ("organizer", "אסנת ברק"),
        ("duration_min", "95"),
        ("title", "לילות ים תיכוניים"),
        ("date", "2027-04-02"),
        ("venue", "היכל התרבות"),
        ("price_nis", "85"),
        ("organizer", "יובל שמעוני"),
        ("duration_min", "140"),
    ]
    .iter()
    .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
    .collect()
}

/// The three `hebrew_*` tasks.
#[must_use]
pub fn hebrew_tasks() -> Vec<EvalTask> {
    vec![
        graded(
            "hebrew_summarize",
            SUMMARIZE_PROMPT,
            Graded::Recall {
                items: strs(SUMMARIZE_ITEMS),
                handles: strs(SUMMARIZE_HANDLES),
                forbidden: Vec::new(),
            },
        ),
        graded(
            "hebrew_mixed",
            MIXED_PROMPT,
            Graded::Recall {
                items: strs(MIXED_ITEMS),
                handles: strs(MIXED_HANDLES),
                forbidden: strs(MIXED_FORBIDDEN),
            },
        ),
        graded(
            "hebrew_json",
            JSON_PROMPT,
            Graded::JsonFields {
                expected: json_expected(),
            },
        )
        .json(),
    ]
}

#[cfg(test)]
#[path = "probe_hebrew_tests.rs"]
mod tests;
