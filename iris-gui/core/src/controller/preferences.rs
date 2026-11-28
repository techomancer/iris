//! The application preferences as fields (the settings field kinds): the
//! interface language and how the guest display is scaled. These are
//! app-wide (D22), so they live in `GuiSettings`, not a machine's config.

use super::app::AppAction;
use super::display::{DisplayAction, DisplayFilter, DisplayMode, PointerMode};
use super::settings::{Field, FieldKind, Value};
use super::Enabled;
use crate::model::settings::GuiSettings;

/// Field ids of the preferences form.
pub mod field {
    pub const LANGUAGE: &str = "prefs.language";
    pub const DISPLAY_MODE: &str = "prefs.display-mode";
    pub const SCALE: &str = "prefs.scale";
    pub const INTEGER_ONLY: &str = "prefs.integer-only";
    pub const FILTER: &str = "prefs.filter";
    pub const POINTER: &str = "prefs.pointer";
}

const MODES: [DisplayMode; 3] = [DisplayMode::Actual, DisplayMode::Fit, DisplayMode::Stretch];
const FILTERS: [DisplayFilter; 3] = [DisplayFilter::Auto, DisplayFilter::Nearest, DisplayFilter::Linear];
const POINTERS: [PointerMode; 2] = [PointerMode::Capture, PointerMode::Follow];

/// The form for `prefs`.
pub fn fields(prefs: &GuiSettings) -> Vec<Field> {
    let langs = crate::model::i18n::languages();
    let mut names = vec![t!("menu-view-language-system")];
    names.extend(langs.iter().map(|(_, n)| n.clone()));
    let lang = langs.iter().position(|(c, _)| *c == prefs.language).map_or(0, |i| i + 1);
    let mode = MODES.iter().position(|m| *m == prefs.display_mode).unwrap_or(0);
    vec![
        Field {
            id: field::LANGUAGE,
            label: t!("menu-view-language"),
            help: None,
            kind: FieldKind::Choice { options: names, selected: lang },
            enabled: Enabled::Yes,
        },
        Field {
            id: field::DISPLAY_MODE,
            label: t!("prefs-display-size"),
            help: None,
            kind: FieldKind::Choice {
                options: vec![t!("prefs-display-actual"), t!("prefs-display-fit"), t!("prefs-display-stretch")],
                selected: mode,
            },
            enabled: Enabled::Yes,
        },
        Field {
            id: field::INTEGER_ONLY,
            label: t!("prefs-integer-only"),
            help: Some(t!("prefs-integer-only-help")),
            kind: FieldKind::Toggle(prefs.integer_only),
            enabled: Enabled::when(prefs.display_mode == DisplayMode::Fit, || t!("prefs-integer-when-fit")),
        },
        Field {
            id: field::POINTER,
            label: t!("prefs-pointer"),
            help: Some(t!("prefs-pointer-help")),
            kind: FieldKind::Choice {
                options: vec![t!("prefs-pointer-capture"), t!("prefs-pointer-follow")],
                selected: POINTERS.iter().position(|m| *m == prefs.pointer_mode).unwrap_or(0),
            },
            enabled: Enabled::Yes,
        },
        Field {
            id: field::FILTER,
            label: t!("prefs-filter"),
            help: None,
            kind: FieldKind::Choice {
                options: vec![t!("prefs-filter-auto"), t!("prefs-filter-nearest"), t!("prefs-filter-linear")],
                selected: FILTERS.iter().position(|f| *f == prefs.display_filter).unwrap_or(0),
            },
            enabled: Enabled::Yes,
        },
    ]
}

/// A tab of the preferences window.
#[derive(Debug, Clone, PartialEq)]
pub struct Tab {
    pub id: &'static str,
    pub title: String,
    pub icon: super::icons::Icon,
    pub fields: Vec<Field>,
}

/// The form for `prefs` in tabs: General (the language), Display (how the
/// guest's screen is drawn and the pointer).
pub fn tabs(prefs: &GuiSettings) -> Vec<Tab> {
    use super::icons::Icon;
    let all = fields(prefs);
    let take = |ids: &[&str]| -> Vec<Field> { all.iter().filter(|f| ids.contains(&f.id)).cloned().collect() };
    vec![
        Tab { id: "prefs.tab.general", title: t!("prefs-tab-general"), icon: Icon::Info, fields: take(&[field::LANGUAGE]) },
        Tab {
            id: "prefs.tab.display",
            title: t!("prefs-tab-display"),
            icon: Icon::DisplayScale,
            fields: take(&[field::DISPLAY_MODE, field::INTEGER_ONLY, field::FILTER, field::POINTER]),
        },
    ]
}

/// The action a preferences control with `id` means.
pub fn action_for(id: &str, value: Value) -> Option<AppAction> {
    let d = AppAction::Display;
    Some(match (id, value) {
        (field::LANGUAGE, Value::Index(0)) => AppAction::SetLanguage(String::new()),
        (field::LANGUAGE, Value::Index(i)) => AppAction::SetLanguage(crate::model::i18n::languages().get(i - 1)?.0.to_string()),
        (field::DISPLAY_MODE, Value::Index(i)) => d(DisplayAction::SetMode(*MODES.get(i)?)),
        (field::SCALE, Value::Number(n)) => d(DisplayAction::SetScale(n as f32)),
        (field::INTEGER_ONLY, Value::Bool(b)) => d(DisplayAction::SetIntegerOnly(b)),
        (field::FILTER, Value::Index(i)) => d(DisplayAction::SetFilter(*FILTERS.get(i)?)),
        (field::POINTER, Value::Index(i)) => d(DisplayAction::SetPointerMode(*POINTERS.get(i)?)),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_field_maps_to_an_action() {
        let prefs = GuiSettings::default();
        let f = fields(&prefs);
        assert_eq!(f.len(), 5);
        assert!(matches!(action_for(field::POINTER, Value::Index(1)), Some(AppAction::Display(DisplayAction::SetPointerMode(PointerMode::Follow)))));
        assert!(matches!(action_for(field::LANGUAGE, Value::Index(0)), Some(AppAction::SetLanguage(l)) if l.is_empty()));
        assert!(matches!(action_for(field::FILTER, Value::Index(1)), Some(AppAction::Display(DisplayAction::SetFilter(DisplayFilter::Nearest)))));
        assert!(matches!(action_for(field::SCALE, Value::Number(2.0)), Some(AppAction::Display(DisplayAction::SetScale(_)))));
        assert!(action_for(field::FILTER, Value::Index(9)).is_none());
    }

    #[test]
    fn the_tabs_hold_every_field_once() {
        let prefs = GuiSettings::default();
        let mut ids: Vec<&str> = tabs(&prefs).iter().flat_map(|t| t.fields.iter().map(|f| f.id)).collect();
        ids.sort();
        let mut all: Vec<&str> = fields(&prefs).iter().map(|f| f.id).collect();
        all.sort();
        assert_eq!(ids, all);
    }
}
