//! Forms, drawn from the controller's schemas (D8): one row per field kind.
//! Nothing here knows what a field means — the core builds the list, the
//! shell draws each kind once and reports a control by its id. [`Form`] does
//! the drawing for the machine settings, the New Machine wizard and the
//! application preferences alike; [`SettingsDialog`] is the machine
//! settings over it, one `AdwPreferencesPage` per section.
//!
//! Edits apply at once and the app autosaves them, so there is nothing to
//! confirm and no Apply button but the complete-TOML editor's.

use std::cell::RefCell;

use adw::prelude::*;
use gtk::gio;
use gtk4 as gtk;
use iris_gui_core::controller::app::AppController;
use iris_gui_core::controller::settings::{
    self, control_id, field as fid, part, DiskRow, FieldKind, ForwardRow, PickKind, Section, Value,
};
use iris_gui_core::controller::Severity;
use libadwaita as adw;

use crate::icons;

/// Draws field kinds into preference groups and reports what changed. The
/// machine settings, the New Machine wizard and the application preferences
/// are all field lists, so they all draw through one of these.
pub struct Form {
    report: Box<dyn Fn(String, Value)>,
    /// The complete-TOML editor, whose text Apply sends.
    toml: RefCell<Option<gtk::TextView>>,
    /// Set while the shell fills controls, so they do not report back.
    syncing: std::cell::Cell<bool>,
}

impl Form {
    pub fn new(report: impl Fn(String, Value) + 'static) -> std::rc::Rc<Self> {
        std::rc::Rc::new(Self { report: Box::new(report), toml: RefCell::default(), syncing: std::cell::Cell::new(false) })
    }

    /// Fill `group` with `fields`, reporting nothing while it is drawn.
    pub fn fill(self: &std::rc::Rc<Self>, group: &adw::PreferencesGroup, fields: &[settings::Field]) {
        self.syncing.set(true);
        for f in fields {
            self.add_field(group, f);
        }
        self.syncing.set(false);
    }

    pub fn toml_text(&self) -> Option<String> {
        let view = self.toml.borrow();
        let buffer = view.as_ref()?.buffer();
        Some(buffer.text(&buffer.start_iter(), &buffer.end_iter(), false).to_string())
    }

    pub fn forget_toml(&self) {
        *self.toml.borrow_mut() = None;
    }

    fn add_field(self: &std::rc::Rc<Self>, group: &adw::PreferencesGroup, f: &settings::Field) {
        let enabled = f.enabled.is_yes();
        let tip = match (&f.help, f.enabled.reason()) {
            (_, Some(why)) => Some(why.to_string()),
            (Some(help), None) => Some(help.clone()),
            _ => None,
        };
        match &f.kind {
            FieldKind::Choice { options, selected } => {
                let row = self.combo(f.id, &f.label, options, *selected);
                finish(group, &row, enabled, tip.as_deref());
            }
            FieldKind::GroupedChoice { groups, selected } => {
                // GTK's combo row has no sections, so the groups become the
                // rows' subtitles: the Mac's pop-up has Workstations and
                // Servers as headings, and prefixing every label with its
                // group just truncates the model's own name away.
                let options: Vec<String> = groups.iter().flat_map(|(_, v)| v.clone()).collect();
                let row = self.combo(f.id, &f.label, &options, *selected);
                let of_group: Vec<String> =
                    groups.iter().flat_map(|(title, v)| v.iter().map(move |_| title.clone())).collect();
                row.set_subtitle(of_group.get(*selected).map(String::as_str).unwrap_or(""));
                finish(group, &row, enabled, tip.as_deref());
            }
            FieldKind::Toggle(on) => {
                let row = adw::SwitchRow::builder().title(&f.label).active(*on).build();
                let me = self.clone();
                let id = f.id;
                row.connect_active_notify(move |r| me.report(id.to_string(), Value::Bool(r.is_active())));
                finish(group, &row, enabled, tip.as_deref());
            }
            FieldKind::Text { value, placeholder, error } => {
                let row = adw::EntryRow::builder().title(&f.label).text(value).build();
                if !placeholder.is_empty() {
                    row.set_tooltip_text(Some(placeholder));
                }
                if error.is_some() {
                    row.add_css_class("error");
                }
                let me = self.clone();
                let id = f.id;
                row.connect_changed(move |r| me.report(id.to_string(), Value::Text(r.text().to_string())));
                finish(group, &row, enabled, error.clone().or(tip).as_deref());
            }
            FieldKind::Number { value, min, max, step, unit } => {
                let row = adw::SpinRow::with_range(*min, *max, *step);
                row.set_title(&f.label);
                row.set_value(*value);
                if !unit.is_empty() {
                    row.set_subtitle(unit);
                }
                let me = self.clone();
                let id = f.id;
                row.connect_value_notify(move |r| me.report(id.to_string(), Value::Number(r.value())));
                finish(group, &row, enabled, tip.as_deref());
            }
            FieldKind::Path { value, pick } => {
                let folder = matches!(pick, PickKind::Folder);
                let row = adw::ActionRow::builder().title(&f.label).subtitle(shown_path(value, folder)).build();
                row.set_subtitle_lines(0);
                let what = match pick {
                    PickKind::Folder => t!("settings-choose"),
                    PickKind::File { .. } => t!("settings-choose"),
                };
                let button = gtk::Button::builder().label(&what).valign(gtk::Align::Center).build();
                let me = self.clone();
                let id = control_id(f.id, None, Some(part::BROWSE));
                button.connect_clicked(move |_| me.report(id.clone(), Value::Bool(true)));
                row.add_suffix(&button);
                finish(group, &row, enabled, tip.as_deref());
            }
            FieldKind::Memory { mb, totals, .. } => {
                let options: Vec<String> = totals.iter().map(|m| t!("config-megabytes", size = *m)).collect();
                let selected = totals.iter().position(|t| t == mb).unwrap_or(0);
                // The action takes megabytes, so the row reports the amount.
                let row = adw::ComboRow::builder().title(&f.label).model(&strings(&options)).selected(selected as u32).build();
                let me = self.clone();
                let id = f.id;
                let totals = totals.clone();
                row.connect_selected_notify(move |r| {
                    if let Some(mb) = totals.get(r.selected() as usize) {
                        me.report(id.to_string(), Value::Number(*mb as f64));
                    }
                });
                finish(group, &row, enabled, tip.as_deref());
            }
            FieldKind::BankGrid { banks, sizes, summary } => {
                let expander = adw::ExpanderRow::builder().title(&f.label).subtitle(summary).build();
                let options: Vec<String> =
                    sizes.iter().map(|s| if *s == 0 { t!("settings-bank-empty") } else { t!("config-megabytes", size = *s) }).collect();
                for (i, bank) in banks.iter().enumerate() {
                    let selected = sizes.iter().position(|s| s == bank).unwrap_or(0);
                    let row = self.combo(
                        Box::leak(control_id(f.id, Some(i), None).into_boxed_str()),
                        &t!("config-ram-bank", bank = i),
                        &options,
                        selected,
                    );
                    row.set_sensitive(enabled);
                    expander.add_row(&row);
                }
                finish(group, &expander, enabled, tip.as_deref());
            }
            FieldKind::DiskTable(rows) => {
                let table = adw::PreferencesGroup::builder().title(&f.label).build();
                table.set_header_suffix(Some(&self.add_disk_buttons(f.id, enabled)));
                for (i, row) in rows.iter().enumerate() {
                    table.add(&self.disk_row(f.id, i, row, enabled));
                }
                group.add(&table);
            }
            FieldKind::PortForwardTable(rows) => {
                let table = adw::PreferencesGroup::builder().title(&f.label).build();
                table.set_header_suffix(Some(&self.add_forward_button(f.id, rows, enabled)));
                for (i, row) in rows.iter().enumerate() {
                    table.add(&self.forward_row(f.id, i, row, enabled));
                }
                group.add(&table);
            }
            FieldKind::Toml { text } => {
                let view = gtk::TextView::new();
                view.set_monospace(true);
                view.buffer().set_text(text);
                view.set_size_request(-1, 320);
                let scroll = gtk::ScrolledWindow::builder().child(&view).height_request(320).build();
                scroll.add_css_class("card");
                let expander = adw::ExpanderRow::builder().title(&f.label).build();
                let holder = adw::ActionRow::new();
                holder.set_child(Some(&scroll));
                expander.add_row(&holder);
                let apply = gtk::Button::builder().label(t!("extra-apply")).halign(gtk::Align::End).build();
                apply.add_css_class("suggested-action");
                let me = self.clone();
                let id = control_id(f.id, None, Some(part::APPLY));
                apply.connect_clicked(move |_| me.report(id.clone(), Value::Bool(true)));
                let apply_row = adw::ActionRow::new();
                apply_row.set_child(Some(&apply));
                expander.add_row(&apply_row);
                *self.toml.borrow_mut() = Some(view);
                finish(group, &expander, enabled, tip.as_deref());
            }
            FieldKind::Note(severity, text) => {
                let row = adw::ActionRow::builder().title(text).build();
                row.set_title_lines(0);
                let icon = match severity {
                    Severity::Info => iris_gui_core::controller::icons::Icon::Info,
                    Severity::Warning => iris_gui_core::controller::icons::Icon::Warning,
                    Severity::Error => iris_gui_core::controller::icons::Icon::Error,
                };
                let image = icons::image(icon, 16, "");
                image.add_css_class(match severity {
                    Severity::Info => "accent",
                    Severity::Warning => "warning",
                    Severity::Error => "error",
                });
                row.add_prefix(&image);
                group.add(&row);
            }
        }
    }

    fn combo(self: &std::rc::Rc<Self>, id: &'static str, label: &str, options: &[String], selected: usize) -> adw::ComboRow {
        let row = adw::ComboRow::builder().title(label).model(&strings(options)).selected(selected as u32).build();
        let me = self.clone();
        row.connect_selected_notify(move |r| me.report(id.to_string(), Value::Index(r.selected() as usize)));
        row
    }

    fn add_disk_buttons(self: &std::rc::Rc<Self>, field: &'static str, enabled: bool) -> gtk::Widget {
        let b = gtk::Box::new(gtk::Orientation::Horizontal, 6);
        for (part, label) in [(part::ADD_DISK, t!("settings-add-disk")), (part::ADD_CDROM, t!("settings-add-cdrom"))] {
            let button = gtk::Button::builder().label(&label).sensitive(enabled).build();
            let me = self.clone();
            let id = control_id(field, None, Some(part));
            button.connect_clicked(move |_| me.report(id.clone(), Value::Bool(true)));
            b.append(&button);
        }
        b.upcast()
    }

    fn disk_row(self: &std::rc::Rc<Self>, field: &'static str, i: usize, row: &DiskRow, enabled: bool) -> adw::ExpanderRow {
        let title = t!("ov-scsi", id = row.name.as_str());
        let expander = adw::ExpanderRow::builder().title(&title).subtitle(shown_path(&row.path, false)).build();
        if let Some(problem) = &row.problem {
            let warn = icons::image(iris_gui_core::controller::icons::Icon::Warning, 16, problem);
            warn.add_css_class("warning");
            warn.set_tooltip_text(Some(problem));
            expander.add_suffix(&warn);
        }

        let kinds = settings::disk_kind_labels();
        let kind = adw::ComboRow::builder()
            .title(t!("settings-disk-kind"))
            .model(&strings(&kinds))
            .selected(settings::disk_kind_index(row) as u32)
            .sensitive(enabled)
            .build();
        let me = self.clone();
        let id = control_id(field, Some(i), Some(part::KIND));
        kind.connect_selected_notify(move |r| me.report(id.clone(), Value::Index(r.selected() as usize)));
        expander.add_row(&kind);

        if row.is_disk() {
            let new = gtk::Button::builder()
                .label(t!("scsi-create-blank"))
                .halign(gtk::Align::End)
                .valign(gtk::Align::Center)
                .sensitive(enabled)
                .build();
            let me = self.clone();
            let id = control_id(field, Some(i), Some(part::NEW_DISK));
            new.connect_clicked(move |_| me.report(id.clone(), Value::Bool(true)));
            let new_row = adw::ActionRow::builder().title(t!("create-disk-title")).build();
            new_row.add_suffix(&new);
            expander.add_row(&new_row);
        }

        let path = adw::EntryRow::builder().title(t!("settings-disk-image")).text(&row.path).sensitive(enabled).build();
        let me = self.clone();
        let id = control_id(field, Some(i), Some(part::PATH));
        path.connect_changed(move |r| me.report(id.clone(), Value::Text(r.text().to_string())));
        let browse = gtk::Button::builder().label(t!("settings-choose")).valign(gtk::Align::Center).build();
        let me = self.clone();
        let pick = control_id(field, Some(i), Some(part::PICK));
        browse.connect_clicked(move |_| me.report(pick.clone(), Value::Bool(true)));
        path.add_suffix(&browse);
        expander.add_row(&path);

        if row.is_daynaport() {
            // A DaynaPort is an Ethernet card on the bus, not storage.
            for (part, label, value) in [
                (part::MAC, t!("config-scsi-mac"), row.mac.clone()),
                (part::SUBNET, t!("config-scsi-subnet"), row.subnet.clone()),
            ] {
                let entry = adw::EntryRow::builder().title(&label).text(&value).sensitive(enabled).build();
                let me = self.clone();
                let id = control_id(field, Some(i), Some(part));
                entry.connect_changed(move |r| me.report(id.clone(), Value::Text(r.text().to_string())));
                expander.add_row(&entry);
            }
            let remove = gtk::Button::builder().label(t!("config-remove")).halign(gtk::Align::End).sensitive(enabled).build();
            remove.add_css_class("destructive-action");
            let me = self.clone();
            let id = control_id(field, Some(i), Some(part::REMOVE));
            remove.connect_clicked(move |_| me.report(id.clone(), Value::Bool(true)));
            let remove_row = adw::ActionRow::new();
            remove_row.set_child(Some(&remove));
            expander.add_row(&remove_row);
            return expander;
        }

        if row.is_cdrom() {
            // The changer: the discs behind the one loaded now.
            for (n, disc) in row.discs.iter().enumerate() {
                let entry = adw::EntryRow::builder()
                    .title(t!("config-scsi-changer-slot", n = n + 1))
                    .text(disc)
                    .sensitive(enabled)
                    .build();
                let me = self.clone();
                let id = control_id(field, Some(i), Some(&format!("{}{n}", part::DISC)));
                entry.connect_changed(move |r| me.report(id.clone(), Value::Text(r.text().to_string())));
                let pick = gtk::Button::builder().label(t!("settings-choose")).valign(gtk::Align::Center).build();
                let me = self.clone();
                let id = control_id(field, Some(i), Some(&format!("{}{n}", part::DISC)));
                pick.connect_clicked(move |_| me.report(id.clone(), Value::Bool(true)));
                entry.add_suffix(&pick);
                expander.add_row(&entry);
            }
            let add = gtk::Button::builder().label(t!("config-scsi-add-disc")).halign(gtk::Align::Start).sensitive(enabled).build();
            let me = self.clone();
            let id = control_id(field, Some(i), Some(part::ADD_DISC));
            add.connect_clicked(move |_| me.report(id.clone(), Value::Bool(true)));
            let add_row = adw::ActionRow::builder().title(t!("config-scsi-changer")).build();
            add_row.add_suffix(&add);
            expander.add_row(&add_row);
        }

        if row.is_disk() {
            let scratch =
                adw::SwitchRow::builder().title(t!("config-scsi-scratch")).active(row.scratch).sensitive(enabled).build();
            let me = self.clone();
            let id = control_id(field, Some(i), Some(part::SCRATCH));
            scratch.connect_active_notify(move |r| me.report(id.clone(), Value::Bool(r.is_active())));
            expander.add_row(&scratch);
            if row.scratch {
                let size = adw::SpinRow::with_range(1.0, 1_048_576.0, 64.0);
                size.set_title(&t!("config-scsi-scratch-size"));
                size.set_value(row.scratch_mb as f64);
                size.set_sensitive(enabled);
                let me = self.clone();
                let id = control_id(field, Some(i), Some(part::SCRATCH_MB));
                size.connect_value_notify(move |r| me.report(id.clone(), Value::Number(r.value())));
                expander.add_row(&size);
            }
            let cache = adw::SwitchRow::builder()
                .title(t!("config-scsi-write-cache"))
                .subtitle(t!("config-scsi-write-cache-tip"))
                .active(row.write_cache)
                .sensitive(enabled)
                .build();
            cache.set_subtitle_lines(0);
            let me = self.clone();
            let id = control_id(field, Some(i), Some(part::WRITE_CACHE));
            cache.connect_active_notify(move |r| me.report(id.clone(), Value::Bool(r.is_active())));
            expander.add_row(&cache);
        }

        if row.can_overlay() {
            let overlay =
                adw::SwitchRow::builder().title(t!("settings-overlay")).active(row.overlay).sensitive(enabled).build();
            let me = self.clone();
            let id = control_id(field, Some(i), Some(part::OVERLAY));
            overlay.connect_active_notify(move |r| me.report(id.clone(), Value::Bool(r.is_active())));
            expander.add_row(&overlay);
        }

        let remove = gtk::Button::builder().label(t!("config-remove")).halign(gtk::Align::End).sensitive(enabled).build();
        remove.add_css_class("destructive-action");
        let me = self.clone();
        let id = control_id(field, Some(i), Some(part::REMOVE));
        remove.connect_clicked(move |_| me.report(id.clone(), Value::Bool(true)));
        let remove_row = adw::ActionRow::new();
        remove_row.set_child(Some(&remove));
        expander.add_row(&remove_row);
        expander
    }

    fn add_forward_button(self: &std::rc::Rc<Self>, field: &'static str, rows: &[ForwardRow], enabled: bool) -> gtk::Widget {
        let menu = gio::Menu::new();
        let Some(app) = crate::app::application() else { return gtk::Box::new(gtk::Orientation::Horizontal, 0).upcast() };
        for (i, (preset, allowed)) in settings::forward_menu(rows).into_iter().enumerate() {
            if !allowed.is_yes() {
                continue;
            }
            let id = format!("{}{}", crate::app::SETTING_PREFIX, control_id(field, None, Some(part::ADD)));
            let item = gio::MenuItem::new(Some(&preset.label), None);
            item.set_detailed_action(&format!("app.{}", crate::menu::action_for(&app, &format!("{id}#{i}"))));
            menu.append_item(&item);
        }
        let custom = gio::Menu::new();
        let id = format!("{}{}", crate::app::SETTING_PREFIX, control_id(field, None, Some(part::ADD)));
        let item = gio::MenuItem::new(Some(&t!("config-forward-custom")), None);
        item.set_detailed_action(&format!("app.{}", crate::menu::action_for(&app, &format!("{id}#custom"))));
        custom.append_item(&item);
        menu.append_section(None, &custom);
        let button = gtk::MenuButton::builder().label(t!("config-forward-add")).menu_model(&menu).sensitive(enabled).build();
        button.upcast()
    }

    fn forward_row(self: &std::rc::Rc<Self>, field: &'static str, i: usize, row: &ForwardRow, enabled: bool) -> adw::ExpanderRow {
        let title = t!("settings-forward-row", host = row.host_port, guest = row.guest_port);
        let expander = adw::ExpanderRow::builder()
            .title(&title)
            .subtitle(if row.tcp { "TCP" } else { "UDP" })
            .build();
        let protocols = vec!["TCP".to_string(), "UDP".to_string()];
        let proto = adw::ComboRow::builder()
            .title(t!("settings-forward-protocol"))
            .model(&strings(&protocols))
            .selected(if row.tcp { 0 } else { 1 })
            .sensitive(enabled)
            .build();
        let me = self.clone();
        let id = control_id(field, Some(i), Some(part::TCP));
        proto.connect_selected_notify(move |r| me.report(id.clone(), Value::Index(r.selected() as usize)));
        expander.add_row(&proto);

        for (part, label, value) in [
            (part::HOST, t!("config-forward-host"), row.host_port),
            (part::GUEST, t!("config-forward-guest"), row.guest_port),
        ] {
            let entry = adw::EntryRow::builder().title(&label).text(value.to_string()).sensitive(enabled).build();
            let me = self.clone();
            let id = control_id(field, Some(i), Some(part));
            entry.connect_changed(move |r| me.report(id.clone(), Value::Text(r.text().to_string())));
            expander.add_row(&entry);
        }
        let any = adw::SwitchRow::builder()
            .title(t!("config-forward-bind-any"))
            .active(row.any_interface)
            .sensitive(enabled)
            .build();
        let me = self.clone();
        let id = control_id(field, Some(i), Some(part::ANY));
        any.connect_active_notify(move |r| me.report(id.clone(), Value::Bool(r.is_active())));
        expander.add_row(&any);

        let remove = gtk::Button::builder().label(t!("config-remove")).halign(gtk::Align::End).sensitive(enabled).build();
        remove.add_css_class("destructive-action");
        let me = self.clone();
        let id = control_id(field, Some(i), Some(part::REMOVE));
        remove.connect_clicked(move |_| me.report(id.clone(), Value::Bool(true)));
        let remove_row = adw::ActionRow::new();
        remove_row.set_child(Some(&remove));
        expander.add_row(&remove_row);
        expander
    }

    /// Report a control from outside the form (a section's own button).
    pub fn report_public(&self, id: String, value: Value) {
        self.report(id, value);
    }

    /// A control changed: tell the shell, unless it was the shell filling it in.
    fn report(&self, id: String, value: Value) {
        if self.syncing.get() {
            return;
        }
        (self.report)(id, value);
    }
}

fn strings(options: &[String]) -> gtk::StringList {
    let list = gtk::StringList::new(&[]);
    for o in options {
        list.append(o);
    }
    list
}

/// A path row's subtitle: the path, or what it is when nothing is chosen.
fn shown_path(value: &str, folder: bool) -> String {
    match (value.trim().is_empty(), folder) {
        (false, _) => value.to_string(),
        (true, true) => t!("wizard-none"),
        (true, false) => t!("ov-empty"),
    }
}

fn finish(group: &adw::PreferencesGroup, row: &impl IsA<gtk::Widget>, enabled: bool, tip: Option<&str>) {
    let row = row.as_ref();
    row.set_sensitive(enabled);
    if let Some(tip) = tip {
        row.set_tooltip_text(Some(tip));
    }
    group.add(row);
}

/// The theme icon for a section's meaning: a settings page's icon is GTK's
/// own furniture, beside the row icons the core names.
pub fn icon_name(icon: iris_gui_core::controller::icons::Icon) -> &'static str {
    use iris_gui_core::controller::icons::Icon as I;
    match icon {
        I::Disk | I::CdRom => "drive-harddisk-symbolic",
        I::Network => "network-wired-symbolic",
        I::DisplayScale => "video-display-symbolic",
        I::Audio => "audio-speakers-symbolic",
        I::Camera => "camera-web-symbolic",
        _ => "preferences-system-symbolic",
    }
}

/// The machine settings: one `AdwPreferencesPage` per section of the
/// controller's schema, drawn by a [`Form`].
pub struct SettingsDialog {
    pub dialog: adw::PreferencesDialog,
    form: std::rc::Rc<Form>,
    /// The sections as last drawn, to resolve a control's action and to
    /// redraw only on a change.
    shown: RefCell<Vec<Section>>,
    /// The pages now in the dialog: AdwPreferencesDialog does not list them,
    /// and adding one twice under the same name is a warning and a muddle.
    pages: RefCell<Vec<adw::PreferencesPage>>,
}

impl SettingsDialog {
    pub fn new() -> std::rc::Rc<Self> {
        let dialog = adw::PreferencesDialog::builder().title(t!("tb-settings")).build();
        dialog.set_search_enabled(true);
        std::rc::Rc::new(Self {
            dialog,
            form: Form::new(|id, value| crate::app::setting(id, value)),
            shown: RefCell::default(),
            pages: RefCell::default(),
        })
    }

    /// Draw the sections the controller holds now.
    pub fn refresh(&self, app: &AppController) {
        let sections = app.settings_sections();
        if *self.shown.borrow() == sections {
            return;
        }
        // Keep the page the user is on across a redraw.
        let open = self.dialog.visible_page().and_then(|p| p.name()).map(|n| n.to_string());
        for page in self.pages.borrow_mut().drain(..) {
            self.dialog.remove(&page);
        }
        self.form.forget_toml();
        for section in &sections {
            let page = self.page(section);
            self.dialog.add(&page);
            self.pages.borrow_mut().push(page);
        }
                if let Some(name) = open {
            self.show_page(&name);
        }
        *self.shown.borrow_mut() = sections;
    }

    pub fn show_page(&self, id: &str) {
        self.dialog.set_visible_page_name(id);
    }

    /// The action a control's report means, from what the dialog shows.
    pub fn action(&self, id: &str, value: Value) -> Option<settings::SettingsAction> {
        let shown = self.shown.borrow();
        let mut disk_keys: Vec<u8> = Vec::new();
        let mut forwards: Vec<ForwardRow> = Vec::new();
        let mut sizes: Vec<u32> = Vec::new();
        for f in shown.iter().flat_map(|s| s.fields.iter()) {
            match &f.kind {
                FieldKind::DiskTable(rows) => disk_keys = rows.iter().map(|r| r.key).collect(),
                FieldKind::PortForwardTable(rows) => forwards = rows.clone(),
                FieldKind::BankGrid { sizes: s, .. } => sizes = s.clone(),
                _ => {}
            }
        }
        let value = match value {
            // A bank's popup lists the sizes; the action takes megabytes.
            Value::Index(i) if id.starts_with(&format!("{}/", fid::BANKS)) => Value::Number(*sizes.get(i)? as f64),
            _ if id == control_id(fid::TOML, None, Some(part::APPLY)) => {
                Value::Text(self.form.toml_text()?)
            }
            v => v,
        };
        settings::action_for(id, value, &disk_keys, &forwards)
    }

    fn page(&self, section: &Section) -> adw::PreferencesPage {
        let page = adw::PreferencesPage::builder().title(&section.title).name(section.id).build();
        page.set_icon_name(Some(icon_name(section.icon)));
        let group = adw::PreferencesGroup::new();
        self.form.fill(&group, &section.fields);
        page.add(&group);
        // The guest clock's Reset: an action on the section, which the core
        // takes as a plain set on the reset field (the schema has no button
        // kind, and adding one would break the AppKit shell's match).
        if section.id == iris_gui_core::controller::ids::settings::CLOCK {
            let button = gtk::Button::builder().label(t!("config-rtc-reset")).halign(gtk::Align::Center).build();
            let form = self.form.clone();
            button.connect_clicked(move |_| form.report_public(fid::RTC_RESET.to_string(), Value::Bool(true)));
            let holder = adw::PreferencesGroup::new();
            let row = adw::ActionRow::new();
            row.set_child(Some(&button));
            holder.add(&row);
            page.add(&holder);
        }
        page
    }
}
