//! The egui shell's effect host: carries out what a controller asks for
//! (dialogs, file pickers, revealing and opening files) and hands the user's
//! answers back as `Answer`s for the controller that asked.

use std::collections::VecDeque;
use std::path::PathBuf;

use eframe::egui;
use iris_gui_core::controller::{Answer, ChoiceSpec, ConfirmSpec, Effect, PickSpec, Severity, TextSpec};

use crate::filedialog;

enum Dialog {
    Confirm(ConfirmSpec),
    Ask { spec: TextSpec, text: String },
    Choose(ChoiceSpec),
    Alert { severity: Severity, title: String, message: String },
}

/// What the app must do after effects were carried out.
#[derive(Default)]
pub struct Outcome {
    pub toasts: Vec<String>,
    /// Answers available at once (file pickers are modal and synchronous).
    pub answers: Vec<Answer>,
    pub quit: bool,
    /// Text to put on the clipboard (egui does that through its context).
    pub clipboard: Option<String>,
    /// A settings page to show (`ids::settings`).
    pub open_settings: Option<&'static str>,
}

#[derive(Default)]
pub struct EffectHost {
    dialogs: VecDeque<Dialog>,
}

impl EffectHost {
    /// Carry out `effects`. Dialogs queue and show one at a time in [`show`].
    pub fn run(&mut self, effects: Vec<Effect>) -> Outcome {
        let mut out = Outcome::default();
        for e in effects {
            match e {
                Effect::PickFile(spec) => out.answers.push(pick(&spec, false)),
                Effect::PickFolder(spec) => out.answers.push(pick(&spec, true)),
                Effect::Confirm(spec) => self.dialogs.push_back(Dialog::Confirm(spec)),
                Effect::AskText(spec) => {
                    let text = spec.initial.clone();
                    self.dialogs.push_back(Dialog::Ask { spec, text });
                }
                Effect::Choose(spec) => self.dialogs.push_back(Dialog::Choose(spec)),
                Effect::Toast(msg) => out.toasts.push(msg),
                Effect::Alert { severity, title, message } => {
                    self.dialogs.push_back(Dialog::Alert { severity, title, message })
                }
                Effect::OpenUrl(url) => crate::host::open_url(&url),
                Effect::OpenPath(path) => crate::host::open_path(&path),
                Effect::RevealPath(path) => crate::config_ui::reveal_in_file_manager(&path.to_string_lossy()),
                Effect::CopyToClipboard(text) => out.clipboard = Some(text),
                Effect::OpenSettings(page) => out.open_settings = Some(page),
                Effect::Quit => out.quit = true,
            }
        }
        out
    }

    /// Draw the front dialog; returns the user's answer once it closes.
    pub fn show(&mut self, ctx: &egui::Context) -> Option<Answer> {
        let front = self.dialogs.front_mut()?;
        let mut answer = None;
        let mut closed = false;
        match front {
            Dialog::Confirm(spec) => {
                window(&spec.title).show(ctx, |ui| {
                    ui.set_max_width(420.0);
                    ui.label(&spec.message);
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        let ok = if spec.destructive {
                            crate::extra_ui::filled_button(&spec.confirm_label, egui::Color32::from_rgb(190, 60, 60))
                        } else {
                            egui::Button::new(&spec.confirm_label)
                        };
                        if ui.add(ok).clicked() {
                            answer = Some(Answer::Confirmed { purpose: spec.purpose, yes: true });
                        }
                        if ui.button(t!("common-cancel")).clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                            answer = Some(Answer::Confirmed { purpose: spec.purpose, yes: false });
                        }
                    });
                });
            }
            Dialog::Ask { spec, text } => {
                window(&spec.title).show(ctx, |ui| {
                    ui.set_max_width(420.0);
                    ui.label(&spec.prompt);
                    let edit = ui.add(egui::TextEdit::singleline(text).desired_width(300.0));
                    if !edit.has_focus() && ui.memory(|m| m.focused().is_none()) {
                        edit.request_focus();
                    }
                    let check = spec.validate(text);
                    match (&check, &spec.error) {
                        (Err(msg), _) if !msg.is_empty() => {
                            ui.colored_label(egui::Color32::LIGHT_RED, msg);
                        }
                        (_, Some(msg)) => {
                            ui.colored_label(egui::Color32::LIGHT_RED, msg);
                        }
                        _ => {}
                    }
                    ui.horizontal(|ui| {
                        let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                        if (ui.add_enabled(check.is_ok(), egui::Button::new(&spec.ok_label)).clicked() || enter)
                            && check.is_ok()
                        {
                            answer = Some(Answer::Text { purpose: spec.purpose, text: Some(text.trim().to_string()) });
                        }
                        if ui.button(t!("common-cancel")).clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                            answer = Some(Answer::Text { purpose: spec.purpose, text: None });
                        }
                    });
                });
            }
            Dialog::Choose(spec) => {
                window(&spec.title).show(ctx, |ui| {
                    ui.set_max_width(460.0);
                    ui.label(&spec.message);
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        for (i, label) in spec.choices.iter().enumerate() {
                            let b = if spec.destructive.contains(&i) {
                                crate::extra_ui::filled_button(label, egui::Color32::from_rgb(190, 60, 60))
                            } else if spec.default == Some(i) {
                                crate::extra_ui::filled_button(label, egui::Color32::from_rgb(60, 110, 190))
                            } else {
                                egui::Button::new(label)
                            };
                            if ui.add(b).clicked() {
                                answer = Some(Answer::Chose { purpose: spec.purpose, index: i });
                            }
                        }
                    });
                    if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                        answer = Some(Answer::Chose { purpose: spec.purpose, index: spec.dismiss });
                    }
                });
            }
            Dialog::Alert { severity, title, message } => {
                window(title).show(ctx, |ui| {
                    ui.set_max_width(460.0);
                    let color = match severity {
                        Severity::Info => ui.visuals().text_color(),
                        Severity::Warning => egui::Color32::from_rgb(230, 170, 60),
                        Severity::Error => egui::Color32::LIGHT_RED,
                    };
                    ui.colored_label(color, message.as_str());
                    if ui.button(t!("common-close")).clicked() {
                        closed = true;
                    }
                });
            }
        }
        if answer.is_some() || closed {
            self.dialogs.pop_front();
        }
        answer
    }
}

fn window(title: &str) -> egui::Window<'static> {
    egui::Window::new(title.to_string())
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
}

fn pick(spec: &PickSpec, folder: bool) -> Answer {
    let start = spec.start_dir.as_ref().map(|d| d.to_string_lossy().into_owned()).unwrap_or_default();
    let mut dialog = filedialog::dialog_at_dir(&spec.title, &start, filedialog::Anchor::Data);
    for (name, exts) in &spec.filters {
        dialog = dialog.add_filter(name.clone(), exts);
    }
    let paths: Vec<PathBuf> = if folder {
        dialog.pick_folder().into_iter().collect()
    } else if spec.save {
        dialog.save_file().into_iter().collect()
    } else {
        dialog.pick_file().into_iter().collect()
    };
    Answer::Picked { purpose: spec.purpose, paths }
}
