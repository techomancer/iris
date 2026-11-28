//! The New Machine wizard: the controller's steps, one `AdwNavigationPage`
//! each, drawn by the same [`Form`](crate::settings::Form) as everything
//! else. The spec lives here because nothing is created until Create.

use std::cell::{Cell, RefCell};

use adw::prelude::*;
use gtk4 as gtk;
use iris_gui_core::controller::new_machine::NewMachineSpec;
use iris_gui_core::controller::settings::Value;
use iris_gui_core::controller::Answer;
use libadwaita as adw;

use crate::settings::Form;

pub struct NewMachineDialog {
    pub dialog: adw::Dialog,
    pub spec: RefCell<NewMachineSpec>,
    step: Cell<usize>,
    stack: gtk::Stack,
    back: gtk::Button,
    next: gtk::Button,
    title: adw::WindowTitle,
    form: std::rc::Rc<Form>,
}

impl NewMachineDialog {
    pub fn new() -> std::rc::Rc<Self> {
        let title = adw::WindowTitle::new(&t!("dialog-new-machine-title"), "");
        let header = adw::HeaderBar::builder().show_end_title_buttons(true).build();
        header.set_title_widget(Some(&title));
        let back = gtk::Button::with_label(&t!("wizard-back"));
        header.pack_start(&back);
        let next = gtk::Button::with_label(&t!("wizard-continue"));
        next.add_css_class("suggested-action");
        header.pack_end(&next);

        let stack = gtk::Stack::builder().transition_type(gtk::StackTransitionType::SlideLeftRight).build();
        let scroll = gtk::ScrolledWindow::builder().hscrollbar_policy(gtk::PolicyType::Never).vexpand(true).child(&stack).build();
        let view = adw::ToolbarView::new();
        view.add_top_bar(&header);
        view.set_content(Some(&scroll));

        let dialog = adw::Dialog::builder().title(t!("dialog-new-machine-title")).content_width(620).content_height(560).build();
        dialog.set_child(Some(&view));

        let me = std::rc::Rc::new(Self {
            dialog,
            spec: RefCell::new(NewMachineSpec::default()),
            step: Cell::new(0),
            stack,
            back,
            next,
            title,
            form: Form::new(|id, value| crate::app::new_machine_control(id, value)),
        });

        let w = std::rc::Rc::downgrade(&me);
        me.back.connect_clicked(move |_| {
            if let Some(me) = w.upgrade() {
                me.go(me.step.get().saturating_sub(1));
            }
        });
        let w = std::rc::Rc::downgrade(&me);
        me.next.connect_clicked(move |_| {
            if let Some(me) = w.upgrade() {
                me.forward();
            }
        });
        me.redraw();
        me
    }

    /// Draw every step again (a choice on one changes the others).
    pub fn redraw(self: &std::rc::Rc<Self>) {
        while let Some(child) = self.stack.first_child() {
            self.stack.remove(&child);
        }
        let steps = self.spec.borrow().steps();
        for (i, (title, fields)) in steps.iter().enumerate() {
            let page = adw::PreferencesPage::builder().title(title).build();
            let group = adw::PreferencesGroup::new();
            self.form.fill(&group, fields);
            page.add(&group);
            self.stack.add_named(&page, Some(&i.to_string()));
        }
        self.go(self.step.get().min(steps.len().saturating_sub(1)));
    }

    fn go(self: &std::rc::Rc<Self>, step: usize) {
        self.step.set(step);
        self.stack.set_visible_child_name(&step.to_string());
        let spec = self.spec.borrow();
        let steps = spec.steps();
        let last = step + 1 >= steps.len();
        self.title.set_title(&t!("dialog-new-machine-title"));
        self.title.set_subtitle(&steps.get(step).map(|(t, _)| t.clone()).unwrap_or_default());
        self.back.set_visible(step > 0);
        self.next.set_label(&if last { t!("dialog-new-machine-create") } else { t!("wizard-continue") });
        let can = if last { spec.can_create() } else { spec.can_continue(step) };
        self.next.set_sensitive(can.is_yes());
        self.next.set_tooltip_text(can.reason());
    }

    /// Next, or Create on the last step.
    fn forward(self: &std::rc::Rc<Self>) {
        let (last, action) = {
            let spec = self.spec.borrow();
            let last = self.step.get() + 1 >= spec.steps().len();
            (last, last.then(|| spec.create()))
        };
        match action {
            Some(action) => {
                self.dialog.close();
                crate::app::dispatch(action);
            }
            None if !last => self.go(self.step.get() + 1),
            None => {}
        }
    }

    /// A control reported a value; `true` when the pages must be drawn again.
    pub fn control(self: &std::rc::Rc<Self>, id: &str, value: Value) -> bool {
        if let Some(effect) = self.spec.borrow().pick(id) {
            crate::app::run_effects(vec![effect]);
            return false;
        }
        let changed = self.spec.borrow_mut().apply(id, value);
        if changed {
            self.redraw();
        } else {
            self.go(self.step.get());
        }
        changed
    }

    /// A file picker answered: `true` when something changed.
    pub fn picked(self: &std::rc::Rc<Self>, answer: &Answer) -> bool {
        let changed = self.spec.borrow_mut().picked(answer);
        if changed {
            self.redraw();
        }
        changed
    }
}
