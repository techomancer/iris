use crate::dev::ng1::rex3::Rex3Context;
use crate::dev::ng1::rex3_flat_fill::FillPlane;

use super::fill_walk::{
    prepare_fill_walk, FillCompletion, FillOperation, FillWalkKind,
};
use super::shade::ShadeDda;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CopyKind {
    Block,
    Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CopyFill {
    pub(super) kind: CopyKind,
    pub(super) plane: FillPlane,
    pub(super) min_x: i32,
    pub(super) min_y: i32,
    pub(super) max_x: i32,
    pub(super) max_y: i32,
    pub(super) walker_x0: i32,
    pub(super) walker_y0: i32,
    pub(super) walker_xsave: i32,
    pub(super) walker_x1: i32,
    pub(super) walker_y1: i32,
    pub(super) first_row_visits: u32,
    pub(super) full_row_visits: u32,
    pub(super) rows: u32,
    pub(super) first_row_complete: bool,
}

#[must_use]
#[derive(Debug)]
pub(super) struct PreparedCopyFill {
    fill: Option<CopyFill>,
    completion: FillCompletion,
    final_shade: Option<ShadeDda>,
}

impl PreparedCopyFill {
    pub(super) fn fill(&self) -> Option<CopyFill> {
        self.fill
    }

    pub(super) fn finish(self, context: &mut Rex3Context) {
        if let Some(shade) = self.final_shade {
            shade.write_back(context);
        }
        self.completion.apply(context);
    }
}

pub(super) fn prepare_copy_fill(
    context: &Rex3Context,
) -> Result<PreparedCopyFill, &'static str> {
    let walk = prepare_fill_walk(context, FillOperation::Copy)?;
    let completion = walk.completion(
        context,
        context.drawmode0.enzpattern(),
        context.drawmode0.enlspattern(),
    );
    let fill = walk.writes_possible(context).then(|| CopyFill {
        kind: if walk.kind == FillWalkKind::Block {
            CopyKind::Block
        } else {
            CopyKind::Span
        },
        plane: walk.plane.expect("writeable copy has a plane"),
        min_x: walk.min_x,
        min_y: walk.min_y,
        max_x: walk.max_x,
        max_y: walk.max_y,
        walker_x0: walk.walker_x0,
        walker_y0: walk.walker_y0,
        walker_xsave: walk.walker_xsave,
        walker_x1: walk.walker_x1,
        walker_y1: walk.walker_y1,
        first_row_visits: walk.first_row_visits,
        full_row_visits: walk.full_row_visits,
        rows: walk.rows,
        first_row_complete: walk.first_row_complete,
    });
    let final_shade = if context.drawmode0.shade() {
        let mut shade = ShadeDda::from_context(context);
        for _ in 0..walk.visits {
            shade.advance();
        }
        Some(shade)
    } else {
        None
    };
    Ok(PreparedCopyFill {
        fill,
        completion,
        final_shade,
    })
}
