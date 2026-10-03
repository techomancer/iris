//! 4x4 matrices, column-major (element (row r, column c) at `c * 4 + r`, the
//! order glLoadMatrix takes), and the matrix stacks.

pub type Mat4 = [f32; 16];

pub const IDENT: Mat4 = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.];

/// a * b.
pub fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut m = [0.0f32; 16];
    for c in 0..4 {
        for r in 0..4 {
            m[c * 4 + r] = (0..4).map(|k| a[k * 4 + r] * b[c * 4 + k]).sum();
        }
    }
    m
}

/// m * v.
pub fn xform(m: &Mat4, v: [f32; 4]) -> [f32; 4] {
    let row = |r: usize| m[r] * v[0] + m[4 + r] * v[1] + m[8 + r] * v[2] + m[12 + r] * v[3];
    [row(0), row(1), row(2), row(3)]
}

/// glOrtho.
pub fn ortho(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> Mat4 {
    let mut m = IDENT;
    m[0] = 2.0 / (r - l);
    m[5] = 2.0 / (t - b);
    m[10] = -2.0 / (f - n);
    m[12] = -(r + l) / (r - l);
    m[13] = -(t + b) / (t - b);
    m[14] = -(f + n) / (f - n);
    m
}

/// glFrustum.
pub fn frustum(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> Mat4 {
    let mut m = [0.0f32; 16];
    m[0] = 2.0 * n / (r - l);
    m[5] = 2.0 * n / (t - b);
    m[8] = (r + l) / (r - l);
    m[9] = (t + b) / (t - b);
    m[10] = -(f + n) / (f - n);
    m[11] = -1.0;
    m[14] = -2.0 * f * n / (f - n);
    m
}

/// glTranslate.
pub fn translate(x: f32, y: f32, z: f32) -> Mat4 {
    let mut m = IDENT;
    m[12] = x;
    m[13] = y;
    m[14] = z;
    m
}

/// glScale.
pub fn scale(x: f32, y: f32, z: f32) -> Mat4 {
    let mut m = IDENT;
    m[0] = x;
    m[5] = y;
    m[10] = z;
    m
}

/// glRotate: `deg` degrees about (x, y, z).
pub fn rotate(deg: f32, x: f32, y: f32, z: f32) -> Mat4 {
    let len = (x * x + y * y + z * z).sqrt();
    if len < 1e-20 {
        return IDENT;
    }
    let (x, y, z) = (x / len, y / len, z / len);
    let (s, c) = deg.to_radians().sin_cos();
    let t = 1.0 - c;
    [
        x * x * t + c, y * x * t + z * s, x * z * t - y * s, 0.0,
        x * y * t - z * s, y * y * t + c, y * z * t + x * s, 0.0,
        x * z * t + y * s, y * z * t - x * s, z * z * t + c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// Upper-left 3x3 of the inverse transpose of `m` (column-major), what
/// normals transform by. Identity if `m` is singular.
pub fn normal_matrix(m: &Mat4) -> [f32; 9] {
    let a = |r: usize, c: usize| m[c * 4 + r];
    let cof = |r0: usize, r1: usize, c0: usize, c1: usize| a(r0, c0) * a(r1, c1) - a(r0, c1) * a(r1, c0);
    // Cofactor matrix of the 3x3: (inverse transpose) = cofactors / det.
    let c = [
        cof(1, 2, 1, 2), -cof(1, 2, 0, 2), cof(1, 2, 0, 1),
        -cof(0, 2, 1, 2), cof(0, 2, 0, 2), -cof(0, 2, 0, 1),
        cof(0, 1, 1, 2), -cof(0, 1, 0, 2), cof(0, 1, 0, 1),
    ];
    let det = a(0, 0) * c[0] + a(0, 1) * c[1] + a(0, 2) * c[2];
    if det.abs() < 1e-30 {
        return [1., 0., 0., 0., 1., 0., 0., 0., 1.];
    }
    // c is row-major cofactors C[r][c]; column-major result element (r, c)
    // = C[r][c] / det at c * 3 + r.
    let mut out = [0.0f32; 9];
    for r in 0..3 {
        for col in 0..3 {
            out[col * 3 + r] = c[r * 3 + col] / det;
        }
    }
    out
}

/// Inverse of `m` (column-major), by cofactors. None if singular.
pub fn invert(m: &Mat4) -> Option<Mat4> {
    let a = |r: usize, c: usize| m[c * 4 + r] as f64;
    let mut inv = [0.0f64; 16];
    // inv[c * 4 + r] = cofactor(r, c) of the transpose = cofactor(c, r) / det.
    let minor = |skip_r: usize, skip_c: usize| -> f64 {
        let rows: Vec<usize> = (0..4).filter(|&r| r != skip_r).collect();
        let cols: Vec<usize> = (0..4).filter(|&c| c != skip_c).collect();
        let e = |i: usize, j: usize| a(rows[i], cols[j]);
        e(0, 0) * (e(1, 1) * e(2, 2) - e(1, 2) * e(2, 1)) - e(0, 1) * (e(1, 0) * e(2, 2) - e(1, 2) * e(2, 0))
            + e(0, 2) * (e(1, 0) * e(2, 1) - e(1, 1) * e(2, 0))
    };
    let mut det = 0.0;
    for c in 0..4 {
        det += a(0, c) * minor(0, c) * if c % 2 == 0 { 1.0 } else { -1.0 };
    }
    if det.abs() < 1e-30 {
        return None;
    }
    for r in 0..4 {
        for c in 0..4 {
            let sign = if (r + c) % 2 == 0 { 1.0 } else { -1.0 };
            // (row r, column c) of the inverse = cofactor(c, r) / det.
            inv[c * 4 + r] = sign * minor(c, r) / det;
        }
    }
    let mut out = [0.0f32; 16];
    for (o, v) in out.iter_mut().zip(inv) {
        *o = v as f32;
    }
    Some(out)
}

/// A GL matrix stack. Plain data, valid zeroed after `init`.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Stack<const N: usize> {
    pub m: [Mat4; N],
    pub top: u32,
}

impl<const N: usize> Stack<N> {
    pub fn init(&mut self) {
        self.top = 0;
        self.m[0] = IDENT;
    }

    /// The top entry's index. State saved in guest-visible memory (a GE's
    /// context RAM) can come back as anything: clamp rather than panic.
    fn t(&self) -> usize {
        (self.top as usize).min(N - 1)
    }

    pub fn get(&self) -> &Mat4 {
        &self.m[self.t()]
    }

    pub fn load(&mut self, m: Mat4) {
        let t = self.t();
        self.m[t] = m;
    }

    /// glMultMatrix: top = top * m.
    pub fn mult(&mut self, m: &Mat4) {
        let t = self.t();
        self.m[t] = mul(&self.m[t], m);
    }

    /// False on overflow (GL_STACK_OVERFLOW; the stack is unchanged).
    pub fn push(&mut self) -> bool {
        let t = self.t();
        if t + 1 >= N {
            return false;
        }
        self.m[t + 1] = self.m[t];
        self.top += 1;
        true
    }

    /// False on underflow.
    pub fn pop(&mut self) -> bool {
        if self.top == 0 {
            return false;
        }
        self.top -= 1;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 4], b: [f32; 4]) -> bool {
        a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn ortho_maps_the_box_to_the_unit_cube() {
        let m = ortho(0.0, 400.0, 0.0, 300.0, -1.0, 1.0);
        assert!(close(xform(&m, [0.0, 0.0, 0.0, 1.0]), [-1.0, -1.0, 0.0, 1.0]));
        assert!(close(xform(&m, [400.0, 300.0, 0.0, 1.0]), [1.0, 1.0, 0.0, 1.0]));
    }

    #[test]
    fn rotate_turns_x_into_y_about_z() {
        let m = rotate(90.0, 0.0, 0.0, 1.0);
        assert!(close(xform(&m, [1.0, 0.0, 0.0, 1.0]), [0.0, 1.0, 0.0, 1.0]));
    }

    #[test]
    fn stack_push_mult_pop() {
        let mut s: Stack<4> = unsafe { std::mem::zeroed() };
        s.init();
        assert!(s.push());
        s.mult(&translate(1.0, 2.0, 3.0));
        assert!(close(xform(s.get(), [0.0, 0.0, 0.0, 1.0]), [1.0, 2.0, 3.0, 1.0]));
        assert!(s.pop());
        assert_eq!(*s.get(), IDENT);
        assert!(!s.pop());
    }

    #[test]
    fn invert_undoes_a_transform() {
        let m = mul(&translate(1.0, 2.0, 3.0), &mul(&rotate(30.0, 1.0, 2.0, 3.0), &scale(2.0, 3.0, 4.0)));
        let i = invert(&m).unwrap();
        let p = xform(&i, xform(&m, [0.5, -1.0, 2.0, 1.0]));
        assert!(close(p, [0.5, -1.0, 2.0, 1.0]), "{p:?}");
    }

    #[test]
    fn normal_matrix_of_a_scale_is_its_inverse() {
        let n = normal_matrix(&scale(2.0, 4.0, 1.0));
        assert_eq!([n[0], n[4], n[8]], [0.5, 0.25, 1.0]);
    }
}
