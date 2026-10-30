//! A raw pointer for the places the graphics code keeps one: the opaque
//! argument a head's callbacks receive (the board that owns the scanout and
//! the GL context they call back into), and the pointers a draw descriptor
//! carries into vertex, texture and plane data for the length of one draw.
//! Each is set up by the code that owns the pointee and used only while it
//! holds the board lock, so no other reference is live meanwhile.

use std::ffi::c_void;
use std::ops::{Deref, DerefMut};

/// A nullable, copyable, unchecked pointer. Dereferencing a null one
/// panics.
#[repr(transparent)]
pub struct P<T: ?Sized>(pub *mut T);

impl<T> P<T> {
    pub const fn null() -> Self {
        P(std::ptr::null_mut())
    }
    pub fn new(r: &mut T) -> Self {
        P(r as *mut T)
    }
    pub fn from_ref(r: &T) -> Self {
        P(r as *const T as *mut T)
    }
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
    /// The pointee, for as long as the caller says: the owner keeps it alive
    /// and unaliased for that long (module doc).
    #[allow(clippy::mut_from_ref)]
    pub fn get<'a>(&self) -> &'a mut T {
        assert!(!self.0.is_null(), "null pointer dereference");
        unsafe { &mut *self.0 }
    }
    pub fn as_opt<'a>(&self) -> Option<&'a mut T> {
        if self.0.is_null() { None } else { Some(unsafe { &mut *self.0 }) }
    }
    pub fn cast<U>(&self) -> P<U> {
        P(self.0 as *mut U)
    }
    pub fn opaque(&self) -> Opaque {
        P(self.0 as *mut c_void)
    }
}

impl<T: ?Sized> Clone for P<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: ?Sized> Copy for P<T> {}
impl<T> Default for P<T> {
    fn default() -> Self {
        P::null()
    }
}
impl<T: ?Sized> PartialEq for P<T> {
    fn eq(&self, o: &Self) -> bool {
        std::ptr::addr_eq(self.0, o.0)
    }
}
impl<T: ?Sized> std::fmt::Debug for P<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "P({:p})", self.0)
    }
}
// Used only under the board lock (module doc).
unsafe impl<T: ?Sized> Send for P<T> {}
unsafe impl<T: ?Sized> Sync for P<T> {}

impl<T> Deref for P<T> {
    type Target = T;
    fn deref(&self) -> &T {
        assert!(!self.0.is_null(), "null pointer dereference");
        unsafe { &*self.0 }
    }
}
impl<T> DerefMut for P<T> {
    fn deref_mut(&mut self) -> &mut T {
        assert!(!self.0.is_null(), "null pointer dereference");
        unsafe { &mut *self.0 }
    }
}

/// `void *opaque`.
pub type Opaque = P<c_void>;

/// The `opaque` a callback was registered with, back as its type.
pub fn from_opaque<'a, T>(o: Opaque) -> &'a mut T {
    o.cast::<T>().get()
}
