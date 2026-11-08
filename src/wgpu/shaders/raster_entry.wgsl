// The raster pass's entry point where the device has no subgroup
// operations: the lane is the invocation's index in the workgroup.

@compute @workgroup_size(64)
fn raster(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) index: u32) {
    raster_tile(wg, index);
}
