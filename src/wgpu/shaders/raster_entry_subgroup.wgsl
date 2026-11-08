// The raster pass's entry point with subgroup operations: the lane is the
// invocation's place in its subgroup, after the subgroups before it, so
// lanes 4k to 4k + 3 are one hardware quad whatever order the device
// gives the workgroup's invocations to its subgroups. Subgroups hold a
// multiple of four invocations.

@compute @workgroup_size(64)
fn raster(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(subgroup_id) sg: u32,
    @builtin(subgroup_size) sg_size: u32,
    @builtin(subgroup_invocation_id) sg_lane: u32,
) {
    raster_tile(wg, sg * sg_size + sg_lane);
}
