# GR2 GL HLE must clip to the view volume and user planes

The GE clips in homogeneous space; the HLE rasterizer only clips to the
window. Vertices keep their clip-space (`Wv::h`) and eye-space (`Wv::e`)
positions and an outcode (`Wv::oc`, computed once in `transform`) for
-w <= x, y, z <= w and the enabled user planes (0x02E / 0x02F, eye space).

Pipeline (gl.rs):
- `GlState::vb` (32 slots): strips / fans / tmeshes cycle through the first
  8, never overwriting what `pv` (last three, swaptmesh swaps two) or
  `vfirst` still refer to; polygons fill the buffer and are drawn whole
  when LOADV|0x065 ends them (chunked from vertex 0 past 32 vertices).
- `gl_poly(indices)`: trivial accept / reject by outcodes; otherwise
  Sutherland-Hodgman on index lists, new vertices in a scratch area
  addressed after the buffer.
- `raster_poly::<SMOOTH, ZS>`: one convex polygon per call, both edge
  chains walked down from the top vertex, attributes interpolated along
  the chains and across rows, one RE3 span per visible window piece.
  Flat shading = the same walk with the provoking colour.
- `gl_line` clips parametrically; points outside any plane are dropped. GL and IRIS GL share the clip volume; their Z ranges differ only in
the viewport transform, after clipping.

The earlier shortcut dropped any primitive with a vertex behind the eye
(w <= 0). Demos with everything in front of the camera never noticed;
Backseat Driver lost its whole road and terrain (big polygons running under
the car) while trees, road markings and the car's hood drew fine. Tests:
`gl_near_plane_clips_ground`, `gl_clip_far_and_user_planes`.

Related HQ2 gotcha: commands whose arguments all go to their own index
(0x02E = enabled, then 0x02E = plane; IRIS GL 0x008) need the repeat taken
as the next argument. Before that fix the second word restarted the
command, so such commands never executed.
