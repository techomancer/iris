# REX3 Rendering Engine Operation

The REX3 (Raster Engine) is a custom ASIC used in the SGI Newport graphics board architecture (such as on the Indy workstation). It functions primarily as a highly capable blitter and line-drawing engine, acting as the interface between the host system and the rest of the graphics subsystem (like the RB2, XMAP9, and RAMDAC).

## Registers

The REX3 chip is controlled via a set of memory-mapped registers, accessible over the GIO64 bus. The base address depends on the GIO64 slot (e.g., `0x1F000000` to `0x1FC00000`).

**Crucial Side Effect (`+ 0x0800` offset):** A major design feature of the REX3 is that the graphics pipeline `GO` command can be implicitly triggered by a memory-mapped write. Accessing *any* register at an offset of `base_address + register_offset + 0x0800` writes the data to the target register AND immediately issues a primitive `GO` command in the same cycle. This avoids requiring a separate write to trigger rasterization.

Below is a detailed breakdown of the REX3 registers grouped by function, including their exact fixed-point memory formats and any specific hardware side effects triggered when reading or writing them.

### 1. Command & Mode Registers
These configure the operational modes and pixel depth logic.
*   `0x0000` **DRAWMODE1**: Pixel data format and logic operations. Sets depth (4, 8, 12, 24-bpp), RGB vs CI mode, plane masks (RGB/CI, Overlay, Popup, Cursor, CID), blend functions, raster operations (LogicOps like SRC, XOR), dithering, and fast-clear enables.
*   `0x0004` **DRAWMODE0**: Draw instruction and addressing mode. Sets opcode (`DRAW`, `READ`, `SCR2SCR`), address mode (`SPAN`, `BLOCK`, `I_LINE`, `F_LINE`, `A_LINE`), hardware iterator setup (`DOSETUP`), pattern enables, stop conditions (`STOPONX`, `STOPONY`), host data overrides, and stipple control.

### 2. Iterator & Coordinate Registers (Fixed-Point Details)
These registers dictate the geometric boundaries of the primitive. Internally, the iterator coordinates are stored in a high-precision two's complement fixed-point format, but the host provides various "views" to optimize different coordinate injection methods (e.g., integers vs floats vs packed arrays).

**Base Internal Format (`16.4(7)`):** The full representation stores 16 bits of integer and 4 bits of fraction. The notation `16.4(7)` indicates that within the 32-bit register, the lowest 7 bits are unused (padding/shifted out), bits 7-10 are the 4 fractional bits, and bits 11-26 are the 16 integer bits (sign-extended).
*   **Full State Views:**
    *   `0x0100` **XSTART**, `0x0104` **YSTART**, `0x0108` **XEND**, `0x010C` **YEND**: Full state (16.4) for context switching or sub-pixel geometry. *Side Effect*: Writing to XSTART also saves the integer portion into the internal `XSAVE` register for carriage-return in BLOCK mode.
*   **Integer-Only Views (16-bit):**
    *   `0x0148` **XSTARTI**: Loads the 16-bit integer portion of XSTART directly.
    *   `0x0110` **XSAVE**: Copy of XSTART integer value, primarily used by the hardware for carriage-return in `BLOCK` addressing mode.
*   **Float/GL Mapping Views (`12.4(7)`):**
    *   `0x0138` **XSTARTF**, `0x013C` **YSTARTF**, `0x0140` **XENDF**, `0x0144` **YENDF**: These registers behave identically to the full state but *force-zero the 4 most significant integer bits*. This accommodates specific GL floating-point to fixed-point conversion biases natively in hardware.
*   **Packed Integer Views (`16,16`):**
    *   `0x0150` **XYSTARTI**, `0x0154` **XYENDI**, `0x0158` **XSTARTENDI**: These accept two 16-bit integers packed into a single 32-bit word, accelerating block and line definitions from the host by cutting bus transfers in half.

### 3. Relative Coordinates & Screen Masks
*   `0x0114` **XYMOVE**: A packed 16,16 (X, Y) signed offset. Added to `XSTART`/`YSTART` during relative operations like `Scr2Scr` (blitting).
*   `0x1324` **XYWIN**: A packed 16,16 screen X, Y offset. Used for window-relative addressing and coordinate biasing.
*   `0x0028` / `0x002C` **SMASK0X** / **SMASK0Y**: Screenmask 0 minimum and maximum boundaries. Window relative (affected by `XYWIN`).
*   `0x1300` to `0x131C` **SMASK1X - SMASK4Y**: Screenmasks 1-4 boundaries. These are absolute screen coordinates and ignore `XYWIN` offsets.

### 4. Bresenham Parameter Registers
These configure the Bresenham rasterizer. If `DOSETUP` is set in `DRAWMODE0`, the REX3 computes these dynamically in 3 to 15 clock cycles. Otherwise, the host can inject them. They use specific fixed-point offsets:
*   `0x0118` **BRESD**: Bresenham primary decision variable ("d" error term). Format `19.8` (19 integer bits, 8 fraction).
*   `0x011C` **BRESS1**: Antialiased Bresenham "s1" coverage term. Format `2.15`.
*   `0x012C` **BRESS2**: Antialiased Bresenham "s2" coverage term. Format `18.8`.
*   `0x0120` **BRESOCTINC1**: Octant and `incr1` increment value. Encodes 3 bits of octant direction alongside a `17.3` step value.
*   `0x0124` **BRESRNDINC2**: Octant rounding mode and `incr2` increment value. Includes 8 bits of rounding rules and an `18.3` step value.
*   `0x0128` **BRESE1**: Bresenham "e1" constant (minor slope) for antialiased line draws. Format `1.15`.

### 5. Color & Shading Registers
REX3 handles colors as multi-part values to enable highly precise, drift-free Gouraud shading along a 1024-pixel span.
*   **Color State (`o12.11` / `o8.11`):**
    *   `0x0200` **COLORRED**, `0x0204` **COLORALPHA**, `0x0208` **COLORGRN**, `0x020C` **COLORBLUE**: The current component values. They store an overflow bit (`o`), an integer (8 or 12 bits depending on depth), and 11 fractional bits. For CI modes, Red acts as the color index up to 12 bits (`o12.11`).
    *   *Side Effect:* Writing to any of these updates both the base color value and the `current` iterating color used by the pipeline. During span/line rendering, the `current` color increments by the slopes, and at the end of a line or block-row, it is reloaded from the base color.
*   **Packed and Convenience Views:**
    *   `0x0224` **COLORI**: Packed 24-bit integer color register. *Side Effect:* Writing to this register populates the RGB state integers but forcefully zeros out their fractional bits. Required for setting flat colors before screen-to-screen blits to avoid fractional rounding drift.
    *   `0x0228` **COLORX**: Color index shade. *Side Effect:* Zeros the overflow bit automatically.
*   **Color Slopes:**
    *   `0x0210` **SLOPERED**, `0x0214` **SLOPEALPHA**, `0x0218` **SLOPEGRN**, `0x021C` **SLOPEBLUE**: Color delta per pixel for shading. Written format is `s(7)12.11` (sign bit, 7 unused, 12 integer, 11 fraction). Can be written as signed magnitude and the hardware converts it to two's complement.
*   **Special Colors:**
    *   `0x0018` **COLORBACK**: Background color used for opaque stippling (drawing "off" bits as solid) or as a constant destination blend color.
    *   `0x001C` **COLORVRAM**: Fastclear color. Injected into VRAM during hardware blockfill passes.

### 6. Stipple, Pattern, & Host Data Registers
*   `0x0008` **LSMODE**: Line stipple mode register. Sets length (`LSLENGTH`) and repeat factor (`LSREPEAT`).
*   `0x000C` **LSPATTERN**: 32-bit line stipple bitmask (MSB = first pixel).
*   `0x0014` **ZPATTERN**: 32-bit area pattern bitmask.
*   `0x003C` **LSSAVE** / `0x0038` **LSRESTORE**: *Side Effect:* Writing to `LSSAVE` (no data payload required) pushes the current `LSPATTERN` and iteration count to shadow registers. Writing to `LSRESTORE` pops them back. Crucial for restoring stipple context when switching between overlapping polygons.
*   `0x0130` **AWEIGHT0** / `0x0134` **AWEIGHT1**: 16x4-bit antialiased line weight table. Represents coverage LUT mappings.
*   `0x0230` **HOSTRW0** / `0x0234` **HOSTRW1**: Data portals for Host PIO (Programmed I/O) and DMA transfers to/from the framebuffer.
*   `0x0220` **WRMASK**: Write mask bitfield controlling which exact planes are modified during a write cycle.

### 7. Configuration, Command Triggers, and Status
*   `0x1330` **CONFIG**: System setup (GIO32 vs GIO64 modes, bus width, FIFO watermarks/trigger depths, VRAM refresh pacing).
*   `0x1338` **STATUS** / `0x133C` **USER_STATUS**: FIFO and busy-state queries. *Side Effect:* Reading `STATUS` acknowledges and clears hardware interrupt bits (VRINT, BFIFO_INT, GFIFO_INT). `USER_STATUS` maps to the exact same data but is non-destructive (does not clear interrupts).
*   `0x0024` **STALL0** / `0x132C` **STALL1**: *Side Effect:* Issuing a write to these addresses stalls the GIO bus/GFIFO output immediately until the graphics rasterizer pipe is entirely idle.
*   `0x0030` **SETUP**: *Side Effect:* Triggers Bresenham calculation for spans/lines *without* firing an iteration (ignoring `DOSETUP`).
*   `0x0034` **STEPZ**: *Side Effect:* Enables `ZPATTERN` test failure simulation for a single iteration (the current pixel).

### 8. Display Control Bus (DCB) Registers
REX3 orchestrates external Display Control Bus hardware (XMAP9, VC2, CMAP RAMDACs) via a bridged internal bus.
*   `0x0238` **DCBMODE**: Defines slave targeting, data width (1 to 4 bytes), synchronous vs asynchronous acknowledges, and wait-state pacing.
*   `0x0240` **DCBDATA0** / `0x0244` **DCBDATA1**: Portals transmitting read/write payloads over the DCB.
*   `0x1340` **DCBRESET**: *Side Effect:* Writing to this abruptly resets the DCB state machine and forcefully flushes the BFIFO (Display Bus FIFO).

### Control Register Bit Definitions

#### DRAWMODE0 (Draw Instruction and Mode)
*   `1:0 OPCODE`: 00=NOOP, 01=READ, 10=DRAW, 11=SCR2SCR.
*   `4:2 ADRMODE`: 000=SPAN, 001=BLOCK, 010=I_LINE, 011=F_LINE, 100=A_LINE.
*   `5 DOSETUP`: Enables iterator setup in hardware for spans/blocks/lines.
*   `6 COLORHOST`: RGB/CI draw source: 0=DDAs, 1=HOSTRW register.
*   `7 ALPHAHOST`: Alpha draw source: 0=DDA, 1=HOSTRW register.
*   `8 STOPONX` / `9 STOPONY`: Specifies execution tests to stop when X or Y endpoint is reached.
*   `10 SKIPFIRST` / `11 SKIPLAST`: Disables drawing of the start/end points (lines only).
*   `12 ENZPATTERN`: Patterning enable.
*   `13 ENLSPATTERN`: Line stipple pattern enable.
*   `14 LSADVLAST`: Enables stipple advance at the end of a line.
*   `15 LENGTH32`: Limits primitive to 32 pixels.
*   `16 ZPOPAQUE` / `17 LSOPAQUE`: Enables opaque (vs transparent) mode for Z/LS patterns.
*   `18 SHADE`: Enables linear shader DDAs.
*   `19 LRONLY`: Aborts primitive if XSTART > XEND.
*   `20 XYOFFSET`: Adds XYMOVE to XSTART, YSTART.
*   `21 CICLAMP`: Enables CI shader clamping.
*   `22 ENDPTFILTER`: Enables hardware endpoint filtering for A_LINE.
*   `23 YSTRIDE`: Enables Y-axis increment/decrement by 2.

#### DRAWMODE1 (Pixel Data Format and Logic Ops)
*   `2:0 PLANES`: Framebuffer planes to write (001=RGB/CI, 010=RGBA, 100=OLAY, 101=PUP, 110=CID).
*   `4:3 DRAWDEPTH`: Drawn depth of VRAM (00=4bpp, 01=8bpp, 10=12bpp, 11=24bpp).
*   `5 DBLSRC`: Double-buffer mode read source.
*   `6 YFLIP`: Origin mapping (0=upper-left, 1=lower-left).
*   `7 RWPACKED`: Enables pixel packing for HOSTRW.
*   `9:8 HOSTDEPTH`: Depth of host transfers (00=4b, 01=8b, 10=12b, 11=32b).
*   `10 RWDOUBLE`: Enables double word 64-bit transfers.
*   `11 SWAPENDIAN`: Reverses byte ordering within packed data.
*   `14:12 COMPARE`: Color compare operator (src > dest, src = dest, src < dest).
*   `15 RGBMODE`: Selects RGB (vs CI) mode for shade, round, and dither.
*   `16 DITHER`: Enables dithering.
*   `17 FASTCLEAR`: Enables fast-clear write mode.
*   `18 BLEND`: Enables blend function.
*   `21:19 SFACTOR` / `24:22 DFACTOR`: Source and Destination blending factors (e.g., ZERO, ONE, SRC_COLOR, SRC_ALPHA).
*   `25 BACKBLEND`: Enables COLORBACK as the destination color for blending.
*   `26 PREFETCH`: Enables host pixel prefetch for reads.
*   `31:28 LOGICOP`: Raster operation (e.g., 0000=ZERO, 0011=SRC, 0110=XOR).

#### LSMODE (Line Stipple Mode)
*   `7:0 LSRCOUNT`: Current repeat down counter.
*   `15:8 LSREPEAT`: Line stipple repeat factor.
*   `23:16 LSRCNTSAVE`: Copy of LSRCOUNT.
*   `27:24 LSLENGTH`: Length of LSPATTERN (from 17 to 32).

#### CLIPMODE (Clipping and CID)
*   `4:0 ENSMASK`: Individual enables for SMASK0 through SMASK4.
*   `12:9 CIDMATCH`: Mask of permitted two-bit CIDs: bit N permits writes to CID N.

#### STATUS / USER_STATUS
*   `2:0 VERSION`: Revision code.
*   `3 GFXBUSY`: Graphics pipeline not idle.
*   `4 BACKBUSY`: Backend pipeline not idle.
*   `5 VRINT`: Vertical retrace interrupt.
*   `6 VIDEOINT`: Video option interrupt.
*   `12:7 GFIFOLEVEL`: Current GFIFO level.
*   `17:13 BFIFOLEVEL`: Current BFIFO level.
*   `18 BFIFO_INT` / `19 GFIFO_INT`: FIFO above depth interrupts.

#### CONFIG (Configuration)
*   `0 GIO32MODE`: GIO32 vs GIO64 protocol format.
*   `1 BUSWIDTH`: Physical width of GIO64 bus (1=64 bits, 0=32 bits).
*   `2 EXTREGXCVR`: External registered transceivers present.
*   `6:3 BFIFODEPTH` / `12:8 GFIFODEPTH`: High/low trigger depths for FIFOs.
*   `16:14 TIMEOUT`: GIO bus timeout interval.
*   `19:17 VREFRESH`: Number of VRAM refresh cycles per transfer.

#### DCBMODE (Display Control Bus Mode)
*   `1:0 DATAWIDTH`: Data width for DCBDATA operations (4, 1, 2, or 3 bytes).
*   `2 ENDATAPACK`: Unpacked vs Packed data transfers.
*   `10:7 DCBADDR`: Slave device address (e.g., XMAP9, VC2, CMAP).
*   `11 ENSYNCACK` / `12 ENASYNCACK`: Acknowledge protocols.

## Coordinate System and Biasing

The REX3 utilizes a heavily biased 16-bit physical coordinate system internally, which is primarily designed to facilitate rapid floating-point to fixed-point transformations natively in the graphics library (GL). 

### Physical Memory Space
The framebuffer supports a region up to 1280 x 1024 pixels. However, physically, this visible region is *not* located at coordinates (0,0). 
Instead, the physical coordinates for the displayable space are offset:
*   **Upper-Left Corner:** `4096, 4096` (0x1000, 0x1000)
*   **Lower-Right Corner:** `5375, 5119` (1279 + 4096, 1023 + 4096)
*   An additional "off-screen" section, 64 pixels wide, exists to the right, extending the maximum X coordinate to `5439`.

Because coordinates are represented internally as 16-bit signed integers (via the `16.4(7)` fixed-point format), the chip provides an addressability of -32768 to +32767. Placing the origin at `4096, 4096` ensures that most typical negative window coordinates do not underflow into negative absolute memory space, removing the immediate need for the host to clip off-screen geometry purely to prevent memory faults.

### Window Relative Origins and XYWIN
Windowing systems (like X11) traditionally treat the upper-left of a window as `(0, 0)`. To achieve this transparently, REX3 relies heavily on the `XYWIN` register.

*   **X11 Coordinate Mapping:** For pure X11 operations, the `XYWIN` register is permanently loaded with the `4096, 4096` bias. All incoming host coordinates are treated as relative to `(0,0)`, and the REX3 hardware adds `XYWIN` automatically to calculate the correct physical memory address.
*   **GL Transformation Biasing:** SGI's GL implementation uses math tricks (magic numbers in IEEE-754 mantissas) to very rapidly convert floating-point vertices to integer coordinates. This process natively introduces a bias. REX3 is hardwired to assume this GL bias naturally maps to `4096`. 
    *   If the GL needs to draw to a window located at screen offset `xrel, yrel`, it programs `XYWIN` with the window's offset relative to the screen origin.
    *   If a specific GL implementation uses a bias *other* than 4096, `XYWIN` must be explicitly compensated: `XYWIN_X = xrel + (GL_bias - 4096)`.

### Coordinate Y-Flipping
To accommodate GL's standard "Y points UP" coordinate system against X11's "Y points DOWN" system, the `DRAWMODE1` register contains the `YFLIP` bit.
*   `YFLIP = 0` (X11): The Y axis increases downwards. Origins are referenced to the upper-left.
*   `YFLIP = 1` (GL): The Y axis increases upwards. Window and screen origins are conceptually referenced from the **lower-left**. 
    *   When `YFLIP` is active, the REX3 hardware negates the incoming Y coordinates (`YSTART`, `YEND`, `YMOVE`, `SMASK0Y`) and effectively *subtracts* them from the `XYWIN` bias. 
    *   Consequently, when in `YFLIP` mode, `XYWIN` must be configured with a special calculation to place the origin at the bottom of the window: `XYWIN_Y = (Screen_Height_In_Pixels - 1 - yrel) + 4096`.

## Clipping and Masking

Framebuffer values are conditionally written as a function of multiple independent clipping pipelines: Sector Clipping, Screen Masking, and CID Masking. 

### Sector Clipping (Absolute Boundaries)
This is an implicit hardware function. REX3 tracks the physical boundaries of the VRAM memory space. Any write operations that generate addresses outside the legal VRAM boundary limits (e.g., `< 4096` or `> 5439` in X) are immediately culled. This allows the host to send grossly oversized primitives without fear of memory corruption. *Reads* are never sector-clipped to maintain simplified DMA behavior.

### Screen Masking (Rectangular Clipping)
Clipping to complex window shapes (e.g., overlapping windows) is accelerated via five hardware screen masks (`SMASK0` through `SMASK4`), selectively enabled by the `CLIPMODE` register.

*   **SMASK0 (Window Relative):** This mask is dedicated to GL clipping. It is affected by the `XYWIN` offset and the `YFLIP` bit. Coordinates programmed into `SMASK0X`/`SMASK0Y` must be biased in the exact same manner as incoming drawing coordinates. 
*   **SMASK1 - SMASK4 (Absolute X11):** These four masks are general-purpose X11 rectangles. Crucially, they are **absolute and unaffected by `XYWIN` or `YFLIP`**. The host software must manually add the `4096, 4096` physical screen bias to the boundary coordinates before loading them into `SMASK1`-`4`.

**Masking Logic:** A pixel is only drawn if it satisfies the following boolean condition:
```
{ (Inside any enabled SMASK1-4) OR (All SMASKS1-4 are disabled) } 
AND 
{ (Inside SMASK0) OR (SMASK0 is disabled) }
```

### CID Masking (Per-Pixel Window IDs)
For complex, non-rectangular window clipping, REX3 relies on Coordinate ID (CID) planes. The frame buffer dedicates physical bitplanes specifically to store a "Window ID" for every pixel on the screen.

The `CIDMATCH` field in `CLIPMODE` is a four-bit mask: bit N permits writes to pixels whose two-bit CID is N. For example, `0010` permits CID `01`. The mask `1111` permits all CIDs; `0000` rejects all writes. Popup bits are separate and do not participate in this check. Drawing and screen-to-screen copies check the destination CID; framebuffer reads do not check CID. See the [REX3 specification](rex3.pdf), table 16 and section 3.3.

## Drawing Modes

The REX3 executes primitives based on the `ADDRMODE` and the operation command (Draw, Read, Scr2Scr) in `DrawMode0`.

### Blocks
A Block is a rectangular area bounded by the start and end coordinates.
*   **Operation**: The REX3 iterates from `ystart` to `yend`, and for each scanline, from `xstart` to `xend`.
*   **Flags**: `STOPONX` and `STOPONY` determine whether the engine stops rendering at the end of the line or the end of the block.
*   **Y-Stride**: If `YSTRIDE` is set in `DrawMode0`, the iteration behavior can account for stride parameters, useful for handling bitmaps with varying pitches.
*   **Host Data**: If `COLORHOST` is set, the REX3 halts drawing and waits for the host to write pixel data to the `hostrw` port.

### Screen-to-Screen (Scr2Scr) Move
A Screen-to-Screen copy is initiated by setting the operation command to `Scr2Scr` with an `ADDRMODE` of Block or Span.
*   **Source**: The `XSTART`/`YSTART` and `XEND`/`YEND` coordinates define the boundaries of the **source** area.
*   **Destination**: The `XYMOVE` register specifies a signed offset to the **destination** area (unlike older SGI chips like REX1 where it offset the source).
*   **Side Effects**: The offset is applied with respect to the window origin and is therefore affected by the `YFLIP` bit. When performing this copy, the `XYOFFSET` bit in `DRAWMODE0` must be explicitly set to `0`. Right-to-Left spans are supported so the host can order operations carefully to prevent overlapping copies from destroying source data.

### Spans
Spans are horizontal lines, typically used for rendering filled polygons (triangle rasterization is done on the host, sending horizontal spans to the REX3).
*   **Operation**: Rasterizes pixels from `xstart` to `xend` along a constant `ystart`.
*   **Interpolation**: The engine applies color slopes (`slopered`, `slopegrn`, etc.) to smoothly shade the span.

### Lines
Lines are rasterized using the Bresenham algorithm.
*   **Integer Lines (`ILINE`)**: Connect integer coordinates.
*   **Fractional Lines (`FLINE`)**: Connect coordinates with sub-pixel precision.
*   **Antialiased Lines (`AFLINE`)**: Connect sub-pixel coordinates while utilizing the `aweight` (alpha weight) table to modulate alpha/coverage values for adjacent pixels, softening the edges of the line.

## Bresenham Parameters and Quadrants

When drawing lines, REX3 relies on Bresenham parameters to determine which pixels to plot. If `DOSETUP` is set in `DrawMode0`, the REX3 calculates these parameters in hardware from the provided start and end coordinates. If not, the host must pre-calculate and provide them.

### Parameters
*   **Octant and Increments (`bresoctinc1`, `bresrndinc2`)**:
    *   Determines the primary direction of the line. The slope's absolute value dictates the major axis (`x_major` or `y_major`).
    *   `incrx1`, `incry1`: Increment applied on a "straight" step (error term doesn't roll over).
    *   `incrx2`, `incry2`: Increment applied on a "diagonal" step (error term rolls over).
    *   For an X-major line going down-right, `incrx1 = 1`, `incry1 = 0`, `incrx2 = 1`, `incry2 = 1`.
*   **Error Variables (`bresd`, `bress1`, `bress2`, `brese1`)**:
    *   `dx` and `dy` are the absolute differences.
    *   `e1` (Major axis diff) and `e2` (Minor axis diff).
    *   `bresd`: The initial decision variable (error term).
    *   `bress1`: The increment to the error term for a straight step (`e1 * 2`).
    *   `bress2`: The increment to the error term for a diagonal step (`(e1 - e2) * 2`).

### Calculation and Utilization
During the setup phase, the hardware computes the octant based on the signs of dx and dy. During rasterization, at each step along the major axis, the engine checks the sign of the decision variable `bresd`.
1.  If `bresd` < 0: The engine takes a step along the major axis only (using `incrx1`, `incry1`) and adds `bress1` to `bresd`.
2.  If `bresd` >= 0: The engine takes a diagonal step (using `incrx2`, `incry2`) and adds `bress2` to `bresd`.

## Color Interpolation

The REX3 supports Gouraud shading across spans and lines.
*   **State Setup**: The host writes the initial starting color (11-bit fractional precision) to `colorred`, `colorgrn`, `colorblue`, and `coloralpha`. It also calculates the per-pixel change and writes this to `slopered`, `slopegrn`, `slopeblue`, and `slopealpha`.
*   **Execution**: At every pixel rendered along the span or line, the pixel is drawn using the integer portion of the color registers. Following the draw, the slopes are added to their respective color registers.
*   **Fractions**: The hardware strictly maintains fractional values, ensuring colors do not drift significantly across large spans. For `Scr2Scr` block blitting in RGB mode, these fractional parts must be explicitly zeroed to prevent unwanted color drifting.

## Logical Layout of Pixel Data in VRAM

From a software developer's perspective, VRAM is accessed linearly but structured as parallel planes. The interpretation of these planes depends on the configuration of `DrawMode1` and the XMAP9 (display generator) settings.

### Planes
The frame buffer memory (controlled by RB2 chips) contains distinct functional planes:
1.  **RGB/CI Planes**: The primary display memory, 8 or 24 bits deep.
2.  **Overlay / Underlay Planes**: Separate bitplanes used for overlaying UI elements without overwriting the primary image.
3.  **Popup Planes**: Used for pop-up menus.
4.  **CID Planes**: Store the window ID per pixel, used for clipping.

### Data Depth and Packing
*   `DRAWDEPTH` dictates how the REX3 addresses memory. It can be 4, 8, 12, or 24 bpp.
    *   **8bpp**: Typically used for Color Index (CI) modes, where the 8-bit value indexes into the CMAP (Color Map / Palette). It can also be a packed RGB mode.
    *   **24bpp**: Truecolor RGB mode (8 bits each for Red, Green, and Blue).
    *   **12bpp/4bpp**: Sub-packed modes for specific underlay/overlay configurations or lower-bandwidth CI.
*   `HOSTDEPTH` and `RWPACKED` dictate how the CPU writes data to the `hostrw` port.
    *   If `RWPACKED` is set and `HOSTDEPTH` is 8 while `DRAWDEPTH` is 8, writing a 32-bit word to `hostrw0` will write 4 consecutive 8-bit pixels horizontally.
    *   If `RWPACKED` is clear, a 32-bit write usually corresponds to a single pixel (with the lower 8 bits or 24 bits used based on depth).

### RGB vs CI Mode
*   **CI Mode (`RGBMODE = 0`)**: The color value generated by the REX3 is treated as a single index. Iterating colors simply steps the index.
*   **RGB Mode (`RGBMODE = 1`)**: The color values from `colorred`, `colorgrn`, and `colorblue` are composited together into a single 24-bit word before being written to the frame buffer planes. The engine ensures the RGB components are shifted and masked into their proper locations.
