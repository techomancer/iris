import asyncio
import sys
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("iris-monitor")

class IrisClient:
    def __init__(self, host='127.0.0.1', port=8888):
        self.host = host
        self.port = port
        self.reader = None
        self.writer = None
        self._lock = asyncio.Lock()

    def _reset(self):
        if self.writer:
            try:
                self.writer.close()
            except Exception:
                pass
        self.reader = None
        self.writer = None

    async def connect(self, retries=10, delay=2.0):
        for attempt in range(retries):
            try:
                self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
                # Read initial banner
                await self.read_until_prompt()
                print(f"Connected to IRIS monitor at {self.host}:{self.port}", file=sys.stderr)
                return
            except Exception as e:
                self._reset()
                print(f"IRIS connect attempt {attempt+1}/{retries} failed: {e}", file=sys.stderr)
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
        raise ConnectionError(f"Could not connect to IRIS monitor at {self.host}:{self.port} after {retries} attempts")

    async def read_until_prompt(self):
        data = b""
        while True:
            chunk = await self.reader.read(4096)
            if not chunk:
                raise ConnectionError("IRIS monitor closed connection")
            data += chunk
            if data.endswith(b"> "):
                return data[:-2].decode('utf-8', errors='replace').strip()

    async def drain_stale(self):
        """Discard anything already waiting on the socket: output that belongs
        to no command we are waiting for. Reading it as the reply to the next
        command is how replies used to come back one command late."""
        while True:
            try:
                chunk = await asyncio.wait_for(self.reader.read(4096), timeout=0.02)
            except asyncio.TimeoutError:
                return
            if not chunk:
                raise ConnectionError("IRIS monitor closed connection")
            print(f"IRIS: discarded {len(chunk)} stale bytes", file=sys.stderr)

    async def send_command(self, cmd):
        async with self._lock:
            for attempt in range(2):
                try:
                    if not self.writer or self.writer.is_closing():
                        await self.connect()
                    await self.drain_stale()
                    self.writer.write(f"{cmd}\n".encode())
                    await self.writer.drain()
                    return await self.read_until_prompt()
                except asyncio.CancelledError:
                    # The call was abandoned with its reply possibly still in
                    # flight: drop the connection so that reply dies with it.
                    self._reset()
                    raise
                except Exception as e:
                    print(f"IRIS command failed (attempt {attempt+1}): {e}", file=sys.stderr)
                    self._reset()
                    if attempt == 0:
                        await self.connect()
            raise ConnectionError("Failed to send command after reconnect")

client = IrisClient()

@mcp.tool()
async def run_command(command: str) -> str:
    """Run a raw command on the IRIS monitor.
    
    Use this for commands not explicitly exposed as tools.
    Common commands: 'tlb dump', 'u' (undo), 'bp list'.
    """
    return await client.send_command(command)

@mcp.tool()
async def read_memory(address: str, count: int = 1, size: str = "w") -> str:
    """Read memory from IRIS.

    Args:
        address: Address, symbol or expression (e.g. '0x80001000', 'sp+16')
        count: Number of items to read
        size: Item size: 'b' (byte), 'h' (half), 'w' (word, default), 'd' (doubleword)
    """
    return await client.send_command(f"mem {address} {count} {size}")

@mcp.tool()
async def read_string(address: str, max_len: int = 256) -> str:
    """Read a null-terminated string from memory."""
    return await client.send_command(f"ms {address} {max_len}")

@mcp.tool()
async def write_memory(address: str, value: str, size: str = "w") -> str:
    """Write value to memory.
    
    Args:
        address: Hex address
        value: Value to write (hex or dec)
        size: Size 'b' (byte), 'h' (half), 'w' (word), 'd' (double)
    """
    return await client.send_command(f"mw {address} {value} {size}")

@mcp.tool()
async def step(count: int = 1) -> str:
    """Step execution by N instructions. Blocks until complete."""
    return await client.send_command(f"step block {count}")

@mcp.tool()
async def next_instruction(count: int = 1) -> str:
    """Step over function calls (next). Blocks until complete."""
    return await client.send_command(f"next block {count}")

@mcp.tool()
async def get_registers() -> str:
    """Dump general purpose registers."""
    return await client.send_command("regs")

@mcp.tool()
async def disassemble(address: str, count: int = 10) -> str:
    """Disassemble instructions.
    
    Args:
        address: Start address (hex) or register name (e.g. 'pc')
        count: Number of instructions
    """
    return await client.send_command(f"dis {address} {count}")

@mcp.tool()
async def backtrace(frames: int = 10) -> str:
    """Show stack backtrace."""
    return await client.send_command(f"bt {frames}")

@mcp.tool()
async def add_breakpoint(address: str, kind: str = "pc", condition: str = None) -> str:
    """Add a breakpoint. Works whether the CPU is running or stopped.

    Args:
        address: Address, symbol, or symbol:N when several map files define the
            name (see symbol_copies)
        kind: 'pc' (execution), 'read', 'write', 'fetch', 'pread', 'pwrite', 'pfetch'
        condition: Optional expression; the breakpoint stops only when it is
            non-zero, e.g. "a0 == 3 && (c0_cause & 0x7c) == 8". Registers by
            name (a0, sp, c0_badvaddr, ...); unknown names are rejected.
    """
    cmd = f"bp add {address} {kind}"
    if condition:
        cmd += f" if {condition}"
    return await client.send_command(cmd)

@mcp.tool()
async def remove_breakpoint(id: int) -> str:
    """Remove a breakpoint by ID."""
    return await client.send_command(f"bp del {id}")

@mcp.tool()
async def list_breakpoints() -> str:
    """List all breakpoints."""
    return await client.send_command("bp list")

@mcp.tool()
async def get_status() -> str:
    """CPU running or stopped, the PC when stopped, and why it last stopped
    (which breakpoint, with its condition and hit count). Never blocks."""
    return await client.send_command("status")

@mcp.tool()
async def wait_for_stop(timeout_ms: int = 10000) -> str:
    """Wait until the CPU stops (a breakpoint or exception catch) or the
    timeout passes, then report where and why. Use after start_cpu, or after
    triggering guest activity some other way (telnet, the console)."""
    return await client.send_command(f"wait {timeout_ms}")

@mcp.tool()
async def start_cpu() -> str:
    """Start (resume) the CPU without waiting. Pair with wait_for_stop."""
    return await client.send_command("start")

@mcp.tool()
async def stop_cpu() -> str:
    """Stop the CPU."""
    return await client.send_command("stop")

@mcp.tool()
async def continue_execution(until: str = None) -> str:
    """Continue execution (run). Blocks until a breakpoint is hit or execution
    stops, with no timeout: prefer start_cpu + wait_for_stop.

    Args:
        until: Optional address to run until (temporary breakpoint)
    """
    if until:
        return await client.send_command(f"run block {until}")
    return await client.send_command("run block")

@mcp.tool()
async def finish_function() -> str:
    """Run until the current function returns. Blocks until complete."""
    return await client.send_command("finish block")

@mcp.tool()
async def read_cop0() -> str:
    """Dump CP0 registers."""
    return await client.send_command("cop0")

@mcp.tool()
async def read_cop1() -> str:
    """Dump CP1 (FPU) registers."""
    return await client.send_command("cop1")

@mcp.tool()
async def translate_address(address: str) -> str:
    """Translate virtual address to physical."""
    return await client.send_command(f"translate {address}")

@mcp.tool()
async def dump_tlb() -> str:
    """Dump TLB entries."""
    return await client.send_command("tlb dump")

@mcp.tool()
async def lookup_symbol(name_or_addr: str) -> str:
    """Look up an address (symbol+offset) or a name. A name lists every copy
    with the map file it came from; pick one elsewhere as name:N."""
    return await client.send_command(f"sym {name_or_addr}")

@mcp.tool()
async def read_eframe(address: str, layout: str = "") -> str:
    """Decode an IRIX exception frame (e.g. the 'ep' in a panic message, or
    s0/a0 at dumpsys): registers, cause/sr, EPC, and a backtrace of the
    interrupted context.

    Args:
        address: Frame address or expression
        layout: '' (from Status.KX), 'o32' (IRIX 5.3, 6.5 on IP22) or '64'
    """
    return await client.send_command(f"eframe {address} {layout}".strip())

@mcp.tool()
async def kernel_log(size: int = 0) -> str:
    """The IRIX kernel console log (putbuf): panic messages and kernel
    printfs. Needs the kernel's symbols loaded. size: only for kernels whose
    putbuf is a fixed array (no putbufsz symbol)."""
    return await client.send_command(f"putbuf {size}" if size else "putbuf")

@mcp.tool()
async def undo(count: int = 1) -> str:
    """Undo the last N instructions."""
    return await client.send_command(f"undo {count}")

@mcp.tool()
async def traceback(count: int = 10) -> str:
    """Show execution traceback (history of executed instructions)."""
    return await client.send_command(f"dt {count}")

if __name__ == "__main__":
    mcp.run()
