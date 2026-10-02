use crate::cpu::mips_core::MipsCore;
use crate::cpu::mips_dis::{self, SymbolTable};

#[derive(Debug, Clone, PartialEq)]
pub enum RegTarget {
    Pc,
    Hi,
    Lo,
    Gpr(u32),
    Cp0(u32),
    Fpr(u32),
}

pub fn parse_reg_target(arg: &str) -> Option<RegTarget> {
    match arg {
        "pc" => return Some(RegTarget::Pc),
        "hi" => return Some(RegTarget::Hi),
        "lo" => return Some(RegTarget::Lo),
        _ => {}
    }

    // GPR symbolic names
    let name = arg.trim_start_matches('$');
    let gpr_idx: u32 = match name {
        "zero" => 0, "at" => 1, "v0" => 2, "v1" => 3,
        "a0" => 4, "a1" => 5, "a2" => 6, "a3" => 7,
        "t0" => 8, "t1" => 9, "t2" => 10, "t3" => 11,
        "t4" => 12, "t5" => 13, "t6" => 14, "t7" => 15,
        "s0" => 16, "s1" => 17, "s2" => 18, "s3" => 19,
        "s4" => 20, "s5" => 21, "s6" => 22, "s7" => 23,
        "t8" => 24, "t9" => 25, "k0" => 26, "k1" => 27,
        "gp" => 28, "sp" => 29, "fp" => 30, "ra" => 31,
        _ => 100,
    };
    if gpr_idx < 32 { return Some(RegTarget::Gpr(gpr_idx)); }

    // Numeric GPR: $N or rN
    if arg.starts_with('$') || arg.starts_with('r') {
        let num_str = name.trim_start_matches('r');
        if let Ok(idx) = num_str.parse::<u32>() {
            if idx < 32 { return Some(RegTarget::Gpr(idx)); }
        }
    }

    // CP0: c0_N or c0_Name
    if let Some(rest) = arg.strip_prefix("c0_") {
        if let Ok(idx) = rest.parse::<u32>() {
            if idx < 32 { return Some(RegTarget::Cp0(idx)); }
        }
        let rest_lc = rest.to_lowercase();
        for i in 0..32u32 {
            if mips_dis::cp0_reg_name(i).to_lowercase() == rest_lc {
                return Some(RegTarget::Cp0(i));
            }
        }
        return None;
    }

    // FPU: fN
    if let Some(rest) = arg.strip_prefix('f') {
        if let Ok(idx) = rest.parse::<u32>() {
            if idx < 32 { return Some(RegTarget::Fpr(idx)); }
        }
    }

    None
}

pub fn read_reg_target(target: &RegTarget, core: &MipsCore) -> u64 {
    match target {
        RegTarget::Pc      => core.pc,
        RegTarget::Hi      => core.hi,
        RegTarget::Lo      => core.lo,
        RegTarget::Gpr(i)  => core.read_gpr(*i),
        RegTarget::Cp0(i)  => core.read_cp0_debug(*i),
        RegTarget::Fpr(i)  => core.fpr[*i as usize],
    }
}

pub fn write_reg_target(target: &RegTarget, core: &mut MipsCore, val: u64) {
    match target {
        RegTarget::Pc      => core.pc = val,
        RegTarget::Hi      => core.hi = val,
        RegTarget::Lo      => core.lo = val,
        RegTarget::Gpr(i)  => core.write_gpr(*i, val),
        RegTarget::Cp0(i)  => core.write_cp0(*i, val),
        RegTarget::Fpr(i)  => core.fpr[*i as usize] = val,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    Neg, Not, BitNot
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryOp {
    Add, Sub, Mul, Div, Rem,
    And, Or, Xor, Shl, Shr,
    Eq, Ne, Lt, Le, Gt, Ge,
    LogAnd, LogOr
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(u64),
    Register(RegTarget),
    Symbol(String),
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
}

impl Expr {
    pub fn eval(&self, core: &MipsCore, symbols: Option<&SymbolTable>) -> Result<u64, String> {
        match self {
            Expr::Literal(val) => Ok(*val),
            Expr::Register(reg) => Ok(read_reg_target(reg, core)),
            Expr::Symbol(name) => {
                if let Some(syms) = symbols {
                    if let Some(addr) = syms.get_addr(name) {
                        // Sign-extend 32-bit kernel addresses
                        if addr <= 0xFFFF_FFFF && (addr & 0x8000_0000) != 0 {
                            Ok(addr | 0xFFFF_FFFF_0000_0000)
                        } else {
                            Ok(addr)
                        }
                    } else {
                        Err(format!("Symbol not found: {}", name))
                    }
                } else {
                    Err(format!("No symbol table available to resolve: {}", name))
                }
            }
            Expr::Unary(op, expr) => {
                let val = expr.eval(core, symbols)?;
                match op {
                    UnaryOp::Neg => Ok((-(val as i64)) as u64),
                    UnaryOp::Not => Ok((val == 0) as u64),
                    UnaryOp::BitNot => Ok(!val),
                }
            }
            Expr::Binary(op, lhs, rhs) => {
                let l = lhs.eval(core, symbols)?;
                let r = rhs.eval(core, symbols)?;
                match op {
                    BinaryOp::Add => Ok(l.wrapping_add(r)),
                    BinaryOp::Sub => Ok(l.wrapping_sub(r)),
                    BinaryOp::Mul => Ok(l.wrapping_mul(r)),
                    BinaryOp::Div => if r == 0 { Err("Division by zero".to_string()) } else { Ok(l.wrapping_div(r)) },
                    BinaryOp::Rem => if r == 0 { Err("Division by zero".to_string()) } else { Ok(l.wrapping_rem(r)) },
                    BinaryOp::And => Ok(l & r),
                    BinaryOp::Or => Ok(l | r),
                    BinaryOp::Xor => Ok(l ^ r),
                    BinaryOp::Shl => Ok(l.wrapping_shl(r as u32)),
                    BinaryOp::Shr => Ok(l.wrapping_shr(r as u32)),
                    BinaryOp::Eq => Ok((l == r) as u64),
                    BinaryOp::Ne => Ok((l != r) as u64),
                    BinaryOp::Lt => Ok((l < r) as u64),
                    BinaryOp::Le => Ok((l <= r) as u64),
                    BinaryOp::Gt => Ok((l > r) as u64),
                    BinaryOp::Ge => Ok((l >= r) as u64),
                    BinaryOp::LogAnd => Ok(((l != 0) && (r != 0)) as u64),
                    BinaryOp::LogOr => Ok(((l != 0) || (r != 0)) as u64),
                }
            }
        }
    }

    pub fn fold(self, symbols: Option<&SymbolTable>) -> Self {
        match self {
            Expr::Symbol(name) => {
                if let Some(syms) = symbols {
                    if let Some(addr) = syms.get_addr(&name) {
                        let val = if addr <= 0xFFFF_FFFF && (addr & 0x8000_0000) != 0 {
                            addr | 0xFFFF_FFFF_0000_0000
                        } else {
                            addr
                        };
                        return Expr::Literal(val);
                    }
                }
                Expr::Symbol(name)
            }
            Expr::Unary(op, expr) => {
                let folded = expr.fold(symbols);
                if let Expr::Literal(val) = folded {
                    let res = match op {
                        UnaryOp::Neg => (-(val as i64)) as u64,
                        UnaryOp::Not => (val == 0) as u64,
                        UnaryOp::BitNot => !val,
                    };
                    Expr::Literal(res)
                } else {
                    Expr::Unary(op, Box::new(folded))
                }
            }
            Expr::Binary(op, lhs, rhs) => {
                let l = lhs.fold(symbols);
                let r = rhs.fold(symbols);
                if let (Expr::Literal(lv), Expr::Literal(rv)) = (&l, &r) {
                    let res = match op {
                        BinaryOp::Add => Some(lv.wrapping_add(*rv)),
                        BinaryOp::Sub => Some(lv.wrapping_sub(*rv)),
                        BinaryOp::Mul => Some(lv.wrapping_mul(*rv)),
                        BinaryOp::Div => if *rv != 0 { Some(lv.wrapping_div(*rv)) } else { None },
                        BinaryOp::Rem => if *rv != 0 { Some(lv.wrapping_rem(*rv)) } else { None },
                        BinaryOp::And => Some(lv & rv),
                        BinaryOp::Or => Some(lv | rv),
                        BinaryOp::Xor => Some(lv ^ rv),
                        BinaryOp::Shl => Some(lv.wrapping_shl(*rv as u32)),
                        BinaryOp::Shr => Some(lv.wrapping_shr(*rv as u32)),
                        BinaryOp::Eq => Some((*lv == *rv) as u64),
                        BinaryOp::Ne => Some((*lv != *rv) as u64),
                        BinaryOp::Lt => Some((*lv < *rv) as u64),
                        BinaryOp::Le => Some((*lv <= *rv) as u64),
                        BinaryOp::Gt => Some((*lv > *rv) as u64),
                        BinaryOp::Ge => Some((*lv >= *rv) as u64),
                        BinaryOp::LogAnd => Some(((*lv != 0) && (*rv != 0)) as u64),
                        BinaryOp::LogOr => Some(((*lv != 0) || (*rv != 0)) as u64),
                    };
                    if let Some(val) = res {
                        return Expr::Literal(val);
                    }
                }
                Expr::Binary(op, Box::new(l), Box::new(r))
            }
            _ => self,
        }
    }

    pub fn parse(input: &str) -> Result<Self, String> {
        let tokens = tokenize(input)?;
        let mut parser = Parser::new(tokens);
        parser.parse_expr(0)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Literal(u64),
    Identifier(String),
    Op(String),
    LParen,
    RParen,
}

fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else if c.is_digit(10) {
            let mut num_str = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_digit(16) || c == 'x' || c == 'X' {
                    num_str.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
            let val = if num_str.starts_with("0x") || num_str.starts_with("0X") {
                let hex_digits = &num_str[2..];
                let raw = u64::from_str_radix(hex_digits, 16)
                    .map_err(|_| format!("Invalid number: {}", num_str))?;
                // Sign-extend if value fits in 32 bits with bit 31 set,
                // unless the user wrote more than 8 hex digits (explicit width).
                if hex_digits.len() <= 8 && raw & 0x8000_0000 != 0 && raw <= 0xFFFF_FFFF {
                    raw | 0xFFFF_FFFF_0000_0000
                } else {
                    raw
                }
            } else if num_str.starts_with("0") && num_str.len() > 1 {
                u64::from_str_radix(&num_str[1..], 8)
                    .map_err(|_| format!("Invalid number: {}", num_str))?
            } else {
                let raw = num_str.parse::<u64>()
                    .map_err(|_| format!("Invalid number: {}", num_str))?;
                // Sign-extend 32-bit decimal literals with bit 31 set.
                if raw & 0x8000_0000 != 0 && raw <= 0xFFFF_FFFF {
                    raw | 0xFFFF_FFFF_0000_0000
                } else {
                    raw
                }
            };
            tokens.push(Token::Literal(val));
        } else if c.is_alphabetic() || c == '_' || c == '$' {
            let mut ident = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_alphanumeric() || c == '_' || c == '$' {
                    ident.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
            tokens.push(Token::Identifier(ident));
        } else {
            match c {
                '(' => { chars.next(); tokens.push(Token::LParen); }
                ')' => { chars.next(); tokens.push(Token::RParen); }
                _ => {
                    // Operators
                    let mut op = String::new();
                    op.push(c);
                    chars.next();
                    if let Some(&next) = chars.peek() {
                        let two_char = format!("{}{}", c, next);
                        if matches!(two_char.as_str(), "<<" | ">>" | "==" | "!=" | "<=" | ">=" | "&&" | "||") {
                            op = two_char;
                            chars.next();
                        }
                    }
                    tokens.push(Token::Op(op));
                }
            }
        }
    }
    Ok(tokens)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn consume(&mut self) -> Option<Token> {
        if self.pos < self.tokens.len() {
            let t = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(t)
        } else {
            None
        }
    }

    fn parse_expr(&mut self, min_prec: u8) -> Result<Expr, String> {
        let mut lhs = self.parse_atom()?;

        while let Some(Token::Op(op_str)) = self.peek() {
            let (prec, assoc_right) = match op_str.as_str() {
                "||" => (1, false),
                "&&" => (2, false),
                "|" => (3, false),
                "^" => (4, false),
                "&" => (5, false),
                "==" | "!=" => (6, false),
                "<" | "<=" | ">" | ">=" => (7, false),
                "<<" | ">>" => (8, false),
                "+" | "-" => (9, false),
                "*" | "/" | "%" => (10, false),
                _ => break,
            };

            if prec < min_prec {
                break;
            }

            let op = match op_str.as_str() {
                "||" => BinaryOp::LogOr, "&&" => BinaryOp::LogAnd,
                "|" => BinaryOp::Or, "^" => BinaryOp::Xor, "&" => BinaryOp::And,
                "==" => BinaryOp::Eq, "!=" => BinaryOp::Ne,
                "<" => BinaryOp::Lt, "<=" => BinaryOp::Le, ">" => BinaryOp::Gt, ">=" => BinaryOp::Ge,
                "<<" => BinaryOp::Shl, ">>" => BinaryOp::Shr,
                "+" => BinaryOp::Add, "-" => BinaryOp::Sub,
                "*" => BinaryOp::Mul, "/" => BinaryOp::Div, "%" => BinaryOp::Rem,
                _ => unreachable!(),
            };

            self.consume(); // consume op
            let next_min_prec = if assoc_right { prec } else { prec + 1 };
            let rhs = self.parse_expr(next_min_prec)?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }

        Ok(lhs)
    }

    fn parse_atom(&mut self) -> Result<Expr, String> {
        match self.consume() {
            Some(Token::Literal(val)) => Ok(Expr::Literal(val)),
            Some(Token::Identifier(name)) => {
                if let Some(reg) = parse_reg_target(&name) {
                    Ok(Expr::Register(reg))
                } else {
                    Ok(Expr::Symbol(name))
                }
            }
            Some(Token::LParen) => {
                let expr = self.parse_expr(0)?;
                if let Some(Token::RParen) = self.consume() {
                    Ok(expr)
                } else {
                    Err("Expected ')'".to_string())
                }
            }
            Some(Token::Op(op)) => {
                // Unary operators
                let unary_op = match op.as_str() {
                    "-" => UnaryOp::Neg,
                    "!" => UnaryOp::Not,
                    "~" => UnaryOp::BitNot,
                    _ => return Err(format!("Unexpected operator: {}", op)),
                };
                let expr = self.parse_atom()?; // Right-associative unary
                Ok(Expr::Unary(unary_op, Box::new(expr)))
            }
            Some(t) => Err(format!("Unexpected token: {:?}", t)),
            None => Err("Unexpected end of expression".to_string()),
        }
    }
}

pub fn parse_and_eval(input: &str, core: &MipsCore, symbols: Option<&SymbolTable>) -> Result<u64, String> {
    let expr = Expr::parse(input)?;
    let folded = expr.fold(symbols);
    folded.eval(core, symbols)
}

pub fn parse_and_fold(input: &str, symbols: Option<&SymbolTable>) -> Result<Expr, String> {
    let expr = Expr::parse(input)?;
    Ok(expr.fold(symbols))
}

/// Parse and evaluate a constant expression (no CPU registers, no symbol table).
/// Useful for command arguments that are numeric expressions like "0xC00" or "1<<10".
pub fn eval_const_expr(input: &str) -> Result<u64, String> {
    let expr = Expr::parse(input)?;
    let folded = expr.fold(None);
    match folded {
        Expr::Literal(val) => Ok(val),
        _ => Err(format!("Expression '{}' could not be reduced to a constant", input)),
    }
}