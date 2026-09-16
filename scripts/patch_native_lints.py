from pathlib import Path

memory = Path("crates/cortex-memory/src/lib.rs")
s = memory.read_text()
old = """            } else if self.prototypes.len() < self.cfg.budget {
                self.append(x);
                structural_change = true;
            } else if self.certified_recompress() {
                self.append(x);
                structural_change = true;
            } else {
"""
new = """            } else if self.prototypes.len() < self.cfg.budget || self.certified_recompress() {
                // Short-circuiting preserves the v0.8 rule: recompression is
                // attempted only when the memory budget is already full.
                self.append(x);
                structural_change = true;
            } else {
"""
if old not in s:
    raise SystemExit("memory lint pattern not found")
memory.write_text(s.replace(old, new))

art = Path("crates/cortex-articulation/src/lib.rs")
s = art.read_text()
old = "if assignment.iter().any(|&j| j == usize::MAX) {"
new = "if assignment.contains(&usize::MAX) {"
if old not in s:
    raise SystemExit("articulation lint pattern not found")
art.write_text(s.replace(old, new))
