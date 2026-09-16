mod math;
mod reasoner;
mod state;

pub use reasoner::CortexReasoner;
pub use state::{Config, Read, Snapshot};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
