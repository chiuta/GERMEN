//! Library-level reproductions of findings of docs/AUDIT.md that no command-line option reaches.
//!
//! The same source runs against the 0.2.0 library and the current one (copy it into the
//! `examples` folder of each and `cargo run --release -p germen-core --example audit_lib`); on
//! 0.2.0 it shows seeds 2k and 2k+1 giving one stream (A-14), `split_point` returning offsets past
//! the end of the text (A-10) and `GptConfig::validate` accepting absurd values (A-02).
use germen_core::config::GptConfig;
use germen_core::data::{Rng, split_point};

fn main() {
    // A-14: seeds that differ only in the lowest bit.
    let mut same = 0;
    for k in 0..1000u64 {
        let (mut a, mut b) = (Rng::new(2 * k), Rng::new(2 * k + 1));
        if (0..8).all(|_| a.next_u64() == b.next_u64()) {
            same += 1;
        }
    }
    println!("A-14 rng: {same} of 1000 pairs of seeds (2k, 2k+1) give identical streams");

    // A-10: the split between training and validation text.
    for f in [0.05f64, 2.0, -1.0, f64::NAN, f64::INFINITY] {
        let r = std::panic::catch_unwind(|| split_point(100, f));
        match r {
            Ok(v) => println!(
                "A-10 split_point(100, {f}) = {v}{}",
                if v > 100 { "   <-- out of range" } else { "" }
            ),
            Err(_) => println!("A-10 split_point(100, {f}) = panic"),
        }
    }

    // A-02: a configuration file with absurd dimensions.
    let json = r#"{"vocab_size":256,"block_size":64,"n_layer":4,"n_head":4,"d_model":1099511627776,"d_ff":512,"rope_theta":10000.0,"norm_eps":0.00001}"#;
    let cfg: GptConfig = serde_json::from_str(json).unwrap();
    println!(
        "A-02 validate() on d_model = 2^40: {:?}",
        cfg.validate().map(|_| "accepted")
    );
    let json = r#"{"vocab_size":256,"block_size":64,"n_layer":4294967296,"n_head":4,"d_model":128,"d_ff":512,"rope_theta":10000.0,"norm_eps":0.00001}"#;
    let cfg: GptConfig = serde_json::from_str(json).unwrap();
    println!(
        "A-02 validate() on n_layer = 2^32: {:?}",
        cfg.validate().map(|_| "accepted")
    );
    let json = r#"{"vocab_size":256,"block_size":64,"n_layer":0,"n_head":4,"d_model":128,"d_ff":512,"rope_theta":10000.0,"norm_eps":0.00001}"#;
    let cfg: GptConfig = serde_json::from_str(json).unwrap();
    println!(
        "A-02 validate() on n_layer = 0: {:?}",
        cfg.validate().map(|_| "accepted")
    );
    let json = r#"{"vocab_size":256,"block_size":64,"n_layer":4,"n_head":4,"d_model":128,"d_ff":512,"rope_theta":-1.0,"norm_eps":0.0}"#;
    let cfg: GptConfig = serde_json::from_str(json).unwrap();
    println!(
        "A-02 validate() on rope_theta = -1, norm_eps = 0: {:?}",
        cfg.validate().map(|_| "accepted")
    );
}
