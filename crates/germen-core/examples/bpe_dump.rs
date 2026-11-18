//! Trains the BPE vocabulary of GERMEN on a file and writes the merges and the tokens of the file
//! (and of an optional second, "probe" file) as JSON, for `scripts/reference_bpe.py` to check with
//! an independent, deliberately naive implementation (docs/AUDIT.md, round 4).
//!
//! `cargo run --release -p germen-core --example bpe_dump -- TRAIN.txt VOCAB OUT.json [PROBE.txt]`
use germen_core::bpe::Bpe;
use serde_json::json;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (train, vocab, out) = (&a[1], a[2].parse::<usize>().unwrap(), &a[3]);
    let data = std::fs::read(train).unwrap();
    let bpe = Bpe::train(&data, vocab, &mut |_, _| {}).unwrap();
    let probe = a
        .get(4)
        .map(|p| std::fs::read(p).unwrap())
        .unwrap_or_default();
    let mut back = Vec::new();
    for id in bpe.encode(&data) {
        back.extend_from_slice(bpe.token_bytes(id));
    }
    let decoded_ok = back == data;
    let doc = json!({
        "merges": bpe.merges(),
        "train_tokens": bpe.encode(&data),
        "probe_tokens": bpe.encode(&probe),
        "decoded_ok": decoded_ok,
    });
    std::fs::write(out, serde_json::to_string(&doc).unwrap()).unwrap();
    println!("{} merges", bpe.merges().len());
}
